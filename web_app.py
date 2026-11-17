"""หน้าเว็บทดสอบโมเดลด้วยกล้องมือถือ (ไม่ใช่ส่วนของงานส่ง)

    pip install -r requirements-web.txt
    python web_app.py --weights weights/

แล้วเปิด URL ที่พิมพ์ออกมาบนมือถือที่ต่อ Wi-Fi วงเดียวกับคอม
ถ่ายรูป → คอมแยกเม็ด + ทำนาย → ได้ภาพวาดกรอบสีตามคลาสกลับมา
"""
import argparse
import base64
import socket
import sys
import threading
import traceback
from collections import Counter

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image, ImageOps

from crop_beans import extract_beans
from infer import CROP_SIZE, InferError, load_model

RESULT_MAX_SIDE = 1280
MAX_UPLOAD_MB = 40
MIN_SIDE = 32           # ภาพที่ด้านสั้นเล็กกว่านี้ไม่ใช่ภาพเมล็ดจริง
# สี BGR ของแต่ละคลาส (แยกกันชัดบนพื้นกระดาษขาว)
CLASS_COLORS = {
    "good": (60, 180, 75),
    "shell": (200, 130, 0),
    "black": (40, 40, 40),
    "husk": (0, 140, 255),
    "broken": (48, 48, 230),
    "parchment": (180, 30, 145),
    "immature": (0, 200, 200),
}

_model_lock = threading.Lock()
HEIC_HINT = "ถ้ามือถือบันทึกภาพเป็น HEIC ให้ตั้งกล้องเป็น JPEG / ‘Most Compatible’"


class UnreadableImage(ValueError):
    """ไฟล์ไม่ใช่ภาพที่ Pillow อ่านได้ (เช่น HEIC)"""


def read_upload(fileobj):
    """อ่านภาพ + EXIF orientation → BGR · ภาพมีพื้นโปร่งใสจะวางบนพื้นขาว (ไม่ใช่ดำ)"""
    with Image.open(fileobj) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        return cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)


def _center_crop(bgr):
    """ใช้กลางภาพแทนเมื่อแยกเม็ดไม่ได้ (แบบเดียวกับ infer.py) · คืน (crop, box)"""
    h, w = bgr.shape[:2]
    s = min(h, w) // 3
    y0, x0 = h // 2 - s, w // 2 - s
    crop = cv2.resize(bgr[y0:y0 + 2 * s, x0:x0 + 2 * s], (CROP_SIZE, CROP_SIZE))
    return crop, (x0, y0, 2 * s, 2 * s, 0)


def _label(img, text, x, y, color, scale, thick):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = max(x, 0), max(y, th + base + 2)       # ไม่ให้ป้ายล้นขอบซ้าย/บน
    cv2.rectangle(img, (x, y - th - base - 2), (x + tw + 4, y), color, cv2.FILLED)
    cv2.putText(img, text, (x + 2, y - base), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), thick, cv2.LINE_AA)


def draw_result(bgr, boxes, bean_classes, cls, conf):
    """วาดกรอบ + ชื่อคลาสรายเม็ด และแถบผลรวมด้านบน · คืนภาพย่อ (ด้านยาว <= 1280)"""
    s = min(1.0, RESULT_MAX_SIDE / max(bgr.shape[:2]))
    vis = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    t = max(2, max(vis.shape[:2]) // 400)
    for (x, y, w, h, _), c in zip(boxes, bean_classes):
        x0, y0, x1, y1 = (int(round(v * s)) for v in (x, y, x + w, y + h))
        cv2.rectangle(vis, (x0, y0), (x1, y1), CLASS_COLORS[c], t)
        _label(vis, c, x0, y0, CLASS_COLORS[c], 0.25 * t, max(1, t // 2))
    _label(vis, f"{cls.upper()} {conf:.0%}", 0, 0, CLASS_COLORS[cls], 0.5 * t, t)
    return vis


def analyze(bundle, fileobj):
    """วิเคราะห์ภาพที่อัปโหลด · คืน dict: cls, confidence, counts, n_beans, warning, image(base64 JPEG)
    อ่านภาพไม่ได้ → ValueError"""
    try:
        bgr = read_upload(fileobj)
    except Exception as e:
        raise UnreadableImage(f"อ่านภาพไม่ได้ ({type(e).__name__})") from e
    if min(bgr.shape[:2]) < MIN_SIDE:
        raise ValueError(f"ภาพเล็กเกินไป ({bgr.shape[1]}x{bgr.shape[0]} px)")

    crops, boxes = extract_beans(bgr, CROP_SIZE, ref_area=bundle["ref_area"], return_boxes=True)
    warning = None
    if not crops:
        crop, box = _center_crop(bgr)
        crops, boxes = [crop], [box]
        warning = "แยกเม็ดไม่ได้ ใช้บริเวณกลางภาพทำนายแทน"

    x = torch.stack([bundle["tf"](cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops])
    with _model_lock, torch.no_grad():
        probs = torch.cat([bundle["model"](b).softmax(1) for b in x.split(32)])
    classes = bundle["classes"]
    mean_p = probs.mean(0)
    k = int(mean_p.argmax())
    bean_classes = [classes[i] for i in probs.argmax(1).tolist()]

    vis = draw_result(bgr, boxes, bean_classes, classes[k], float(mean_p[k]))
    ok, jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("แปลงภาพผลเป็น JPEG ไม่สำเร็จ")
    return {
        "cls": classes[k],
        "confidence": float(mean_p[k]),
        "counts": dict(Counter(bean_classes).most_common()),
        "n_beans": len(crops),
        "warning": warning,
        "image": base64.b64encode(jpg.tobytes()).decode("ascii"),
    }


# ----------------------------------------------------------------------------- web
PAGE = """<!doctype html>
<html lang="th">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ทดสอบเมล็ดกาแฟ</title>
<style>
  :root { --bg:#f6f3ee; --card:#fff; --ink:#2b2118; --muted:#7a6a5a; --accent:#6b4226; --err:#c0392b; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:system-ui,-apple-system,"Segoe UI",sans-serif; background:var(--bg); color:var(--ink); }
  main { max-width:640px; margin:0 auto; padding:16px; }
  h1 { font-size:1.3rem; margin:8px 0 16px; }
  .btn { display:flex; align-items:center; justify-content:center; width:100%; min-height:56px;
         margin-bottom:10px; border-radius:12px; font-size:1.1rem; font-weight:600; cursor:pointer;
         border:2px solid var(--accent); background:var(--accent); color:#fff; }
  .btn.alt { background:transparent; color:var(--accent); }
  .btn input { display:none; }
  #status { color:var(--muted); text-align:center; min-height:1.5em; margin:12px 0; }
  .card { background:var(--card); border-radius:14px; padding:16px; box-shadow:0 1px 4px #0001; }
  #cls { font-size:2.2rem; font-weight:800; letter-spacing:.02em; }
  #conf { color:var(--muted); }
  #counts { display:flex; flex-wrap:wrap; gap:6px; margin:10px 0; }
  .chip { padding:4px 10px; border-radius:999px; color:#fff; font-size:.9rem; }
  #result img { width:100%; border-radius:10px; margin-top:8px; }
  .warn { color:#a66300; margin-top:6px; }
  .error { background:#fdecea; color:var(--err); border-radius:12px; padding:12px; }
  [hidden] { display:none !important; }
</style>
</head>
<body>
<main>
  <h1>☕ ทดสอบจำแนกเมล็ดกาแฟ</h1>
  <label class="btn">📷 ถ่ายรูป
    <input type="file" accept="image/*" capture="environment">
  </label>
  <label class="btn alt">🖼️ เลือกจากคลังภาพ
    <input type="file" accept="image/*">
  </label>
  <div id="status"></div>
  <div id="error" class="error" hidden></div>
  <div id="result" class="card" hidden>
    <div id="cls"></div>
    <div id="conf"></div>
    <div id="warn" class="warn" hidden></div>
    <div id="counts"></div>
    <img id="img" alt="ผลการตรวจจับรายเม็ด">
  </div>
</main>
<script>
const COLORS = __COLORS__;
const $ = id => document.getElementById(id);
const inputs = document.querySelectorAll("input[type=file]");
function busy(on) {               // กันส่งรูปซ้อนระหว่างรอผล (ผลรูปเก่าจะไม่ทับรูปใหม่)
  inputs.forEach(i => { i.disabled = on; i.parentElement.style.opacity = on ? 0.5 : 1; });
}
async function send(file) {
  busy(true);
  $("error").hidden = true; $("result").hidden = true;
  $("status").textContent = "กำลังส่งรูปและทำนาย…";
  const fd = new FormData(); fd.append("image", file);
  try {
    const r = await fetch("predict", { method: "POST", body: fd });
    const j = await r.json().catch(() => ({ error: "เซิร์ฟเวอร์ตอบกลับผิดรูปแบบ (HTTP " + r.status + ")" }));
    if (!r.ok) throw Object.assign(new Error(j.error || ("HTTP " + r.status)), { hint: j.hint });
    $("cls").textContent = j.cls.toUpperCase();
    $("cls").style.color = COLORS[j.cls];
    $("conf").textContent = "ความมั่นใจ " + Math.round(j.confidence * 100) + "% · " + j.n_beans + " เม็ด";
    $("warn").hidden = !j.warning; $("warn").textContent = j.warning ? "⚠️ " + j.warning : "";
    $("counts").replaceChildren(...Object.entries(j.counts).map(([c, n]) => {
      const s = document.createElement("span"); s.className = "chip";
      s.style.background = COLORS[c]; s.textContent = c + " " + n; return s; }));
    $("img").src = "data:image/jpeg;base64," + j.image;
    $("result").hidden = false; $("status").textContent = "";
  } catch (e) {
    $("status").textContent = "";
    $("error").textContent = "ทำนายไม่สำเร็จ: " + e.message + (e.hint ? " — " + e.hint : "");
    $("error").hidden = false;
  } finally {
    busy(false);
  }
}
inputs.forEach(inp => inp.addEventListener("change", () => {
  if (inp.files[0]) send(inp.files[0]);
  inp.value = "";
}));
</script>
</body>
</html>
"""


def _css_color(bgr):
    b, g, r = bgr
    return f"rgb({r},{g},{b})"


def create_app(bundle):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
    colors = "{" + ",".join(f'"{c}":"{_css_color(v)}"' for c, v in CLASS_COLORS.items()) + "}"
    page = PAGE.replace("__COLORS__", colors)

    @app.get("/")
    def index():
        return page

    @app.post("/predict")
    def predict():
        f = request.files.get("image")
        if f is None or not f.filename:
            return jsonify(error="ไม่พบไฟล์ภาพในคำขอ"), 400
        try:
            return jsonify(analyze(bundle, f.stream))
        except UnreadableImage as e:
            return jsonify(error=str(e), hint=HEIC_HINT), 400
        except ValueError as e:
            return jsonify(error=str(e)), 400

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error=f"ไฟล์ใหญ่เกิน {MAX_UPLOAD_MB} MB"), 413

    @app.errorhandler(500)
    def server_error(e):
        orig = getattr(e, "original_exception", None) or e
        traceback.print_exception(orig, file=sys.stderr)
        return jsonify(error=f"เกิดข้อผิดพลาดในการประมวลผลภาพ ({type(orig).__name__})"), 500

    return app


def lan_ip():
    """IP ของคอมในวง LAN (UDP connect ไม่ได้ส่งข้อมูลจริง)"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def main():
    ap = argparse.ArgumentParser(description="หน้าเว็บทดสอบโมเดลด้วยกล้องมือถือ")
    ap.add_argument("--weights", default="weights", help="โฟลเดอร์ weights/ (หรือไฟล์ .pt)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--threads", type=int, default=4, help="จำนวน CPU threads")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    try:
        bundle = load_model(args.weights)
    except InferError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"\n  เปิดบนมือถือ (Wi-Fi วงเดียวกัน): http://{lan_ip()}:{args.port}\n", flush=True)
    create_app(bundle).run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
