"""ทำนายคลาสเมล็ดกาแฟ 1 คลาสต่อภาพ (Day 1) · รันบน CPU (Raspberry Pi)

    python infer.py --input test/ --weights weights/ --out outputs/

--input   โฟลเดอร์ภาพทดสอบ หรือไฟล์ภาพไฟล์เดียว
--weights โฟลเดอร์ weights (หาไฟล์โมเดลเอง) หรือ path ไฟล์ .pt โดยตรงก็ได้
--out     โฟลเดอร์ผลลัพธ์ (สร้างให้ถ้ายังไม่มี)

ขั้นตอนต่อภาพ: อ่านภาพ + EXIF orientation → แยกเม็ด (crop_beans.extract_beans)
→ จำแนกทีละเม็ด → เฉลี่ยความน่าจะเป็นทุกเม็ด → คลาสที่สูงสุดคือผลของภาพ
ไม่มี augmentation / TTA และไม่ปรับโมเดลระหว่างรัน

ต้องมี crop_beans.py อยู่โฟลเดอร์เดียวกับไฟล์นี้
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torchvision as tv
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crop_beans import extract_beans, load_image  # noqa: E402

CLASSES = ["good", "shell", "black", "husk", "broken", "parchment", "immature"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}
WEIGHT_EXTS = (".pt", ".pth")
PREFERRED_WEIGHT = "model.pt"
CROP_SIZE = 256               # ต้องตรงกับ --size ของ crop_beans.py ตอน train


class InferError(Exception):
    pass


# ----------------------------------------------------------------------------- weights
def find_weight_file(weights):
    """รับโฟลเดอร์หรือไฟล์ แล้วคืน path ไฟล์ weight ที่จะใช้ (เลือกแบบคงที่ทุกครั้ง)"""
    p = Path(weights)
    if p.is_file():
        return p
    if not p.is_dir():
        raise InferError(f"ไม่พบ weights: {p}")
    if (p / PREFERRED_WEIGHT).is_file():
        return p / PREFERRED_WEIGHT
    cands = sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in WEIGHT_EXTS)
    if not cands:
        raise InferError(f"ไม่พบไฟล์ weight ({', '.join(WEIGHT_EXTS)}) ใน {p}")
    if len(cands) > 1:
        print(f"[warn] พบ weight หลายไฟล์ เลือก {cands[0].name}", file=sys.stderr)
    return cands[0]


def build_model(arch, num_classes):
    m = getattr(tv.models, arch)(weights=None)
    if arch.startswith(("efficientnet", "mobilenet")):
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif arch.startswith(("resnet", "shufflenet")):
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise InferError(f"ไม่รองรับ arch {arch}")
    return m


def load_model(weights):
    """โหลดโมเดล + config จากโฟลเดอร์ weights (หรือไฟล์ .pt)"""
    wfile = find_weight_file(weights)
    ckpt = torch.load(wfile, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        cfg = {k: v for k, v in ckpt.items() if k != "state_dict"}
        state = ckpt["state_dict"]
    else:                                   # state_dict ล้วน → อ่าน config.json ข้างกัน
        cfg_path = wfile.parent / "config.json"
        if not cfg_path.is_file():
            raise InferError(f"{wfile.name} ไม่มี config และไม่พบ {cfg_path}")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        state = ckpt
    classes = cfg.get("classes", CLASSES)
    if sorted(classes) != sorted(CLASSES):
        raise InferError(f"รายชื่อคลาสใน weights ไม่ตรงกับ spec: {classes}")

    model = build_model(cfg["arch"], len(classes))
    model.load_state_dict(state)
    model.eval()
    size = int(cfg.get("img_size", 224))
    tf = T.Compose([T.ToTensor(), T.Resize((size, size), antialias=True),
                    T.Normalize(cfg.get("mean", [0.485, 0.456, 0.406]),
                                cfg.get("std", [0.229, 0.224, 0.225]))])
    print(f"weights: {wfile}  arch={cfg['arch']}  img_size={size}  "
          f"ref_area={cfg.get('ref_area')}", file=sys.stderr)
    return {"model": model, "classes": classes, "tf": tf, "ref_area": cfg.get("ref_area")}


# ----------------------------------------------------------------------------- predict
@torch.no_grad()
def predict_image(bundle, path):
    """คืน (ชื่อคลาส, จำนวนเม็ดที่ใช้โหวต)"""
    try:
        bgr = load_image(path)
    except Exception as e:
        raise InferError(f"อ่านภาพไม่ได้: {path} ({e})") from e

    crops = extract_beans(bgr, CROP_SIZE, ref_area=bundle["ref_area"])
    if not crops:
        # หาเม็ดไม่เจอ: ใช้กลางภาพแทน (ไม่ข้ามภาพ) และแจ้งเตือน
        print(f"[warn] {Path(path).name}: แยกเม็ดไม่ได้ ใช้บริเวณกลางภาพแทน", file=sys.stderr)
        h, w = bgr.shape[:2]
        s = min(h, w) // 3
        crops = [cv2.resize(bgr[h // 2 - s:h // 2 + s, w // 2 - s:w // 2 + s], (CROP_SIZE, CROP_SIZE))]

    x = torch.stack([bundle["tf"](cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops])
    probs = torch.cat([bundle["model"](b).softmax(1) for b in x.split(32)])
    k = int(probs.mean(0).argmax())
    return bundle["classes"][k], len(crops)


# ----------------------------------------------------------------------------- main
def list_images(inp):
    p = Path(inp)
    if p.is_file():
        if p.suffix.lower() not in IMG_EXTS:
            raise InferError(f"ไฟล์ไม่ใช่ภาพที่รองรับ: {p}")
        return [p]
    if not p.is_dir():
        raise InferError(f"ไม่พบ input: {p}")
    imgs = sorted((f for f in p.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS),
                  key=lambda f: f.name)
    if not imgs:
        raise InferError(f"ไม่พบภาพ (.jpg/.jpeg/.png) ใน {p}")
    names = [f.name for f in imgs]
    if len(set(names)) != len(names):
        raise InferError("พบชื่อไฟล์ input ซ้ำกัน")
    return imgs


def run(args):
    images = list_images(args.input)
    bundle = load_model(args.weights)

    out = Path(args.out)
    if out.resolve() == Path(args.input).resolve():
        raise InferError("--out ต้องแยกจาก --input")
    for c in CLASSES:
        (out / c).mkdir(parents=True, exist_ok=True)

    rows = []
    for img in images:
        cls, n = predict_image(bundle, img)
        shutil.copy2(img, out / cls / img.name)
        rows.append((img.name, cls))
        print(f"{img.name},{cls}  ({n} เม็ด)", file=sys.stderr)

    # ตรวจรูปแบบผลลัพธ์ก่อนเขียน
    if len(rows) != len(images) or len({r[0] for r in rows}) != len(rows):
        raise InferError("จำนวนแถวผลลัพธ์ไม่ตรงกับจำนวนภาพ")
    if any(c not in CLASSES for _, c in rows):
        raise InferError("พบชื่อคลาสที่ไม่อยู่ใน spec")
    with open(out / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "class"])
        w.writerows(rows)
    print(f"เสร็จ {len(rows)} ภาพ -> {out / 'predictions.csv'}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="Coffee bean classification (Day 1)")
    ap.add_argument("--input", required=True, help="โฟลเดอร์ภาพทดสอบ หรือไฟล์ภาพ")
    ap.add_argument("--weights", required=True, help="โฟลเดอร์ weights/ (หรือไฟล์ .pt)")
    ap.add_argument("--out", required=True, help="โฟลเดอร์บันทึกผล")
    ap.add_argument("--threads", type=int, default=4, help="จำนวน CPU threads")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    try:
        run(args)
    except InferError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
