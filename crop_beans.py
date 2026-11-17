"""แยกเมล็ดกาแฟออกจากภาพทีละเม็ด (OpenCV ล้วน ใช้ได้ทั้งตอน train และ infer)

ใช้งาน:
    python crop_beans.py --data . --out crops

อ่านภาพจากโฟลเดอร์คลาส <data>/<class>/*.jpg แล้วบันทึก
    crops/<class>/<ชื่อภาพ>__NN.jpg   ภาพ crop รายเม็ด (label สืบทอดจากภาพ)
    crops/index.csv                  ข้อมูล crop + ภาพต้นทาง (ใช้แบ่ง split ตามภาพ)
    crops/_debug/<class>_<ชื่อภาพ>.jpg ภาพวาดกรอบไว้ตรวจว่าแยกเม็ดถูกไหม
    crops/seg_stats.json             ขนาดเม็ดเดี่ยว (ref_area) ให้ infer ใช้แยกเม็ดในกอง

ฟังก์ชัน load_image() และ extract_beans() ให้ infer.py import ไปใช้ซ้ำได้
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

CLASSES = ["good", "shell", "black", "husk", "broken", "parchment", "immature"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}

WORK_SIZE = 1536        # ด้านยาวของภาพที่ใช้หา mask (เม็ด ~35-40 px ที่ขนาดนี้)
MIN_AREA_FRAC = 0.15    # ตัดชิ้นที่เล็กกว่า 15% ของขนาดเม็ดปกติ (ฝุ่น/จุด)
CROP_PAD = 0.25         # ขยายกรอบรอบเม็ด 25%
SPLIT_AREA_RATIO = 1.6  # ก้อนที่ใหญ่กว่า 1.6 เท่าของเม็ดปกติ ถือว่าเป็นหลายเม็ดติดกัน
WINDOW_STRIDE = 0.6     # ระยะเลื่อนหน้าต่าง (เท่าของเส้นผ่านศูนย์กลางเม็ด) สำหรับกองที่แยกไม่ออก
DEFAULT_REF_AREA = 3500 # ขนาดเม็ดเดี่ยวโดยประมาณ (px ที่ WORK_SIZE) ใช้เมื่อยังไม่รู้ ref_area
MAX_HOLE_FRAC = 0.5    # เติมรูในเม็ดที่เล็กกว่า 50% ของเม็ด (ช่องว่างระหว่างเม็ดในกองไม่ถูกเติม)


def load_image(path):
    """อ่านภาพ + ใช้ EXIF orientation แล้วคืนเป็น BGR (uint8)"""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def _fit_background(L, iters=3):
    """ประมาณความสว่างพื้นกระดาษด้วยพื้นผิวกำลังสอง (รองรับแสงไม่สม่ำเสมอ/เงา)"""
    h, w = L.shape
    step = max(1, min(h, w) // 200)
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    v = L[::step, ::step].ravel()
    x = xs.ravel() / w
    y = ys.ravel() / h
    A = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], 1)
    keep = np.ones_like(v, dtype=bool)
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(A[keep], v[keep], rcond=None)
        resid = v - A @ coef
        sd = np.std(resid[keep]) + 1e-6
        keep = resid > -2.5 * sd          # ตัดพิกเซลที่มืดกว่าพื้น (เมล็ด) ออกจากการ fit
    gy, gx = np.mgrid[0:h, 0:w]
    gx = gx / w
    gy = gy / h
    return (coef[0] + coef[1] * gx + coef[2] * gy + coef[3] * gx * gx
            + coef[4] * gx * gy + coef[5] * gy * gy), keep, step


def bean_mask(small, max_hole=MAX_HOLE_FRAC * DEFAULT_REF_AREA):
    """mask ของเมล็ดจากภาพย่อ: เมล็ด = มืดกว่าพื้น หรือ มีสีต่างจากพื้นกระดาษ"""
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab = cv2.GaussianBlur(lab, (5, 5), 0)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    bgL, keep, step = _fit_background(L)
    bg_a = np.median(a[::step, ::step].ravel()[keep])
    bg_b = np.median(b[::step, ::step].ravel()[keep])

    dark = np.clip(bgL - L, 0, None)
    chroma = np.sqrt((a - bg_a) ** 2 + (b - bg_b) ** 2)
    score = dark + 2.0 * chroma
    score8 = np.clip(score * 2, 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(score8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(thr, 30)                    # กัน Otsu ต่ำเกินเมื่อพื้นเยอะมาก
    mask = (score8 > thr).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    # เติมรูเล็กในเม็ด (เช่น ร่องกลางสีอ่อน / เมล็ดกะลาสีอ่อน)
    # รูใหญ่ = ช่องว่างระหว่างเม็ดในกอง → ไม่เติม เพื่อให้ watershed ยังแยกได้
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is not None:
        for c, h in zip(cnts, hier[0]):
            if h[3] >= 0 and cv2.contourArea(c) < max_hole:
                cv2.drawContours(mask, [c], -1, 255, cv2.FILLED)
    return mask


def _components(mask):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    h, w = mask.shape
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches = x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1
        out.append((i, area, touches))
    return lab, out


def _ref_area(mask, lab, comps):
    """ขนาดเม็ดเดี่ยวโดยประมาณ: median ของชิ้นที่นูน (solidity สูง) ไม่แตะขอบภาพ"""
    areas = []
    for i, area, touches in comps:
        if touches:
            continue
        m = (lab == i).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.contourArea(cv2.convexHull(cnts[0]))
        if hull > 0 and area / hull > 0.9:
            areas.append(area)
    if not areas:
        areas = [a for _, a, t in comps if not t]
    return float(np.median(areas)) if areas else None


def split_touching(mask, ref_area):
    """แยกเม็ดที่ติดกันด้วย distance transform + watershed"""
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    r = np.sqrt(ref_area / np.pi)
    win = max(3, int(r * 0.9) | 1)
    peaks = (dist == cv2.dilate(dist, np.ones((win, win), np.uint8))) & (dist > 0.45 * r)
    peaks = cv2.dilate(peaks.astype(np.uint8), np.ones((3, 3), np.uint8))
    n, markers = cv2.connectedComponents(peaks)
    markers = markers + 1
    markers[(mask > 0) & (peaks == 0)] = 0      # บริเวณที่ยังไม่รู้ ให้ watershed ตัดสิน
    markers[mask == 0] = 1                       # พื้นหลัง
    color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color, markers.astype(np.int32))
    return markers                               # 1 = พื้น, -1 = เส้นแบ่ง, >=2 = เม็ด


def extract_beans(bgr, out_size=256, ref_area=None, return_boxes=False):
    """คืน list ของภาพ crop รายเม็ด (BGR, out_size x out_size)

    ref_area: ขนาดเม็ดเดี่ยว (พิกเซลที่ WORK_SIZE) ถ้า None จะประมาณจากภาพเอง
    """
    H, W = bgr.shape[:2]
    s = WORK_SIZE / max(H, W)
    small = cv2.resize(bgr, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
    mask = bean_mask(small, MAX_HOLE_FRAC * (ref_area or DEFAULT_REF_AREA))
    lab, comps = _components(mask)
    # ทิ้งชิ้นที่แตะขอบภาพ (ขอบโต๊ะ/เงา) ออกก่อน
    for i, _, touches in comps:
        if touches:
            mask[lab == i] = 0
    lab, comps = _components(mask)
    if ref_area is None:
        ref_area = _ref_area(mask, lab, comps)
    if ref_area is None:
        return ([], []) if return_boxes else []

    # ชิ้นขนาดเม็ดเดียวไม่ต้องแยก (กันแยกเม็ดยาวเป็น 2 ชิ้น) · watershed เฉพาะก้อนใหญ่
    markers = np.ones(mask.shape, np.int32)
    big = np.zeros_like(mask)
    next_id = 2
    for i, area, _ in comps:
        if area > SPLIT_AREA_RATIO * ref_area:
            big[lab == i] = 255
        else:
            markers[lab == i] = next_id
            next_id += 1
    if big.any():
        ws = split_touching(big, ref_area)
        sel = ws >= 2
        markers[sel] = ws[sel] + next_id
    # รวมรายการกรอบ (cx, cy, side, area) ในพิกัดภาพย่อ
    diam = 2 * np.sqrt(ref_area / np.pi)
    cands = []
    for m_id in np.unique(markers):
        if m_id <= 1:
            continue
        region = (markers == m_id).astype(np.uint8)
        area = int(region.sum())
        if area < MIN_AREA_FRAC * ref_area:
            continue
        x, y, bw, bh = cv2.boundingRect(region)
        if area <= SPLIT_AREA_RATIO * ref_area:
            cands.append((x + bw / 2, y + bh / 2, max(bw, bh), area))
            continue
        # ยังเป็นก้อนใหญ่ (กองแน่น แยกไม่ออก) → เลื่อนหน้าต่างขนาดเม็ดทั่วก้อน
        stride = WINDOW_STRIDE * diam
        half = diam / 2
        for wy in np.arange(y + half, y + bh - half + 1e-6, stride) if bh > diam else [y + bh / 2]:
            for wx in np.arange(x + half, x + bw - half + 1e-6, stride) if bw > diam else [x + bw / 2]:
                y0, y1 = int(wy - half), int(wy + half)
                x0, x1 = int(wx - half), int(wx + half)
                cover = region[max(0, y0):y1, max(0, x0):x1].mean()
                if cover >= 0.5:
                    cands.append((wx, wy, diam, int(cover * diam * diam)))

    crops, boxes = [], []
    for cxs, cys, side_s, area in cands:
        cx, cy = cxs / s, cys / s
        side = side_s / s * (1 + 2 * CROP_PAD)
        x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
        x1, y1 = int(round(cx + side / 2)), int(round(cy + side / 2))
        # crop แบบ pad ด้วยสีขอบ ถ้ากรอบล้นภาพ
        crop = cv2.copyMakeBorder(
            bgr[max(0, y0):min(H, y1), max(0, x0):min(W, x1)],
            max(0, -y0), max(0, y1 - H), max(0, -x0), max(0, x1 - W),
            cv2.BORDER_REPLICATE)
        crops.append(cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA))
        boxes.append((x0, y0, x1 - x0, y1 - y0, area))
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i][1], boxes[i][0]))
    crops = [crops[i] for i in order]
    boxes = [boxes[i] for i in order]
    return (crops, boxes) if return_boxes else crops


def draw_debug(bgr, boxes, path, label=""):
    vis = bgr.copy()
    t = max(2, bgr.shape[1] // 600)
    for k, (x, y, w, h, _) in enumerate(boxes):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), t)
        cv2.putText(vis, str(k), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, t * 0.6, (0, 0, 255), t)
    cv2.putText(vis, f"{label} n={len(boxes)}", (20, 20 + 30 * t),
                cv2.FONT_HERSHEY_SIMPLEX, t * 0.9, (255, 0, 0), t)
    s = 1200 / max(vis.shape[:2])
    vis = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), vis)


def main():
    ap = argparse.ArgumentParser(description="crop เมล็ดกาแฟรายเม็ดจากภาพ train")
    ap.add_argument("--data", default=".", help="โฟลเดอร์ที่มีโฟลเดอร์ย่อยตามคลาส")
    ap.add_argument("--out", default="crops", help="โฟลเดอร์ผลลัพธ์")
    ap.add_argument("--size", type=int, default=256, help="ขนาดภาพ crop")
    args = ap.parse_args()

    data, out = Path(args.data), Path(args.out)
    (out / "_debug").mkdir(parents=True, exist_ok=True)
    rows = []
    for cls in CLASSES:
        cdir = data / cls
        if not cdir.is_dir():
            print(f"[warn] ไม่พบโฟลเดอร์คลาส {cdir}", file=sys.stderr)
            continue
        (out / cls).mkdir(parents=True, exist_ok=True)
        for p in sorted(cdir.iterdir()):
            if p.suffix.lower() not in IMG_EXTS:
                continue
            bgr = load_image(p)
            crops, boxes = extract_beans(bgr, args.size, return_boxes=True)
            for k, (c, b) in enumerate(zip(crops, boxes)):
                name = f"{p.stem}__{k:02d}.jpg"
                cv2.imwrite(str(out / cls / name), c, [cv2.IMWRITE_JPEG_QUALITY, 95])
                rows.append([f"{cls}/{name}", cls, f"{cls}/{p.name}", *b])
            draw_debug(bgr, boxes, out / "_debug" / f"{cls}_{p.stem}.jpg", f"{cls}/{p.name}")
            print(f"{cls}/{p.name}: {len(crops)} เม็ด")

    with open(out / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["crop", "class", "source", "x", "y", "w", "h", "area"])
        w.writerows(rows)

    # ขนาดเม็ดเดี่ยวจากภาพ train → ใช้ตอน infer ภาพกอง (ซึ่งประมาณจากภาพเองไม่ได้)
    if rows:
        stats = {"work_size": WORK_SIZE, "ref_area": float(np.median([r[-1] for r in rows]))}
        with open(out / "seg_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"ref_area = {stats['ref_area']:.0f} px (ที่ WORK_SIZE={WORK_SIZE}) -> {out / 'seg_stats.json'}")
    print(f"รวม {len(rows)} crops -> {out}")


if __name__ == "__main__":
    main()
