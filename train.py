"""Fine-tune backbone จาก ImageNet บนภาพ crop รายเม็ด (ผลจาก crop_beans.py)

ขั้นตอน:
    python crop_beans.py --data . --out crops          # 1) crop เม็ด
    python train.py --mode cv                          # 2) วัดความแม่นแบบ leave-one-layout-out
    python train.py --mode full --out weights          # 3) train ด้วยข้อมูลทั้งหมด แล้วบันทึก weights/

ผลลัพธ์ของ --mode full:
    weights/model.pt     {arch, state_dict, classes, img_size, mean, std}
    weights/config.json  ค่าเดียวกัน (ไม่มี state_dict) + ref_area สำหรับแยกเม็ดตอน infer

การ validate (--mode cv): แต่ละคลาสมี 3 ภาพ (circle / grid / square) → 3 fold
แต่ละ fold กันภาพ layout หนึ่งของทุกคลาสไว้เป็น val (7 ภาพ) แล้ว train กับอีก 2 layout
รายงานทั้งความแม่นรายเม็ด และรายภาพ (โหวตด้วยค่าเฉลี่ยความน่าจะเป็นของทุกเม็ด)
"""
import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
from torchvision import transforms as T

CLASSES = ["good", "shell", "black", "husk", "broken", "parchment", "immature"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ----------------------------------------------------------------------------- model
def build_model(arch, num_classes, pretrained=True):
    """สร้าง backbone จาก torchvision แล้วเปลี่ยนชั้นสุดท้ายเป็น num_classes
    คืน (model, backbone_params, head_params)"""
    w = "IMAGENET1K_V1" if pretrained else None
    m = getattr(tv.models, arch)(weights=w)
    if arch.startswith(("efficientnet", "mobilenet")):
        last = m.classifier[-1]
        m.classifier[-1] = nn.Linear(last.in_features, num_classes)
        head, body = m.classifier, m.features
    elif arch.startswith(("resnet", "shufflenet")):
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        head = m.fc
        body = nn.ModuleList([c for n, c in m.named_children() if n != "fc"])
    else:
        raise ValueError(f"ไม่รองรับ arch {arch}")
    return m, body, head


# ----------------------------------------------------------------------------- data
def crop_mask(img):
    """mask เมล็ดในภาพ crop (พื้นกระดาษ = ค่ามัธยฐานของขอบภาพ) ใช้ตอนแปะเม็ดข้างเคียง"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate([lab[:8].reshape(-1, 3), lab[-8:].reshape(-1, 3),
                             lab[:, :8].reshape(-1, 3), lab[:, -8:].reshape(-1, 3)])
    bg = np.median(border, 0)
    dark = np.clip(bg[0] - lab[..., 0], 0, None)
    chroma = np.sqrt((lab[..., 1] - bg[1]) ** 2 + (lab[..., 2] - bg[2]) ** 2)
    score = np.clip((dark + 2 * chroma) * 2, 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (score > max(thr, 30)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # เก็บเฉพาะชิ้นที่ใหญ่ที่สุด (ตัวเม็ดกลางภาพ)
    n, lab_cc, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (lab_cc == k).astype(np.uint8)
    return m.astype(bool)


def rand_affine(img, rng, scale=(0.85, 1.15), shift=0.08):
    """หมุน 0-360° + flip + ย่อขยาย + เลื่อน โดยเติมขอบด้วยพื้นกระดาษ (ไม่ใช่สีดำ)"""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(0, 360), rng.uniform(*scale))
    M[:, 2] += rng.uniform(-shift, shift, 2) * (w, h)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.5:
        out = out[:, ::-1]
    return np.ascontiguousarray(out)


def paste_neighbors(img, pool, rng, max_n=3):
    """จำลองกอง: แปะเม็ดอื่น (คลาสใดก็ได้) ชิด/ทับขอบเม็ดกลาง · label ยังเป็นของเม็ดกลาง"""
    h, w = img.shape[:2]
    out = img.copy()
    for _ in range(rng.integers(1, max_n + 1)):
        nb_img, nb_mask = pool[rng.integers(len(pool))]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(0, 360), rng.uniform(0.85, 1.15))
        ang = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(0.42, 0.75) * w
        M[:, 2] += (dist * np.cos(ang), dist * np.sin(ang))
        nb = cv2.warpAffine(nb_img, M, (w, h), flags=cv2.INTER_LINEAR)
        nm = cv2.warpAffine(nb_mask.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST).astype(bool)
        out[nm] = nb[nm]
    return out


class BeanDataset(torch.utils.data.Dataset):
    def __init__(self, items, img_size, train, pool=None, repeat=1, neighbor_p=0.6, seed=0):
        self.items = items            # list of (img_bgr, label, source)
        self.train = train
        self.pool = pool or []
        self.repeat = repeat if train else 1
        self.neighbor_p = neighbor_p
        self.seed = seed
        self.epoch = 0
        self.post = T.Compose([
            T.ToTensor(),
            T.Resize((img_size, img_size), antialias=True),
            *([T.RandomApply([T.ColorJitter(0.15, 0.15, 0.1, 0.02)], p=0.8),
               T.RandomApply([T.GaussianBlur(5, (0.1, 1.5))], p=0.3)] if train else []),
            T.Normalize(MEAN, STD),
        ])

    def __len__(self):
        return len(self.items) * self.repeat

    def __getitem__(self, i):
        img, label, _ = self.items[i % len(self.items)]
        if self.train:
            rng = np.random.default_rng((self.seed, self.epoch, i))
            img = rand_affine(img, rng)          # หมุนก่อน (เติมขอบด้วยพื้นกระดาษ)
            if self.pool and rng.random() < self.neighbor_p:
                img = paste_neighbors(img, self.pool, rng)   # แล้วค่อยแปะเม็ดข้างเคียง
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.post(rgb), label


def load_crops(crops_dir):
    crops_dir = Path(crops_dir)
    items = []
    with open(crops_dir / "index.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["class"] not in CLASSES:
                continue
            img = cv2.imread(str(crops_dir / r["crop"]))
            if img is None:
                raise FileNotFoundError(crops_dir / r["crop"])
            items.append((img, CLASSES.index(r["class"]), r["source"]))
    if not items:
        raise SystemExit(f"ไม่พบ crop ใน {crops_dir} — รัน crop_beans.py ก่อน")
    return items


def layout_of(source):
    """'good/grid_good.jpg' -> 'grid' (ใช้แบ่ง fold ตาม layout)"""
    return Path(source).stem.split("_")[0]


# ----------------------------------------------------------------------------- train / eval
def class_weights(items):
    counts = np.bincount([y for _, y, _ in items], minlength=len(CLASSES)).astype(float)
    w = counts.sum() / (len(CLASSES) * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)


def train_model(train_items, args, device, log_prefix=""):
    torch.manual_seed(args.seed)
    model, body, head = build_model(args.arch, len(CLASSES))
    model.to(device)
    pool = [(img, crop_mask(img)) for img, _, _ in train_items]
    ds = BeanDataset(train_items, args.img_size, True, pool, args.repeat, args.neighbor_p, args.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                                     drop_last=True, persistent_workers=args.workers > 0)
    crit = nn.CrossEntropyLoss(weight=class_weights(train_items).to(device), label_smoothing=0.1)

    def make_opt(stage):
        if stage == 1:     # ตรึง backbone · train หัวอย่างเดียว
            body.requires_grad_(False)
            opt = torch.optim.AdamW(head.parameters(), lr=args.lr_head, weight_decay=1e-4)
            n = args.freeze_epochs
        else:              # ปลดทั้งหมด · backbone lr ต่ำ
            body.requires_grad_(True)
            opt = torch.optim.AdamW([{"params": body.parameters(), "lr": args.lr_body},
                                     {"params": head.parameters(), "lr": args.lr_head}],
                                    weight_decay=1e-4)
            n = args.epochs - args.freeze_epochs
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, n * len(dl)))
        return opt, sched

    opt, sched = make_opt(1)
    for ep in range(args.epochs):
        if ep == args.freeze_epochs:
            opt, sched = make_opt(2)
        ds.epoch = ep
        model.train()
        if ep < args.freeze_epochs:
            body.eval()                     # BatchNorm ของ backbone ไม่ขยับระหว่างตรึง
        tot, correct, n = 0.0, 0, 0
        t0 = time.time()
        for x, y in dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = crit(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
        if (ep + 1) % args.log_every == 0 or ep == args.epochs - 1:
            print(f"{log_prefix}epoch {ep + 1:3d}/{args.epochs}  loss {tot / n:.4f}  "
                  f"train-acc {correct / n:.3f}  ({time.time() - t0:.1f}s)")
    return model


@torch.no_grad()
def evaluate(model, items, args, device):
    """คืน (ความแม่นรายเม็ด, ความแม่นรายภาพ, รายละเอียดรายภาพ)"""
    model.eval()
    ds = BeanDataset(items, args.img_size, False)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
    probs = torch.cat([model(x.to(device)).softmax(1).cpu() for x, _ in dl])
    labels = torch.tensor([y for _, y, _ in items])
    bean_acc = (probs.argmax(1) == labels).float().mean().item()

    by_img = defaultdict(list)
    for p, (_, y, src) in zip(probs, items):
        by_img[src].append((p, y))
    rows = []
    for src, lst in sorted(by_img.items()):
        mean_p = torch.stack([p for p, _ in lst]).mean(0)
        pred, y = int(mean_p.argmax()), lst[0][1]
        rows.append((src, CLASSES[y], CLASSES[pred], float(mean_p[pred]), len(lst)))
    img_acc = float(np.mean([r[1] == r[2] for r in rows]))
    return bean_acc, img_acc, rows, probs.argmax(1).numpy(), labels.numpy()


def print_confusion(y_true, y_pred):
    cm = np.zeros((len(CLASSES), len(CLASSES)), int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    w = max(len(c) for c in CLASSES)
    print("confusion (แถว=จริง, คอลัมน์=ทำนาย) รายเม็ด")
    print(" " * (w + 1) + " ".join(f"{c[:5]:>5}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"{c:>{w}} " + " ".join(f"{v:5d}" for v in cm[i]))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="train ตัวจำแนกเมล็ดกาแฟรายเม็ด")
    ap.add_argument("--mode", choices=["cv", "full"], default="cv")
    ap.add_argument("--crops", default="crops")
    ap.add_argument("--out", default="weights")
    ap.add_argument("--arch", default="efficientnet_b0",
                    help="efficientnet_b0 | mobilenet_v3_small | mobilenet_v3_large | resnet18 | shufflenet_v2_x1_0")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--freeze-epochs", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=8, help="วนชุด train กี่รอบต่อ epoch (augment ต่างกันทุกรอบ)")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-body", type=float, default=1e-4)
    ap.add_argument("--neighbor-p", type=float, default=0.6, help="โอกาสแปะเม็ดข้างเคียง (จำลองกอง)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=5)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = load_crops(args.crops)
    print(f"device={device}  arch={args.arch}  crops={len(items)}  "
          f"images={len({s for _, _, s in items})}")

    if args.mode == "cv":
        layouts = sorted({layout_of(s) for _, _, s in items})
        all_rows, yt, yp, bean_accs = [], [], [], []
        for lay in layouts:
            tr = [it for it in items if layout_of(it[2]) != lay]
            va = [it for it in items if layout_of(it[2]) == lay]
            print(f"\n=== fold: val = {lay}  (train {len(tr)} / val {len(va)} crops)")
            model = train_model(tr, args, device, f"[{lay}] ")
            bean_acc, img_acc, rows, p, t = evaluate(model, va, args, device)
            bean_accs.append(bean_acc)
            all_rows += rows
            yt += list(t)
            yp += list(p)
            print(f"[{lay}] val รายเม็ด {bean_acc:.3f} · รายภาพ {img_acc:.3f}")
            for src, y, pred, conf, n in rows:
                print(f"    {'OK ' if y == pred else 'XX '} {src:35s} -> {pred:9s} ({conf:.2f}, {n} เม็ด)")
        img_acc = np.mean([r[1] == r[2] for r in all_rows])
        print(f"\n===== CV สรุป ({args.arch})")
        print(f"รายเม็ด  {np.mean(bean_accs):.3f}  (แต่ละ fold {', '.join(f'{a:.3f}' for a in bean_accs)})")
        print(f"รายภาพ   {img_acc:.3f}  ({sum(r[1] == r[2] for r in all_rows)}/{len(all_rows)} ภาพ)")
        print_confusion(yt, yp)
    else:
        model = train_model(items, args, device)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        cfg = {"arch": args.arch, "classes": CLASSES, "img_size": args.img_size,
               "mean": MEAN, "std": STD}
        seg = Path(args.crops) / "seg_stats.json"
        if seg.exists():
            cfg.update(json.loads(seg.read_text(encoding="utf-8")))
        else:
            print(f"[warn] ไม่พบ {seg} — infer จะประมาณขนาดเม็ดจากภาพเอง")
        torch.save({**cfg, "state_dict": model.cpu().state_dict()}, out / "model.pt")
        (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"บันทึก {out / 'model.pt'} และ {out / 'config.json'}")


if __name__ == "__main__":
    main()
