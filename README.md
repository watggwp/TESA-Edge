# Day 1 · จำแนกคุณภาพเมล็ดกาแฟโรบัสต้า (Classification)

ทำนาย **1 คลาสต่อภาพ** จาก 7 คลาส: `good`, `shell`, `black`, `husk`, `broken`, `parchment`, `immature`
รันบน Raspberry Pi 4 ด้วย CPU

## วิธีการโดยสรุป

ภาพมีขนาด 3072x4096 แต่เมล็ดแต่ละเม็ดกินพื้นที่ภาพเพียงเล็กน้อย ถ้าย่อทั้งภาพ เม็ดจะเล็กจนเห็นรายละเอียดไม่ชัด จึงจำแนก **ทีละเม็ด** แล้วโหวตเป็นผลของภาพ

1. **แยกเม็ด** (`crop_beans.py`, OpenCV): ประมาณความสว่างพื้นกระดาษด้วยพื้นผิวกำลังสอง (รองรับแสงไม่สม่ำเสมอ) → พิกเซลที่มืดกว่าหรือสีต่างจากพื้น = เมล็ด → แยกเม็ดที่ติดกันด้วย distance transform + watershed → กองที่แน่นจนแยกไม่ออก ใช้หน้าต่างขนาดเม็ดเลื่อนทั่วกอง
2. **จำแนกรายเม็ด**: `efficientnet_b0` (pretrained ImageNet) fine-tune บนภาพ crop รายเม็ด
3. **โหวต**: เฉลี่ยความน่าจะเป็นของทุกเม็ดในภาพ → คลาสที่ได้ค่าสูงสุดคือผลของภาพ

## การติดตั้ง

ใช้ Raspberry Pi OS **64-bit** (PyTorch ไม่มีให้ 32-bit) และ Python 3.9 ขึ้นไป

```sh
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` มีเฉพาะ dependency ของ `infer.py`: `torch`, `torchvision`, `opencv-python-headless`, `numpy`, `Pillow`
(ทดสอบกับ torch 2.11.0, torchvision 0.26.0, opencv 4.11.0, numpy 2.2.6, Pillow 11.3.0 บน Python 3.12)

## คำสั่งรัน

```sh
python infer.py --input test/ --weights weights/ --out outputs/
```

| Argument | ความหมาย |
| :---- | :---- |
| `--input` | โฟลเดอร์ภาพทดสอบ (`.jpg` / `.jpeg` / `.png` ไม่สนตัวพิมพ์เล็ก/ใหญ่) หรือไฟล์ภาพไฟล์เดียว |
| `--weights` | โฟลเดอร์ `weights/` — `infer.py` หาไฟล์โมเดลเอง (รับ path ไฟล์ `.pt` โดยตรงได้ด้วย) |
| `--out` | โฟลเดอร์ผลลัพธ์ สร้างให้หากยังไม่มี (ต้องไม่ใช่โฟลเดอร์เดียวกับ `--input`) |
| `--threads` | (ไม่บังคับ) จำนวน CPU threads ค่าเริ่มต้น 4 |

ผลลัพธ์:

```
outputs/
├── predictions.csv      # UTF-8, header filename,class · หนึ่งแถวต่อหนึ่งภาพ
├── good/  shell/  black/  husk/  broken/  parchment/  immature/   # สำเนาภาพตามคลาสที่ทำนาย
```

พฤติกรรมโปรแกรม:

- ประมวลผลภาพเรียงตามชื่อไฟล์ ผลจึงออกมาลำดับเดิมทุกครั้ง
- ภาพต้นฉบับถูก **คัดลอก** ไปโฟลเดอร์คลาส ไม่ถูกย้ายหรือลบ
- log ความคืบหน้า (ชื่อภาพ, คลาส, จำนวนเม็ด) พิมพ์ออก stderr ไม่ปนกับผลลัพธ์
- ถ้าภาพใดแยกเม็ดไม่ได้เลย จะใช้บริเวณกลางภาพทำนายแทนและพิมพ์ `[warn]` (ไม่ข้ามภาพ)
- ถ้าไม่พบ input / weights, ไม่มีภาพ, อ่านภาพไม่ได้, ชื่อไฟล์ซ้ำ หรือผลลัพธ์ผิดรูปแบบ จะแสดง `ERROR: ...` และจบด้วย exit code 1 · สำเร็จจบด้วย exit code 0

## Preprocessing (ตอน infer)

ทำแบบคงที่ ไม่มี random augmentation หรือ TTA และไม่ปรับโมเดลระหว่างรัน

1. อ่านภาพด้วย Pillow และหมุนตาม EXIF orientation
2. ย่อสำเนาภาพให้ด้านยาว 1536 px เพื่อหา mask เมล็ด แล้ว crop แต่ละเม็ดจาก **ภาพเต็มความละเอียด** (กรอบสี่เหลี่ยมจัตุรัส ขยายขอบ 25%) ย่อเป็น 256x256
3. แปลงเป็น RGB → resize 224x224 → normalize ด้วย mean `[0.485, 0.456, 0.406]` std `[0.229, 0.224, 0.225]`

## ไฟล์ที่ต้องใช้

| ไฟล์ | หน้าที่ |
| :---- | :---- |
| `infer.py` | โปรแกรมทำนาย (`load_model()`, `predict_image()`) |
| `crop_beans.py` | **จำเป็น** — ฟังก์ชันแยกเม็ดที่ `infer.py` import ต้องอยู่โฟลเดอร์เดียวกับ `infer.py` |
| `weights/best.pt` | weights ของโมเดล + config (arch, คลาส, ขนาด input, mean/std, `ref_area`) |
| `weights/config.json` | config ชุดเดียวกันแบบอ่านได้ (ใช้เมื่อไฟล์ weight เป็น state_dict ล้วน) |

`ref_area` (= 3457 px ที่ภาพด้านยาว 1536) คือขนาดเม็ดเดี่ยวที่วัดจากภาพ train ใช้แยกเม็ดในภาพกองตอน infer

ลำดับการหาไฟล์ weight ใน `--weights`: ใช้ `best.pt` ก่อน ถ้าไม่มีจะใช้ไฟล์ `.pt` / `.pth` ไฟล์แรกตามลำดับชื่อ (ค้นในโฟลเดอร์ย่อยด้วย)

## การ train (ไม่ต้องใช้ตอนตรวจ)

ต้องมี GPU หรือ CPU ที่เร็วพอ และต้องต่ออินเทอร์เน็ตครั้งแรกเพื่อโหลด ImageNet weights

```sh
# ภาพ train อยู่ในโฟลเดอร์ตามคลาส: good/ shell/ black/ husk/ broken/ parchment/ immature/
python crop_beans.py --data . --out crops          # 1) crop เม็ด -> crops/ (ดูผลแยกเม็ดใน crops/_debug/)
python train.py --mode cv                          # 2) วัดความแม่นแบบ leave-one-layout-out (3 fold)
python train.py --mode full --out weights          # 3) train ด้วยข้อมูลทั้งหมด -> weights/
```

ผลของขั้นที่ 1:

| ไฟล์ | เนื้อหา |
| :---- | :---- |
| `crops/<class>/<ภาพ>__NN.jpg` | ภาพ crop รายเม็ด (label ใช้ของภาพต้นทาง) |
| `crops/index.csv` | รายการ crop + ภาพต้นทาง (ใช้แบ่ง fold ตามภาพ) |
| `crops/_debug/` | ภาพวาดกรอบรอบเม็ด ไว้ตรวจว่าแยกเม็ดถูกไหม |
| `crops/seg_stats.json` | `ref_area` ที่ `train.py --mode full` ใส่ลงใน weights |

รายละเอียดการ train: ตรึง backbone 5 epoch (lr หัว 1e-3) แล้วปลดทั้งหมดจนครบ 30 epoch (lr backbone 1e-4) · AdamW + cosine LR · CrossEntropy + label smoothing 0.1 + class weights · augmentation: หมุน 0–360°, flip, ย่อขยาย/เลื่อน, ColorJitter เบา ๆ, blur และ **จำลองกอง** โดยแปะเม็ดอื่นชิด/ทับขอบเม็ดกลาง

`train.py` เลือก backbone อื่นได้ด้วย `--arch` (`mobilenet_v3_small`, `mobilenet_v3_large`, `resnet18`, `shufflenet_v2_x1_0`) ดู option ทั้งหมดด้วย `python train.py -h`

ผล CV (แต่ละคลาสมี 3 ภาพ layout circle / grid / square · กัน layout หนึ่งของทุกคลาสไว้เป็น val ทีละ fold): ถูกรายภาพ 20/21 (95%), รายเม็ด 83%

## ทดสอบด้วยกล้องมือถือ (ไม่ใช่ส่วนของงานส่ง)

`web_app.py` เปิดหน้าเว็บให้มือถือถ่ายรูปส่งมาให้คอมทำนาย แล้วได้ภาพวาดกรอบสีตามคลาสรายเม็ดกลับไป

```sh
pip install -r requirements-web.txt
python web_app.py --weights weights/          # --port 8000 (ค่าเริ่มต้น)
```

โปรแกรมพิมพ์ URL เช่น `http://192.168.1.103:8000` → เปิดบนมือถือที่ต่อ **Wi-Fi วงเดียวกับคอม** → กด 📷 ถ่ายรูป หรือ 🖼️ เลือกจากคลังภาพ

- Windows อาจถามให้อนุญาต Python ผ่าน Firewall ครั้งแรก — ต้องกดอนุญาต (Private network) มือถือจึงเข้าได้
- ภาพไม่ถูกบันทึกลงเครื่อง · ถ้ามือถือบันทึกภาพเป็น HEIC ให้ตั้งกล้องเป็น JPEG
- ทดสอบ: `python -m unittest tests.test_web_app -v`

ไม่ต้องใส่ `web_app.py`, `requirements-web.txt`, `tests/` และ `docs/` ใน zip ที่ส่ง

## โครงสร้างไฟล์ส่งงาน

```
team_XX_day1/
├── README.md
├── requirements.txt
├── train.py
├── crop_beans.py
├── infer.py
├── weights/
│   ├── best.pt
│   └── config.json
└── outputs/
```

ส่งเป็นไฟล์ zip · ไม่ต้องใส่ภาพ train, `crops/`, `__pycache__/` และ PDF โจทย์
