# สเปก infer.py สำหรับผู้เข้าแข่งขัน — Day 1

## 1\. คำสั่งมาตรฐาน

```sh
python infer.py --input test/ --weights weights/ --out outputs/
```

&nbsp;

| Argument | ชนิด | จำเป็น | ความหมาย |
| :---- | :---- | :---- | :---- |
| `--input` | path | ใช่ | โฟลเดอร์ภาพทดสอบ หรือไฟล์ภาพหนึ่งไฟล์ |
| `--weights` | path | ใช่ | weights/ ที่ผู้เข้าแข่งขันส่ง |
| `--out` | path | ใช่ | โฟลเดอร์บันทึกผล สร้างให้หากยังไม่มี |

รูปแบบนามสกุล weights ใช้ตามโมเดลจริงได้ เช่น `.pt`, `.onnx`, `.joblib` โดย `infer.py` ต้องโหลดและใช้งานได้ตามคำสั่งข้างต้น หากต้องมีไฟล์ประกอบ ให้รวมไว้ใน `weights/` และระบุใน README ไม่ต้องเปลี่ยนคำสั่งมาตรฐาน

## 2\. ข้อมูลเข้าและการทำนาย

- รองรับภาพ `.jpg`, `.jpeg`, `.png` โดยไม่สนตัวพิมพ์เล็ก/ใหญ่  
- หนึ่งภาพต้องได้ผลทำนาย **หนึ่งคลาส** ไม่ใช่ผลลัพธ์รายเม็ด  
- อ่านภาพและใช้ EXIF orientation ให้ถูกต้องก่อน preprocessing  
- ทำนายด้วย weights ที่โหลดมา ไม่ฝึกหรือปรับโมเดลจาก test ระหว่างรัน  
- สำหรับสเปกนี้ ไม่ทำ random augmentation หรือ test-time augmentation ใช้ preprocessing แบบคงที่ตามโมเดล เช่น resize, padding, normalization ได้  
- รันบน Raspberry Pi 4 ด้วย CPU

## 3\. ชื่อคลาสที่ส่งออก

ใช้ข้อความภาษาอังกฤษตัวพิมพ์เล็กให้ตรงกับรายการนี้เท่านั้น:

```
good
shell
black
husk
broken
parchment
immature
```

&nbsp;

## 4\. ผลลัพธ์บังคับ

```
outputs/
├── predictions.csv
├── good/
├── shell/
├── black/
├── husk/
├── broken/
├── parchment/
└── immature/
```

คัดลอกภาพต้นฉบับไปยังโฟลเดอร์ของคลาสที่ทำนาย ไม่ย้ายหรือลบไฟล์ต้นฉบับ

### predictions.csv

ไฟล์ UTF-8 มี header และคอลัมน์ตามลำดับนี้:

```csv
filename,class
image_001.jpg,good
image_002.jpg,black
```

&nbsp;

| Field | ชนิด | เงื่อนไข |
| :---- | :---- | :---- |
| `filename` | string | ชื่อไฟล์พร้อมนามสกุล ไม่ใช่ absolute path |
| `class` | enum | หนึ่งใน 7 คลาสข้างต้น |

ต้องมีหนึ่งแถวต่อหนึ่งภาพ ไม่มีแถวซ้ำ และไม่มีภาพตกหล่น ส่งออกเฉพาะสองคอลัมน์ `filename,class` ไม่ต้องส่ง confidence ไม่ต้องจับเวลาหรือสร้าง `timings.csv` ผู้จัดการแข่งขันจะจับเวลาเอง

## 5\. พฤติกรรมโปรแกรม

- เรียงไฟล์ก่อนประมวลผลเพื่อให้ลำดับผลสม่ำเสมอ  
- ไม่ต้องเปิดหน้าต่าง GUI หรือรอผู้ใช้กดปุ่ม  
- จบสำเร็จด้วย exit code `0`  
- หาก input/weights ไม่พบ ไม่มีภาพ อ่านภาพไม่ได้ หรือผลลัพธ์ผิดรูปแบบ ให้แสดง error และจบด้วย exit code ที่ไม่ใช่ `0` ไม่ข้ามภาพเงียบ ๆ  
- ใช้ชื่อไฟล์ input ที่ไม่ซ้ำกัน และเก็บ output แยกจาก input

## 6\. โครงสร้างไฟล์ส่งงาน

```
team_XX_day1/
├── README.md
├── requirements.txt
├── train.py
├── infer.py
├── weights/
│   └── best.pt
└── outputs/
```

## Folder Weights สามารถใช้ได้มากกว่า 1 model&nbsp;

README ต้องบอกวิธีติดตั้ง คำสั่งรัน preprocessing และไฟล์ประกอบ weights ที่ต้องใช้ ส่วน `requirements.txt` ต้องติดตั้ง dependency ที่ใช้จริงได้บนระบบทดสอบ(requirements สำหรับ infer เท่านั้นไม่ต้องใส่ requirement ของ train)

`infer.py` ที่แนบเป็น **แม่แบบ interface** ผู้เข้าแข่งขันต้องเติม `load_model()` และ `predict_image()` ให้ใช้งานกับโมเดลของตนเองก่อนส่ง

&nbsp;

ไฟล์ที่ส่งให้บีบอัดเป็นไฟล์ zip ก่อนเท่านั้น