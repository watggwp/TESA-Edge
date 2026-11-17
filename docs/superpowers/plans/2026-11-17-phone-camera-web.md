# Phone Camera Test Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** หน้าเว็บที่มือถือถ่ายรูปเมล็ดกาแฟ ส่งให้คอมทำนาย แล้วได้รูปวาดกรอบ + คลาสกลับมา

**Architecture:** `web_app.py` ไฟล์เดียว: ส่วนวิเคราะห์ภาพ (`analyze`) ใช้ `infer.load_model` + `crop_beans.extract_beans` แล้ววาดผล; ส่วน Flask มี `GET /` (หน้า HTML inline) และ `POST /predict` (JSON) ฟังที่ `0.0.0.0`

**Tech Stack:** Python 3.12, Flask 3.x, PyTorch/torchvision, OpenCV, Pillow · ทดสอบด้วย `unittest` (stdlib, ไม่มี pytest ในเครื่อง)

**Spec:** `docs/superpowers/specs/2026-11-17-phone-camera-web-design.md`

## Global Constraints

- ห้ามแก้ `infer.py`, `crop_beans.py`, `requirements.txt` (ไฟล์ส่งงาน)
- dependency ใหม่อยู่ใน `requirements-web.txt` = `-r requirements.txt` + `flask>=3.0`
- ต้องใช้ผ่าน http ธรรมดา: `<input type="file" accept="image/*" capture="environment">`
- ไม่บันทึกภาพที่อัปโหลดลงดิสก์
- คลาสของภาพ = argmax ของค่าเฉลี่ย softmax ทุกเม็ด (เหมือน `infer.predict_image`)
- ภาพผลย่อให้ด้านยาว ≤ 1280 px, ส่งเป็น JPEG base64

## Review Focus

- ภาพจากมือถือมี EXIF หมุน 90° → กรอบต้องตรงกับภาพที่หมุนแล้ว (`load_image` รับ BytesIO) — test `test_exif_rotated_upload`
- ภาพไม่มีเมล็ด (พื้นเปล่า) → ไม่ crash, ได้คลาส + `warning` — test `test_blank_image_falls_back`
- ไฟล์ที่ไม่ใช่ภาพ / ไม่มีไฟล์ → HTTP 400 + `error` ภาษาไทย — tests `test_predict_rejects_non_image`, `test_predict_missing_file`
- ภาพใหญ่ 12 MP จากมือถือ → Flask ต้องรับได้ (`MAX_CONTENT_LENGTH` = 40 MB) — covered by test with real 3072x4096 train image
- มือถือส่ง HEIC → Pillow อ่านไม่ได้ → ตกไปเคส 400 (ระบุในหน้าเว็บว่าให้ตั้งกล้องเป็น JPEG ถ้าเจอ)

---

### Task 1: วิเคราะห์ภาพ + วาดผล

**Files:**
- Create: `web_app.py` (ส่วน analyze)
- Test: `tests/test_web_app.py`

**Interfaces:**
- Consumes: `infer.load_model(weights) -> dict{model, classes, tf, ref_area}`, `crop_beans.load_image(path_or_filelike) -> BGR`, `crop_beans.extract_beans(bgr, 256, ref_area=..., return_boxes=True) -> (crops, boxes[(x,y,w,h,area)])`
- Produces: `analyze(bundle: dict, fileobj) -> dict` keys: `cls: str`, `confidence: float`, `counts: dict[str,int]` (เฉพาะคลาสที่ > 0, เรียงมากไปน้อย), `n_beans: int`, `warning: str|None`, `image: str` (base64 JPEG) · raises `ValueError` เมื่ออ่านภาพไม่ได้

- [ ] **Step 1: เขียน test** (`setUpClass` โหลด `load_model("weights")` ครั้งเดียว)
  - `test_black_image`: `analyze(bundle, open("black/circle_black.jpg","rb"))` → `cls == "black"`, `n_beans >= 5`, `sum(counts.values()) == n_beans`, `warning is None`, base64 decode ได้ JPEG ด้านยาว ≤ 1280
  - `test_exif_rotated_upload`: เขียนภาพเดียวกันใหม่ (ย่อ 1/2) ด้วย EXIF Orientation=6 ลง BytesIO → ภาพผลกว้าง < สูง กลับด้านจากพิกเซลดิบ (ยืนยันว่าใช้ EXIF)
  - `test_blank_image_falls_back`: ภาพสีขาวล้วน 1000x1000 → มี `cls` ใน 7 คลาส, `n_beans == 1`, `warning` ไม่ว่าง
  - `test_garbage_raises`: `BytesIO(b"not an image")` → `ValueError`
- [ ] **Step 2:** `python -m unittest tests.test_web_app -v` → FAIL (import error)
- [ ] **Step 3:** เขียน `analyze` ใน `web_app.py`: ทำนายเป็น batch 32 แบบ `infer.predict_image`; fallback กลางภาพแบบเดียวกับ infer (box = สี่เหลี่ยมกลางภาพ); วาดกรอบหนา ~ด้านยาว/400 สีตาม `CLASS_COLORS` (BGR, 7 สีแยกชัด) + ป้าย `ชื่อคลาส` บนพื้นสีเดียวกัน; แถบหัวภาพเขียน `<CLS> <conf%>`; ห่อ `with torch.no_grad()` และ `threading.Lock` ระดับโมดูล
- [ ] **Step 4:** run test → PASS 4/4

### Task 2: Flask server + หน้ามือถือ + เอกสาร

**Files:**
- Modify: `web_app.py` (เพิ่ม `create_app`, `main`)
- Create: `requirements-web.txt`
- Modify: `README.md` (เพิ่มหัวข้อสั้น "ทดสอบด้วยกล้องมือถือ (ไม่ใช่ส่วนงานส่ง)")
- Test: `tests/test_web_app.py`

**Interfaces:**
- Consumes: `analyze` จาก Task 1
- Produces: `create_app(bundle) -> flask.Flask`; CLI `python web_app.py --weights weights/ [--port 8000] [--threads 4]`

- [ ] **Step 1: เขียน test** (ใช้ `app.test_client()`)
  - `test_index_has_camera_input`: `GET /` 200, HTML มี `capture="environment"` และ `accept="image/*"`
  - `test_predict_ok`: POST `/predict` ไฟล์ `good/grid_good.jpg` field `image` → 200, JSON มี `cls`, `image`
  - `test_predict_rejects_non_image`: ไฟล์ `b"hello"` → 400, `error` ไม่ว่าง
  - `test_predict_missing_file`: POST ไม่มีไฟล์ → 400
- [ ] **Step 2:** run → FAIL
- [ ] **Step 3:** เขียน `create_app` (`MAX_CONTENT_LENGTH = 40 MB`, ValueError → 400) + หน้า HTML inline: ปุ่ม "📷 ถ่ายรูป" (capture) และ "🖼️ เลือกจากคลังภาพ" (ไม่มี capture), `fetch` อัตโนมัติเมื่อเลือกไฟล์, สถานะ "กำลังทำนาย…", แสดงคลาสตัวใหญ่ + % + จำนวนเม็ดต่อคลาส + ภาพผล, error เป็นกล่องแดง; viewport มือถือ, ปุ่มสูง ≥ 56px. `main()` หา IP วง LAN ด้วย UDP socket connect ไป `8.8.8.8` (ไม่ส่งข้อมูลจริง) แล้วพิมพ์ `เปิดบนมือถือ: http://<ip>:<port>`; `app.run(host="0.0.0.0", threaded=True)`
- [ ] **Step 4:** run test → PASS 8/8
- [ ] **Step 5:** ทดสอบจริง: รัน server background, `curl -F image=@black/grid_black.jpg http://127.0.0.1:8000/predict` ได้ JSON `cls`; เขียน `requirements-web.txt` + หัวข้อ README (คำสั่งรัน, Wi-Fi เดียวกัน, Windows Firewall อาจต้องอนุญาต Python)
