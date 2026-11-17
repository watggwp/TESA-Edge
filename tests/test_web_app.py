"""ทดสอบ web_app.py · รัน: python -m unittest tests.test_web_app -v (จากโฟลเดอร์โปรเจกต์)"""
import base64
import io
import sys
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import web_app  # noqa: E402
from infer import CLASSES, load_model  # noqa: E402


def decode_image(b64):
    return Image.open(io.BytesIO(base64.b64decode(b64)))


class AnalyzeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_model(ROOT / "weights")

    def test_black_image(self):
        with open(ROOT / "black" / "circle_black.jpg", "rb") as f:
            r = web_app.analyze(self.bundle, f)
        self.assertEqual(r["cls"], "black")
        self.assertGreaterEqual(r["n_beans"], 5)
        self.assertEqual(sum(r["counts"].values()), r["n_beans"])
        self.assertIsNone(r["warning"])
        self.assertTrue(0 < r["confidence"] <= 1)
        im = decode_image(r["image"])
        self.assertEqual(im.format, "JPEG")
        self.assertLessEqual(max(im.size), 1280)

    def test_exif_rotated_upload(self):
        src = Image.open(ROOT / "black" / "circle_black.jpg")
        src = src.resize((src.width // 2, src.height // 2))
        raw_w, raw_h = src.size
        exif = Image.Exif()
        exif[0x0112] = 6                      # Orientation: หมุน 90° ตามเข็ม
        buf = io.BytesIO()
        src.save(buf, "JPEG", exif=exif)
        buf.seek(0)
        r = web_app.analyze(self.bundle, buf)
        w, h = decode_image(r["image"]).size
        # ภาพผลต้องหมุนตาม EXIF แล้ว → สัดส่วนกลับด้านจากพิกเซลดิบ
        self.assertEqual(w > h, raw_h > raw_w)

    def test_blank_image_falls_back(self):
        buf = io.BytesIO()
        Image.new("RGB", (1000, 1000), "white").save(buf, "JPEG")
        buf.seek(0)
        r = web_app.analyze(self.bundle, buf)
        self.assertIn(r["cls"], CLASSES)
        self.assertEqual(r["n_beans"], 1)
        self.assertTrue(r["warning"])

    def test_transparent_png_is_white_not_black(self):
        buf = io.BytesIO()
        Image.new("RGBA", (600, 600), (0, 0, 0, 0)).save(buf, "PNG")
        buf.seek(0)
        r = web_app.analyze(self.bundle, buf)
        px = decode_image(r["image"]).convert("L").getpixel((300, 590))
        self.assertGreater(px, 200)           # พื้นโปร่งใส → ขาว ไม่ใช่ดำ

    def test_label_near_left_edge_is_visible(self):
        import numpy as np
        bgr = np.full((400, 400, 3), 255, np.uint8)
        vis = web_app.draw_result(bgr, [(-200, 200, 260, 120, 0)], ["broken"], "broken", 0.9)
        # ป้ายชื่อต้องเริ่มที่ขอบซ้าย (x=0..) ไม่ล้นออกนอกภาพ
        row = vis[185:196, 0:6].reshape(-1, 3)    # เหนือเส้นกรอบ (y=200) = พื้นที่ป้ายชื่อ
        self.assertTrue((np.abs(row.astype(int) - web_app.CLASS_COLORS["broken"]).sum(1) < 30).any())

    def test_garbage_raises(self):
        with self.assertRaises(ValueError):
            web_app.analyze(self.bundle, io.BytesIO(b"not an image"))


class AppTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = web_app.create_app(load_model(ROOT / "weights")).test_client()

    def test_index_has_camera_input(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        html = r.get_data(as_text=True)
        self.assertIn('capture="environment"', html)
        self.assertIn('accept="image/*"', html)

    def test_predict_ok(self):
        with open(ROOT / "good" / "grid_good.jpg", "rb") as f:
            r = self.client.post("/predict", data={"image": (f, "grid_good.jpg")},
                                 content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        j = r.get_json()
        self.assertIn(j["cls"], CLASSES)
        self.assertTrue(j["image"])

    def test_predict_rejects_non_image(self):
        r = self.client.post("/predict", data={"image": (io.BytesIO(b"hello"), "x.jpg")},
                             content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)
        self.assertTrue(r.get_json()["error"])
        self.assertIn("HEIC", r.get_json()["hint"])   # คำแนะนำ HEIC เฉพาะกรณีอ่านภาพไม่ได้

    def test_predict_tiny_image_is_400(self):
        buf = io.BytesIO()
        Image.new("RGB", (2, 2), "white").save(buf, "PNG")
        buf.seek(0)
        r = self.client.post("/predict", data={"image": (buf, "tiny.png")},
                             content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)
        self.assertTrue(r.get_json()["error"])

    def test_unexpected_error_returns_json_500(self):
        from unittest import mock
        with mock.patch.object(web_app, "analyze", side_effect=RuntimeError("boom")):
            r = self.client.post("/predict", data={"image": (io.BytesIO(b"x"), "a.jpg")},
                                 content_type="multipart/form-data")
        self.assertEqual(r.status_code, 500)
        self.assertTrue(r.get_json()["error"])

    def test_predict_missing_file(self):
        r = self.client.post("/predict", data={}, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)
        self.assertTrue(r.get_json()["error"])
        self.assertNotIn("hint", r.get_json())


if __name__ == "__main__":
    unittest.main()
