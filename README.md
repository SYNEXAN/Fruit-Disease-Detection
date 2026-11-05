---
title: Fruit Disease Detection & AI Diagnosis
emoji: 🥭
colorFrom: emerald
colorTo: green
sdk: gradio
sdk_version: 6.27.0
python_version: '3.10'
app_file: app.py
pinned: false
license: mit
---

# 🥭 Fruit Disease Detection & AI Diagnosis System

Hệ thống Trí tuệ Nhân tạo hỗ trợ kiểm định, phát hiện dấu hiệu bệnh trên bề mặt trái cây sau thu hoạch (**Xoài**, **Cam**, **Lựu**) kết hợp cơ chế giải thích hình ảnh **Focused Lesion Grad-CAM** và Bác sĩ Nông sản AI (**Gemini 2.5**).

---

## 🌟 Điểm Nổi Bật & Đóng Góp Kỹ Thuật

- **Kiến trúc mô hình**: **ConvNeXtV2-Tiny Advanced CMTL** (Học đa nhiệm có hướng dẫn phân cấp: Nhận diện Loài cây + Nhận diện Bệnh hại).
- **Hàm mất mát ArcFace Margin Loss**: Phân tách hình học siêu cầu ($s=32.0, m=0.50$ rad), triệt tiêu **100% hiện tượng nhầm lẫn loài cây chéo** (0% cross-species leakage).
- **Cơ chế Attention Pooling**: Gom tụ đặc trưng tổn thương cực đại (Spatial Peak-Lesion Attention) kết hợp cổng phản hồi 2 chiều (Lesion-to-Species Gating).
- **Độ chính xác vượt trội (SOTA)**:
  - **Overall Test Accuracy**: **98.28%** (trên 3.026 ảnh kiểm thử độc lập).
  - **Species Accuracy**: **100.00%**.
  - **Độ chính xác từng loài**: Xoài (**98.46%**), Lựu (**99.61%**), Cam (**99.04%**).
- **Bản đồ nhiệt Grad-CAM trực quan**: Tích hợp chuẩn hóa Min-Max Scaling và thuật toán lọc phông trắng, định vị chính xác ổ bệnh trên bề mặt quả mà không bị loang quầng sáng.
- **Trợ lý Bác sĩ Nông sản (Gemini 2.5 Flash)**: Tự động phân tích triệu chứng, cảnh báo mức độ nguy hiểm (*Nguy hiểm / Trung bình / An toàn*), nhận định tác nhân nấm/vi khuẩn nghi ngờ và đề xuất phác đồ xử lý sau thu hoạch chi tiết.

---

## 📂 Danh mục 15 Nhóm Bệnh & Trạng Thái Phân Loại:

### 🥭 Xoài (Mango - *Mangifera indica*):
1. `mango_Healthy`: Quả khỏe mạnh
2. `mango_Anthracnose`: Bệnh thán thư (*Colletotrichum spp.*)
3. `mango_Bacterial_Canker`: Bệnh loét vi khuẩn (*Xanthomonas campestris*)
4. `mango_Powdery_Mildew`: Bệnh phấn trắng (*Oidium mangiferae*)
5. `mango_Scab`: Bệnh vảy nấm ghẻ (*Elsinoë mangiferae*)
6. `mango_Sooty_Mould`: Bệnh bồ hóng muội đen (*Capnodium spp.*)
7. `mango_Stem_End_Rot`: Bệnh thối đầu cuống quả (*Lasiodiplodia theobromae*)

### 🍊 Cam (Orange - *Citrus sinensis*):
8. `orange_Healthy`: Quả khỏe mạnh
9. `orange_Citrus_Canker`: Bệnh loét vi khuẩn (*Xanthomonas citri*)
10. `orange_Citrus_Melanose`: Bệnh đốm sẹo gôm đen (*Diaporthe citri*)

### 🫐 Lựu (Pomegranate - *Punica granatum*):
11. `pomegranate_Healthy`: Quả khỏe mạnh
12. `pomegranate_Alternaria`: Bệnh đốm thối đen Alternaria (*Alternaria alternata*)
13. `pomegranate_Anthracnose`: Bệnh thán thư quả lựu (*Colletotrichum gloeosporioides*)
14. `pomegranate_Bacterial_Blight`: Bệnh cháy vi khuẩn (*Xanthomonas axonopodis pv. punicae*)
15. `pomegranate_Cercospora`: Bệnh đốm mắt chim (*Cercospora punicae*)

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Cục Bộ (Local Deployment)

```bash
# 1. Clone repository
git clone https://github.com/SYNEXAN/Fruit-Disease-Detection.git
cd Fruit-Disease-Detection

# 2. Cài đặt môi trường ảo Python
python -m venv .venv
source .venv/bin/activate  # Trên Linux/macOS
# .venv\Scripts\activate   # Trên Windows

# 3. Cài đặt các thư viện cần thiết
pip install -r requirements.txt

# 4. Thiết lập khóa API Gemini (tùy chọn để kích hoạt bác sĩ AI)
export GEMINI_API_KEY="your_api_key_here"  # Linux/macOS
# set GEMINI_API_KEY="your_api_key_here"   # Windows CMD

# 5. Khởi chạy ứng dụng Web Gradio
python app.py
```
Truy cập giao diện tại: `http://localhost:7860`

---

## 🌐 Triển Khai Đám Mây (CI/CD)

Repository này được cấu hình sẵn cho quy trình tự động Build & Deploy trên:
- **Render**: Thông qua `render.yaml` (hỗ trợ Git LFS và PyTorch CPU tối ưu).
- **Hugging Face Spaces**: SDK Gradio với giao diện tương thích hoàn toàn.
