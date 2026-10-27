---
title: Fruit Disease Detection & Diagnosis
emoji: 🍎
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.27.0
python_version: '3.11'
app_file: app.py
pinned: false
license: mit
---

# 🍎 Fruit Disease Detection & Diagnosis

Hệ thống AI nhận diện và chẩn đoán bệnh trên 4 loại cây ăn trái kinh tế: **Táo (Apple)**, **Ổi (Guava)**, **Xoài (Mango)** và **Lựu (Pomegranate)**.

## 🌟 Tính Năng Nổi Bật
- **Mô hình**: **EfficientNetB0** (Fine-tuned với Transfer Learning)
- **Độ chính xác**: **Test Accuracy: 88.61%**, **F1-Score: 88.48%**
- **Quy mô phân loại**: **17 Lớp bệnh và quả khỏe mạnh**
- **Chẩn đoán tiếng Việt**: Cung cấp mô tả triệu chứng và phác đồ điều trị nông nghiệp chi tiết cho từng loại bệnh.
- **Hỗ trợ thiết bị**: Tối ưu hóa với TensorFlow Lite (TFLite) cho tốc độ suy luận dưới 50ms trên CPU tiêu chuẩn.
- **Giao diện đa kênh**: Hỗ trợ tải ảnh tĩnh (Drag & drop) và quét camera trực tiếp qua Webcam.

## 📂 Danh mục 17 Lớp Bệnh:
1. `APPLE_Blotch_Apple`: Bệnh đốm lá đốm quả Marssonina
2. `APPLE_Healthy_Apple`: Táo khỏe mạnh
3. `APPLE_Rot_Apple`: Bệnh thối quả (Black/Bitter Rot)
4. `APPLE_Scab_Apple`: Bệnh vảy nấm Apple Scab
5. `GUAVA_Anthracnose_Guava`: Bệnh thán thư ổi
6. `GUAVA_Fruitfly_Guava`: Bị ruồi đục quả phá hại
7. `GUAVA_Healthy_Guava`: Ổi khỏe mạnh
8. `MANGO_Alternaria_Mango`: Bệnh đốm nấm Alternaria xoài
9. `MANGO_Anthracnose_Mango`: Bệnh thán thư xoài
10. `MANGO_Black Mould Rot (Aspergillus)_Mango`: Bệnh thối mốc đen Aspergillus
11. `MANGO_Healthy_Mango`: Xoài khỏe mạnh
12. `MANGO_Stem and Rot (Lasiodiplodia)_Mango`: Bệnh thối cuống thối thân xoài
13. `POMEGRANATE_Alternaria_Pomegranate`: Bệnh đốm thối đen Alternaria
14. `POMEGRANATE_Anthracnose_Pomegranate`: Bệnh thán thư quả lựu
15. `POMEGRANATE_Bacterial_Blight_Pomegranate`: Bệnh cháy vi khuẩn lựu
16. `POMEGRANATE_Cercospora_Pomegranate`: Bệnh đốm mắt chim Cercospora
17. `POMEGRANATE_Healthy_Pomegranate`: Lựu khỏe mạnh
