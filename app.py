"""
Hugging Face Spaces - Fruit Disease Detection & Diagnosis App
Framework: Gradio 6
Model: EfficientNetB0 (Fine-tuned, 88.61% Accuracy, 88.48% F1-score)
Compatible with: Hugging Face CPU basic & ZeroGPU
"""

import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from PIL import Image
import numpy as np

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Disable Gradio SSR mode for container stability
os.environ["GRADIO_SSR_MODE"] = "false"

# Hugging Face ZeroGPU compatibility
try:
    import spaces
except ImportError:
    class _MockSpaces:
        @staticmethod
        def GPU(fn=None, duration=None):
            def decorator(f):
                return f
            return decorator(fn) if fn else decorator
    spaces = _MockSpaces()

import gradio as gr

from inference import predict_image, resolve_model_path, warmup_model
from classes import FRUIT_CLASSES, FRUIT_VN_MAP, DISEASE_INFO
from gemini_service import explain_diagnosis_with_gemini

# Sample images directory
TEST_IMAGES_DIR = ROOT_DIR / "test-images"
SAMPLE_IMAGES = []
if TEST_IMAGES_DIR.is_dir():
    SAMPLE_IMAGES = [
        [str(TEST_IMAGES_DIR / f)]
        for f in sorted(os.listdir(TEST_IMAGES_DIR))
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
    ]

# ============================================================
# Core Prediction Callback Function
# ============================================================
@spaces.GPU
def run_diagnosis(input_img, conf_threshold=0.75):
    """
    Handle inference request from Gradio interface.
    Executes fast unified DL + Grad-CAM (~0.49s), then Gemini deep analysis (~2s),
    and returns complete 3-feature diagnosis all at once.
    """
    if input_img is None:
        return None, "⚠️ **Vui lòng chọn hoặc tải một ảnh trái cây lên để phân tích.**"

    result = predict_image(input_img, conf_threshold=conf_threshold, generate_cam=True)

    best = result["best"]
    info = result["disease_info"]
    pathogen = result.get("pathogen_info", {})
    gradcam_img = result.get("gradcam_image")

    # Kiểm tra nếu độ tin cậy chưa đạt ngưỡng yêu cầu
    confidence_val = best.get("confidence", 0.0)
    if confidence_val < conf_threshold:
        threshold_pct = int(round(conf_threshold * 100)) if conf_threshold <= 1.0 else int(conf_threshold)
        warning_md = f"""
### ⚠️ Chưa Đủ Độ Tin Cậy Để Kết Luận ({best.get('percentage', 'N/A')} < {threshold_pct}%):
- **Tình trạng nhận diện**: **Chưa xác định rõ loại bệnh**

💡 **Khuyến nghị**:
- Hình ảnh có thể bị mờ, thiếu sáng, góc chụp quá xa hoặc triệu chứng chưa rõ ràng.
- Vui lòng chụp hoặc tải lại ảnh cận cảnh, lấy nét trực tiếp vào vết bệnh để hệ thống chẩn đoán chính xác.
"""
        return gradcam_img, warning_md

    # Chuẩn bị dữ liệu tác nhân đối chiếu từ CSDL (dùng làm cơ sở cho Gemini hoặc fallback)
    tac_nhan_list = pathogen.get("tac_nhan", [])
    if tac_nhan_list:
        tac_nhan_str = ", ".join([f"*{tn}*" for tn in tac_nhan_list])
    else:
        tac_nhan_str = "Không có tác nhân gây hại (Cây/quả khỏe mạnh)"

    loai_tac_nhan = pathogen.get("loai", "Không có")
    ghi_chu = pathogen.get("ghi_chu", "")
    ghi_chu_md = f"\n- **Ghi chú thêm**: {ghi_chu}" if ghi_chu else ""
    initial_pathogen_md = f"- **Nhóm tác nhân**: {loai_tac_nhan}\n- **Tác nhân chính**: {tac_nhan_str}{ghi_chu_md}"

    # Gọi Bác sĩ Nông sản (Gemini) để hoàn tất trọn vẹn cả 3 chức năng
    try:
        gemini_result = explain_diagnosis_with_gemini(
            gradcam_image=gradcam_img,
            best_prediction=best,
            pathogen_info=pathogen,
            disease_info=info
        )
        explanation_text = gemini_result.get("explanation", "").strip()
        pathogen_analysis_text = gemini_result.get("pathogen_analysis", "").strip()
        advice_text = gemini_result.get("advice", "").strip()

        final_diagnosis_md = f"""
### 🎯 Kết Quả Dự Đoán:
- **Tình trạng**: **{best.get('vn_name')}**
- **Độ tin cậy**: **{best.get('percentage', 'N/A')}**
- **Giải thích kết quả**: {explanation_text}

---

#### 🔬 Tác Nhân Gây Bệnh:
{pathogen_analysis_text}

---

#### 📦 Đề xuất xử lý:
{advice_text}
"""
        return gradcam_img, final_diagnosis_md.strip()

    except Exception as e:
        print(f"[App] Lỗi phân tích Gemini: {e}")
        fallback_md = f"""
### 🎯 Kết Quả Dự Đoán:
- **Tình trạng**: **{best.get('vn_name')}**
- **Độ tin cậy**: **{best.get('percentage', 'N/A')}**

---

#### 🔬 Tác Nhân Gây Bệnh:
{initial_pathogen_md}

---

#### 📦 Đề xuất xử lý:
- **Phân loại**: {info.get('severity', 'Cảnh báo')}
- **Xử lý**: {info.get('treatment', 'Theo dõi sát diễn biến vết bệnh và kiểm tra bảo quản.')}
"""
        return gradcam_img, fallback_md.strip()


# ============================================================
# Gradio User Interface Definition
# ============================================================
CUSTOM_CSS = """
/* App styling customized like electronic_component_detection */
.main-header {
    text-align: center;
    padding: 20px 10px 10px;
    background: linear-gradient(135deg, #10b981 0%, #059669 50%, #047857 100%);
    color: white;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(5, 150, 105, 0.2);
}
.main-header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 6px;
    color: #ffffff;
}
.main-header p {
    font-size: 1.05rem;
    opacity: 0.95;
    margin-bottom: 10px;
}
.badge-bar {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 8px;
}
.badge {
    background: rgba(255, 255, 255, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.35);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
}
.diagnosis-box {
    border-left: 5px solid #10b981;
    background-color: #f8fafc;
    padding: 15px;
    border-radius: 8px;
    margin-top: 10px;
}
/* Đồng bộ khung ảnh hiển thị */
.gradio-image, .gradio-image img, .image-container img {
    max-height: 380px !important;
    object-fit: contain !important;
}
/* Thu hẹp khoảng cách giữa các khối chẩn đoán */
.diagnosis-container hr {
    margin: 8px 0 !important;
    border: none !important;
    border-top: 1px solid #e2e8f0 !important;
}
.diagnosis-container h3 {
    margin-top: 4px !important;
    margin-bottom: 6px !important;
    font-size: 1.15rem !important;
}
.diagnosis-container h4 {
    margin-top: 8px !important;
    margin-bottom: 4px !important;
    font-size: 1.05rem !important;
}
.diagnosis-container ul {
    margin-top: 2px !important;
    margin-bottom: 4px !important;
    padding-left: 18px !important;
}
.diagnosis-container li {
    margin-bottom: 3px !important;
    line-height: 1.45 !important;
}
.diagnosis-container p {
    margin-top: 2px !important;
    margin-bottom: 4px !important;
    line-height: 1.45 !important;
}
"""

with gr.Blocks(title="Nhận Diện & Chẩn Đoán Bệnh Cây Ăn Quả") as demo:
    # Header
    gr.HTML("""
    <div class="main-header">
        <h1>🍎 Nhận Diện & Chẩn Đoán Bệnh Cây Ăn Quả</h1>
        <p>Hệ thống AI nhận diện và chẩn đoán bệnh trên cây ăn quả (Táo, Ổi, Xoài, Lựu) sử dụng <strong>EfficientNetB0</strong></p>
        <div class="badge-bar">
            <span class="badge">🎯 Độ chính xác kiểm thử: 88.61%</span>
            <span class="badge">⚡ Điểm F1-Score: 88.48%</span>
            <span class="badge">🍏 17 Lớp bệnh & quả</span>
            <span class="badge">🚀 Tối ưu hóa mô hình TFLite</span>
        </div>
    </div>
    """)

    # Main Tabs
    with gr.Tabs():
        # Tab 1: Image Upload
        with gr.TabItem("🖼️ Tải Ảnh Lên Phân Tích", id="tab_image"):
            with gr.Row():
                with gr.Column(scale=1) as left_col:
                    img_input = gr.Image(type="pil", label="Tải ảnh quả", height=380)
                    btn_predict = gr.Button("🔍 Phân Tích & Chẩn Đoán", variant="primary", size="lg")

                with gr.Column(scale=1):
                    img_output = gr.Image(type="numpy", label="Ảnh phân tích", height=380)
                    diagnosis_output = gr.Markdown("💡 *Tải ảnh lên và bấm 'Phân Tích & Chẩn Đoán' để xem kết quả dự đoán.*", elem_classes=["diagnosis-container"])

            # Đặt Thư viện ảnh mẫu ngay bên dưới nút bấm ở Cột bên trái
            if SAMPLE_IMAGES:
                with left_col:
                    gr.Markdown("### 📂 Thư Viện Ảnh Mẫu (Click để thử nhanh):")
                    gr.Examples(
                        examples=SAMPLE_IMAGES,
                        inputs=img_input,
                        outputs=[img_output, diagnosis_output],
                        fn=run_diagnosis,
                        cache_examples=False
                    )

            btn_predict.click(
                fn=run_diagnosis,
                inputs=[img_input],
                outputs=[img_output, diagnosis_output]
            )

        # Tab 2: Webcam Stream
        with gr.TabItem("📹 Quét Trực Tiếp Qua Camera", id="tab_cam"):
            with gr.Row():
                with gr.Column(scale=1):
                    cam_input = gr.Image(sources=["webcam"], type="pil", label="Bật Camera chụp trực tiếp quả/lá", height=380)
                    btn_cam_predict = gr.Button("📸 Chụp & Nhận Diện Ngay", variant="primary", size="lg")
                with gr.Column(scale=1):
                    cam_img_out = gr.Image(type="numpy", label="Ảnh phân tích", height=380)
                    cam_diag_out = gr.Markdown("💡 *Chụp ảnh qua camera để xem kết quả dự đoán.*", elem_classes=["diagnosis-container"])

            btn_cam_predict.click(
                fn=run_diagnosis,
                inputs=[cam_input],
                outputs=[cam_img_out, cam_diag_out]
            )

        # Tab 3: Handbook & Classes
        with gr.TabItem("📋 Sổ Tay 17 Lớp Bệnh Trái Cây", id="tab_handbook"):
            gr.Markdown("""
### Danh Sách 17 Lớp Bệnh Được Huấn Luyện Trong Mô Hình:

| STT | Loại Quả | Tên Bệnh (Tiếng Anh) | Tên Tiếng Việt | Mức Độ |
| :---: | :---: | :--- | :--- | :---: |
| 1 | 🍎 Táo | `APPLE_Blotch_Apple` | Bệnh đốm lá đốm quả Marssonina | 🟡 Trung bình |
| 2 | 🍎 Táo | `APPLE_Healthy_Apple` | Táo khỏe mạnh | 🟢 An toàn |
| 3 | 🍎 Táo | `APPLE_Rot_Apple` | Bệnh thối quả (Black/Bitter Rot) | 🔴 Nguy hiểm |
| 4 | 🍎 Táo | `APPLE_Scab_Apple` | Bệnh vảy nấm Apple Scab | 🟡 Trung bình |
| 5 | 🍈 Ổi | `GUAVA_Anthracnose_Guava` | Bệnh thán thư ổi | 🟡 Trung bình |
| 6 | 🍈 Ổi | `GUAVA_Fruitfly_Guava` | Bị ruồi đục quả phá hại | 🔴 Nguy hiểm |
| 7 | 🍈 Ổi | `GUAVA_Healthy_Guava` | Ổi khỏe mạnh | 🟢 An toàn |
| 8 | 🥭 Xoài | `MANGO_Alternaria_Mango` | Bệnh đốm nấm Alternaria xoài | 🟡 Trung bình |
| 9 | 🥭 Xoài | `MANGO_Anthracnose_Mango` | Bệnh thán thư xoài | 🟡 Trung bình |
| 10 | 🥭 Xoài | `MANGO_Black Mould Rot (Aspergillus)_Mango` | Bệnh thối mốc đen Aspergillus | 🔴 Nguy hiểm |
| 11 | 🥭 Xoài | `MANGO_Healthy_Mango` | Xoài khỏe mạnh | 🟢 An toàn |
| 12 | 🥭 Xoài | `MANGO_Stem and Rot (Lasiodiplodia)_Mango` | Bệnh thối cuống thối thân xoài | 🔴 Nguy hiểm |
| 13 | 🫐 Lựu | `POMEGRANATE_Alternaria_Pomegranate` | Bệnh đốm thối đen Alternaria | 🟡 Trung bình |
| 14 | 🫐 Lựu | `POMEGRANATE_Anthracnose_Pomegranate` | Bệnh thán thư quả lựu | 🟡 Trung bình |
| 15 | 🫐 Lựu | `POMEGRANATE_Bacterial_Blight_Pomegranate` | Bệnh cháy vi khuẩn lựu | 🔴 Nguy hiểm |
| 16 | 🫐 Lựu | `POMEGRANATE_Cercospora_Pomegranate` | Bệnh đốm mắt chim Cercospora | 🟡 Trung bình |
| 17 | 🫐 Lựu | `POMEGRANATE_Healthy_Pomegranate` | Lựu khỏe mạnh | 🟢 An toàn |
            """)

if __name__ == "__main__":
    print("[App] Pre-warming model and tensor execution graphs...")
    warmup_model()
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        theme=gr.themes.Soft(primary_hue="emerald"),
        css=CUSTOM_CSS
    )
