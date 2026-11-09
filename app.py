"""
Hugging Face Spaces - Fruit Disease Detection & Diagnosis App
Framework: Gradio 6
Model: ConvNeXt-V2 Advanced CMTL (98.28% Accuracy, 15 Classes)
Architecture: Dual-Task ArcFace Species Margin + Spatial Peak-Lesion Attention Pooling + Gating
"""

import os
import sys

# Configure low-memory footprint and single-threaded execution for cloud containers (Render Free Tier 512MB RAM)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MALLOC_ARENA_MAX"] = "1"
os.environ["MALLOC_TRIM_THRESHOLD_"] = "65536"
os.environ["MALLOC_MMAP_THRESHOLD_"] = "65536"

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

from classes import FRUIT_CLASSES, FRUIT_VN_MAP, DISEASE_INFO
from gemini_service import explain_diagnosis_with_gemini

# ============================================================
# High-Confidence Sample Images Selection
# ============================================================
TEST_IMAGES_DIR = ROOT_DIR / "test-images"
SAMPLE_CANDIDATES = [
    "test-images/3fruits/mango_anthracnose.jpg",
    "test-images/3fruits/mango_bacterial_canker.jpg",
    "test-images/ripe-mango-isolated-on-white-background-F2G431.jpg",
    "test-images/ripe-mango-on-white-background-CPTPNN.jpg",
    "test-images/3fruits/orange_citrus_canker.jpg",
    "test-images/3fruits/orange_citrus_melanose.jpg",
    "test-images/istockphoto-174652957-1024x1024.jpg",
    "test-images/OIP (6).webp",
    "test-images/3fruits/pomegranate_alternaria.jpg",
    "test-images/3fruits/pomegranate_bacterial_blight.jpg",
    "test-images/3fruits/pomegranate_healthy.jpg",
    "test-images/7_Luu_Chay_Vi_Khuan.jpg"
]

SAMPLE_IMAGES = []
for rel in SAMPLE_CANDIDATES:
    p = ROOT_DIR / rel
    if p.is_file():
        SAMPLE_IMAGES.append([str(p)])

# Fallback to scanning if specific candidates not found
if not SAMPLE_IMAGES and TEST_IMAGES_DIR.is_dir():
    SAMPLE_IMAGES = [
        [str(TEST_IMAGES_DIR / f)]
        for f in sorted(os.listdir(TEST_IMAGES_DIR))
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
    ]

# ============================================================
# Core Prediction Callback Function
# ============================================================
def get_process_rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for l in f:
                if l.startswith("VmRSS:"):
                    return float(l.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def cleanup_temp_memory():
    """Giải phóng triệt để RAM trên Render Free Tier."""
    try:
        import gc
        gc.collect()
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


@spaces.GPU
def run_diagnosis(input_img, conf_threshold=0.75):
    """
    Handle inference request from Gradio interface.
    Executes in 2 progressive stages:
    - Stage 1 (~1.05s): Immediately yields Grad-CAM image, class, and confidence with status notice.
    - Stage 2 (~2s): Deep Gemini analysis yields full pathogen & advice.
    """
    if input_img is None:
        yield None, "⚠️ **Vui lòng chọn hoặc tải một ảnh trái cây lên để phân tích.**"
        return

    # Unpack dict if Gradio passes file metadata dict
    if isinstance(input_img, dict):
        input_img = input_img.get("path") or input_img.get("url")
        if not input_img:
            yield None, "⚠️ **Vui lòng chọn hoặc tải một ảnh trái cây hợp lệ.**"
            return

    cleanup_temp_memory()
    rss_in = get_process_rss_mb()
    if rss_in > 0:
        print(f"[App] Diagnosis start - Current RSS: {rss_in:.1f} MB", flush=True)

    from inference import predict_image
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
        yield gradcam_img, warning_md.strip()
        cleanup_temp_memory()
        return

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

    # ⚡ YIELD GIAI ĐOẠN 1: Hiển thị ngay ảnh Grad-CAM và kết quả dự đoán sau ~1s
    stage1_md = f"""
### 🎯 Kết Quả Dự Đoán:
- **Tình trạng**: **{best.get('vn_name')}**
- **Độ tin cậy**: **{best.get('percentage', 'N/A')}**

---
⏳ *AI đang phân tích...*
"""
    yield gradcam_img, stage1_md.strip()

    # ⚡ GIAI ĐOẠN 2: Gọi Bác sĩ Nông sản (Gemini) để hoàn tất trọn vẹn cả 3 chức năng
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
        yield gradcam_img, final_diagnosis_md.strip()

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
        yield gradcam_img, fallback_md.strip()
    finally:
        cleanup_temp_memory()
        rss_out = get_process_rss_mb()
        if rss_out > 0:
            print(f"[App] Diagnosis finish - RSS: {rss_out:.1f} MB", flush=True)


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
.image-container img, [data-testid="image"] img {
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
        <p>Hệ thống AI nhận diện và chẩn đoán bệnh trên cây ăn quả (Xoài, Cam, Lựu) sử dụng <strong>ConvNeXt-V2 Advanced CMTL</strong></p>
        <div class="badge-bar">
            <span class="badge">🎯 Độ chính xác kiểm thử: 98.28%</span>
            <span class="badge">⚡ Điểm F1-Score: 98.26%</span>
            <span class="badge">🥭 15 Lớp bệnh & quả (Xoài, Cam, Lựu)</span>
            <span class="badge">🚀 Dual-Task CMTL & ArcFace Margin</span>
        </div>
    </div>
    """)

    # Main Tabs
    with gr.Tabs():
        # Tab 1: Image Upload
        with gr.TabItem("🖼️ Tải Ảnh Lên Phân Tích", id="tab_image"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(type="filepath", label="Tải ảnh quả", height=380)
                    btn_predict = gr.Button("🔍 Phân Tích & Chẩn Đoán", variant="primary", size="lg")
                    if SAMPLE_IMAGES:
                        gr.Markdown("### 📂 Thư Viện Ảnh Mẫu (Click ảnh mẫu rồi bấm Phân Tích):")
                        gr.Examples(
                            examples=SAMPLE_IMAGES,
                            inputs=img_input,
                            label="Chọn ảnh mẫu thử nhanh"
                        )

                with gr.Column(scale=1):
                    img_output = gr.Image(type="numpy", label="Ảnh phân tích", height=380)
                    diagnosis_output = gr.Markdown("💡 *Tải ảnh lên và bấm 'Phân Tích & Chẩn Đoán' để xem kết quả dự đoán.*", elem_classes=["diagnosis-container"])

            btn_predict.click(
                fn=run_diagnosis,
                inputs=[img_input],
                outputs=[img_output, diagnosis_output],
                concurrency_limit=1
            )

        # Tab 2: Webcam Stream
        with gr.TabItem("📹 Quét Trực Tiếp Qua Camera", id="tab_cam"):
            with gr.Row():
                with gr.Column(scale=1):
                    cam_input = gr.Image(sources=["webcam"], type="filepath", label="Bật Camera chụp trực tiếp quả/lá", height=380)
                    btn_cam_predict = gr.Button("📸 Chụp & Nhận Diện Ngay", variant="primary", size="lg")
                with gr.Column(scale=1):
                    cam_img_out = gr.Image(type="numpy", label="Ảnh phân tích", height=380)
                    cam_diag_out = gr.Markdown("💡 *Chụp ảnh qua camera để xem kết quả dự đoán.*", elem_classes=["diagnosis-container"])

            btn_cam_predict.click(
                fn=run_diagnosis,
                inputs=[cam_input],
                outputs=[cam_img_out, cam_diag_out],
                concurrency_limit=1
            )

        # Tab 3: Handbook & Classes
        with gr.TabItem("📋 Sổ Tay 15 Lớp Bệnh Trái Cây", id="tab_handbook"):
            gr.Markdown("""
### Danh Sách 15 Lớp Bệnh Được Huấn Luyện Trong Mô Hình (ConvNeXt-V2 Advanced CMTL):

| STT | Loại Quả | Tên Lớp Kỹ Thuật | Tên Bệnh Tiếng Việt | Mức Độ |
| :---: | :---: | :--- | :--- | :---: |
| 1 | 🥭 Xoài | `mango_Anthracnose` | Bệnh thán thư xoài | 🔴 Nguy hiểm |
| 2 | 🥭 Xoài | `mango_Bacterial_Canker` | Bệnh loét vi khuẩn xoài | 🔴 Nguy hiểm |
| 3 | 🥭 Xoài | `mango_Healthy` | Xoài khỏe mạnh | 🟢 An toàn |
| 4 | 🥭 Xoài | `mango_Powdery_Mildew` | Bệnh phấn trắng xoài | 🟡 Trung bình |
| 5 | 🥭 Xoài | `mango_Scab` | Bệnh vảy nấm ghẻ xoài | 🟡 Trung bình |
| 6 | 🥭 Xoài | `mango_Sooty_Mould` | Bệnh bồ hóng muội đen xoài | 🟡 Trung bình |
| 7 | 🥭 Xoài | `mango_Stem_End_Rot` | Bệnh thối đầu cuống quả xoài | 🔴 Nguy hiểm |
| 8 | 🍊 Cam | `orange_Citrus_Canker` | Bệnh loét vi khuẩn trên cam | 🔴 Nguy hiểm |
| 9 | 🍊 Cam | `orange_Citrus_Melanose` | Bệnh đốm sẹo gôm đen Melanose | 🟡 Trung bình |
| 10 | 🍊 Cam | `orange_Healthy` | Cam khỏe mạnh | 🟢 An toàn |
| 11 | 🫐 Lựu | `pomegranate_Alternaria` | Bệnh đốm thối đen Alternaria | 🔴 Nguy hiểm |
| 12 | 🫐 Lựu | `pomegranate_Anthracnose` | Bệnh thán thư quả lựu | 🟡 Trung bình |
| 13 | 🫐 Lựu | `pomegranate_Bacterial_Blight` | Bệnh cháy vi khuẩn trên lựu | 🔴 Nguy hiểm |
| 14 | 🫐 Lựu | `pomegranate_Cercospora` | Bệnh đốm mắt chim Cercospora | 🟡 Trung bình |
| 15 | 🫐 Lựu | `pomegranate_Healthy` | Lựu khỏe mạnh | 🟢 An toàn |
            """)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"[App] Pre-warming model and memory...", flush=True)
    try:
        from inference import warmup_model
        warmup_model()
        import gc
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
    except Exception as _e:
        print(f"[App] Warmup notice: {_e}", flush=True)

    print(f"[App] Starting Gradio web server on 0.0.0.0:{port}...", flush=True)
    demo.queue(default_concurrency_limit=1, max_size=4).launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        theme=gr.themes.Soft(primary_hue="emerald"),
        css=CUSTOM_CSS
    )
