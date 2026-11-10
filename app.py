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
def run_diagnosis(input_img, orig_cache=None, conf_threshold=0.50):
    """
    Handle inference request from Gradio interface.
    Executes in 2 progressive stages:
    - Stage 1 (~1.05s): Immediately yields Grad-CAM image, class, and confidence with status notice.
    - Stage 2 (~2s): Deep Gemini analysis yields full pathogen & advice.
    """
    # Xác định ảnh gốc hợp lệ: nếu input_img là đường dẫn thì lưu vào cache
    if isinstance(input_img, str) and input_img:
        actual_img = input_img
        cached_orig = input_img
    elif orig_cache:
        actual_img = orig_cache
        cached_orig = orig_cache
    else:
        actual_img = input_img
        cached_orig = input_img

    if actual_img is None:
        yield None, "**Vui lòng chọn hoặc tải một ảnh trái cây lên để phân tích.**", None
        return

    # Unpack dict if Gradio passes file metadata dict
    if isinstance(actual_img, dict):
        actual_img = actual_img.get("path") or actual_img.get("url")
        if not actual_img:
            yield None, "**Vui lòng chọn hoặc tải một ảnh trái cây hợp lệ.**", None
            return
        cached_orig = actual_img

    cleanup_temp_memory()
    rss_in = get_process_rss_mb()
    if rss_in > 0:
        print(f"[App] Diagnosis start - Current RSS: {rss_in:.1f} MB", flush=True)

    from inference import predict_image
    result = predict_image(actual_img, conf_threshold=conf_threshold, generate_cam=True)

    best = result["best"]
    info = result["disease_info"]
    pathogen = result.get("pathogen_info", {})
    gradcam_img = result.get("gradcam_image")

    # Kiểm tra nếu độ tin cậy chưa đạt ngưỡng yêu cầu
    confidence_val = best.get("confidence", 0.0)
    if confidence_val < conf_threshold:
        threshold_pct = int(round(conf_threshold * 100)) if conf_threshold <= 1.0 else int(conf_threshold)
        warning_md = f"""
### Chưa Đủ Độ Tin Cậy Để Kết Luận ({best.get('percentage', 'N/A')} < {threshold_pct}%):
- **Tình trạng nhận diện**: **Chưa xác định rõ loại bệnh**

**Khuyến nghị**:
- Hình ảnh có thể bị mờ, thiếu sáng, góc chụp quá xa hoặc triệu chứng chưa rõ ràng.
- Vui lòng chụp hoặc tải lại ảnh cận cảnh, lấy nét trực tiếp vào vết bệnh để hệ thống chẩn đoán chính xác.
"""
        yield gradcam_img, warning_md.strip(), cached_orig
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

    # ⚡ YIELD GIAI ĐOẠN 1: Hiển thị ngay ảnh Grad-CAM và kết quả dự đoán sau ~0.5s
    stage1_md = f"""
### Kết Quả Dự Đoán:
- **Tình trạng**: **{best.get('vn_name')}**
- **Độ tin cậy**: **{best.get('percentage', 'N/A')}**

---
*Đang hoàn tất phân tích chi tiết...*
"""
    yield gradcam_img, stage1_md.strip(), cached_orig

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
### Kết Quả Dự Đoán:
- **Tình trạng**: **{best.get('vn_name')}**
- **Độ tin cậy**: **{best.get('percentage', 'N/A')}**
- **Giải thích kết quả**: {explanation_text}

---

#### Tác Nhân Gây Bệnh:
{pathogen_analysis_text}

---

#### Đề xuất xử lý:
{advice_text}
"""
        yield gradcam_img, final_diagnosis_md.strip(), cached_orig

    except Exception as e:
        print(f"[App] Lỗi phân tích Gemini: {e}")
        fallback_md = f"""
### Kết Quả Dự Đoán:
- **Tình trạng**: **{best.get('vn_name')}**
- **Độ tin cậy**: **{best.get('percentage', 'N/A')}**

---

#### Tác Nhân Gây Bệnh:
{initial_pathogen_md}

---

#### Đề xuất xử lý:
- **Phân loại**: {info.get('severity', 'Cảnh báo')}
- **Xử lý**: {info.get('treatment', 'Theo dõi sát diễn biến vết bệnh và kiểm tra bảo quản.')}
"""
        yield gradcam_img, fallback_md.strip(), cached_orig
    finally:
        del gradcam_img, result
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
    padding: 16px 12px;
    background: linear-gradient(135deg, #10b981 0%, #059669 50%, #047857 100%);
    color: white;
    border-radius: 12px;
    margin-bottom: 16px;
    box-shadow: 0 4px 15px rgba(5, 150, 105, 0.2);
}
.main-header h1 {
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0;
    color: #ffffff;
    letter-spacing: 0.5px;
    text-wrap: balance !important;
}
@media (max-width: 768px) {
    .main-header {
        padding: 12px 10px !important;
        margin-bottom: 10px !important;
        border-radius: 8px !important;
    }
    .main-header h1 {
        font-size: 1.25rem !important;
        line-height: 1.35 !important;
        text-wrap: balance !important;
    }
    .tab-nav button, button[role="tab"] {
        font-size: 0.88rem !important;
        padding: 6px 12px !important;
    }
}
@media (max-width: 480px) {
    .main-header {
        padding: 10px 8px !important;
        margin-bottom: 8px !important;
        border-radius: 8px !important;
    }
    .main-header h1 {
        font-size: 1.05rem !important;
        line-height: 1.35 !important;
        text-wrap: balance !important;
        letter-spacing: 0 !important;
    }
    /* Tối ưu thanh 3 nút chọn nguồn: Tải ảnh, Chụp ảnh, Dán ảnh */
    .source-selection, [data-testid="source-select"] {
        gap: 4px !important;
        padding: 4px 2px !important;
    }
    .source-selection button, [data-testid="source-select"] button {
        height: 28px !important;
        padding: 3px 6px !important;
        gap: 3px !important;
        font-size: 0.76rem !important;
        white-space: nowrap !important;
    }
    .source-selection button svg, [data-testid="source-select"] button svg {
        width: 14px !important;
        height: 14px !important;
    }
    .source-selection button[aria-label*="Upload"]::after,
    [data-testid="source-select"] button[aria-label*="Upload"]::after,
    .source-selection button[aria-label*="camera"]::after,
    [data-testid="source-select"] button[aria-label*="camera"]::after,
    .source-selection button[aria-label*="clipboard"]::after,
    [data-testid="source-select"] button[aria-label*="clipboard"]::after {
        font-size: 0.76rem !important;
        margin-left: 2px !important;
    }
    /* Thu gọn nút công cụ góc phải trên khung ảnh để nhường chỗ */
    [data-testid="image"] .icon-button,
    [data-testid="image"] button[aria-label="Clear"],
    [data-testid="image"] button[aria-label="Download"],
    [data-testid="image"] button[aria-label="Share"],
    [data-testid="image"] button[aria-label="Fullscreen"] {
        width: 24px !important;
        height: 24px !important;
        padding: 2px !important;
    }
    /* Chiều cao khung ảnh cân đối với màn hình 667px của iPhone SE */
    .image-container img, [data-testid="image"] img {
        max-height: 310px !important;
    }
    #image-input-box, #tab_detect [data-testid="image"] {
        min-height: 290px !important;
    }
    .prompt h2 {
        font-size: 1rem !important;
    }
    .prompt p {
        font-size: 0.82rem !important;
        line-height: 1.4 !important;
    }
    #btn-predict-action {
        font-size: 0.95rem !important;
        padding: 10px 14px !important;
    }
    .tab-nav button, button[role="tab"] {
        font-size: 0.82rem !important;
        padding: 5px 10px !important;
    }
}
@media (max-width: 360px) {
    .main-header h1 {
        font-size: 0.95rem !important;
    }
    .source-selection button, [data-testid="source-select"] button {
        padding: 2px 4px !important;
        font-size: 0.72rem !important;
    }
    .source-selection button[aria-label*="Upload"]::after,
    [data-testid="source-select"] button[aria-label*="Upload"]::after,
    .source-selection button[aria-label*="camera"]::after,
    [data-testid="source-select"] button[aria-label*="camera"]::after,
    .source-selection button[aria-label*="clipboard"]::after,
    [data-testid="source-select"] button[aria-label*="clipboard"]::after {
        font-size: 0.72rem !important;
    }
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
/* Tối ưu hiển thị tab điều hướng: vừa vặn kích thước chữ, dồn về bên trái */
.tab-nav {
    display: flex !important;
    flex-wrap: nowrap !important;
    justify-content: flex-start !important;
    align-items: center !important;
    overflow-x: auto !important;
    gap: 8px !important;
    border-bottom: 2px solid #e2e8f0 !important;
    padding-bottom: 2px !important;
    width: 100% !important;
    margin-bottom: 4px !important;
}
.tab-nav button, button[role="tab"] {
    flex: 0 0 auto !important;
    width: auto !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    padding: 8px 16px !important;
    white-space: nowrap !important;
    text-align: center !important;
}
@media (max-width: 768px) {
    .tab-nav button, button[role="tab"] {
        font-size: 0.85rem !important;
        padding: 6px 12px !important;
    }
}
/* Thu hẹp khoảng cách giữa thanh Tab và Khung Tải Ảnh */
.tab-wrapper {
    padding-bottom: 0 !important;
    margin-bottom: 0 !important;
}
.tabitem {
    padding-top: 6px !important;
    padding-bottom: 6px !important;
}
.tabitem > div:first-child, [data-testid="image"] {
    margin-top: 0 !important;
}
/* Đưa thanh 3 nút chọn nguồn (Upload / Camera / Clipboard) lên đầu khung ảnh và thêm nhãn text */
.upload-wrapper {
    display: flex !important;
    flex-direction: column !important;
}
.source-selection, [data-testid="source-select"] {
    order: -1 !important;
    border-top: none !important;
    border-bottom: 1px solid #e2e8f0 !important;
    background: #f8fafc !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 6px 4px !important;
    margin-bottom: 4px !important;
    display: flex !important;
    justify-content: center !important;
    gap: 8px !important;
    height: auto !important;
}
.source-selection button, [data-testid="source-select"] button {
    width: auto !important;
    height: 32px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 4px 10px !important;
    margin: 0 !important;
    border-radius: 6px !important;
    gap: 6px !important;
    cursor: pointer !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
.source-selection button svg, [data-testid="source-select"] button svg {
    width: 17px !important;
    height: 17px !important;
    flex-shrink: 0 !important;
}
.source-selection button[aria-label*="Upload"]::after,
[data-testid="source-select"] button[aria-label*="Upload"]::after {
    content: " Tải ảnh" !important;
    font-size: 0.85rem !important;
    line-height: 1 !important;
    margin-left: 2px !important;
}
.source-selection button[aria-label*="camera"]::after,
[data-testid="source-select"] button[aria-label*="camera"]::after {
    content: " Chụp ảnh" !important;
    font-size: 0.85rem !important;
    line-height: 1 !important;
    margin-left: 2px !important;
}
.source-selection button[aria-label*="clipboard"]::after,
[data-testid="source-select"] button[aria-label*="clipboard"]::after {
    content: " Dán ảnh" !important;
    font-size: 0.85rem !important;
    line-height: 1 !important;
    margin-left: 2px !important;
}
.source-selection button.selected, [data-testid="source-select"] button.selected {
    background: rgba(16, 185, 129, 0.12) !important;
    color: #059669 !important;
    font-weight: 600 !important;
}
.source-selection button:hover, [data-testid="source-select"] button:hover {
    background: #f1f5f9 !important;
}
/* Thu hẹp khoảng cách giữa tiêu đề 'Ảnh mẫu' và danh sách ảnh cuộn ngang */
.sample-header {
    margin-top: 8px !important;
    margin-bottom: 0 !important;
    padding: 0 !important;
}
.sample-header h3 {
    margin-top: 0 !important;
    margin-bottom: 2px !important;
    padding: 0 !important;
    font-size: 1.1rem !important;
}
/* Ẩn hoàn toàn thanh nhãn hamburger của khối ảnh mẫu */
#sample-images-dataset .label {
    display: none !important;
}
#sample-images-dataset, .gradio-container .contain #sample-images-dataset {
    margin-top: -15px !important;
    padding-top: 0 !important;
}
/* Bố trí ảnh mẫu theo 1 hàng ngang và cuộn ngang */
#sample-images-dataset .gallery {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    -webkit-overflow-scrolling: touch !important;
    gap: 10px !important;
    padding: 4px 2px 8px 2px !important;
}
#sample-images-dataset .gallery-item {
    flex: 0 0 auto !important;
    width: 72px !important;
    height: 72px !important;
    border-radius: 8px !important;
    border: 2px solid #e2e8f0 !important;
    overflow: hidden !important;
    cursor: pointer !important;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease !important;
}
#sample-images-dataset .gallery-item:hover {
    border-color: #10b981 !important;
    transform: scale(1.06) !important;
    box-shadow: 0 4px 10px rgba(16, 185, 129, 0.25) !important;
}
#sample-images-dataset .gallery-item img {
    width: 100% !important;
    height: 100% !important;
    object-fit: cover !important;
}
/* Ẩn hoàn toàn khối phân trang */
#sample-images-dataset .paginate {
    display: none !important;
}
/* Tinh chỉnh thanh cuộn ngang thanh mảnh */
#sample-images-dataset .gallery::-webkit-scrollbar {
    height: 5px !important;
}
#sample-images-dataset .gallery::-webkit-scrollbar-track {
    background: #f1f5f9 !important;
    border-radius: 4px !important;
}
#sample-images-dataset .gallery::-webkit-scrollbar-thumb {
    background: #cbd5e1 !important;
    border-radius: 4px !important;
}
#sample-images-dataset .gallery::-webkit-scrollbar-thumb:hover {
    background: #10b981 !important;
}
/* Tối ưu responsive bảng Sổ tay bệnh trên màn hình nhỏ / di động */
#tab_handbook, #handbook-content {
    overflow-x: auto !important;
    -webkit-overflow-scrolling: touch !important;
    max-width: 100% !important;
}
#handbook-content table, #tab_handbook table {
    width: 100% !important;
    min-width: 620px !important;
    border-collapse: collapse !important;
    margin-top: 10px !important;
    font-size: 0.92rem !important;
}
#handbook-content th, #tab_handbook th {
    background: #f8fafc !important;
    color: #0f172a !important;
    font-weight: 600 !important;
    padding: 10px 14px !important;
    border-bottom: 2px solid #e2e8f0 !important;
    white-space: nowrap !important;
    text-align: left !important;
}
#handbook-content th:first-child, #handbook-content td:first-child,
#tab_handbook th:first-child, #tab_handbook td:first-child {
    text-align: center !important;
}
#handbook-content td, #tab_handbook td {
    padding: 9px 14px !important;
    border-bottom: 1px solid #f1f5f9 !important;
    vertical-align: middle !important;
}
#handbook-content tr:hover, #tab_handbook tr:hover {
    background: #f8fafc !important;
}
#handbook-content td:nth-child(1),
#handbook-content td:nth-child(2),
#handbook-content td:nth-child(3),
#handbook-content td:nth-child(5),
#tab_handbook td:nth-child(1),
#tab_handbook td:nth-child(2),
#tab_handbook td:nth-child(3),
#tab_handbook td:nth-child(5) {
    white-space: nowrap !important;
}
#handbook-content td code, #tab_handbook td code {
    font-size: 0.84rem !important;
    padding: 2px 6px !important;
    background: #f1f5f9 !important;
    border-radius: 4px !important;
    color: #0369a1 !important;
    white-space: nowrap !important;
}
/* Định dạng và căn chỉnh khu vực kéo thả / tải ảnh tiếng Việt */
.prompt h2 {
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    color: #059669 !important;
    text-align: center !important;
    margin-bottom: 2px !important;
}
.prompt p {
    white-space: pre-line !important;
    text-align: center !important;
    line-height: 1.6 !important;
    margin-top: 4px !important;
    color: #64748b !important;
}
/* Ẩn hoàn toàn footer mặc định của Gradio (Use via API, Built with Gradio, Settings) */
footer, .gradio-container footer, [data-testid="footer"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    overflow: hidden !important;
}
/* Giới hạn kích thước khung ảnh và nút phân tích trên màn hình lớn */
#image-input-box,
#tab_detect [data-testid="image"] {
    max-width: 680px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    width: 100% !important;
}
#btn-predict-action {
    max-width: 680px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    display: block !important;
    width: 100% !important;
}
"""

with gr.Blocks(title="Hệ thống nhận diện và chẩn đoán bệnh cây ăn quả") as demo:
    # Header
    gr.HTML("""
    <div class="main-header">
        <h1>Hệ thống nhận diện và chẩn đoán bệnh cây ăn quả</h1>
    </div>
    """)

    # Main Tabs
    with gr.Tabs(overflow_behavior="wrap"):
        # Tab 1: Image & Camera Diagnosis (Bố cục 1 cột chuẩn ứng dụng)
        with gr.TabItem("Nhận diện", id="tab_detect"):
            orig_state = gr.State(value=None)
            img_input = gr.Image(
                sources=["upload", "webcam", "clipboard"],
                type="filepath",
                show_label=False,
                height=380,
                buttons=["download", "share", "fullscreen"],
                placeholder="# Kéo thả ảnh vào đây\n- hoặc -\nBấm để tải ảnh lên",
                elem_id="image-input-box"
            )
            btn_predict = gr.Button("Phân Tích & Chẩn Đoán", variant="primary", size="lg", elem_id="btn-predict-action")

            if SAMPLE_IMAGES:
                gr.Markdown("### Ảnh mẫu", elem_classes=["sample-header"])
                gr.Examples(
                    examples=SAMPLE_IMAGES,
                    inputs=img_input,
                    label="",
                    elem_id="sample-images-dataset",
                    examples_per_page=100
                )

            diagnosis_output = gr.Markdown(
                "*Tải ảnh lên hoặc chụp camera rồi bấm 'Phân Tích & Chẩn Đoán' để xem kết quả.*",
                elem_classes=["diagnosis-container"]
            )

            btn_predict.click(
                fn=run_diagnosis,
                inputs=[img_input, orig_state],
                outputs=[img_input, diagnosis_output, orig_state],
                concurrency_limit=1
            )

        # Tab 2: Handbook & Classes
        with gr.TabItem("Sổ tay bệnh", id="tab_handbook"):
            gr.Markdown("""
### Danh Sách 15 Lớp Bệnh Được Huấn Luyện Trong Mô Hình (ConvNeXt-V2 Advanced CMTL):

| STT | Loại Quả | Tên Lớp Kỹ Thuật | Tên Bệnh Tiếng Việt | Mức Độ |
| :---: | :---: | :--- | :--- | :---: |
| 1 | Xoài | `mango_Anthracnose` | Bệnh thán thư xoài | ⚪ Nguy hiểm |
| 2 | Xoài | `mango_Bacterial_Canker` | Bệnh loét vi khuẩn xoài | ⚪ Nguy hiểm |
| 3 | Xoài | `mango_Healthy` | Xoài khỏe mạnh | ⚪ An toàn |
| 4 | Xoài | `mango_Powdery_Mildew` | Bệnh phấn trắng xoài | ⚪ Trung bình |
| 5 | Xoài | `mango_Scab` | Bệnh vảy nấm ghẻ xoài | ⚪ Trung bình |
| 6 | Xoài | `mango_Sooty_Mould` | Bệnh bồ hóng muội đen xoài | ⚪ Trung bình |
| 7 | Xoài | `mango_Stem_End_Rot` | Bệnh thối đầu cuống quả xoài | ⚪ Nguy hiểm |
| 8 | Cam | `orange_Citrus_Canker` | Bệnh loét vi khuẩn trên cam | ⚪ Nguy hiểm |
| 9 | Cam | `orange_Citrus_Melanose` | Bệnh đốm sẹo gôm đen Melanose | ⚪ Trung bình |
| 10 | Cam | `orange_Healthy` | Cam khỏe mạnh | ⚪ An toàn |
| 11 | Lựu | `pomegranate_Alternaria` | Bệnh đốm thối đen Alternaria | ⚪ Nguy hiểm |
| 12 | Lựu | `pomegranate_Anthracnose` | Bệnh thán thư quả lựu | ⚪ Trung bình |
| 13 | Lựu | `pomegranate_Bacterial_Blight` | Bệnh cháy vi khuẩn trên lựu | ⚪ Nguy hiểm |
| 14 | Lựu | `pomegranate_Cercospora` | Bệnh đốm mắt chim Cercospora | ⚪ Trung bình |
| 15 | Lựu | `pomegranate_Healthy` | Lựu khỏe mạnh | ⚪ An toàn |
            """, elem_id="handbook-content")

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
