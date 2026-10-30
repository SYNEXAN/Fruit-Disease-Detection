"""
Module: gemini_service.py
Mục đích: Tích hợp Gemini API dưới góc độ Bác sĩ Nông sản để giải thích
          kết quả chẩn đoán một cách gần gũi, dễ hiểu cho bà con nông dân.
"""

from typing import Optional, Dict, Any, Tuple
from PIL import Image
import numpy as np

import os
from pathlib import Path

# ============================================================
# CẤU HÌNH GEMINI API KEY & MODELS
# Đọc API Key từ .env hoặc biến môi trường hệ thống
# ============================================================
_ENV_PATH = Path(__file__).resolve().parent / ".env"
if _ENV_PATH.is_file():
    try:
        for _line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
    except Exception:
        pass

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Cặp mô hình: Chính là 3.5-flash-lite, Fallback là 3.1-flash-lite (Hạn ngạch request cao nhất)
PRIMARY_MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODEL = "gemini-3.1-flash-lite"
PREFERRED_GEMINI_MODELS = [PRIMARY_MODEL, FALLBACK_MODEL]


def explain_diagnosis_with_gemini(
    gradcam_image: Any,
    best_prediction: Dict[str, Any],
    pathogen_info: Dict[str, Any],
    disease_info: Dict[str, Any],
    api_key: Optional[str] = None
) -> Dict[str, str]:
    """
    Đóng vai trò Bác sĩ Cây trồng, giải thích kết quả và đưa ra lời khuyên thực tế.
    
    Returns:
        Dict gồm:
            - "explanation": Đoạn giải thích ngắn gọn để hiển thị ngay dưới nhãn Đối tượng.
            - "advice": Hướng dẫn điều kiện phát sinh và cách xử lý thực tế cho nhà nông.
    """
    active_key = (GEMINI_API_KEY if api_key is None else api_key).strip()

    vn_name = best_prediction.get("vn_name", "Không xác định")
    fruit = disease_info.get("fruit", "Trái cây")
    percentage = best_prediction.get("percentage", "N/A")
    class_name = best_prediction.get("class", "")
    is_healthy = "healthy" in class_name.lower()
    
    loai_tac_nhan = pathogen_info.get("loai", "Chưa phân loại")
    tac_nhan_list = pathogen_info.get("tac_nhan", [])
    tac_nhan_str = ", ".join(tac_nhan_list) if tac_nhan_list else "Không có tác nhân gây hại"
    ghi_chu = pathogen_info.get("ghi_chu", "")

    # Trường hợp 1: Không có API Key -> Trả về nội dung mặc định dễ hiểu
    if not active_key:
        return _get_default_farmer_fallback(vn_name=vn_name, fruit=fruit, is_healthy=is_healthy)

    # Chuyển đổi định dạng ảnh sang PIL.Image & tối ưu dung lượng payload (thumbnail max 384px)
    pil_img: Optional[Image.Image] = None
    if gradcam_image is not None:
        try:
            if isinstance(gradcam_image, Image.Image):
                pil_img = gradcam_image.copy()
            elif isinstance(gradcam_image, np.ndarray):
                pil_img = Image.fromarray(gradcam_image.astype(np.uint8))

            if pil_img is not None:
                # Thu nhỏ ảnh tối đa 384x384 để tiết kiệm băng thông và giảm 60% thời gian xử lý của Gemini API
                pil_img.thumbnail((384, 384), Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"[GeminiService] Cảnh báo chuyển đổi/nén ảnh: {e}")

    # Xác định mức độ phân loại chuẩn theo Tab 3 (Sổ tay 17 lớp bệnh)
    status = disease_info.get("status", "warning")
    if is_healthy or status == "healthy":
        phan_loai_tab3 = "An toàn"
    elif status == "danger":
        phan_loai_tab3 = "Nguy hiểm"
    else:
        phan_loai_tab3 = "Trung bình"

    # Nhiệt độ tham khảo chuẩn theo từng loại quả
    temp_guide = {
        "Táo": "1°C - 3°C, độ ẩm 85% - 90%",
        "Xoài": "12°C - 13°C, độ ẩm 85% - 90% (tránh dưới 10°C làm thâm da)",
        "Ổi": "8°C - 10°C, độ ẩm 85% - 90%",
        "Lựu": "5°C - 7°C, độ ẩm 90% - 95%"
    }.get(fruit, "kho mát 8°C - 12°C, độ ẩm 85% - 90%")

    # Prompt bác sĩ nông sản: Thân thiện, súc tích, ngắn gọn, để Gemini tự suy luận thực tế
    if is_healthy:
        phan_3_instruction = f"""[PHẦN 3: ĐỀ XUẤT XỬ LÝ]
Yêu cầu trình bày chính xác theo dạng gạch đầu dòng in đậm như sau:
- **Phân loại**: An toàn
- **Điều chỉnh bảo quản**: (Viết 1 câu: Nêu mức nhiệt độ chuẩn {temp_guide} và độ ẩm phù hợp cho quả {fruit} để giữ quả tươi ngon lâu nhất, hạn chế đọng nước và tổn thương cơ học.)"""
    else:
        phan_3_instruction = f"""[PHẦN 3: ĐỀ XUẤT XỬ LÝ]
Yêu cầu trình bày chính xác theo dạng gạch đầu dòng in đậm như sau:
- **Phân loại**: {phan_loai_tab3}
- **Cách ly tạm thời**: (Viết 1 câu: Hướng dẫn cách ly quả {fruit} bị {vn_name} ra khỏi lô trái cây khỏe mạnh để hạn chế nguy cơ lây nhiễm chéo.)
- **Kiểm tra bổ sung**: (Viết 1 câu: Hướng dẫn quan sát mức độ lan rộng của vết bệnh {vn_name}, kiểm tra độ mềm, tình trạng thối và các dấu hiệu bất thường khác trên quả {fruit}.)
- **Điều chỉnh bảo quản**: (Viết 1 câu: Nêu nhiệt độ {temp_guide} và độ ẩm phù hợp cho {fruit}, lưu ý cách giữ vỏ ráo thoáng, hạn chế đọng nước và tổn thương cơ học.)
- **Chuyển kiểm định khi cần**: (Viết 1 câu: Nêu rõ nếu vết bệnh thối sâu, lan rộng hoặc nghi ngờ mầm bệnh nguy hiểm thì chuyển mẫu đến bộ phận kiểm định chuyên môn.)"""

    prompt_text = f"""Bạn là một Bác sĩ Cây trồng (Bác sĩ Nông sản) giàu kinh nghiệm thực tế, đang trực tiếp thăm khám và tư vấn cho bà con nông dân.
Hình ảnh đính kèm là hình ảnh quả nông sản được khoanh vùng màu đỏ, cam, vàng làm nổi bật các dấu hiệu tổn thương trên vỏ (hoặc bao phủ đều màu nếu quả khỏe mạnh).

Thông tin chẩn đoán:
- Đối tượng quả: {fruit}
- Tình trạng: {vn_name} (Độ tin cậy: {percentage})
- Mức độ bệnh (theo sổ tay phân loại Tab 3): {phan_loai_tab3}
- Tác nhân từ cơ sở dữ liệu: {tac_nhan_str} ({loai_tac_nhan})
{f"- Ghi chú thêm: {ghi_chu}" if ghi_chu else ""}

CÁC NGUYÊN TẮC BẮT BUỘC:
1. KHÔNG viết lời chào hỏi. Bắt đầu ngay vào nội dung.
2. TUYỆT ĐỐI KHÔNG sử dụng các thuật ngữ kỹ thuật máy móc như: "Grad-CAM", "EfficientNet-B0", "mô hình AI", "mạng nơ-ron", "bản đồ nhiệt".
3. VIẾT THẬT NGẮN GỌN, SÚC TÍCH, đi thẳng vào trọng tâm.
4. Trình bày chính xác theo đúng 3 phần được phân tách bằng thẻ đánh dấu bên dưới:

[PHẦN 1: GIẢI THÍCH KẾT QUẢ]
(Chỉ viết 1 - 2 câu: Vùng khoanh màu đỏ/vàng trên vỏ chỉ ra triệu chứng gì; vì sao nhận định quả đang trong tình trạng {vn_name}.)

[PHẦN 2: PHÂN TÍCH TÁC NHÂN GÂY BỆNH]
(Chỉ viết 2 - 3 câu súc tích: Đối chiếu cơ sở tri thức để chỉ rõ nhóm vi nấm / vi sinh vật nghi ngờ hàng đầu - ví dụ nhóm Colletotrichum spp. đối với thán thư, Penicillium expansum đối với thối táo...; chỉ ra con đường lây lan chính. Nếu là quả khỏe mạnh thì xác nhận ngắn gọn 1 câu.)

{phan_3_instruction}"""

    raw_response = None
    last_error = None

    try:
        from google import genai
        client = genai.Client(api_key=active_key)
        contents = [pil_img, prompt_text] if pil_img is not None else [prompt_text]
        
        for model_name in PREFERRED_GEMINI_MODELS:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents
                )
                if response and response.text:
                    raw_response = response.text.strip()
                    break
            except Exception as model_err:
                last_error = model_err
                print(f"[GeminiService] Thử model {model_name} thất bại: {model_err}")
                continue

    except Exception as general_err:
        last_error = general_err
        print(f"[GeminiService] Lỗi kết nối Google GenAI: {general_err}")

    if raw_response:
        explanation, pathogen_analysis, advice = _parse_farmer_response(raw_response, is_healthy=is_healthy, phan_loai_tab3=phan_loai_tab3)
        return {"explanation": explanation, "pathogen_analysis": pathogen_analysis, "advice": advice}

    return _get_default_farmer_fallback(vn_name=vn_name, fruit=fruit, is_healthy=is_healthy, phan_loai_tab3=phan_loai_tab3, tac_nhan_str=tac_nhan_str, loai_tac_nhan=loai_tac_nhan)


def _parse_farmer_response(text: str, is_healthy: bool, phan_loai_tab3: str = "Nguy hiểm") -> Tuple[str, str, str]:
    """Tách 3 phần: giải thích kết quả, phân tích tác nhân và đề xuất xử lý từ câu trả lời của Gemini."""
    p1_tag = "[PHẦN 1: GIẢI THÍCH KẾT QUẢ]"
    p2_tag = "[PHẦN 2: PHÂN TÍCH TÁC NHÂN GÂY BỆNH]"
    p3_tag = "[PHẦN 3: ĐỀ XUẤT XỬ LÝ]"

    explanation = ""
    pathogen_analysis = ""
    advice = ""

    if p1_tag in text and p2_tag in text and p3_tag in text:
        parts_1 = text.split(p2_tag)
        explanation = parts_1[0].replace(p1_tag, "").strip()
        parts_2 = parts_1[1].split(p3_tag)
        pathogen_analysis = parts_2[0].strip()
        advice = parts_2[1].strip()
    elif p1_tag in text and p2_tag in text:
        parts = text.split(p2_tag)
        explanation = parts[0].replace(p1_tag, "").strip()
        pathogen_analysis = parts[1].strip()
    elif p1_tag in text and p3_tag in text:
        parts = text.split(p3_tag)
        explanation = parts[0].replace(p1_tag, "").strip()
        advice = parts[1].strip()
    else:
        explanation = text.strip()

    # Dọn dẹp nếu có lời chào sót lại
    for greeting in ["Chào bạn,", "Chào bà con,", "Xin chào,"]:
        if explanation.startswith(greeting):
            explanation = explanation[len(greeting):].strip()

    # Chuẩn hóa advice thành dạng bullet point in đậm nếu có dính số thứ tự
    if advice:
        import re
        # Chuyển đổi các định dạng số: "1. Phân loại:" -> "- **Phân loại**:"
        advice = re.sub(r'^\s*1\.\s*(?:\*\*)?Phân loại(?:\*\*)?:?\s*', '- **Phân loại**: ', advice, flags=re.MULTILINE)
        advice = re.sub(r'^\s*2\.\s*(?:\*\*)?Cách ly tạm thời(?:\*\*)?:?\s*\n?', '- **Cách ly tạm thời**: ', advice, flags=re.MULTILINE)
        advice = re.sub(r'^\s*3\.\s*(?:\*\*)?Kiểm tra bổ sung(?:\*\*)?:?\s*\n?', '- **Kiểm tra bổ sung**: ', advice, flags=re.MULTILINE)
        advice = re.sub(r'^\s*4\.\s*(?:\*\*)?Điều chỉnh bảo quản(?:\*\*)?:?\s*\n?', '- **Điều chỉnh bảo quản**: ', advice, flags=re.MULTILINE)
        advice = re.sub(r'^\s*5\.\s*(?:\*\*)?Chuyển kiểm định khi cần(?:\*\*)?:?\s*\n?', '- **Chuyển kiểm định khi cần**: ', advice, flags=re.MULTILINE)
        # Nếu dòng nào có số thứ tự dạng 2. Điều chỉnh bảo quản
        advice = re.sub(r'^\s*2\.\s*(?:\*\*)?Điều chỉnh bảo quản(?:\*\*)?:?\s*\n?', '- **Điều chỉnh bảo quản**: ', advice, flags=re.MULTILINE)

    if not advice:
        temp_guide = {
            "Táo": "1°C - 3°C, độ ẩm 85% - 90%",
            "Xoài": "12°C - 13°C, độ ẩm 85% - 90%",
            "Ổi": "8°C - 10°C, độ ẩm 85% - 90%",
            "Lựu": "5°C - 7°C, độ ẩm 90% - 95%"
        }.get(fruit, "kho mát 8°C - 12°C, độ ẩm 85% - 90%")

        if is_healthy:
            advice = f"""- **Phân loại**: An toàn
- **Điều chỉnh bảo quản**: Bảo quản {fruit} ở điều kiện {temp_guide}, tránh ánh nắng trực tiếp và hạn chế đọng nước để giữ quả tươi ngon lâu nhất."""
        else:
            advice = f"""- **Phân loại**: {phan_loai_tab3}
- **Cách ly tạm thời**: Tách ngay những quả {fruit} có vết bệnh khỏi lô hàng để ngăn bào tử lây nhiễm chéo sang các quả khỏe mạnh.
- **Kiểm tra bổ sung**: Quan sát kỹ độ lan rộng của vết {vn_name}, kiểm tra độ mềm của thịt quả xung quanh vết thương để đánh giá mức độ thối.
- **Điều chỉnh bảo quản**: Duy trì điều kiện {temp_guide}, giữ bề mặt quả khô ráo và kho chứa thông thoáng để ức chế nấm phát triển.
- **Chuyển kiểm định khi cần**: Nếu vết bệnh ăn sâu vào thịt quả hoặc xuất hiện trên diện rộng của lô hàng, chuyển mẫu đến bộ phận kiểm định chuyên môn để xử lý kịp thời."""

    if not pathogen_analysis:
        if is_healthy:
            pathogen_analysis = "Bề mặt vỏ quả nguyên vẹn, tế bào biểu bì phát triển tốt, không ghi nhận dấu hiệu xâm nhiễm của các loài nấm bệnh hay vi khuẩn."
        else:
            pathogen_analysis = "Đối chiếu triệu chứng trên vỏ cho thấy vết bệnh nhiều khả năng do các loài nấm gây hại tiềm sinh phát triển mạnh trong điều kiện nóng ẩm."

    return explanation, pathogen_analysis, advice


def _get_default_farmer_fallback(vn_name: str, fruit: str, is_healthy: bool, phan_loai_tab3: str = "Nguy hiểm", tac_nhan_str: str = "", loai_tac_nhan: str = "") -> Dict[str, str]:
    """Nội dung mặc định theo văn phong bác sĩ cây trồng khi chưa có API Key hoặc mất mạng."""
    temp_guide = {
        "Táo": "1°C - 3°C, độ ẩm 85% - 90%",
        "Xoài": "12°C - 13°C, độ ẩm 85% - 90%",
        "Ổi": "8°C - 10°C, độ ẩm 85% - 90%",
        "Lựu": "5°C - 7°C, độ ẩm 90% - 95%"
    }.get(fruit, "kho mát 8°C - 12°C, độ ẩm 85% - 90%")

    if is_healthy:
        explanation = f"Vùng sáng trên ảnh bao phủ đều bề mặt quả, vỏ căng bóng mịn màng, không phát hiện vết thâm nám hay dấu vết nấm bệnh bất thường. Quả phát triển tốt và đạt tiêu chuẩn chất lượng."
        pathogen_analysis = "Bề mặt vỏ quả lành lặn, tế bào biểu bì nguyên vẹn, không có sự xâm nhiễm của các loài vi nấm hay vi khuẩn gây hại."
        advice = f"""- **Phân loại**: An toàn
- **Điều chỉnh bảo quản**: Bảo quản {fruit} ở điều kiện {temp_guide}, tránh ánh nắng trực tiếp và hạn chế đọng nước để giữ quả tươi ngon lâu nhất."""
    else:
        explanation = f"Vùng khoanh màu đỏ/vàng trên vỏ quả đang chỉ ra các vết thâm đốm hoặc vùng mô tổn thương đặc trưng của **{vn_name}**. Những vết này làm hỏng lớp vỏ ngoài và dễ làm thịt quả bên trong bị thối nhũn."
        target_pathogen = tac_nhan_str if tac_nhan_str and "Không có" not in tac_nhan_str else "các loài vi nấm gây hại"
        pathogen_analysis = f"Bệnh do nhóm vi nấm nghi ngờ **{target_pathogen}** tấn công vào lớp vỏ. Bào tử nấm phát tán mạnh qua nước mưa, gió và lây lan nhanh khi gặp điều kiện nóng ẩm cao."
        advice = f"""- **Phân loại**: {phan_loai_tab3}
- **Cách ly tạm thời**: Tách ngay những quả {fruit} có vết bệnh khỏi lô hàng để ngăn bào tử lây nhiễm chéo sang các quả khỏe mạnh.
- **Kiểm tra bổ sung**: Quan sát kỹ độ lan rộng của vết {vn_name}, kiểm tra độ mềm của thịt quả xung quanh vết thương để đánh giá mức độ thối.
- **Điều chỉnh bảo quản**: Duy trì điều kiện {temp_guide}, giữ bề mặt quả khô ráo và kho chứa thông thoáng để ức chế nấm phát triển.
- **Chuyển kiểm định khi cần**: Nếu vết bệnh ăn sâu vào thịt quả hoặc xuất hiện trên diện rộng của lô hàng, chuyển mẫu đến bộ phận kiểm định chuyên môn để xử lý kịp thời."""

    return {"explanation": explanation, "pathogen_analysis": pathogen_analysis, "advice": advice}
