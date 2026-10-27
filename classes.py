"""
Fruit Disease Classification - Complete 17-Class Knowledge Base & Pathogen Database
Integrated with TacNhan.json and EfficientNetB0_model_labels.txt
Classes across: Apple (Táo), Guava (Ổi), Mango (Xoài), Pomegranate (Lựu)
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TAC_NHAN_PATH = BASE_DIR / "TacNhan.json"

FRUIT_CLASSES = [
    "APPLE_Blotch_Apple",
    "APPLE_Healthy_Apple",
    "APPLE_Rot_Apple",
    "APPLE_Scab_Apple",
    "GUAVA_Anthracnose_Guava",
    "GUAVA_Fruitfly_Guava",
    "GUAVA_Healthy_Guava",
    "MANGO_Alternaria_Mango",
    "MANGO_Anthracnose_Mango",
    "MANGO_Black Mould Rot (Aspergillus)_Mango",
    "MANGO_Healthy_Mango",
    "MANGO_Stem and Rot (Lasiodiplodia)_Mango",
    "POMEGRANATE_Alternaria_Pomegranate",
    "POMEGRANATE_Anthracnose_Pomegranate",
    "POMEGRANATE_Bacterial_Blight_Pomegranate",
    "POMEGRANATE_Cercospora_Pomegranate",
    "POMEGRANATE_Healthy_Pomegranate"
]

FRUIT_VN_MAP = {
    "APPLE_Blotch_Apple": "🍎 Táo - Bệnh đốm lá đốm quả (Marssonina Blotch)",
    "APPLE_Healthy_Apple": "🍎 Táo - Khỏe mạnh (Healthy Apple)",
    "APPLE_Rot_Apple": "🍎 Táo - Bệnh thối quả (Black Rot / Bitter Rot)",
    "APPLE_Scab_Apple": "🍎 Táo - Bệnh vảy nấm (Apple Scab)",
    "GUAVA_Anthracnose_Guava": "🍈 Ổi - Bệnh thán thư (Anthracnose)",
    "GUAVA_Fruitfly_Guava": "🍈 Ổi - Bị ruồi đục quả tấn công (Fruitfly Damage)",
    "GUAVA_Healthy_Guava": "🍈 Ổi - Khỏe mạnh (Healthy Guava)",
    "MANGO_Alternaria_Mango": "🥭 Xoài - Bệnh đốm lá đốm quả do nấm Alternaria",
    "MANGO_Anthracnose_Mango": "🥭 Xoài - Bệnh thán thư (Anthracnose)",
    "MANGO_Black Mould Rot (Aspergillus)_Mango": "🥭 Xoài - Bệnh thối mốc đen (Aspergillus Black Mould)",
    "MANGO_Healthy_Mango": "🥭 Xoài - Khỏe mạnh (Healthy Mango)",
    "MANGO_Stem and Rot (Lasiodiplodia)_Mango": "🥭 Xoài - Bệnh thối đầu cuống quả (Stem-end Rot)",
    "POMEGRANATE_Alternaria_Pomegranate": "🫐 Lựu - Bệnh đốm đen thối đen Alternaria",
    "POMEGRANATE_Anthracnose_Pomegranate": "🫐 Lựu - Bệnh thán thư (Anthracnose)",
    "POMEGRANATE_Bacterial_Blight_Pomegranate": "🫐 Lựu - Bệnh cháy vi khuẩn (Bacterial Blight)",
    "POMEGRANATE_Cercospora_Pomegranate": "🫐 Lựu - Bệnh đốm mắt chim Cercospora",
    "POMEGRANATE_Healthy_Pomegranate": "🫐 Lựu - Khỏe mạnh (Healthy Pomegranate)"
}

DISEASE_INFO = {
    "APPLE_Blotch_Apple": {
        "status": "warning",
        "fruit": "Táo",
        "severity": "Trung bình",
        "description": "Vết bệnh màu nâu sẫm hoặc xám đen, viền răng cưa không đều trên vỏ táo và mặt lá, làm giảm phẩm cấp quả.",
        "treatment": "Cắt tỉa cành thông thoáng, thu dọn tàn dư lá rụng. Phun luân phiên thuốc gốc đồng hoặc Captan, Mancozeb."
    },
    "APPLE_Healthy_Apple": {
        "status": "healthy",
        "fruit": "Táo",
        "severity": "Không có",
        "description": "Quả táo phát triển tốt, bề mặt căng bóng đều màu, không có dấu hiệu nấm bệnh hay côn trùng gây hại.",
        "treatment": "Tiếp tục quy trình tưới tiêu và bổ sung phân bón trung vi lượng cân đối."
    },
    "APPLE_Rot_Apple": {
        "status": "danger",
        "fruit": "Táo",
        "severity": "Cao",
        "description": "Vùng thối mềm màu nâu đen lan tròn nhanh chóng, phần thịt quả bên trong bị phân hủy nhũn chua.",
        "treatment": "Thu gom tiêu hủy ngay các quả thối để tránh lây nhiễm. Phun phòng ngừa Thiophanate-methyl hoặc Fludioxonil sau đợt mưa kéo dài."
    },
    "APPLE_Scab_Apple": {
        "status": "warning",
        "fruit": "Táo",
        "severity": "Trung bình",
        "description": "Các vết đốm màu xanh ô liu dần chuyển sang màu đen như lớp vảy mốc, làm quả bị nứt nẻ và dị dạng.",
        "treatment": "Thu gom lá khô cuối vụ. Phun Myclobutanil hoặc Difenoconazole vào giai đoạn nụ hoa và quả non."
    },
    "GUAVA_Anthracnose_Guava": {
        "status": "warning",
        "fruit": "Ổi",
        "severity": "Trung bình",
        "description": "Đốm bệnh tròn lõm màu nâu xám trên vỏ ổi, có hạt bào tử hồng nhạt khi độ ẩm không khí cao.",
        "treatment": "Bao trái sớm. Phun thuốc trừ nấm gốc Azoxystrobin, Difenoconazole hoặc Copper Hydroxide."
    },
    "GUAVA_Fruitfly_Guava": {
        "status": "danger",
        "fruit": "Ổi",
        "severity": "Rất cao",
        "description": "Vết châm nhỏ của ruồi đục quả, bên trong có ấu trùng (dòi) ăn rỗng thịt quả làm quả nhanh thối và rụng non.",
        "treatment": "Bao trái bằng túi chuyên dụng khi quả cỡ ngón tay cái. Dùng bẫy dẫn dụ Methyl Eugenol diệt ruồi trưởng thành."
    },
    "GUAVA_Healthy_Guava": {
        "status": "healthy",
        "fruit": "Ổi",
        "severity": "Không có",
        "description": "Quả ổi căng bóng, màu xanh sáng tự nhiên, cấu trúc thịt quả giòn và lành lặn.",
        "treatment": "Bao bọc bảo vệ quả chống côn trùng và bón phân hữu cơ định kỳ."
    },
    "MANGO_Alternaria_Mango": {
        "status": "warning",
        "fruit": "Xoài",
        "severity": "Trung bình",
        "description": "Đốm bệnh hình tròn hoặc bầu dục màu nâu sậm, có các vòng đồng tâm trên vỏ xoài làm quả mất giá trị.",
        "treatment": "Phun phòng trừ nấm Mancozeb hoặc Chlorothalonil khi thời tiết ấm ẩm."
    },
    "MANGO_Anthracnose_Mango": {
        "status": "warning",
        "fruit": "Xoài",
        "severity": "Trung bình - Cao",
        "description": "Các mảng đốm đen loang lổ trên vỏ xoài, hoa và lá non, gây thui hoa và rụng quả non hàng loạt.",
        "treatment": "Tỉa cành tạo độ thông thoáng. Phun phòng trừ bằng Prochloraz, Difenoconazole trước và sau khi trổ hoa."
    },
    "MANGO_Black Mould Rot (Aspergillus)_Mango": {
        "status": "danger",
        "fruit": "Xoài",
        "severity": "Cao",
        "description": "Vết bệnh phủ lớp bột mốc đen mịn như nhung trên bề mặt quả bị trầy xước trong quá trình thu hoạch.",
        "treatment": "Tránh làm trầy xước vỏ khi thu hái và vận chuyển. Rửa sạch quả sau thu hoạch bằng nước ấm."
    },
    "MANGO_Healthy_Mango": {
        "status": "healthy",
        "fruit": "Xoài",
        "severity": "Không có",
        "description": "Quả xoài phát triển toàn diện, vỏ láng mịn, màu sắc tự nhiên không tì vết.",
        "treatment": "Duy trì bao quả và chế độ bón phân Kali giúp vỏ xoài sáng đẹp."
    },
    "MANGO_Stem and Rot (Lasiodiplodia)_Mango": {
        "status": "danger",
        "fruit": "Xoài",
        "severity": "Cao",
        "description": "Bắt đầu thối sẫm màu từ phần cuống quả lan rộng xuống thân quả sau thu hoạch và tồn trữ.",
        "treatment": "Cắt cuống dài 1-2cm khi thu hái, xử lý quả bằng nước ấm 52°C trong 5 phút trước khi đóng gói."
    },
    "POMEGRANATE_Alternaria_Pomegranate": {
        "status": "warning",
        "fruit": "Lựu",
        "severity": "Trung bình",
        "description": "Bệnh thối tim Alternaria xâm nhập từ đài hoa làm các hạt lựu bên trong chuyển màu đen thối chua.",
        "treatment": "Phun thuốc phòng trừ nấm ngay từ giai đoạn rụng cánh hoa."
    },
    "POMEGRANATE_Anthracnose_Pomegranate": {
        "status": "warning",
        "fruit": "Lựu",
        "severity": "Trung bình",
        "description": "Các vết đốm trũng tròn màu nâu sẫm trên vỏ quả lựu, làm quả kém phát triển và sượng thịt.",
        "treatment": "Phun Copper Oxychloride hoặc Carbendazim định kỳ 15 ngày/lần trong mùa mưa."
    },
    "POMEGRANATE_Bacterial_Blight_Pomegranate": {
        "status": "danger",
        "fruit": "Lựu",
        "severity": "Rất cao",
        "description": "Vết bệnh màu đen góc cạnh, quả bị nứt dọc hình chữ Y hoặc chữ L, rỉ dịch vi khuẩn nguy hiểm.",
        "treatment": "Cắt bỏ cành bệnh đem tiêu hủy xa vườn. Phun luân phiên Streptomycin Sulfate và thuốc gốc đồng."
    },
    "POMEGRANATE_Cercospora_Pomegranate": {
        "status": "warning",
        "fruit": "Lựu",
        "severity": "Trung bình",
        "description": "Các đốm tròn nhỏ viền đen tâm xám (mắt chim) xuất hiện dày đặc trên lá và vỏ lựu.",
        "treatment": "Phun phòng Mancozeb hoặc Difenoconazole khi phát hiện các đốm đầu tiên."
    },
    "POMEGRANATE_Healthy_Pomegranate": {
        "status": "healthy",
        "fruit": "Lựu",
        "severity": "Không có",
        "description": "Quả lựu căng mọng, vỏ đỏ hồng hoặc vàng bóng, không có khuyết tật hay đốm bệnh.",
        "treatment": "Tưới nước điều độ giữ ẩm đều, tránh để đất khô kiệt rồi tưới đột ngột làm nứt quả."
    }
}

# ============================================================
# TacNhan.json Integration: Load & Map Pathogen Data
# ============================================================
PATHOGEN_MAP = {}
if TAC_NHAN_PATH.is_file():
    try:
        with open(TAC_NHAN_PATH, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            for key, val in raw_data.items():
                nhan_goc = val.get("nhan_goc")
                if nhan_goc:
                    PATHOGEN_MAP[nhan_goc] = val
        print(f"[classes.py] Loaded {len(PATHOGEN_MAP)} pathogen entries from TacNhan.json")
    except Exception as e:
        print(f"[classes.py] Warning: Error reading TacNhan.json: {e}")


def get_pathogen_info(class_name: str) -> dict:
    """Retrieve detailed pathogen cause information for a specific class."""
    if class_name in PATHOGEN_MAP:
        return PATHOGEN_MAP[class_name]
    return {
        "nhan_goc": class_name,
        "ten_tieng_viet": FRUIT_VN_MAP.get(class_name, class_name),
        "tac_nhan": [],
        "loai": "Không có",
        "ghi_chu": ""
    }
