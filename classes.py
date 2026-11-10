"""
Fruit Disease Classification - Complete 15-Class Knowledge Base & Pathogen Database
Covers 3 Fruit Types: Mango (Xoài), Orange (Cam), Pomegranate (Lựu)
Synchronized with ConvNeXt-V2 Advanced CMTL (Version 6 SOTA) and TacNhan.json
"""

import json
from pathlib import Path
from typing import Dict, Any, List

BASE_DIR = Path(__file__).resolve().parent
TAC_NHAN_PATH = BASE_DIR / "TacNhan.json"

FRUIT_CLASSES: List[str] = [
    "mango_Anthracnose",
    "mango_Bacterial_Canker",
    "mango_Healthy",
    "mango_Powdery_Mildew",
    "mango_Scab",
    "mango_Sooty_Mould",
    "mango_Stem_End_Rot",
    "orange_Citrus_Canker",
    "orange_Citrus_Melanose",
    "orange_Healthy",
    "pomegranate_Alternaria",
    "pomegranate_Anthracnose",
    "pomegranate_Bacterial_Blight",
    "pomegranate_Cercospora",
    "pomegranate_Healthy"
]

FRUIT_GROUPS: Dict[str, List[int]] = {
    "mango": [0, 1, 2, 3, 4, 5, 6],
    "orange": [7, 8, 9],
    "pomegranate": [10, 11, 12, 13, 14]
}

FRUIT_NAMES_VN: Dict[str, str] = {
    "mango": "Xoài",
    "orange": "Cam",
    "pomegranate": "Lựu"
}

FRUIT_VN_MAP: Dict[str, str] = {
    "mango_Anthracnose": "Xoài - Bệnh thán thư (Anthracnose)",
    "mango_Bacterial_Canker": "Xoài - Bệnh loét vi khuẩn (Bacterial Canker)",
    "mango_Healthy": "Xoài - Quả khỏe mạnh",
    "mango_Powdery_Mildew": "Xoài - Bệnh phấn trắng (Powdery Mildew)",
    "mango_Scab": "Xoài - Bệnh vảy nấm ghẻ (Mango Scab)",
    "mango_Sooty_Mould": "Xoài - Bệnh bồ hóng muội đen (Sooty Mould)",
    "mango_Stem_End_Rot": "Xoài - Bệnh thối đầu cuống quả (Stem-End Rot)",
    "orange_Citrus_Canker": "Cam - Bệnh loét vi khuẩn (Citrus Canker)",
    "orange_Citrus_Melanose": "Cam - Bệnh đốm sẹo gôm đen Melanose",
    "orange_Healthy": "Cam - Quả khỏe mạnh",
    "pomegranate_Alternaria": "Lựu - Bệnh đốm thối đen Alternaria",
    "pomegranate_Anthracnose": "Lựu - Bệnh thán thư quả lựu",
    "pomegranate_Bacterial_Blight": "Lựu - Bệnh cháy vi khuẩn (Bacterial Blight)",
    "pomegranate_Cercospora": "Lựu - Bệnh đốm mắt chim Cercospora",
    "pomegranate_Healthy": "Lựu - Quả khỏe mạnh"
}

DISEASE_INFO: Dict[str, Dict[str, Any]] = {
    "mango_Anthracnose": {
        "status": "danger",
        "fruit": "Xoài",
        "severity": "Cao",
        "description": "Vết đốm đen tròn hoặc bất định, hơi lõm trên vỏ quả xoài non và chín. Trong điều kiện ẩm ướt, bề mặt vết bệnh xuất hiện các khối bào tử màu hồng cam.",
        "treatment": "Bao trái sớm khi quả đạt kích thước ngón chân cái. Thu gom tiêu hủy quả rụng. Phun luân phiên thuốc gốc Mancozeb, Difenoconazole hoặc Azoxystrobin."
    },
    "mango_Bacterial_Canker": {
        "status": "danger",
        "fruit": "Xoài",
        "severity": "Cao",
        "description": "Các vết nứt hình sao hoặc vết đốm đen dạng mắt lưới góc cạnh, có dịch vi khuẩn nhờn ứa ra trên vỏ quả, sau đó khô lại thành vảy cứng.",
        "treatment": "Cắt tỉa tạo tán thông thoáng, tránh để cọ xát gây vết thương cơ giới. Phun phòng trừ bằng thuốc gốc đồng (Copper Hydroxide, Copper Oxychloride) hoặc Kasugamycin."
    },
    "mango_Healthy": {
        "status": "healthy",
        "fruit": "Xoài",
        "severity": "Không có",
        "description": "Quả xoài phát triển bình thường, vỏ căng bóng, đồng đều màu sắc, không có vết thâm nhiễm nấm khuẩn hay vết chích hút côn trùng.",
        "treatment": "Duy trì chế độ dinh dưỡng NPK cân đối, bổ sung canxi-bo để vỏ quả chắc khỏe, chống nứt."
    },
    "mango_Powdery_Mildew": {
        "status": "warning",
        "fruit": "Xoài",
        "severity": "Trung bình",
        "description": "Lớp bột trắng như phấn mịn bao phủ bề mặt vỏ quả non, làm quả bị chai sần, méo mó và rụng hàng loạt.",
        "treatment": "Phun phòng ngừa từ giai đoạn ra hoa đến khi đậu quả non bằng lưu huỳnh vi sinh (Sulfur), Hexaconazole hoặc Dinocap."
    },
    "mango_Scab": {
        "status": "warning",
        "fruit": "Xoài",
        "severity": "Trung bình",
        "description": "Vết bệnh ban đầu là chấm xám nhỏ, sau phát triển thành mảng sẹo vảy sần sùi màu nâu xám lồi lên như da cóc, làm quả mất giá trị thương phẩm.",
        "treatment": "Cắt bỏ cành sâu bệnh sau thu hoạch. Phun thuốc gốc đồng hoặc Carbendazim giai đoạn quả non."
    },
    "mango_Sooty_Mould": {
        "status": "warning",
        "fruit": "Xoài",
        "severity": "Trung bình",
        "description": "Màng muội đen bám dày như lớp nhung than trên bề mặt quả, cản trở quang hợp và hô hấp. Lớp muội có thể cạo tróc ra.",
        "treatment": "Diệt trừ triệt để các loài côn trùng tiết mật ngọt như rệp sáp, rầy chổng cánh bằng dầu khoáng hoặc thuốc trừ rầy. Phun nước áp lực mạnh rửa trôi muội than."
    },
    "mango_Stem_End_Rot": {
        "status": "danger",
        "fruit": "Xoài",
        "severity": "Rất cao",
        "description": "Vùng thối màu nâu sẫm hoặc tím đen loang nhanh từ cuống quả lan dần xuống thân quả, phần thịt bên trong bị thối rữa và bốc mùi chua.",
        "treatment": "Thu hái chừa cuống 5-10cm, tránh bẻ sát cuống làm chảy mủ. Xử lý nước nóng 52°C trong 5 phút sau thu hoạch, bảo quản nơi thoáng mát."
    },
    "orange_Citrus_Canker": {
        "status": "danger",
        "fruit": "Cam",
        "severity": "Cao",
        "description": "Các vết sẹo nổi gồ ghề dạng nốt sần màu nâu xám, xung quanh có quầng vàng sáng đặc trưng (halo). Vết bệnh có tâm trũng nứt nẻ.",
        "treatment": "Phun thuốc gốc đồng định kỳ khi cây ra đọt non và quả mới tượng. Phòng trừ sâu vẽ bùa vì đường đục của sâu là cửa ngõ cho vi khuẩn xâm nhập."
    },
    "orange_Citrus_Melanose": {
        "status": "warning",
        "fruit": "Cam",
        "severity": "Trung bình",
        "description": "Vô số nốt sần nhỏ li ti màu nâu đen bám ráp như giấy nhám trên vỏ quả, thường xếp thành vệt dài chảy dọc theo dòng nước mưa.",
        "treatment": "Cắt tỉa triệt để toàn bộ cành khô, cành chết trong tán vì đây là ổ lưu tồn nấm. Phun thuốc trừ nấm gốc Azoxystrobin hoặc Difenoconazole sau khi hoa tàn."
    },
    "orange_Healthy": {
        "status": "healthy",
        "fruit": "Cam",
        "severity": "Không có",
        "description": "Vỏ quả cam láng mịn, màu sắc sáng khỏe, túi tinh dầu đều đặn, không có vết thâm sẹo nấm bệnh.",
        "treatment": "Bổ sung phân hữu cơ hoai mục, duy trì độ ẩm đất và phun bổ sung vi lượng Bo, Kẽm để quả mọng nước, vỏ bóng."
    },
    "pomegranate_Alternaria": {
        "status": "danger",
        "fruit": "Lựu",
        "severity": "Cao",
        "description": "Các vết tròn thâm đen lõm sâu trên vỏ lựu, lấn sâu vào bên trong làm các hạt lựu bị thối đen, nhũn nát trong khi vỏ ngoài có thể chỉ bị đốm nhỏ.",
        "treatment": "Bao quả bằng túi chuyên dụng. Phun phòng thuốc trừ nấm Difenoconazole, Iprodione hoặc Chlorothalonil sau khi đậu quả non."
    },
    "pomegranate_Anthracnose": {
        "status": "warning",
        "fruit": "Lựu",
        "severity": "Trung bình",
        "description": "Đốm bệnh tròn màu nâu đen với các vòng tròn đồng tâm, quả bị khô xơ teo tóp và nứt vỏ sớm.",
        "treatment": "Cắt tỉa thông thoáng, dọn dẹp quả bệnh dưới gốc. Phun Carbendazim hoặc Mancozeb luân phiên."
    },
    "pomegranate_Bacterial_Blight": {
        "status": "danger",
        "fruit": "Lựu",
        "severity": "Rất cao",
        "description": "Vết bệnh thâm đen dạng góc cạnh hoặc hình sao, xung quanh có quầng ủng nước, quả bị nứt nẻ sâu hình chữ Y hoặc chữ L, làm quả biến dạng hoàn toàn.",
        "treatment": "Tiêu hủy ngay các quả và cành bị bệnh. Phun luân phiên Streptomycin Sulfate kết hợp gốc đồng (Copper Hydroxide)."
    },
    "pomegranate_Cercospora": {
        "status": "warning",
        "fruit": "Lựu",
        "severity": "Trung bình",
        "description": "Vết đốm nhỏ tròn màu nâu viền sẫm, tâm có màu xám trắng giống mắt chim rải rác trên vỏ lựu.",
        "treatment": "Phun thuốc trừ nấm gốc Thiophanate-methyl, Azoxystrobin hoặc Difenoconazole."
    },
    "pomegranate_Healthy": {
        "status": "healthy",
        "fruit": "Lựu",
        "severity": "Không có",
        "description": "Vỏ quả lựu tròn đều, căng bóng, đài hoa khép tự nhiên, hạt mẩy chắc không có dấu hiệu nấm khuẩn.",
        "treatment": "Tưới tiêu hợp lý, tránh để đất quá khô rồi tưới dồn dập gây nứt quả. Bổ sung Kali thời kỳ nuôi quả."
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


def get_disease_info(class_name: str) -> dict:
    """Retrieve detailed disease info dictionary for a given 15-class label."""
    return DISEASE_INFO.get(class_name, {
        "status": "info",
        "fruit": "Trái cây",
        "severity": "Không xác định",
        "description": "Chưa có mô tả chi tiết cho lớp bệnh này.",
        "treatment": "Vui lòng kiểm tra và theo dõi sát diễn biến tại vườn."
    })
