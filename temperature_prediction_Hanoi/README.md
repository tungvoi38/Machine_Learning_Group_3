# Temperature Prediction — Hanoi

## Tổng quan
Dự án này dự đoán nhiệt độ tại Hà Nội bằng các bước chuẩn hóa dữ liệu, tạo đặc trưng, huấn luyện mô hình và trực quan hóa kết quả. Mục tiêu là cung cấp pipeline có thể tái sử dụng cho thí nghiệm và triển khai.

## Cấu trúc dự án
```
temperature_prediction_Hanoi
├── data
│   ├── raw               # Dữ liệu thô (không chỉnh sửa)
│   ├── processed         # Dữ liệu đã xử lý sẵn sàng cho mô hình
│   └── external          # Dữ liệu từ nguồn bên ngoài
├── notebooks
│   ├── 01-exploration.ipynb
│   └── 02-modeling.ipynb
├── src
│   ├── __init__.py
│   ├── data
│   │   └── make_dataset.py
│   ├── features
│   │   └── build_features.py
│   ├── models
│   │   └── train_model.py
│   └── visualization
│       └── visualize.py
├── tests
│   └── test_basic.py
├── results
│   ├── figures
│   └── models
├── configs
│   └── config.yaml
├── environment.yml
├── requirements.txt
├── .gitignore
└── README.md
```

## Yêu cầu
- Hệ điều hành: Ubuntu (dev container được cấu hình sẵn trong môi trường phát triển)
- Python 3.8+ (hoặc phiên bản tương thích được khai báo trong environment.yml)
- Công cụ: git, pytest, jupyter (có sẵn trong dev container theo cấu hình)

## Cách cài đặt nhanh
1. Vào thư mục dự án:
   cd /workspaces/Machine_Learning_Group_3/temperature_prediction_Hanoi

2. Tạo môi trường ảo và cài dependencies:
   - Sử dụng pip:
     python -m venv .venv
     source .venv/bin/activate
     pip install -r requirements.txt
   - Hoặc dùng conda với environment.yml:
     conda env create -f environment.yml
     conda activate <env-name>

## Chạy notebook và mở trình duyệt
- Khởi động Jupyter Lab/Notebook:
  jupyter lab --no-browser --port=8888
- Mở trên máy chủ host (từ dev container sử dụng biến môi trường):
  "$BROWSER" http://localhost:8888

## Chạy test
- Từ thư mục dự án:
  pytest -q

## Quy trình chính (ví dụ)
1. Đặt dữ liệu thô vào data/raw.
2. Chạy script tạo dataset:
   python -m src.data.make_dataset
3. Tạo đặc trưng:
   python -m src.features.build_features
4. Huấn luyện mô hình:
   python -m src.models.train_model
5. Kết quả và hình ảnh lưu trong thư mục results/

## Ghi chú
- Giữ data/raw nguyên gốc; chỉ lưu các file đã xử lý vào data/processed.
- Cấu hình (đường dẫn, tham số) nằm trong configs/config.yaml.
- Nếu muốn tạo repo và đẩy lên GitHub, sử dụng gh CLI hoặc thiết lập remote bằng git.

## Liên hệ
Mô tả ngắn, hướng dẫn sử dụng và phần mở rộng có thể cập nhật theo nhu cầu thực nghiệm.