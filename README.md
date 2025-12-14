# AI Training Service

Dịch vụ AI Training Service cung cấp các API để train và predict các mô hình machine learning cho hệ thống e-commerce, bao gồm:

- **Recommendation Model (NeuMF)**: Mô hình gợi ý sản phẩm dựa trên collaborative filtering
- **Trending Model (LightGBM)**: Mô hình dự đoán xu hướng bán hàng
- **Next Item Model**: Mô hình dự đoán sản phẩm tiếp theo dựa trên lịch sử mua hàng

## 📋 Mục lục

- [Kiến trúc](#kiến-trúc)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt và chạy trên Local](#cài-đặt-và-chạy-trên-local)
- [Build và Deploy với Docker](#build-và-deploy-với-docker)
- [API Endpoints](#api-endpoints)
- [Cấu hình](#cấu-hình)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)

## 🏗️ Kiến trúc

### Tổng quan

Service được xây dựng bằng Flask, cung cấp RESTful API để:

1. **Training**: Train các mô hình ML cho từng store
2. **Prediction**: Sử dụng mô hình đã train để đưa ra dự đoán
3. **Caching**: Cache kết quả prediction để tăng hiệu suất

### Các thành phần chính

```
┌─────────────────────────────────────────────────────────┐
│                    Flask API Server                      │
│  (training_service.py - Port 5001)                      │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
┌───────▼──────┐  ┌───────▼──────┐  ┌───────▼──────┐
│ Recommendation│  │   Trending   │  │  Next Item   │
│   Trainer     │  │   Trainer    │  │   Trainer    │
│  (NeuMF)      │  │  (LightGBM)  │  │ (Sequential) │
└───────┬──────┘  └───────┬──────┘  └───────┬──────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
┌───────▼──────┐  ┌───────▼──────┐  ┌───────▼──────┐
│   Database   │  │ Data Extract │  │ Model Cache  │
│  (PostgreSQL)│  │   (SQL)      │  │  Manager     │
└──────────────┘  └──────────────┘  └──────────────┘
```

### Models

#### 1. Recommendation Model (NeuMF)

- **Kiến trúc**: Neural Matrix Factorization
- **Input**: User-Item interactions (orders, cart)
- **Output**: Recommendation scores cho từng user-item pair
- **Use case**: Gợi ý sản phẩm cho khách hàng

#### 2. Trending Model (LightGBM)

- **Kiến trúc**: Gradient Boosting Decision Tree
- **Input**: Time series features (sales, lags, rolling stats)
- **Output**: Predicted sales cho sản phẩm
- **Use case**: Dự đoán sản phẩm đang trending

#### 3. Next Item Model

- **Kiến trúc**: Sequential pattern matching
- **Input**: Purchase sequences của khách hàng
- **Output**: Xác suất sản phẩm tiếp theo
- **Use case**: "Khách hàng mua X thường mua Y"

## 💻 Yêu cầu hệ thống

- **Python**: 3.11+
- **PostgreSQL**: 12+
- **RAM**: Tối thiểu 4GB (khuyến nghị 8GB+)
- **Disk**: Tối thiểu 10GB cho models và data
- **GPU**: Tùy chọn (để tăng tốc training)

## 🚀 Cài đặt và chạy trên Local

### Bước 1: Clone repository

```bash
cd ai-service
```

### Bước 2: Tạo virtual environment

```bash
python3.11 -m venv venv
source venv/bin/activate  # Trên Windows: venv\Scripts\activate
```

### Bước 3: Cài đặt dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Bước 4: Cấu hình môi trường

Tạo file `.env` từ template:

```bash
cp env.example .env
```

Chỉnh sửa file `.env` với thông tin database của bạn:

```env
DB_HOST=localhost
DB_PORT=5433
DB_NAME=testocm
DB_USER=zalolog
DB_PASSWORD=123456
FLASK_PORT=5001
FLASK_DEBUG=true
```

### Bước 5: Đảm bảo PostgreSQL đang chạy

```bash
# Kiểm tra kết nối
psql -h localhost -p 5433 -U zalolog -d testocm
```

### Bước 6: Tạo thư mục lưu models

```bash
mkdir -p results
```

### Bước 7: Chạy service

```bash
cd api
python training_service.py
```

Service sẽ chạy tại: `http://localhost:5001`

### Bước 8: Kiểm tra health check

```bash
curl http://localhost:5001/health
```

Kết quả mong đợi:

```json
{
  "status": "healthy",
  "service": "ai-training-service",
  "version": "2.0",
  "config_loaded": true
}
```

## 🐳 Build và Deploy với Docker

### Option 1: Sử dụng Docker Compose (Khuyến nghị)

#### Bước 1: Cấu hình môi trường

Tạo file `.env`:

```bash
cp env.example .env
# Chỉnh sửa các biến môi trường nếu cần
```

#### Bước 2: Build và chạy

```bash
# Build và start tất cả services
docker-compose up -d

# Xem logs
docker-compose logs -f ai-service

# Kiểm tra status
docker-compose ps
```

#### Bước 3: Kiểm tra service

```bash
curl http://localhost:5001/health
```

#### Các lệnh hữu ích

```bash
# Stop services
docker-compose down

# Stop và xóa volumes (xóa data)
docker-compose down -v

# Rebuild sau khi thay đổi code
docker-compose up -d --build

# Xem logs real-time
docker-compose logs -f ai-service

# Vào container
docker-compose exec ai-service bash
```

### Option 2: Build Docker image thủ công

#### Bước 1: Build image

```bash
docker build -t ai-training-service:latest .
```

#### Bước 2: Chạy container

```bash
docker run -d \
  --name ai-service \
  -p 5001:5001 \
  -e DB_HOST=your_db_host \
  -e DB_PORT=5432 \
  -e DB_NAME=testocm \
  -e DB_USER=zalolog \
  -e DB_PASSWORD=123456 \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/data:/app/data \
  ai-training-service:latest
```

#### Bước 3: Kiểm tra

```bash
docker logs ai-service
curl http://localhost:5001/health
```

### Production Deployment

#### 1. Build production image

```bash
docker build -t ai-training-service:prod .
```

#### 2. Tag và push lên registry (nếu cần)

```bash
docker tag ai-training-service:prod your-registry/ai-training-service:v1.0.0
docker push your-registry/ai-training-service:v1.0.0
```

#### 3. Deploy với docker-compose.prod.yml

Tạo file `docker-compose.prod.yml`:

```yaml
version: "3.8"

services:
  ai-service:
    image: ai-training-service:prod
    container_name: ai-training-service-prod
    restart: always
    ports:
      - "5001:5001"
    env_file:
      - .env.prod
    volumes:
      - ./results:/app/results
      - ./data:/app/data
    networks:
      - ai-network
    healthcheck:
      test:
        [
          "CMD",
          "python",
          "-c",
          "import requests; requests.get('http://localhost:5001/health')",
        ]
      interval: 30s
      timeout: 10s
      retries: 3

networks:
  ai-network:
    driver: bridge
```

Chạy:

```bash
docker-compose -f docker-compose.prod.yml up -d
```

## 📡 API Endpoints

### Health Check

```http
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "service": "ai-training-service",
  "version": "2.0",
  "config_loaded": true
}
```

### Train Recommendation Model

```http
POST /train/recommendation
Content-Type: application/json

{
  "store_id": 1
}
```

**Response:**

```json
{
  "success": true,
  "store_id": 1,
  "model_type": "recommendation",
  "metrics": {
    "train_loss": 0.45,
    "val_loss": 0.52,
    "cached_users": 150,
    "model_version": "v1.0"
  },
  "model_path": "results/store_1/recommendation/neumf_model.pth"
}
```

### Train Trending Model

```http
POST /train/trending
Content-Type: application/json

{
  "store_id": 1
}
```

**Response:**

```json
{
  "success": true,
  "store_id": 1,
  "model_type": "trending",
  "metrics": {
    "rmse": 2.34,
    "mae": 1.89,
    "cached_products": 500,
    "model_version": "v1.0"
  },
  "model_path": "results/store_1/trending/lightgbm_model.txt"
}
```

### Train Next Item Model

```http
POST /train/next-item
Content-Type: application/json

{
  "store_id": 1
}
```

**Response:**

```json
{
  "status": "success",
  "store_id": 1,
  "model_path": "results/store_1/next_item/next_item_model.pkl",
  "metrics": {
    "total_patterns": 1250,
    "avg_confidence": 0.75
  }
}
```

### Predict Recommendations

```http
POST /predict/recommendations
Content-Type: application/json

{
  "store_id": 1,
  "user_id": 123,
  "n": 10
}
```

**Response:**

```json
{
  "store_id": 1,
  "user_id": 123,
  "recommendations": [
    {
      "item_id": 456,
      "variant_id": 456,
      "score": 0.89,
      "rank": 1
    }
  ]
}
```

### Predict Trending

```http
POST /predict/trending
Content-Type: application/json

{
  "store_id": 1,
  "n": 20
}
```

**Response:**

```json
{
  "store_id": 1,
  "trending": [
    {
      "item_id": 789,
      "variant_id": 789,
      "predicted_sales": 45.6,
      "trend_score": 2.3,
      "rank": 1
    }
  ]
}
```

### Predict Next Items

```http
POST /predict/next-items
Content-Type: application/json

{
  "store_id": 1,
  "item_history": [100, 200, 300],
  "top_k": 10
}
```

**Response:**

```json
{
  "store_id": 1,
  "input_history": [100, 200, 300],
  "predictions": [
    {
      "item_id": 400,
      "probability": 0.85,
      "confidence": 0.92,
      "support": 45
    }
  ]
}
```

## ⚙️ Cấu hình

Tất cả cấu hình được quản lý qua biến môi trường trong file `.env`:

### Database

- `DB_HOST`: Host PostgreSQL (default: localhost)
- `DB_PORT`: Port PostgreSQL (default: 5433)
- `DB_NAME`: Tên database (default: testocm)
- `DB_USER`: Username (default: zalolog)
- `DB_PASSWORD`: Password

### Flask

- `FLASK_HOST`: Host để bind (default: 0.0.0.0)
- `FLASK_PORT`: Port để listen (default: 5001)
- `FLASK_DEBUG`: Debug mode (default: false)

### Model Hyperparameters

Xem file `.env.example` để biết tất cả các hyperparameters có thể cấu hình.

## 📁 Cấu trúc thư mục

```
ai-service/
├── api/                          # API service code
│   ├── training_service.py      # Flask app chính
│   ├── config.py                # Configuration management
│   ├── database.py              # Database utilities
│   ├── data_extraction.py       # Data extraction từ DB
│   ├── cache_manager.py         # Cache management
│   └── trainers/                # Model trainers
│       ├── recommendation.py    # NeuMF trainer
│       ├── trending.py          # LightGBM trainer
│       └── next_item.py         # Sequential trainer
├── src/                         # Source code
│   ├── data/                    # Data preprocessing
│   └── evaluation/              # Model evaluation
├── scripts/                     # Utility scripts
├── data/                        # Data files
│   └── splits/                  # Train/test splits
├── results/                     # Trained models (generated)
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker image definition
├── docker-compose.yml           # Docker Compose config
├── env.example                  # Environment template
└── README.md                    # This file
```

## 🔧 Troubleshooting

### Lỗi kết nối database

```bash
# Kiểm tra PostgreSQL đang chạy
psql -h localhost -p 5433 -U zalolog -d testocm

# Kiểm tra biến môi trường
echo $DB_HOST
```

### Lỗi thiếu dependencies

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Lỗi out of memory khi training

- Giảm `RECOMMENDATION_BATCH_SIZE`
- Giảm `RECOMMENDATION_EPOCHS`
- Tăng RAM hoặc sử dụng GPU

### Model không được lưu

- Kiểm tra quyền ghi vào thư mục `results/`
- Kiểm tra disk space: `df -h`

## 📝 Notes

- Models được lưu theo store: `results/store_{store_id}/{model_type}/`
- Cache được quản lý tự động với expiry time
- Training logs được lưu vào database table `ai_training_log`
- Service status được cập nhật trong `store_ai_status`

## 📄 License

[Thêm license nếu có]

## 👥 Contributors

[Thêm contributors nếu có]
