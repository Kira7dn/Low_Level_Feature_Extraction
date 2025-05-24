# Sử dụng base image Python 3.10 slim với Debian Bullseye
FROM python:3.10-slim-bullseye

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt hệ thống dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tesseract-ocr \
    libgl1-mesa-glx \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Sao chép mã nguồn từ ngữ cảnh build (cung cấp bởi GitHub Actions)
COPY . .

# Cài đặt Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Mở port 80 (HTTP mặc định)
EXPOSE 80

# Chạy ứng dụng với Gunicorn + Uvicorn cho production
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:80"]