# Sử dụng base image Python 3.10 slim với Debian Bullseye
FROM python:3.10-slim-bullseye

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt hệ thống dependencies và git
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tesseract-ocr \
    libgl1-mesa-glx \
    git \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Clone repository từ GitHub
RUN git clone https://github.com/Kira7dn/Low_Level_Feature_Extraction.git . \
    && apt-get purge -y git \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Tạo user không root để tăng bảo mật
RUN useradd -m appuser
USER appuser

# Mở port 80 (HTTP mặc định)
EXPOSE 80

# Chạy ứng dụng với Gunicorn + Uvicorn cho production
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:80"]