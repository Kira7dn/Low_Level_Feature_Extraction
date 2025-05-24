# Docker Deployment Guide for Image Processing Application

## Prerequisites
- Docker (version 20.10 or higher)
- Docker Compose (version 1.29 or higher)
- Git

## Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/image-processing-app.git
cd image-processing-app
```

### 2. Environment Configuration
Copy the `.env.example` to `.env` and modify as needed:
```bash
cp .env.example .env
```

### 3. Build and Run
```bash
# Build the containers
docker-compose build

# Start the application
docker-compose up -d
```

### 4. Accessing the Application
- Application API: `http://localhost:8000`
- Grafana Monitoring: `http://localhost:3000`

## Docker Compose Services
- **app**: Main image processing application
- **monitoring**: Grafana dashboard for performance metrics

## Performance Monitoring
Performance metrics are stored in `./performance_metrics` and can be analyzed through Grafana.

## Troubleshooting
- Check container logs: `docker-compose logs app`
- Restart services: `docker-compose restart`
- Rebuild containers: `docker-compose up --build`

## Resource Management
- CPU Limit: 2 cores
- Memory Limit: 2GB
- Restart Policy: Always restart unless explicitly stopped

## Security Notes
- Use strong, unique passwords
- Regularly update Docker and dependencies
- Limit network access
- Monitor container health

## Development vs Production
- Use `docker-compose.yml` for development
- For production, create a separate `docker-compose.prod.yml` with stricter settings
