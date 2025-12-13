# Production Deployment Guide

This guide covers deploying WakeGen in production environments.

## Quick Start with Docker Compose

### Prerequisites
- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- 8GB RAM minimum (16GB for GPU providers)
- SSD storage recommended

### 1. Basic Deployment

```yaml
# docker-compose.yml
version: '3.8'

services:
  wakegen:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./output:/app/output
      - ./config:/app/config
    environment:
      - WAKEGEN_API_KEY=your-secure-key-here
      - MINIMAX_API_KEY=${MINIMAX_API_KEY}
    restart: unless-stopped
```

```bash
# Start the service
docker compose up -d

# View logs
docker compose logs -f wakegen
```

### 2. With Nginx Reverse Proxy

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  wakegen:
    build: .
    expose:
      - "8000"
    volumes:
      - wakegen-output:/app/output
    environment:
      - WAKEGEN_API_KEY=${WAKEGEN_API_KEY}
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
    depends_on:
      - wakegen
    restart: unless-stopped

volumes:
  wakegen-output:
```

### 3. Nginx Configuration

```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream wakegen {
        server wakegen:8000;
    }

    server {
        listen 80;
        server_name wakegen.example.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl;
        server_name wakegen.example.com;

        ssl_certificate /etc/nginx/certs/fullchain.pem;
        ssl_certificate_key /etc/nginx/certs/privkey.pem;

        location / {
            proxy_pass http://wakegen;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }

        # WebSocket support
        location /ws {
            proxy_pass http://wakegen;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
}
```

## SSL with Let's Encrypt

```bash
# Install certbot
apt install certbot

# Get certificate
certbot certonly --standalone -d wakegen.example.com

# Copy to certs directory
cp /etc/letsencrypt/live/wakegen.example.com/* ./certs/
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `WAKEGEN_API_KEY` | Yes | API key for web UI authentication |
| `MINIMAX_API_KEY` | No | MiniMax TTS API key |
| `WAKEGEN_OUTPUT_DIR` | No | Output directory (default: `./output`) |
| `WAKEGEN_DEBUG` | No | Enable debug mode (`true`/`false`) |

## GPU Support (NVIDIA)

```yaml
services:
  wakegen:
    build: .
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Requires:
- NVIDIA GPU with 6GB+ VRAM
- nvidia-container-toolkit installed

## Monitoring

### Health Check

```bash
curl http://localhost:8000/api/health
# {"status": "ok", "service": "wakegen-web", "version": "1.0.0"}
```

### Docker Health Check

```yaml
services:
  wakegen:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Troubleshooting

### Common Issues

1. **Port already in use**: Change port mapping in docker-compose.yml
2. **GPU not detected**: Ensure nvidia-container-toolkit is installed
3. **WebSocket disconnects**: Check nginx proxy_read_timeout settings
