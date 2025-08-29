Complete Deployment Guide on AWS - Speakwell Coach app

Step 1: Set Up AWS Infrastructure

1.1 Launch EC2 Instance
1.	Log into AWS Console and navigate to EC2
2.	Launch Instance:
o	Name: speakwell-coach-prod
o	AMI: Ubuntu Server 24.04 LTS
o	Instance Type: t3.xlarge (4 vCPU, 16 GB RAM) - needed for AI processing
o	Key Pair: Create new or use existing
o	Security Group: Create new with these rules: 
o	SSH (22)     - Your IP onlyHTTP (80)    - 0.0.0.0/0HTTPS (443)  - 0.0.0.0/0
o	Storage: 50 GB gp3 (for models and temporary files)
3.	Launch the instance and note the public IP address
1.2 Set Up S3 Bucket
1.	Navigate to S3 in AWS Console
2.	Create bucket:
o	Name: speakwell-videos-[your-unique-suffix]
o	Region: Same as your EC2 instance
o	Block public access: Keep enabled
o	Versioning: Disabled
o	Encryption: Enable with SSE-S3
3.	Configure CORS:
[
  {
    "AllowedHeaders": ["*"],
    "AllowedMethods": ["GET", "PUT", "POST", "HEAD"],
    "AllowedOrigins": ["*"],
    "ExposeHeaders": ["ETag"]
  }
]
1.3 Create IAM User
1.	Navigate to IAM → Users → Create User
2.	User name: speakwell-app
3.	Attach policies directly:
o	Create custom policy:
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-bucket-name",
        "arn:aws:s3:::your-bucket-name/*"
      ]
    }
  ]
}
4.	Create Access Keys → Save the credentials securely

Step 2: Connect to Your EC2 Instance
# Connect via SSH
ssh -i your-key.pem ubuntu@your-ec2-public-ip

# Update system
sudo apt update && sudo apt upgrade -y
Step 3: Install System Dependencies
# Install essential packages
sudo apt install -y \
    python3 python3-pip python3-venv python3-dev \
    build-essential cmake pkg-config \
    libopencv-dev libboost-all-dev \
    libopenblas-dev liblapack-dev \
    libx11-dev libgtk-3-dev \
    redis-server postgresql postgresql-contrib \
    nginx supervisor git curl wget \
    ffmpeg unzip

# Install Node.js (for any build tools)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

Step 4: Set Up Application Directory
# Create application user
sudo useradd -m -s /bin/bash speakwell
sudo usermod -aG sudo speakwell

# Create application directory
sudo mkdir -p /opt/speakwell
sudo chown speakwell:speakwell /opt/speakwell

# Switch to application user
sudo su - speakwell
cd /opt/speakwell

Step 5: Set Up Python Environment and Code
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Create project structure
mkdir -p static templates logs uploads models tests

# Create requirements.txt file
cat > requirements.txt << 'EOF'
Flask==2.3.3
Flask-CORS==4.0.0
Flask-SQLAlchemy==3.0.5
Flask-Migrate==4.0.5
Flask-JWT-Extended==4.5.3
Werkzeug==2.3.7
gunicorn==21.2.0
SQLAlchemy==2.0.21
alembic==1.12.0
celery==5.3.1
redis==4.6.0
boto3==1.28.57
botocore==1.31.57
torch==2.0.1
torchvision==0.15.2
torchaudio==2.0.2
transformers==4.33.2
opencv-python==4.8.0.76
mediapipe==0.10.3
librosa==0.10.1
face-recognition==1.3.0
dlib==19.24.2
spacy==3.6.1
scikit-learn==1.3.0
numpy==1.24.3
scipy==1.11.2
soundfile==0.12.1
audioread==3.0.0
Pillow==10.0.0
imageio==2.31.3
pandas==2.0.3
python-dotenv==1.0.0
requests==2.31.0
psycopg2-binary==2.9.7
cryptography==41.0.4
bcrypt==4.0.1
EOF

# Install Python packages (this will take 10-15 minutes)
pip install --upgrade pip
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

Step 6: Create Application Files

6.1 Create config.py
cat > config.py << 'EOF'
import os
from datetime import timedelta

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'postgresql://speakwell:password@localhost/speakwell'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or 'jwt-secret-change-in-production'
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)
    AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
    AWS_REGION = os.environ.get('AWS_REGION') or 'us-east-1'
    S3_BUCKET = os.environ.get('S3_BUCKET')
    REDIS_HOST = os.environ.get('REDIS_HOST') or 'localhost'
    REDIS_PORT = os.environ.get('REDIS_PORT') or 6379
    REDIS_DB = os.environ.get('REDIS_DB') or 0
    CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL') or 'redis://localhost:6379/0'
    CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND') or 'redis://localhost:6379/0'
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB
EOF

6.2 Create Environment File

cat > .env << 'EOF'
FLASK_APP=app.py
FLASK_ENV=production
SECRET_KEY=your-very-secure-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-key-here
DATABASE_URL=postgresql://speakwell:speakwell123@localhost/speakwell
AWS_ACCESS_KEY_ID=YOUR_AWS_ACCESS_KEY
AWS_SECRET_ACCESS_KEY=YOUR_AWS_SECRET_KEY
AWS_REGION=us-east-1
S3_BUCKET=your-s3-bucket-name
REDIS_HOST=localhost
REDIS_PORT=6379
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
EOF

# Update with your actual credentials
nano .env

6.3 Create Main Application Files

Copy the app.py and ai_analysis.py files from our previous artifacts to the server:
# Create app.py (copy content from the Flask backend artifact)
nano app.py

# Create ai_analysis.py (copy content from the AI analysis artifact)
nano ai_analysis.py

6.4 Create Frontend

# Copy the frontend HTML file
cp /path/to/your/frontend.html static/index.html

Step 7: Set Up Database
# Exit speakwell user temporarily
exit

# Configure PostgreSQL
sudo -u postgres psql << 'EOF'
CREATE USER speakwell WITH PASSWORD 'speakwell123';
CREATE DATABASE speakwell OWNER speakwell;
GRANT ALL PRIVILEGES ON DATABASE speakwell TO speakwell;
\q
EOF

# Back to speakwell user
sudo su - speakwell
cd /opt/speakwell
source venv/bin/activate

# Initialize database
export FLASK_APP=app.py
flask db init
flask db migrate -m "Initial migration"
flask db upgrade

Step 8: Configure Redis

# Exit speakwell user
exit

# Start and enable Redis
sudo systemctl enable redis-server
sudo systemctl start redis-server
sudo systemctl status redis-server

Step 9: Configure Nginx

# Create Nginx configuration
sudo tee /etc/nginx/sites-available/speakwell << 'EOF'
server {
    listen 80;
    server_name your-domain.com www.your-domain.com;  # Replace with your domain or EC2 IP

    client_max_body_size 100M;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    location /static {
        alias /opt/speakwell/static;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
EOF

# Enable the site
sudo ln -sf /etc/nginx/sites-available/speakwell /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test configuration
sudo nginx -t

# Start Nginx
sudo systemctl enable nginx
sudo systemctl restart nginx

Step 10: Configure Supervisor

# Flask app configuration
sudo tee /etc/supervisor/conf.d/speakwell-flask.conf << 'EOF'
[program:speakwell-flask]
command=/opt/speakwell/venv/bin/gunicorn -w 4 -b 127.0.0.1:5000 app:app --timeout 300
directory=/opt/speakwell
user=speakwell
autostart=true
autorestart=true
stopasgroup=true
killasgroup=true
environment=PATH="/opt/speakwell/venv/bin"
stdout_logfile=/opt/speakwell/logs/flask.log
stderr_logfile=/opt/speakwell/logs/flask_error.log
EOF

# Celery worker configuration
sudo tee /etc/supervisor/conf.d/speakwell-celery.conf << 'EOF'
[program:speakwell-celery]
command=/opt/speakwell/venv/bin/celery -A app.celery worker --loglevel=info --concurrency=2
directory=/opt/speakwell
user=speakwell
autostart=true
autorestart=true
stopasgroup=true
killasgroup=true
environment=PATH="/opt/speakwell/venv/bin"
stdout_logfile=/opt/speakwell/logs/celery.log
stderr_logfile=/opt/speakwell/logs/celery_error.log
EOF

# Update supervisor
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start all

Step 11: Set Up SSL (Optional but Recommended)
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate (replace with your domain)
sudo certbot --nginx -d your-domain.com -d www.your-domain.com

# Test auto-renewal
sudo certbot renew --dry-run

Step 12: Configure Firewall
# Set up UFW firewall
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS
sudo ufw --force enable

Step 13: Test Your Deployment
13.1 Health Check
curl http://your-server-ip/api/health
Expected response:
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "version": "1.0.0"
}
13.2 Check Services
# Check all services
sudo supervisorctl status
sudo systemctl status nginx
sudo systemctl status redis-server
sudo systemctl status postgresql

# Check logs
sudo tail -f /opt/speakwell/logs/flask.log
sudo tail -f /opt/speakwell/logs/celery.log

13.3 Test Frontend
Visit http://your-server-ip or https://your-domain.com in your browser.
