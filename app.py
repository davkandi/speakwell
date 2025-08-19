# app.py - Main Flask Application
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import uuid
import redis
from celery import Celery
from datetime import datetime, timedelta
import boto3
from botocore.exceptions import ClientError
import logging
from config import Config

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize extensions
db = SQLAlchemy(app)
migrate = Migrate(app, db)
jwt = JWTManager(app)
CORS(app)

# Initialize Redis for Celery
redis_client = redis.Redis(host=app.config['REDIS_HOST'], port=app.config['REDIS_PORT'], db=0)

# Initialize Celery
celery = Celery(
    app.import_name,
    broker=app.config['CELERY_BROKER_URL'],
    backend=app.config['CELERY_RESULT_BACKEND']
)
celery.conf.update(app.config)

# Initialize AWS S3
s3_client = boto3.client(
    's3',
    aws_access_key_id=app.config['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=app.config['AWS_SECRET_ACCESS_KEY'],
    region_name=app.config['AWS_REGION']
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    analyses = db.relationship('VideoAnalysis', backref='user', lazy=True)

class VideoAnalysis(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)  # Allow anonymous uploads
    filename = db.Column(db.String(255), nullable=False)
    s3_key = db.Column(db.String(255), nullable=True)  # Made optional for S3 archival
    local_path = db.Column(db.String(255), nullable=True)  # Local file path for processing
    file_size = db.Column(db.Integer, nullable=False)
    duration = db.Column(db.Float, nullable=True)
    status = db.Column(db.String(20), default='pending')  # pending, processing, completed, failed
    progress = db.Column(db.Integer, default=0)
    current_step = db.Column(db.String(100), default='Processing Video')
    
    # Analysis Results (JSON fields)
    emotion_analysis = db.Column(db.JSON, nullable=True)
    sentiment_analysis = db.Column(db.JSON, nullable=True)
    eye_contact_analysis = db.Column(db.JSON, nullable=True)
    vocal_variety_analysis = db.Column(db.JSON, nullable=True)
    body_language_analysis = db.Column(db.JSON, nullable=True)
    filler_words_analysis = db.Column(db.JSON, nullable=True)
    transcript = db.Column(db.Text, nullable=True)
    overall_score = db.Column(db.Integer, nullable=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    started_at = db.Column(db.DateTime, nullable=True)
    completed_at = db.Column(db.DateTime, nullable=True)
    error_message = db.Column(db.Text, nullable=True)

# Helper functions
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_to_s3(file, filename):
    """Upload file to S3 and return the key"""
    try:
        s3_key = f"videos/{datetime.utcnow().strftime('%Y/%m/%d')}/{filename}"
        s3_client.upload_fileobj(
            file,
            app.config['S3_BUCKET'],
            s3_key,
            ExtraArgs={'ContentType': 'video/mp4'}
        )
        return s3_key
    except ClientError as e:
        logger.error(f"S3 upload error: {e}")
        return None

def get_s3_url(s3_key):
    """Generate a presigned URL for S3 object"""
    try:
        return s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': app.config['S3_BUCKET'], 'Key': s3_key},
            ExpiresIn=3600
        )
    except ClientError as e:
        logger.error(f"S3 URL generation error: {e}")
        return None

# API Routes

@app.route('/')
def index():
    """Serve the frontend"""
    return send_from_directory('static', 'index.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    })

# Authentication Routes
@app.route('/api/auth/register', methods=['POST'])
def register():
    """User registration"""
    data = request.get_json()
    
    if not data or not all(k in data for k in ('email', 'password', 'full_name')):
        return jsonify({'error': 'Missing required fields'}), 400
    
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already registered'}), 400
    
    user = User(
        email=data['email'],
        password_hash=generate_password_hash(data['password']),
        full_name=data['full_name']
    )
    
    db.session.add(user)
    db.session.commit()
    
    access_token = create_access_token(identity=user.id)
    
    return jsonify({
        'access_token': access_token,
        'user': {
            'id': user.id,
            'email': user.email,
            'full_name': user.full_name
        }
    }), 201

@app.route('/api/auth/login', methods=['POST'])
def login():
    """User login"""
    data = request.get_json()
    
    if not data or not all(k in data for k in ('email', 'password')):
        return jsonify({'error': 'Missing email or password'}), 400
    
    user = User.query.filter_by(email=data['email']).first()
    
    if not user or not check_password_hash(user.password_hash, data['password']):
        return jsonify({'error': 'Invalid credentials'}), 401
    
    access_token = create_access_token(identity=user.id)
    
    return jsonify({
        'access_token': access_token,
        'user': {
            'id': user.id,
            'email': user.email,
            'full_name': user.full_name
        }
    })

@app.route('/api/auth/me')
@jwt_required()
def get_current_user():
    """Get current user info"""
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    return jsonify({
        'user': {
            'id': user.id,
            'email': user.email,
            'full_name': user.full_name
        }
    })

# Video Upload and Analysis Routes
@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Upload video file and save locally for processing"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload MP4, AVI, MOV, or MKV files'}), 400
    
    # Check file size (100MB limit)
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to beginning
    
    if file_size > 100 * 1024 * 1024:  # 100MB
        return jsonify({'error': 'File too large. Maximum size is 100MB'}), 400
    
    # Generate secure filename
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{filename}"
    
    # Save video locally for immediate processing
    upload_dir = os.path.join(app.root_path, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    local_path = os.path.join(upload_dir, unique_filename)
    file.save(local_path)
    
    # Upload to S3 in background for archival (optional)
    file.seek(0)  # Reset for S3 upload
    s3_key = upload_to_s3(file, unique_filename)
    
    # Get current user if authenticated
    user_id = None
    try:
        user_id = get_jwt_identity()
    except:
        pass  # Allow anonymous uploads
    
    # Create video analysis record with local path
    analysis = VideoAnalysis(
        user_id=user_id,
        filename=filename,
        s3_key=s3_key,  # Keep for archival
        file_size=file_size
    )
    
    # Store local path temporarily in database (we'll add this field)
    analysis.local_path = local_path
    
    db.session.add(analysis)
    db.session.commit()
    
    return jsonify({
        'video_id': analysis.id,
        'filename': filename,
        'file_size': file_size,
        'status': 'uploaded'
    }), 201

@app.route('/api/analyze', methods=['POST'])
def start_analysis():
    """Start video analysis"""
    data = request.get_json()
    
    if not data or 'video_id' not in data:
        return jsonify({'error': 'Missing video_id'}), 400
    
    analysis = VideoAnalysis.query.get(data['video_id'])
    if not analysis:
        return jsonify({'error': 'Video not found'}), 404
    
    if analysis.status != 'pending':
        return jsonify({'error': 'Video already being processed or completed'}), 400
    
    # Start background analysis task
    task = analyze_video_task.delay(analysis.id)
    
    # Update status
    analysis.status = 'processing'
    analysis.started_at = datetime.utcnow()
    db.session.commit()
    
    return jsonify({
        'video_id': analysis.id,
        'status': 'processing',
        'task_id': task.id
    })

@app.route('/api/results/<video_id>')
def get_analysis_results(video_id):
    """Get analysis results"""
    analysis = VideoAnalysis.query.get(video_id)
    if not analysis:
        return jsonify({'error': 'Video not found'}), 404
    
    # Check if user has access (if authenticated)
    try:
        user_id = get_jwt_identity()
        if user_id and analysis.user_id and analysis.user_id != user_id:
            return jsonify({'error': 'Access denied'}), 403
    except:
        pass  # Allow anonymous access for anonymous uploads
    
    if analysis.status == 'pending':
        return jsonify({
            'status': 'pending',
            'message': 'Analysis not started yet'
        })
    
    if analysis.status == 'processing':
        return jsonify({
            'status': 'processing',
            'progress': analysis.progress,
            'current_step': analysis.current_step
        })
    
    if analysis.status == 'failed':
        return jsonify({
            'status': 'failed',
            'error': analysis.error_message
        })
    
    if analysis.status == 'completed':
        return jsonify({
            'status': 'completed',
            'data': {
                'video_id': analysis.id,
                'filename': analysis.filename,
                'duration': analysis.duration,
                'overall_score': analysis.overall_score,
                'emotion': analysis.emotion_analysis,
                'sentiment': analysis.sentiment_analysis,
                'eyeContact': analysis.eye_contact_analysis,
                'vocalVariety': analysis.vocal_variety_analysis,
                'bodyLanguage': analysis.body_language_analysis,
                'fillerWords': analysis.filler_words_analysis,
                'transcript': analysis.transcript,
                'created_at': analysis.created_at.isoformat(),
                'completed_at': analysis.completed_at.isoformat()
            }
        })

@app.route('/api/analyses')
@jwt_required()
def get_user_analyses():
    """Get user's analysis history"""
    user_id = get_jwt_identity()
    
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    
    analyses = VideoAnalysis.query.filter_by(user_id=user_id)\
        .order_by(VideoAnalysis.created_at.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)
    
    return jsonify({
        'analyses': [{
            'id': a.id,
            'filename': a.filename,
            'status': a.status,
            'overall_score': a.overall_score,
            'created_at': a.created_at.isoformat(),
            'completed_at': a.completed_at.isoformat() if a.completed_at else None
        } for a in analyses.items],
        'total': analyses.total,
        'pages': analyses.pages,
        'current_page': page
    })

# Import analysis module at module level for Celery
try:
    from ai_analysis import LocalVideoAnalyzer
except ImportError as e:
    logger.error(f"Failed to import ai_analysis module: {e}")
    LocalVideoAnalyzer = None

# Celery Tasks
@celery.task(bind=True)
def analyze_video_task(self, analysis_id):
    """Background task to analyze video using local file"""
    if LocalVideoAnalyzer is None:
        raise Exception("LocalVideoAnalyzer not available - check ai_analysis module")
    
    # Create Flask application context for database access
    with app.app_context():
        try:
            # Get analysis record
            analysis = VideoAnalysis.query.get(analysis_id)
            if not analysis:
                raise Exception("Analysis record not found")
        
            # Check if local file exists
            if not analysis.local_path or not os.path.exists(analysis.local_path):
                raise Exception("Local video file not found")
            
            # Update progress
            def update_progress(progress, step):
                with app.app_context():
                    analysis_update = VideoAnalysis.query.get(analysis_id)
                    if analysis_update:
                        analysis_update.progress = progress
                        analysis_update.current_step = step
                        db.session.commit()
            
            # Initialize AI analyzer with local file path
            analyzer = LocalVideoAnalyzer(
                video_path=analysis.local_path,
                progress_callback=update_progress
            )
            
            # Update status to processing
            analysis.status = 'processing'
            analysis.started_at = datetime.utcnow()
            db.session.commit()
            
            # Run analysis
            results = analyzer.analyze()
            
            # Update database with results
            analysis.emotion_analysis = results['emotion']
            analysis.sentiment_analysis = results['sentiment']
            analysis.eye_contact_analysis = results['eye_contact']
            analysis.vocal_variety_analysis = results['vocal_variety']
            analysis.body_language_analysis = results['body_language']
            analysis.filler_words_analysis = results['filler_words']
            analysis.transcript = results['transcript']
            analysis.overall_score = results['overall_score']
            analysis.duration = results['duration']
            analysis.status = 'completed'
            analysis.completed_at = datetime.utcnow()
            analysis.progress = 100
            analysis.current_step = 'Analysis Complete'
            
            db.session.commit()
            
            # Clean up local file after analysis
            try:
                if os.path.exists(analysis.local_path):
                    os.remove(analysis.local_path)
                    analysis.local_path = None
                    db.session.commit()
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup local file: {cleanup_error}")
            
            logger.info(f"Analysis completed for video {analysis_id}")
            return {'status': 'completed', 'analysis_id': analysis_id}
        
        except Exception as e:
            logger.error(f"Analysis failed for video {analysis_id}: {str(e)}")
            
            # Update database with error
            analysis = VideoAnalysis.query.get(analysis_id)
            if analysis:
                analysis.status = 'failed'
                analysis.error_message = str(e)
                db.session.commit()
            
            raise self.retry(exc=e, countdown=60, max_retries=3)

# Error Handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({'error': 'File too large'}), 413

# Database initialization function (Flask 2.3+ compatible)
def create_tables():
    """Initialize database tables"""
    with app.app_context():
        db.create_all()

# Initialize on first request
@app.before_request
def ensure_tables():
    """Ensure tables exist before handling requests"""
    if not hasattr(app, '_tables_created'):
        create_tables()
        app._tables_created = True

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
