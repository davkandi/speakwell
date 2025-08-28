SpeakWell is an AI-powered public speaking coaching application that analyzes video presentations and provides comprehensive feedback. Here's an overview of the codebase:

  Architecture & Technology Stack

  Backend (Flask):
  - Flask web framework with JWT authentication
  - PostgreSQL database with SQLAlchemy ORM
  - Redis for task queuing with Celery for background processing
  - AWS S3 integration for video storage
  - Real-time analysis using multiple AI models

  Frontend:
  - Single-page application with vanilla JavaScript
  - Modern CSS with custom design system
  - Responsive UI with drag-and-drop video upload
  - Real-time progress tracking and results visualisation

  Core Features

  Video Analysis Pipeline:
  1. Emotion Analysis - Facial expression recognition using computer vision
  2. Eye Contact Tracking - MediaPipe-based gaze detection
  3. Speech Analysis - Whisper model for transcription with sentiment
  analysis
  4. Vocal Variety - Pitch, pace, volume, and clarity assessment
  5. Body Language - Posture and gesture analysis using pose estimation
  6. Filler Word Detection - Context-aware identification of speech
  disfluencies

  User Experience:
  - Anonymous uploads supported (no registration required)
  - Comprehensive scoring with weighted metrics
  - Interactive results with highlighted transcript
  - Performance summaries with strengths and improvement areas

  Key Components

  app.py - Main Flask application with:
  - Authentication routes (/api/auth/*)
  - Video upload handling (/api/upload)
  - Analysis orchestration (/api/analyze)
  - Results retrieval (/api/results/<video_id>)

  ai_analysis.py - AI processing pipeline with specialised analyzers:
  - EmotionAnalyzer - Custom CNN emotion classification
  - EyeContactAnalyzer - MediaPipe face mesh analysis
  - SpeechAnalyzer - Whisper transcription + sentiment analysis
  - BodyLanguageAnalyzer - MediaPipe pose estimation
  - LocalVideoAnalyzer - Orchestrates all analysis modules

  static/index.html - Complete frontend application with:
  - Modern dark theme design system
  - Drag-and-drop upload interface
  - Real-time progress visualization
  - Comprehensive results dashboard

  Database Schema

  Users table - Authentication and user management
  VideoAnalysis table - Stores analysis results with JSON fields for each
  metric

  Processing Flow

  1. User uploads video → stored locally and optionally in S3
  2. Celery task starts background analysis
  3. Each analyzer processes different aspects simultaneously
  4. Progress updates sent to frontend via polling
  5. Results compiled into comprehensive score and recommendations
  6. Local files cleaned up after processing

  The application demonstrates sophisticated video analysis capabilities
  with a professional user interface, making it suitable for public
  speaking training and assessment.
