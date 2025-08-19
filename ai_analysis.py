# ai_analysis.py - AI Analysis Module
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
import mediapipe as mp
import librosa
import boto3
import tempfile
import os
import re
from datetime import timedelta
import logging
from typing import Dict, List, Tuple, Callable
import spacy
from collections import Counter
import face_recognition
import dlib

logger = logging.getLogger(__name__)

class EmotionAnalyzer:
    """Facial emotion analysis using computer vision"""
    
    def __init__(self):
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load pre-trained emotion model (you can replace with a custom model)
        self.emotion_model = self._load_emotion_model()
    
    def _load_emotion_model(self):
        """Load emotion recognition model"""
        # For demonstration, using a simple model
        # In production, use a proper pre-trained emotion recognition model
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, len(self.emotions))
        model.eval()
        return model
    
    def analyze_frame(self, frame):
        """Analyze emotions in a single frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        emotions = []
        for (x, y, w, h) in faces:
            # Extract face region from original color frame
            face_color = frame[y:y+h, x:x+w]
            face_color = cv2.resize(face_color, (224, 224))  # ResNet expects 224x224
            
            # Convert BGR to RGB and normalize
            face_rgb = cv2.cvtColor(face_color, cv2.COLOR_BGR2RGB)
            face_rgb = face_rgb.astype('float32') / 255.0
            
            # Add batch dimension and rearrange to (batch, channels, height, width)
            face_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).unsqueeze(0)
            
            with torch.no_grad():
                prediction = self.emotion_model(face_tensor)
                emotion_probs = torch.softmax(prediction, dim=1).squeeze().numpy()
            
            emotion_dict = {emotion: float(prob) for emotion, prob in zip(self.emotions, emotion_probs)}
            emotions.append(emotion_dict)
        
        return emotions
    
    def analyze_video(self, video_path, progress_callback=None):
        """Analyze emotions throughout the video"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        all_emotions = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analyze every 30th frame to reduce processing time
            if frame_count % 30 == 0:
                emotions = self.analyze_frame(frame)
                if emotions:
                    all_emotions.extend(emotions)
                
                if progress_callback:
                    progress = (frame_count / total_frames) * 20  # 20% of total progress
                    progress_callback(int(progress), "Analyzing facial expressions")
            
            frame_count += 1
        
        cap.release()
        
        # Aggregate results
        return self._aggregate_emotions(all_emotions)
    
    def _aggregate_emotions(self, emotions_list):
        """Aggregate emotion data across all frames"""
        if not emotions_list:
            return {
                'dominant': 'neutral',
                'confidence': 0,
                'breakdown': [{'emotion': emotion, 'percentage': 0} for emotion in self.emotions]
            }
        
        # Average emotions across all detections
        emotion_sums = {emotion: 0 for emotion in self.emotions}
        for emotions in emotions_list:
            for emotion, prob in emotions.items():
                emotion_sums[emotion] += prob
        
        total_detections = len(emotions_list)
        emotion_averages = {emotion: sum_val / total_detections for emotion, sum_val in emotion_sums.items()}
        
        # Find dominant emotion
        dominant_emotion = max(emotion_averages, key=emotion_averages.get)
        confidence = emotion_averages[dominant_emotion] * 100
        
        # Create breakdown
        breakdown = [
            {'emotion': emotion.capitalize(), 'percentage': round(prob * 100, 1)}
            for emotion, prob in sorted(emotion_averages.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return {
            'dominant': dominant_emotion.capitalize(),
            'confidence': round(confidence, 1),
            'breakdown': breakdown
        }

class EyeContactAnalyzer:
    """Eye contact and gaze analysis"""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def analyze_video(self, video_path, progress_callback=None):
        """Analyze eye contact throughout the video"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        eye_contact_frames = 0
        total_analyzed_frames = 0
        gaze_durations = []
        current_gaze_duration = 0
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analyze every 10th frame
            if frame_count % 10 == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    is_looking = self._is_looking_at_camera(results.multi_face_landmarks[0], frame.shape)
                    
                    if is_looking:
                        eye_contact_frames += 1
                        current_gaze_duration += 1
                    else:
                        if current_gaze_duration > 0:
                            gaze_durations.append(current_gaze_duration * 10 / fps)  # Convert to seconds
                            current_gaze_duration = 0
                    
                    total_analyzed_frames += 1
                
                if progress_callback:
                    progress = 20 + (frame_count / total_frames) * 15  # 15% of total progress
                    progress_callback(int(progress), "Analyzing eye contact")
            
            frame_count += 1
        
        cap.release()
        
        # Add final gaze duration if applicable
        if current_gaze_duration > 0:
            gaze_durations.append(current_gaze_duration * 10 / fps)
        
        # Calculate metrics
        if total_analyzed_frames == 0:
            return {
                'percentage': 0,
                'rating': 'Poor',
                'avgDuration': 0
            }
        
        percentage = (eye_contact_frames / total_analyzed_frames) * 100
        avg_duration = np.mean(gaze_durations) if gaze_durations else 0
        
        # Determine rating
        if percentage >= 70:
            rating = 'Excellent'
        elif percentage >= 50:
            rating = 'Good'
        elif percentage >= 30:
            rating = 'Fair'
        else:
            rating = 'Poor'
        
        return {
            'percentage': round(percentage, 1),
            'rating': rating,
            'avgDuration': round(avg_duration, 1)
        }
    
    def _is_looking_at_camera(self, landmarks, frame_shape):
        """Determine if person is looking at camera based on eye landmarks"""
        h, w = frame_shape[:2]
        
        # Get eye landmarks
        left_eye = [landmarks.landmark[i] for i in [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]]
        right_eye = [landmarks.landmark[i] for i in [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]]
        
        # Calculate eye aspect ratios and gaze direction
        # This is a simplified approach - in production, use more sophisticated gaze estimation
        left_center = np.mean([(p.x * w, p.y * h) for p in left_eye], axis=0)
        right_center = np.mean([(p.x * w, p.y * h) for p in right_eye], axis=0)
        
        # Calculate if eyes are approximately centered (looking forward)
        face_center_x = w / 2
        eye_center_x = (left_center[0] + right_center[0]) / 2
        
        # Threshold for "looking at camera"
        threshold = w * 0.1  # 10% of frame width
        return abs(eye_center_x - face_center_x) < threshold

class SpeechAnalyzer:
    """Speech-to-text and vocal analysis"""
    
    def __init__(self):
        # Initialize Whisper for speech recognition
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
        
        # Initialize sentiment analysis
        self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                         model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        
        # Load spaCy for NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Filler words to detect
        self.filler_words = {
            'um', 'uh', 'uhm', 'like', 'you know', 'so', 'actually', 
            'basically', 'literally', 'totally', 'right', 'okay', 'well'
        }
    
    def extract_audio(self, video_path):
        """Extract audio from video using FFmpeg"""
        import subprocess
        import tempfile
        
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        
        try:
            # Use FFmpeg to extract audio (use full path)
            cmd = [
                '/usr/bin/ffmpeg', '-i', video_path, 
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', '16000',  # Sample rate 16kHz
                '-ac', '1',  # Mono
                '-y',  # Overwrite output
                temp_audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"FFmpeg failed: {result.stderr}")
            
            # Load the extracted audio
            audio, sr = librosa.load(temp_audio_path, sr=16000)
            
            return audio, sr
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_audio_path)
            except:
                pass
    
    def transcribe_audio(self, audio, sr, progress_callback=None):
        """Convert speech to text using Whisper"""
        if progress_callback:
            progress_callback(35, "Converting speech to text")
        
        # Prepare audio for Whisper
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = self.model.generate(inputs["input_features"])
        
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        if progress_callback:
            progress_callback(50, "Processing speech content")
        
        return transcription
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of the speech"""
        # Split text into chunks for analysis
        chunks = [text[i:i+512] for i in range(0, len(text), 512)]
        
        sentiments = []
        for chunk in chunks:
            if chunk.strip():
                result = self.sentiment_analyzer(chunk)[0]
                sentiments.append(result)
        
        # Aggregate results
        if not sentiments:
            return {
                'overall': 'Neutral',
                'score': 0,
                'positivity': 50
            }
        
        # Calculate average sentiment
        positive_scores = [s['score'] for s in sentiments if s['label'] == 'LABEL_2']  # Positive
        negative_scores = [s['score'] for s in sentiments if s['label'] == 'LABEL_0']  # Negative
        neutral_scores = [s['score'] for s in sentiments if s['label'] == 'LABEL_1']   # Neutral
        
        avg_positive = np.mean(positive_scores) if positive_scores else 0
        avg_negative = np.mean(negative_scores) if negative_scores else 0
        avg_neutral = np.mean(neutral_scores) if neutral_scores else 0
        
        # Determine overall sentiment
        if avg_positive > avg_negative and avg_positive > avg_neutral:
            overall = 'Positive'
            score = avg_positive
            positivity = min(100, (avg_positive * 100) + 50)
        elif avg_negative > avg_positive and avg_negative > avg_neutral:
            overall = 'Negative'
            score = -avg_negative
            positivity = max(0, 50 - (avg_negative * 50))
        else:
            overall = 'Neutral'
            score = 0
            positivity = 50
        
        return {
            'overall': overall,
            'score': round(score, 3),
            'positivity': round(positivity, 1)
        }
    
    def detect_filler_words(self, text):
        """Detect and count filler words in speech"""
        # Normalize text
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        total_words = len(words)
        
        # Count filler words
        filler_counts = Counter()
        for word in words:
            if word in self.filler_words:
                filler_counts[word] += 1
        
        # Handle multi-word fillers
        for phrase in ['you know', 'i mean', 'kind of', 'sort of']:
            phrase_count = len(re.findall(r'\b' + phrase + r'\b', text_lower))
            if phrase_count > 0:
                filler_counts[phrase] = phrase_count
        
        total_fillers = sum(filler_counts.values())
        filler_percentage = (total_fillers / total_words * 100) if total_words > 0 else 0
        
        # Convert to list of dictionaries
        words_list = [
            {'word': word, 'count': count}
            for word, count in filler_counts.most_common(10)
        ]
        
        return {
            'count': total_fillers,
            'percentage': round(filler_percentage, 2),
            'words': words_list
        }
    
    def analyze_vocal_variety(self, audio, sr, progress_callback=None):
        """Analyze vocal characteristics"""
        if progress_callback:
            progress_callback(55, "Analyzing vocal patterns")
        
        # Extract audio features
        # Pitch analysis
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        # Tempo and rhythm
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
        
        # Volume analysis (RMS energy)
        rms = librosa.feature.rms(y=audio)[0]
        
        # Spectral features for clarity
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        
        # Calculate metrics
        pitch_variance = np.var(pitch_values) if pitch_values else 0
        pitch_range = np.ptp(pitch_values) if pitch_values else 0
        volume_variance = np.var(rms)
        clarity_score = np.mean(spectral_centroids)
        
        # Normalize scores to 0-100
        pace_score = min(100, max(0, (tempo - 60) / 120 * 100))  # Optimal around 120-180 BPM
        volume_score = min(100, volume_variance * 1000)  # Scaled variance
        pitch_score = min(100, pitch_variance / 1000)   # Scaled variance
        clarity_score = min(100, clarity_score / 5000)  # Scaled clarity
        
        return {
            'pace': round(pace_score, 1),
            'volume': round(volume_score, 1),
            'pitch': round(pitch_score, 1),
            'clarity': round(clarity_score, 1)
        }

class BodyLanguageAnalyzer:
    """Body language and posture analysis"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def analyze_video(self, video_path, progress_callback=None):
        """Analyze body language throughout the video"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        posture_scores = []
        gesture_count = 0
        energy_scores = []
        frame_count = 0
        
        prev_hand_positions = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analyze every 15th frame
            if frame_count % 15 == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Pose analysis
                pose_results = self.pose.process(rgb_frame)
                hand_results = self.hands.process(rgb_frame)
                
                if pose_results.pose_landmarks:
                    # Analyze posture
                    posture_score = self._analyze_posture(pose_results.pose_landmarks)
                    posture_scores.append(posture_score)
                    
                    # Calculate energy level based on movement
                    energy = self._calculate_energy(pose_results.pose_landmarks)
                    energy_scores.append(energy)
                
                # Count gestures
                if hand_results.multi_hand_landmarks:
                    current_positions = []
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        # Get hand center position
                        hand_center = self._get_hand_center(hand_landmarks)
                        current_positions.append(hand_center)
                    
                    if prev_hand_positions:
                        # Check for significant movement (gesture)
                        for i, pos in enumerate(current_positions):
                            if i < len(prev_hand_positions):
                                distance = np.linalg.norm(np.array(pos) - np.array(prev_hand_positions[i]))
                                if distance > 0.05:  # Threshold for gesture detection
                                    gesture_count += 1
                    
                    prev_hand_positions = current_positions
                
                if progress_callback:
                    progress = 65 + (frame_count / total_frames) * 15  # 15% of total progress
                    progress_callback(int(progress), "Analyzing body language")
            
            frame_count += 1
        
        cap.release()
        
        # Calculate metrics
        avg_posture = np.mean(posture_scores) if posture_scores else 50
        avg_energy = np.mean(energy_scores) if energy_scores else 50
        gestures_per_minute = (gesture_count / (total_frames / fps)) * 60 if fps > 0 else 0
        
        # Determine posture description
        if avg_posture >= 80:
            posture_desc = "Excellent"
        elif avg_posture >= 60:
            posture_desc = "Good"
        elif avg_posture >= 40:
            posture_desc = "Fair"
        else:
            posture_desc = "Poor"
        
        return {
            'posture': posture_desc,
            'gestures': round(gestures_per_minute, 1),
            'energy': round(avg_energy, 1)
        }
    
    def _analyze_posture(self, landmarks):
        """Analyze posture quality from pose landmarks"""
        # Get key points
        nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        
        # Calculate shoulder alignment
        shoulder_slope = abs(left_shoulder.y - right_shoulder.y)
        
        # Calculate head position relative to shoulders
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        head_alignment = abs(nose.x - shoulder_center_x)
        
        # Calculate spine alignment
        hip_center_x = (left_hip.x + right_hip.x) / 2
        spine_alignment = abs(shoulder_center_x - hip_center_x)
        
        # Score based on alignment (lower values = better posture)
        posture_score = 100 - (shoulder_slope * 500 + head_alignment * 300 + spine_alignment * 200)
        return max(0, min(100, posture_score))
    
    def _calculate_energy(self, landmarks):
        """Calculate energy level based on body movement"""
        # Get key points for movement calculation
        key_points = [
            landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST],
            landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST],
            landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER],
            landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER],
        ]
        
        # Calculate variance in positions (proxy for energy/movement)
        positions = [(p.x, p.y) for p in key_points]
        center = np.mean(positions, axis=0)
        distances = [np.linalg.norm(np.array(pos) - center) for pos in positions]
        energy = np.var(distances) * 1000  # Scale for percentage
        
        return min(100, max(0, energy))
    
    def _get_hand_center(self, hand_landmarks):
        """Get center position of hand"""
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        return (np.mean(x_coords), np.mean(y_coords))

class VideoAnalyzer:
    """Main video analysis orchestrator"""
    
    def __init__(self, s3_bucket, s3_key, progress_callback=None):
        self.s3_bucket = s3_bucket
        self.s3_key = s3_key
        self.progress_callback = progress_callback
        
        # Initialize analyzers
        self.emotion_analyzer = EmotionAnalyzer()
        self.eye_contact_analyzer = EyeContactAnalyzer()
        self.speech_analyzer = SpeechAnalyzer()
        self.body_language_analyzer = BodyLanguageAnalyzer()
        
        # Initialize AWS S3
        self.s3_client = boto3.client('s3')
    
    def analyze(self):
        """Run complete video analysis"""
        try:
            # Download video from S3
            if self.progress_callback:
                self.progress_callback(5, "Downloading video")
            
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                self.s3_client.download_fileobj(self.s3_bucket, self.s3_key, temp_file)
                video_path = temp_file.name
            
            # Get video duration
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            if self.progress_callback:
                self.progress_callback(10, "Starting analysis")
            
            # Run all analyses
            results = {}
            
            # 1. Emotion Analysis (10-30%)
            results['emotion'] = self.emotion_analyzer.analyze_video(video_path, self.progress_callback)
            
            # 2. Eye Contact Analysis (30-45%)
            results['eye_contact'] = self.eye_contact_analyzer.analyze_video(video_path, self.progress_callback)
            
            # 3. Speech Analysis (45-70%)
            audio, sr = self.speech_analyzer.extract_audio(video_path)
            transcript = self.speech_analyzer.transcribe_audio(audio, sr, self.progress_callback)
            results['transcript'] = transcript
            results['sentiment'] = self.speech_analyzer.analyze_sentiment(transcript)
            results['filler_words'] = self.speech_analyzer.detect_filler_words(transcript)
            results['vocal_variety'] = self.speech_analyzer.analyze_vocal_variety(audio, sr, self.progress_callback)
            
            # 4. Body Language Analysis (70-85%)
            results['body_language'] = self.body_language_analyzer.analyze_video(video_path, self.progress_callback)
            
            # 5. Calculate Overall Score (85-95%)
            if self.progress_callback:
                self.progress_callback(85, "Calculating overall score")
            
            results['overall_score'] = self._calculate_overall_score(results)
            results['duration'] = round(duration, 2)
            
            # Cleanup
            os.unlink(video_path)
            
            if self.progress_callback:
                self.progress_callback(100, "Analysis complete")
            
            return results
            
        except Exception as e:
            # Cleanup on error
            if 'video_path' in locals():
                try:
                    os.unlink(video_path)
                except:
                    pass
            raise e
    
    def _calculate_overall_score(self, results):
        """Calculate overall speaking performance score"""
        scores = []
        weights = []
        
        # Emotion Analysis (weight: 20%)
        if results.get('emotion', {}).get('confidence'):
            emotion_score = results['emotion']['confidence']
            scores.append(emotion_score)
            weights.append(20)
        
        # Eye Contact (weight: 25%)
        if results.get('eye_contact', {}).get('percentage'):
            eye_score = results['eye_contact']['percentage']
            scores.append(eye_score)
            weights.append(25)
        
        # Sentiment (weight: 15%)
        if results.get('sentiment', {}).get('positivity'):
            sentiment_score = results['sentiment']['positivity']
            scores.append(sentiment_score)
            weights.append(15)
        
        # Vocal Variety (weight: 20%)
        if results.get('vocal_variety'):
            vocal_scores = list(results['vocal_variety'].values())
            vocal_avg = np.mean(vocal_scores)
            scores.append(vocal_avg)
            weights.append(20)
        
        # Body Language (weight: 15%)
        if results.get('body_language', {}).get('energy'):
            body_score = results['body_language']['energy']
            scores.append(body_score)
            weights.append(15)
        
        # Filler Words (weight: 5% - penalty)
        if results.get('filler_words', {}).get('percentage'):
            filler_penalty = min(20, results['filler_words']['percentage'] * 2)  # Max 20 point penalty
            filler_score = 100 - filler_penalty
            scores.append(filler_score)
            weights.append(5)
        
        # Calculate weighted average
        if scores and weights:
            total_weight = sum(weights)
            weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
            overall_score = weighted_sum / total_weight
            return round(max(0, min(100, overall_score)))
        
        return 50  # Default score if no analysis available


class LocalVideoAnalyzer:
    """Simplified video analyzer that works with local files (no S3 downloads)"""
    
    def __init__(self, video_path, progress_callback=None):
        self.video_path = video_path
        self.progress_callback = progress_callback
        
        # Initialize analyzers
        self.emotion_analyzer = EmotionAnalyzer()
        self.eye_contact_analyzer = EyeContactAnalyzer()
        self.speech_analyzer = SpeechAnalyzer()
        self.body_language_analyzer = BodyLanguageAnalyzer()
    
    def analyze(self):
        """Run complete video analysis on local file"""
        try:
            if not os.path.exists(self.video_path):
                raise Exception(f"Video file not found: {self.video_path}")
            
            if self.progress_callback:
                self.progress_callback(5, "Starting analysis")
            
            # Get video duration
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            if self.progress_callback:
                self.progress_callback(10, "Processing video")
            
            # Run all analyses
            results = {}
            
            # 1. Emotion Analysis (10-25%)
            results['emotion'] = self.emotion_analyzer.analyze_video(self.video_path, self.progress_callback)
            
            # 2. Eye Contact Analysis (25-40%)  
            results['eye_contact'] = self.eye_contact_analyzer.analyze_video(self.video_path, self.progress_callback)
            
            # 3. Speech Analysis (40-70%)
            audio, sr = self.speech_analyzer.extract_audio(self.video_path)
            transcript = self.speech_analyzer.transcribe_audio(audio, sr, self.progress_callback)
            results['transcript'] = transcript
            results['sentiment'] = self.speech_analyzer.analyze_sentiment(transcript)
            results['filler_words'] = self.speech_analyzer.detect_filler_words(transcript)
            results['vocal_variety'] = self.speech_analyzer.analyze_vocal_variety(audio, sr, self.progress_callback)
            
            # 4. Body Language Analysis (70-85%)
            results['body_language'] = self.body_language_analyzer.analyze_video(self.video_path, self.progress_callback)
            
            # 5. Calculate Overall Score (85-95%)
            if self.progress_callback:
                self.progress_callback(85, "Calculating overall score")
            
            results['overall_score'] = self._calculate_overall_score(results)
            results['duration'] = round(duration, 2)
            
            if self.progress_callback:
                self.progress_callback(100, "Analysis complete")
            
            return results
            
        except Exception as e:
            raise e
    
    def _calculate_overall_score(self, results):
        """Calculate overall speaking performance score"""
        scores = []
        weights = []
        
        # Emotion Analysis (weight: 20%)
        if results.get('emotion', {}).get('confidence'):
            emotion_score = results['emotion']['confidence']
            scores.append(emotion_score)
            weights.append(20)
        
        # Eye Contact (weight: 25%)
        if results.get('eye_contact', {}).get('percentage'):
            eye_score = results['eye_contact']['percentage']
            scores.append(eye_score)
            weights.append(25)
        
        # Sentiment (weight: 15%)
        if results.get('sentiment', {}).get('positivity'):
            sentiment_score = results['sentiment']['positivity']
            scores.append(sentiment_score)
            weights.append(15)
        
        # Vocal Variety (weight: 20%)
        if results.get('vocal_variety'):
            vocal_scores = list(results['vocal_variety'].values())
            vocal_avg = np.mean(vocal_scores)
            scores.append(vocal_avg)
            weights.append(20)
        
        # Body Language (weight: 15%)
        if results.get('body_language', {}).get('energy'):
            body_score = results['body_language']['energy']
            scores.append(body_score)
            weights.append(15)
        
        # Filler Words (weight: 5% - penalty)
        if results.get('filler_words', {}).get('percentage'):
            filler_penalty = min(20, results['filler_words']['percentage'] * 2)  # Max 20 point penalty
            filler_score = 100 - filler_penalty
            scores.append(filler_score)
            weights.append(5)
        
        # Calculate weighted average
        if scores and weights:
            total_weight = sum(weights)
            weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
            overall_score = weighted_sum / total_weight
            return round(max(0, min(100, overall_score)))
        
        return 50  # Default score if no analysis available
