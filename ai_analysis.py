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
            
            # Analyze every 60th frame for speed (2-3x faster)
            if frame_count % 60 == 0:
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
                            # Convert analyzed frames to actual time
                            duration_seconds = (current_gaze_duration * 10) / fps  # 10 = frame sampling rate
                            
                            # Only count durations longer than 0.5 seconds (filter noise)
                            if duration_seconds >= 0.5:
                                gaze_durations.append(duration_seconds)
                            
                            current_gaze_duration = 0
                    
                    total_analyzed_frames += 1
                
                if progress_callback:
                    progress = 20 + (frame_count / total_frames) * 15  # 15% of total progress
                    progress_callback(int(progress), "Analyzing eye contact")
            
            frame_count += 1
        
        cap.release()
        
        # Add final gaze duration if applicable
        if current_gaze_duration > 0:
            duration_seconds = (current_gaze_duration * 10) / fps
            # Only count durations longer than 0.5 seconds (filter noise)
            if duration_seconds >= 0.5:
                gaze_durations.append(duration_seconds)
        
        # Calculate metrics
        if total_analyzed_frames == 0:
            return {
                'percentage': 0,
                'rating': 'Poor',
                'avgDuration': 0
            }
        
        percentage = (eye_contact_frames / total_analyzed_frames) * 100
        
        # Calculate average duration with better fallback logic
        if gaze_durations:
            avg_duration = np.mean(gaze_durations)
        else:
            # If no meaningful durations detected but eye contact exists, estimate
            if eye_contact_frames > 0:
                # Estimate duration based on continuous eye contact assumption
                total_video_seconds = total_frames / fps if fps > 0 else 1
                estimated_eye_contact_time = (eye_contact_frames / total_analyzed_frames) * total_video_seconds
                # Assume if high percentage but no durations, it might be continuous
                if percentage > 80:
                    avg_duration = max(2.0, estimated_eye_contact_time / 3)  # Conservative estimate
                else:
                    avg_duration = max(1.0, estimated_eye_contact_time / 5)  # Multiple shorter periods
            else:
                avg_duration = 0
        
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
        
        # Get key facial landmarks for more robust gaze detection
        # Use specific eye corner and pupil landmarks
        left_eye_inner = landmarks.landmark[133]   # Left eye inner corner
        left_eye_outer = landmarks.landmark[33]    # Left eye outer corner  
        right_eye_inner = landmarks.landmark[362]  # Right eye inner corner
        right_eye_outer = landmarks.landmark[263]  # Right eye outer corner
        
        # Get nose tip for reference
        nose_tip = landmarks.landmark[1]
        
        # Calculate eye centers more accurately
        left_eye_center_x = (left_eye_inner.x + left_eye_outer.x) / 2 * w
        right_eye_center_x = (right_eye_inner.x + right_eye_outer.x) / 2 * w
        avg_eye_center_x = (left_eye_center_x + right_eye_center_x) / 2
        
        # Calculate face center using nose as reference
        nose_x = nose_tip.x * w
        face_center_x = w / 2
        
        # Multi-factor eye contact detection
        # Factor 1: Eyes should be approximately centered relative to face
        eye_face_alignment = abs(avg_eye_center_x - nose_x) / w
        
        # Factor 2: Face should be roughly centered in frame
        face_frame_alignment = abs(nose_x - face_center_x) / w
        
        # Factor 3: Eye symmetry (both eyes should be at similar distance from nose)
        eye_symmetry = abs(left_eye_center_x - nose_x - (nose_x - right_eye_center_x)) / w
        
        # More lenient thresholds for natural speaking scenarios
        return (eye_face_alignment < 0.15 and    # Eyes aligned with face
                face_frame_alignment < 0.2 and   # Face reasonably centered  
                eye_symmetry < 0.1)               # Eyes symmetric

class SpeechAnalyzer:
    """Speech-to-text and vocal analysis"""
    
    def __init__(self, model_size="small"):
        # Initialize Whisper for speech recognition (small for speed, medium for accuracy)
        model_name = f"openai/whisper-{model_size}"
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        
        # Move to GPU if available for faster processing
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        logger.info(f"Using device: {self.device} for Whisper model")
        
        # Set forced decoder IDs for English transcription
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="en", task="transcribe")
        
        # Initialize sentiment analysis
        self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                         model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        
        # Load spaCy for NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Filler words to detect (single words only)
        self.single_filler_words = {
            'um', 'uh', 'uhm', 'like', 'so', 'actually', 
            'basically', 'literally', 'totally', 'right', 'okay', 'well'
        }
        
        # Multi-word fillers with contextual patterns
        self.multi_word_fillers = [
            r'\byou know\b(?!\s+(?:what|the|that|how|why|when|where))',  # "you know" not followed by question words
            r'\bi mean\b',
            r'\bi think\b(?!\s+(?:that|about|of))',  # "i think" not followed by specific objects
            r'\bkind of\b',
            r'\bsort of\b'
        ]
    
    def extract_audio(self, video_path):
        """Extract audio from video using FFmpeg with noise reduction"""
        import subprocess
        import tempfile
        
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        
        try:
            # Use FFmpeg for fast basic audio extraction
            cmd = [
                '/usr/bin/ffmpeg', '-i', video_path, 
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', '16000',  # Sample rate 16kHz
                '-ac', '1',  # Mono
                '-af', 'volume=2.0',  # Basic volume normalization only
                '-y',  # Overwrite output
                temp_audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"FFmpeg failed: {result.stderr}")
            
            # Load the extracted audio with minimal preprocessing
            audio, sr = librosa.load(temp_audio_path, sr=16000)
            
            # Basic preprocessing - just trim silence and normalize
            if len(audio) > 0:
                audio, _ = librosa.effects.trim(audio, top_db=20)
                audio = librosa.util.normalize(audio) * 0.8
            
            return audio, sr
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_audio_path)
            except:
                pass
    
    def _preprocess_audio(self, audio, sr):
        """Advanced audio preprocessing for better transcription quality"""
        if len(audio) == 0:
            return audio
        
        # 1. Remove silence at the beginning and end with more aggressive trimming
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=25, frame_length=2048, hop_length=512)
        
        if len(audio_trimmed) == 0:
            return audio
        
        # 2. Apply noise reduction using spectral gating
        # Get noise profile from first 0.5 seconds (likely to contain noise)
        noise_sample_length = min(int(0.5 * sr), len(audio_trimmed) // 4)
        if noise_sample_length > 0:
            noise_sample = audio_trimmed[:noise_sample_length]
            noise_power = np.mean(noise_sample ** 2)
            
            # Simple noise gate - reduce audio below noise threshold
            gate_threshold = noise_power * 3  # 3x noise level
            audio_gated = np.where(audio_trimmed ** 2 > gate_threshold, audio_trimmed, audio_trimmed * 0.3)
            audio_trimmed = audio_gated
        
        # 3. Dynamic range compression for more consistent volume
        def compress_audio(audio, ratio=4.0, threshold=0.5):
            """Apply compression to reduce dynamic range"""
            abs_audio = np.abs(audio)
            compressed = np.where(
                abs_audio > threshold,
                threshold + (abs_audio - threshold) / ratio,
                abs_audio
            )
            return np.sign(audio) * compressed
        
        audio_compressed = compress_audio(audio_trimmed)
        
        # 4. Normalize to optimal level for Whisper (not too loud, not too quiet)
        audio_normalized = librosa.util.normalize(audio_compressed) * 0.8
        
        # 5. Apply gentle pre-emphasis to boost higher frequencies (helps with clarity)
        pre_emphasis = 0.97
        audio_emphasized = np.append(audio_normalized[0], audio_normalized[1:] - pre_emphasis * audio_normalized[:-1])
        
        return audio_emphasized
    
    def transcribe_audio(self, audio, sr, progress_callback=None):
        """Convert speech to text using Whisper with chunking for long audio"""
        if progress_callback:
            progress_callback(35, "Converting speech to text")
        
        # Optimized chunking for speed vs accuracy balance
        chunk_length_samples = 15 * sr  # 15 seconds for faster processing
        audio_length = len(audio)
        
        if audio_length <= chunk_length_samples:
            # Short audio - process directly
            return self._transcribe_chunk(audio, sr)
        
        # Long audio - process in chunks with minimal overlap
        overlap_samples = 1 * sr  # 1 second overlap for speed
        transcriptions = []
        
        num_chunks = int(np.ceil(audio_length / (chunk_length_samples - overlap_samples)))
        
        for i in range(num_chunks):
            start_idx = i * (chunk_length_samples - overlap_samples)
            end_idx = min(start_idx + chunk_length_samples, audio_length)
            
            chunk = audio[start_idx:end_idx]
            
            # Skip very short chunks (less than 1 second)
            if len(chunk) < sr:
                continue
            
            chunk_transcript = self._transcribe_chunk(chunk, sr)
            
            if chunk_transcript.strip():
                # Remove overlap words from previous chunk
                if i > 0 and transcriptions:
                    chunk_transcript = self._remove_overlap(transcriptions[-1], chunk_transcript)
                
                transcriptions.append(chunk_transcript.strip())
            
            # Update progress
            if progress_callback:
                progress = 35 + (i + 1) / num_chunks * 15  # 35-50% progress
                progress_callback(int(progress), f"Transcribing audio chunk {i+1}/{num_chunks}")
        
        # Combine all transcriptions
        full_transcript = " ".join(transcriptions)
        
        if progress_callback:
            progress_callback(50, "Processing speech content")
        
        logger.info(f"Transcribed {num_chunks} chunks, total length: {len(full_transcript)} characters")
        return full_transcript
    
    def _transcribe_chunk(self, audio_chunk, sr):
        """Transcribe a single audio chunk"""
        try:
            # Prepare audio for Whisper and move to device
            inputs = self.processor(
                audio_chunk, 
                sampling_rate=sr, 
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate transcription with speed-optimized parameters
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    inputs["input_features"],
                    forced_decoder_ids=self.forced_decoder_ids,
                    max_new_tokens=200,  # Reduced for speed
                    num_beams=1,  # Greedy decoding for speed
                    temperature=0.0,  # Deterministic output
                    do_sample=False,
                    suppress_tokens=[-1],  # Suppress end-of-sequence token
                    return_dict_in_generate=False
                )
            
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            return transcription.strip()
            
        except Exception as e:
            logger.error(f"Error transcribing chunk: {e}")
            return ""
    
    def _remove_overlap(self, prev_transcript, curr_transcript):
        """Remove overlapping words between consecutive chunks with smart detection"""
        prev_words = prev_transcript.split()
        curr_words = curr_transcript.split()
        
        if len(prev_words) < 3 or len(curr_words) < 3:
            return curr_transcript
        
        # Look for exact phrase matches first (more reliable)
        max_overlap_check = min(8, len(prev_words), len(curr_words))
        
        for overlap_len in range(max_overlap_check, 2, -1):  # Start with longer overlaps
            if overlap_len > len(prev_words) or overlap_len > len(curr_words):
                continue
                
            prev_end = prev_words[-overlap_len:]
            curr_start = curr_words[:overlap_len]
            
            # Check for exact match (case insensitive)
            if [w.lower().strip('.,!?') for w in prev_end] == [w.lower().strip('.,!?') for w in curr_start]:
                logger.info(f"Found exact overlap of {overlap_len} words: {' '.join(curr_start)}")
                return " ".join(curr_words[overlap_len:])
        
        # Fallback: Look for partial matches with word similarity
        last_words = prev_words[-5:]
        
        for i in range(min(8, len(curr_words))):
            curr_start = curr_words[:i+3] if i+3 < len(curr_words) else curr_words
            
            # Count similar words (handling punctuation)
            similarity_count = 0
            for prev_word in last_words:
                prev_clean = prev_word.lower().strip('.,!?')
                for curr_word in curr_start:
                    curr_clean = curr_word.lower().strip('.,!?')
                    if prev_clean == curr_clean and len(prev_clean) > 2:  # Ignore very short words
                        similarity_count += 1
                        break
            
            # If more than half the words are similar, consider it overlap
            if similarity_count >= len(last_words) // 2 and similarity_count >= 2:
                logger.info(f"Found partial overlap of {similarity_count} similar words")
                return " ".join(curr_words[i+1:])
        
        return curr_transcript
    
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
        """Detect and count filler words in speech with context awareness"""
        # Normalize text
        text_lower = text.lower()
        # Clean up transcript artifacts
        text_clean = re.sub(r'\[.*?\]|\(.*?\)', '', text_lower)  # Remove [APPLAUSE], (static) etc
        
        words = re.findall(r'\b\w+\b', text_clean)
        total_words = len(words)
        
        filler_counts = Counter()
        processed_positions = set()
        
        # First, find multi-word fillers to avoid double counting
        for pattern in self.multi_word_fillers:
            matches = re.finditer(pattern, text_clean, re.IGNORECASE)
            for match in matches:
                phrase = match.group().strip()
                filler_counts[phrase] += 1
                # Mark positions of words in this phrase as processed
                start_pos = len(re.findall(r'\b\w+\b', text_clean[:match.start()]))
                phrase_words = re.findall(r'\b\w+\b', phrase)
                for i in range(len(phrase_words)):
                    processed_positions.add(start_pos + i)
        
        # Then count single word fillers, excluding those already counted in phrases  
        for i, word in enumerate(words):
            if i not in processed_positions and word in self.single_filler_words:
                filler_counts[word] += 1
        
        total_fillers = sum(filler_counts.values())
        filler_percentage = (total_fillers / total_words * 100) if total_words > 0 else 0
        
        # Convert to list of dictionaries, sorted by count
        words_list = [
            {'word': word, 'count': count}
            for word, count in filler_counts.most_common(10)
            if count > 0  # Only include words that were actually found
        ]
        
        return {
            'count': total_fillers,
            'percentage': round(filler_percentage, 2),
            'words': words_list,
            'highlighted_text': self._highlight_filler_words(text_clean, filler_counts)
        }
    
    def _highlight_filler_words(self, text, filler_counts):
        """Create highlighted version of text with filler words marked"""
        highlighted = text
        
        # Highlight all filler words/phrases that were actually found
        for filler_word in filler_counts.keys():
            if filler_counts[filler_word] > 0:
                # Check if it's a multi-word filler
                if ' ' in filler_word:
                    # Multi-word filler - use exact phrase matching
                    pattern = r'\b' + re.escape(filler_word) + r'\b'
                    highlighted = re.sub(pattern, r'<mark class="filler-word">\g<0></mark>', highlighted, flags=re.IGNORECASE)
                else:
                    # Single word filler
                    pattern = r'\b' + re.escape(filler_word) + r'\b'
                    highlighted = re.sub(pattern, r'<mark class="filler-word">\g<0></mark>', highlighted, flags=re.IGNORECASE)
        
        return highlighted
    
    def analyze_vocal_variety(self, audio, sr, progress_callback=None):
        """Analyze vocal characteristics"""
        if progress_callback:
            progress_callback(55, "Analyzing vocal patterns")
        
        # Handle empty or invalid audio
        if len(audio) == 0:
            logger.warning("Empty audio provided to vocal variety analysis")
            return {
                'pace': 20.0,
                'volume': 20.0,
                'pitch': 20.0,
                'clarity': 20.0
            }
        
        try:
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
            
        except Exception as e:
            logger.error(f"Error in vocal variety analysis: {e}")
            return {
                'pace': 30.0,
                'volume': 30.0, 
                'pitch': 30.0,
                'clarity': 30.0
            }
        
        # Calculate raw metrics
        pitch_variance = np.var(pitch_values) if pitch_values else 0
        pitch_range = np.ptp(pitch_values) if pitch_values else 0
        volume_variance = np.var(rms)
        volume_mean = np.mean(rms)
        clarity_score = np.mean(spectral_centroids)
        
        # Realistic scoring with balanced ranges
        # 1. Pace: Optimal speaking rate is 120-180 BPM (most people speak 100-200)
        if tempo < 80:
            pace_score = 30  # Too slow
        elif tempo < 110:
            pace_score = 45 + (tempo - 80) / 30 * 15  # Slow (45-60%)
        elif tempo < 140:  # Sweet spot
            pace_score = 65 + (tempo - 110) / 30 * 20  # Good (65-85%)
        elif tempo < 180:
            pace_score = 75 - abs(tempo - 160) / 20 * 10  # Decent (65-75%)
        else:
            pace_score = max(40, 75 - (tempo - 180) / 20 * 10)  # Too fast
        
        # 2. Volume: More conservative dynamic range expectations
        volume_dynamic_range = np.ptp(rms) if len(rms) > 0 else 0
        volume_level = volume_mean if volume_mean > 0 else 0
        if volume_level < 0.005:  # Very quiet
            volume_score = 25
        elif volume_level < 0.02:  # Quiet
            volume_score = 35 + (volume_level - 0.005) / 0.015 * 20  # 35-55%
        elif volume_level < 0.1:  # Good level
            volume_score = 55 + (volume_level - 0.02) / 0.08 * 15  # 55-70%
        else:  # Strong level
            volume_score = 70 + min(15, volume_dynamic_range * 1000)  # 70-85% max
        
        # 3. Pitch: More realistic variance expectations for natural speech
        if pitch_variance < 50:
            pitch_score = 30  # Very monotone
        elif pitch_variance < 200:
            pitch_score = 40 + (pitch_variance - 50) / 150 * 20  # Some variety (40-60%)
        elif pitch_variance < 800:
            pitch_score = 60 + (pitch_variance - 200) / 600 * 20  # Good variety (60-80%)
        else:
            pitch_score = 75 + min(10, (pitch_variance - 800) / 1000 * 10)  # Excellent (75-85%)
        
        # 4. Clarity: Realistic spectral centroid ranges for speech
        if clarity_score < 800:
            clarity_score = 25  # Very unclear
        elif clarity_score < 1500:
            clarity_score = 35 + (clarity_score - 800) / 700 * 25  # Poor (35-60%)
        elif clarity_score < 2500:
            clarity_score = 60 + (clarity_score - 1500) / 1000 * 20  # Good (60-80%)
        else:
            clarity_score = 75 + min(10, (clarity_score - 2500) / 1000 * 10)  # Excellent (75-85%)
        
        # Ensure all scores are within 0-100 range
        pace_score = max(0, min(100, pace_score))
        volume_score = max(0, min(100, volume_score)) 
        pitch_score = max(0, min(100, pitch_score))
        clarity_score = max(0, min(100, clarity_score))
        
        return {
            'pace': round(float(pace_score), 1),
            'volume': round(float(volume_score), 1),
            'pitch': round(float(pitch_score), 1),
            'clarity': round(float(clarity_score), 1)
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
        prev_body_positions = None  # Track body movement for energy
        
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
                    
                    # Calculate energy level based on movement between frames
                    energy = self._calculate_energy_movement(pose_results.pose_landmarks, prev_body_positions)
                    if energy is not None:  # Only add if we have movement data
                        energy_scores.append(energy)
                    
                    # Update previous positions for next frame
                    prev_body_positions = self._extract_body_positions(pose_results.pose_landmarks)
                
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
    
    def _extract_body_positions(self, landmarks):
        """Extract key body positions for movement tracking"""
        key_points = [
            landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST],
            landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST],
            landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW],
            landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW],
            landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER],
            landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER],
            landmarks.landmark[self.mp_pose.PoseLandmark.NOSE],
        ]
        
        return [(p.x, p.y, p.z) for p in key_points]
    
    def _calculate_energy_movement(self, landmarks, prev_positions):
        """Calculate energy level based on actual movement between frames"""
        if prev_positions is None:
            return None  # Need previous frame for movement calculation
        
        current_positions = self._extract_body_positions(landmarks)
        
        # Calculate movement distances for each body part
        total_movement = 0
        movement_count = 0
        
        for i, (current, previous) in enumerate(zip(current_positions, prev_positions)):
            # Calculate 3D distance moved
            distance = np.sqrt(
                (current[0] - previous[0])**2 + 
                (current[1] - previous[1])**2 + 
                (current[2] - previous[2])**2
            )
            
            # Weight different body parts (hands and arms move more in gestures)
            if i < 4:  # Wrists and elbows - more important for energy
                weight = 2.0
            elif i < 6:  # Shoulders - moderate importance  
                weight = 1.5
            else:  # Head - less important for body energy
                weight = 1.0
            
            total_movement += distance * weight
            movement_count += weight
        
        # Average weighted movement
        avg_movement = total_movement / movement_count if movement_count > 0 else 0
        
        # Scale to 0-100 percentage (typical movement range is 0-0.1 in normalized coordinates)
        # Multiply by 500 to get reasonable scores (0.02 movement = 10%, 0.1 movement = 50%)
        energy_score = min(100, avg_movement * 500)
        
        # Apply minimum baseline (even minimal movement should show some energy)
        energy_score = max(20, energy_score)
        
        return energy_score
    
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
        if results.get('filler_words', {}).get('percentage') is not None:
            filler_percentage = results['filler_words']['percentage']
            filler_penalty = min(20, filler_percentage * 2)  # Max 20 point penalty
            filler_score = 100 - filler_penalty
            scores.append(filler_score)
            weights.append(5)
            logger.info(f"Filler words: {filler_percentage}% -> score: {filler_score}")
        
        # Calculate weighted average
        logger.info(f"Overall score calculation - Scores: {scores}, Weights: {weights}")
        if scores and weights:
            total_weight = sum(weights)
            weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
            overall_score = weighted_sum / total_weight
            final_score = round(max(0, min(100, overall_score)))
            logger.info(f"Calculated overall score: {final_score} (from weighted avg: {overall_score})")
            return final_score
        
        logger.warning("No valid scores found for overall calculation, returning default score of 50")
        return 50  # Default score if no analysis available


class LocalVideoAnalyzer:
    """Simplified video analyzer that works with local files (no S3 downloads)"""
    
    def __init__(self, video_path, progress_callback=None):
        self.video_path = video_path
        self.progress_callback = progress_callback
        
        # Initialize analyzers with speed optimization
        self.emotion_analyzer = EmotionAnalyzer()
        self.eye_contact_analyzer = EyeContactAnalyzer()
        self.speech_analyzer = SpeechAnalyzer(model_size="small")  # Use fast model
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
            logger.info("Starting emotion analysis")
            results['emotion'] = self.emotion_analyzer.analyze_video(self.video_path, self.progress_callback)
            logger.info(f"Emotion analysis result: {results['emotion']}")
            
            # 2. Eye Contact Analysis (25-40%)  
            logger.info("Starting eye contact analysis")
            results['eye_contact'] = self.eye_contact_analyzer.analyze_video(self.video_path, self.progress_callback)
            logger.info(f"Eye contact analysis result: {results['eye_contact']}")
            
            # 3. Speech Analysis (40-70%)
            logger.info("Starting speech analysis - extracting audio")
            audio, sr = self.speech_analyzer.extract_audio(self.video_path)
            logger.info(f"Audio extracted: {len(audio)} samples at {sr}Hz")
            
            logger.info("Transcribing audio")
            transcript = self.speech_analyzer.transcribe_audio(audio, sr, self.progress_callback)
            logger.info(f"Transcript result: '{transcript}' (length: {len(transcript)} chars)")
            results['transcript'] = transcript
            
            logger.info("Analyzing sentiment")
            results['sentiment'] = self.speech_analyzer.analyze_sentiment(transcript)
            logger.info(f"Sentiment analysis result: {results['sentiment']}")
            
            logger.info("Detecting filler words")
            results['filler_words'] = self.speech_analyzer.detect_filler_words(transcript)
            logger.info(f"Filler words result: {results['filler_words']}")
            
            logger.info("Analyzing vocal variety")
            results['vocal_variety'] = self.speech_analyzer.analyze_vocal_variety(audio, sr, self.progress_callback)
            logger.info(f"Vocal variety result: {results['vocal_variety']}")
            
            # 4. Body Language Analysis (70-85%)
            logger.info("Starting body language analysis")
            results['body_language'] = self.body_language_analyzer.analyze_video(self.video_path, self.progress_callback)
            logger.info(f"Body language result: {results['body_language']}")
            
            # 5. Calculate Overall Score (85-95%)
            if self.progress_callback:
                self.progress_callback(85, "Calculating overall score")
            
            logger.info("Calculating overall score")
            results['overall_score'] = self._calculate_overall_score(results)
            results['duration'] = round(duration, 2)
            
            logger.info(f"Final results summary: emotion={results.get('emotion', {}).get('dominant', 'N/A')}, transcript_len={len(results.get('transcript', ''))}, vocal_variety_keys={list(results.get('vocal_variety', {}).keys())}, overall_score={results['overall_score']}")
            
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
        if results.get('filler_words', {}).get('percentage') is not None:
            filler_percentage = results['filler_words']['percentage']
            filler_penalty = min(20, filler_percentage * 2)  # Max 20 point penalty
            filler_score = 100 - filler_penalty
            scores.append(filler_score)
            weights.append(5)
            logger.info(f"Filler words: {filler_percentage}% -> score: {filler_score}")
        
        # Calculate weighted average
        logger.info(f"Overall score calculation - Scores: {scores}, Weights: {weights}")
        if scores and weights:
            total_weight = sum(weights)
            weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
            overall_score = weighted_sum / total_weight
            final_score = round(max(0, min(100, overall_score)))
            logger.info(f"Calculated overall score: {final_score} (from weighted avg: {overall_score})")
            return final_score
        
        logger.warning("No valid scores found for overall calculation, returning default score of 50")
        return 50  # Default score if no analysis available
