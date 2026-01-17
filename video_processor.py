import os
import numpy as np
import librosa
from moviepy import VideoFileClip
from pathlib import Path
import warnings
import json
from PIL import Image
import whisper
from transformers import pipeline
warnings.filterwarnings('ignore')


class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.audio_path = None
        self.video_clip = None
        self.audio_data = None
        self.sample_rate = None
        self.temp_files = []
        self.transcript = None
        self.sentiment_scores = []
        
        # Initialize AI models (lazy loading)
        self.whisper_model = None
        self.sentiment_analyzer = None
        
    def extract_audio(self):
        try:
            self.video_clip = VideoFileClip(self.video_path)
            audio_path = "temp/audio.wav"
            self.audio_path = audio_path
            
            # Extract audio
            self.video_clip.audio.write_audiofile(
                audio_path,
                fps=22050,
                nbytes=2,
                codec='pcm_s16le'
            )
            
            self.temp_files.append(audio_path)
            return audio_path
            
        except Exception as e:
            raise Exception(f"Error extracting audio: {str(e)}")
    
    def transcribe_audio(self):
        try:
            print("Loading Whisper model...")
            if self.whisper_model is None:
                self.whisper_model = whisper.load_model("base")
            
            print("Transcribing audio with Whisper...")
            result = self.whisper_model.transcribe(
                self.audio_path,
                language="en",
                task="transcribe",
                verbose=False
            )
            
            self.transcript = result
            return result
            
        except Exception as e:
            print(f"Warning: Whisper transcription failed: {str(e)}")
            return None
    
    def analyze_sentiment(self):
        try:
            if self.transcript is None or 'segments' not in self.transcript:
                return []
            
            print("Analyzing sentiment with AI...")
            if self.sentiment_analyzer is None:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=-1  # CPU
                )
            
            sentiment_timeline = []
            for segment in self.transcript['segments']:
                text = segment['text'].strip()
                if text:
                    # Get sentiment and score
                    sentiment = self.sentiment_analyzer(text[:512])[0]  # Limit to 512 chars
                    
                    sentiment_timeline.append({
                        'start': segment['start'],
                        'end': segment['end'],
                        'text': text,
                        'sentiment': sentiment['label'],
                        'score': sentiment['score'],
                        'emphasis_score': self._calculate_emphasis(text)
                    })
            
            self.sentiment_scores = sentiment_timeline
            return sentiment_timeline
            
        except Exception as e:
            print(f"Warning: Sentiment analysis failed: {str(e)}")
            return []
    
    def _calculate_emphasis(self, text):
        score = 0.5
        
        # Exclamation marks indicate excitement
        score += text.count('!') * 0.1
        
        # Questions can be engaging
        score += text.count('?') * 0.05
        
        # ALL CAPS words indicate emphasis
        words = text.split()
        caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 2) / max(len(words), 1)
        score += caps_ratio * 0.2
        
        # Important keywords
        important_words = ['important', 'key', 'critical', 'must', 'never', 'always',
                          'remember', 'note', 'crucial', 'essential', 'amazing', 'wow']
        for word in important_words:
            if word in text.lower():
                score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def detect_emotional_peaks(self, num_peaks=3, min_gap=60, use_ai=True):
        try:
            if use_ai:
                self.transcribe_audio()
                self.analyze_sentiment()
            
            # Step 2: Audio feature extraction (existing code)
            # Load audio
            self.audio_data, self.sample_rate = librosa.load(
                self.audio_path,
                sr=22050
            )
            
            # Calculate RMS energy (loudness) over time
            frame_length = 2048
            hop_length = 512
            rms = librosa.feature.rms(
                y=self.audio_data,
                frame_length=frame_length,
                hop_length=hop_length
            )[0]
            
            # Convert frame indices to time
            times = librosa.frames_to_time(
                np.arange(len(rms)),
                sr=self.sample_rate,
                hop_length=hop_length
            )
            
            # Smooth the RMS curve
            from scipy.ndimage import gaussian_filter1d
            rms_smooth = gaussian_filter1d(rms, sigma=10)
            
            # Calculate spectral centroid (brightness/energy of audio)
            spectral_centroid = librosa.feature.spectral_centroid(
                y=self.audio_data,
                sr=self.sample_rate,
                hop_length=hop_length
            )[0]
            
            # Normalize both features
            rms_norm = (rms_smooth - np.min(rms_smooth)) / (np.max(rms_smooth) - np.min(rms_smooth) + 1e-6)
            centroid_norm = (spectral_centroid - np.min(spectral_centroid)) / (np.max(spectral_centroid) - np.min(spectral_centroid) + 1e-6)
            
            # Calculate tempo/rhythm strength
            onset_env = librosa.onset.onset_strength(y=self.audio_data, sr=self.sample_rate, hop_length=hop_length)
            onset_norm = (onset_env - np.min(onset_env)) / (np.max(onset_env) - np.min(onset_env) + 1e-6)
            
            # Step 3: MULTIMODAL FUSION - Combine audio + text sentiment
            if use_ai and self.sentiment_scores:
                # Create sentiment score timeline aligned with audio
                sentiment_timeline = np.zeros(len(rms_norm))
                
                for seg in self.sentiment_scores:
                    start_idx = int(seg['start'] * self.sample_rate / hop_length)
                    end_idx = int(seg['end'] * self.sample_rate / hop_length)
                    
                    # Sentiment contribution
                    sentiment_boost = 0.0
                    if seg['sentiment'] == 'POSITIVE':
                        sentiment_boost = seg['score'] * 0.3  # Positive emotion
                    else:
                        sentiment_boost = (1 - seg['score']) * 0.15  # Strong negative can also be engaging
                    
                    # Add emphasis from text features
                    emphasis_boost = seg['emphasis_score'] * 0.2
                    
                    # Apply to timeline
                    if start_idx < len(sentiment_timeline) and end_idx <= len(sentiment_timeline):
                        sentiment_timeline[start_idx:end_idx] += sentiment_boost + emphasis_boost
                
                # Normalize sentiment
                if np.max(sentiment_timeline) > 0:
                    sentiment_norm = sentiment_timeline / np.max(sentiment_timeline)
                else:
                    sentiment_norm = sentiment_timeline
                
                # MULTIMODAL SCORE: Audio (60%) + Text Sentiment (40%)
                emotional_score = (0.3 * rms_norm + 0.15 * centroid_norm + 
                                 0.15 * onset_norm + 0.4 * sentiment_norm)
            else:
                # Fallback to audio-only if AI not available
                emotional_score = 0.5 * rms_norm + 0.2 * centroid_norm + 0.3 * onset_norm
            
            # Find peaks
            peaks = []
            min_gap_frames = int(min_gap * self.sample_rate / hop_length)
            
            # Get video duration
            video_duration = self.video_clip.duration
            
            # Sort by emotional score and pick top peaks
            peak_candidates = []
            for i in range(len(emotional_score)):
                timestamp = times[i]
                # Ensure we don't pick peaks too close to start or end
                if timestamp > 10 and timestamp < video_duration - 70:
                    peak_candidates.append((timestamp, emotional_score[i]))
            
            # Sort by score (descending)
            peak_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Select peaks with minimum gap constraint
            selected_peaks = []
            for timestamp, score in peak_candidates:
                # Check if this peak is far enough from already selected peaks
                if all(abs(timestamp - existing) >= min_gap for existing in selected_peaks):
                    selected_peaks.append(timestamp)
                    if len(selected_peaks) >= num_peaks:
                        break
            
            # Sort peaks chronologically
            selected_peaks.sort()
            
            # Get confidence scores for selected peaks
            peak_scores = []
            for peak in selected_peaks:
                peak_idx = int(peak * self.sample_rate / hop_length)
                if peak_idx < len(emotional_score):
                    peak_scores.append(emotional_score[peak_idx])
                else:
                    peak_scores.append(0.5)
            
            # If we didn't find enough peaks, add some from middle sections
            if len(selected_peaks) < num_peaks:
                video_duration = self.video_clip.duration
                step = video_duration / (num_peaks + 1)
                for i in range(1, num_peaks + 1):
                    candidate = step * i
                    if all(abs(candidate - existing) >= min_gap for existing in selected_peaks):
                        selected_peaks.append(candidate)
                        if len(selected_peaks) >= num_peaks:
                            break
                selected_peaks.sort()
                # Recalculate scores for added peaks
                peak_scores = []
                for peak in selected_peaks:
                    peak_idx = int(peak * self.sample_rate / hop_length)
                    if peak_idx < len(emotional_score):
                        peak_scores.append(emotional_score[peak_idx])
                    else:
                        peak_scores.append(0.5)
            
            # Extract keywords for each peak
            peak_keywords = []
            for peak in selected_peaks[:num_peaks]:
                keywords = self._extract_keywords_at_timestamp(peak)
                peak_keywords.append(keywords)
            
            return (selected_peaks[:num_peaks], peak_scores[:num_peaks], peak_keywords)
            
        except Exception as e:
            raise Exception(f"Error detecting emotional peaks: {str(e)}")
    
    def _extract_keywords_at_timestamp(self, timestamp):
        if not self.sentiment_scores:
            return "High-energy moment"
        
        # Find segments around this timestamp (±5 seconds)
        relevant_segments = [
            seg for seg in self.sentiment_scores
            if abs((seg['start'] + seg['end']) / 2 - timestamp) < 5
        ]
        
        if not relevant_segments:
            return "Engaging content"
        
        # Get the most emphasized segment
        best_seg = max(relevant_segments, key=lambda x: x['emphasis_score'])
        
        # Extract key phrases (first few words)
        text = best_seg['text'].strip()
        words = text.split()[:8]  # First 8 words
        return ' '.join(words) + ("..." if len(text.split()) > 8 else "")
    
    def extract_clips(self, peaks, clip_duration=60, output_dir="output"):
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            
            clips_info = []
            video_duration = self.video_clip.duration
            
            # Unpack peaks and scores
            peak_times, peak_scores = peaks
            
            for i, (peak, score) in enumerate(zip(peak_times, peak_scores), 1):
                # Calculate start and end times
                # Center the clip around the peak
                start_time = max(0, peak - clip_duration / 2)
                end_time = min(video_duration, start_time + clip_duration)
                
                # Adjust start if end hit the boundary
                if end_time == video_duration:
                    start_time = max(0, end_time - clip_duration)
                
                # Extract clip (use subclipped for moviepy 2.x)
                clip = self.video_clip.subclipped(start_time, end_time)
                
                # Output path
                output_path = output_dir / f"reel_{i}.mp4"
                
                # Write video file
                clip.write_videofile(
                    str(output_path),
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile=f'temp/temp_audio_{i}.m4a',
                    remove_temp=True
                )
                
                # Generate thumbnail
                thumbnail_path = output_dir / f"reel_{i}_thumb.jpg"
                thumbnail_time = (start_time + end_time) / 2  # Middle of clip
                frame = clip.get_frame(thumbnail_time - start_time)
                thumbnail = Image.fromarray(frame)
                thumbnail.thumbnail((320, 180))  # 16:9 aspect ratio
                thumbnail.save(str(thumbnail_path), quality=85)
                
                # Collect clip information
                clip_info = {
                    'id': i,
                    'path': str(output_path),
                    'thumbnail': str(thumbnail_path),
                    'peak_time': peak,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': clip_duration,
                    'confidence_score': float(score),
                    'file_size': os.path.getsize(output_path)
                }
                clips_info.append(clip_info)
                
                clip.close()
            
            # Create metadata JSON file
            metadata_path = output_dir / "clips_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    'total_clips': len(clips_info),
                    'clips': clips_info,
                    'generation_settings': {
                        'clip_duration': clip_duration,
                        'num_clips_requested': len(peak_times)
                    }
                }, f, indent=2)
            
            return clips_info
            
        except Exception as e:
            raise Exception(f"Error extracting clips: {str(e)}")
    
    def cleanup(self):
        try:
            if self.video_clip:
                self.video_clip.close()
            
            # Clean temp files
            for temp_file in self.temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    
        except Exception as e:
            print(f"Warning: Error during cleanup: {str(e)}")
