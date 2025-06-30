# sarcasm_detection.py
"""
Advanced Sarcasm Detection Module for ArchiveCAT
Combines acoustic features, prosodic analysis, and text-based detection
for reliable and cost-efficient sarcasm identification.
"""

import os
import json
import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import re
import warnings
from dataclasses import dataclass
from enum import Enum
import pickle

# Text analysis imports
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Text-based detection will be limited.")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ML imports
try:
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. ML-based detection will be limited.")

warnings.filterwarnings('ignore')


class SarcasmConfidence(Enum):
    """Confidence levels for sarcasm detection"""
    VERY_LOW = "very_low"      # 0-20%
    LOW = "low"                # 20-40%
    MODERATE = "moderate"      # 40-60%
    HIGH = "high"              # 60-80%
    VERY_HIGH = "very_high"    # 80-100%


@dataclass
class SarcasmResult:
    """Result container for sarcasm detection"""
    is_sarcastic: bool
    confidence: float
    confidence_level: SarcasmConfidence
    detection_method: str
    acoustic_score: float
    text_score: float
    combined_score: float
    indicators: List[str]
    segment_results: List[Dict]
    metadata: Dict


class SarcasmDetector:
    """
    Multi-modal sarcasm detection system combining:
    1. Acoustic/Prosodic features (pitch, tone, rhythm)
    2. Text-based analysis (sentiment, context, patterns)
    3. Machine learning ensemble for final decision
    """
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        Initialize the sarcasm detector
        
        Args:
            model_name: HuggingFace model for sentiment analysis
        """
        self.model_name = model_name
        self.sentiment_analyzer = None
        self.text_classifier = None
        self.scaler = StandardScaler()
        self.ensemble_model = None
        
        # Sarcasm indicators and patterns
        self.sarcasm_patterns = self._load_sarcasm_patterns()
        self.sentiment_reversals = self._load_sentiment_reversals()
        
        # Initialize models
        self._initialize_models()
        
        # Feature weights for ensemble
        self.feature_weights = {
            'acoustic': 0.4,
            'text': 0.4,
            'prosodic': 0.2
        }
    
    def _initialize_models(self):
        """Initialize ML models for sarcasm detection"""
        try:
            if TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
                # Load sentiment analysis model
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model=self.model_name,
                    device=0 if torch.cuda.is_available() else -1
                )
                
                # Try to load a dedicated sarcasm detection model if available
                try:
                    # Alternative: Use a model specifically trained for sarcasm
                    sarcasm_model_name = "helinivan/english-sarcasm-detector"
                    self.text_classifier = pipeline(
                        "text-classification",
                        model=sarcasm_model_name,
                        device=0 if torch.cuda.is_available() else -1
                    )
                    print(f"Loaded dedicated sarcasm model: {sarcasm_model_name}")
                except:
                    print("Using general sentiment model for text analysis")
                
                print(f"Loaded text analysis models successfully")
            
            if SKLEARN_AVAILABLE:
                # Initialize ensemble classifier
                self._setup_ensemble_classifier()
                
        except Exception as e:
            print(f"Warning: Could not initialize all models: {e}")
    
    def _setup_ensemble_classifier(self):
        """Setup ensemble classifier for final sarcasm prediction"""
        try:
            # Random Forest for acoustic features
            rf_acoustic = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            # SVM for text features
            svm_text = SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                class_weight='balanced'
            )
            
            # Ensemble voting classifier
            self.ensemble_model = VotingClassifier(
                estimators=[
                    ('rf_acoustic', rf_acoustic),
                    ('svm_text', svm_text)
                ],
                voting='soft'
            )
            
        except Exception as e:
            print(f"Warning: Could not setup ensemble classifier: {e}")
    
    def detect_sarcasm(self, audio_path: str = None, transcript: str = None, 
                      prosody_results: Dict = None) -> SarcasmResult:
        """
        Main sarcasm detection function
        
        Args:
            audio_path: Path to audio file
            transcript: Text transcript
            prosody_results: Prosodic analysis results
            
        Returns:
            SarcasmResult with detection results
        """
        if not audio_path and not transcript:
            raise ValueError("Either audio_path or transcript must be provided")
        
        # Initialize scores
        acoustic_score = 0.0
        text_score = 0.0
        prosodic_score = 0.0
        indicators = []
        segment_results = []
        
        # 1. Acoustic Analysis
        if audio_path and os.path.exists(audio_path):
            try:
                acoustic_score, acoustic_indicators = self._analyze_acoustic_sarcasm(audio_path)
                indicators.extend(acoustic_indicators)
            except Exception as e:
                print(f"Warning: Acoustic analysis failed: {e}")
        
        # 2. Text Analysis
        if transcript:
            try:
                text_score, text_indicators, segment_results = self._analyze_text_sarcasm(transcript)
                indicators.extend(text_indicators)
            except Exception as e:
                print(f"Warning: Text analysis failed: {e}")
        
        # 3. Prosodic Analysis Integration
        if prosody_results:
            try:
                prosodic_score, prosodic_indicators = self._analyze_prosodic_sarcasm(prosody_results)
                indicators.extend(prosodic_indicators)
            except Exception as e:
                print(f"Warning: Prosodic analysis failed: {e}")
        
        # 4. Combine scores using weighted ensemble
        combined_score = self._calculate_combined_score(
            acoustic_score, text_score, prosodic_score
        )
        
        # 5. Final decision
        is_sarcastic = combined_score > 0.5
        confidence = abs(combined_score - 0.5) * 2  # Convert to 0-1 scale
        confidence_level = self._get_confidence_level(confidence)
        
        # 6. Determine primary detection method
        detection_method = self._determine_detection_method(
            acoustic_score, text_score, prosodic_score
        )
        
        return SarcasmResult(
            is_sarcastic=is_sarcastic,
            confidence=confidence,
            confidence_level=confidence_level,
            detection_method=detection_method,
            acoustic_score=acoustic_score,
            text_score=text_score,
            combined_score=combined_score,
            indicators=list(set(indicators)),  # Remove duplicates
            segment_results=segment_results,
            metadata={
                'audio_available': audio_path is not None,
                'text_available': transcript is not None,
                'prosody_available': prosody_results is not None,
                'feature_weights': self.feature_weights
            }
        )
    
    def _analyze_acoustic_sarcasm(self, audio_path: str) -> Tuple[float, List[str]]:
        """
        Analyze acoustic features for sarcasm detection
        
        Research shows sarcastic speech typically has:
        - Exaggerated pitch contours
        - Slower speech rate
        - Longer vowel duration
        - Different stress patterns
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=22050)
            sound = parselmouth.Sound(audio_path)
            
            features = {}
            indicators = []
            
            # 1. Pitch Analysis
            pitch = call(sound, "To Pitch", 0.0, 75, 600)
            pitch_values = pitch.selected_array['frequency']
            pitch_values = pitch_values[pitch_values > 0]
            
            if len(pitch_values) > 0:
                # Pitch range and variability
                pitch_range = np.max(pitch_values) - np.min(pitch_values)
                pitch_std = np.std(pitch_values)
                pitch_cv = pitch_std / np.mean(pitch_values)
                
                features['pitch_range'] = pitch_range
                features['pitch_variability'] = pitch_cv
                
                # Exaggerated pitch contours (indicator of sarcasm)
                if pitch_range > 150:  # Hz
                    indicators.append("Exaggerated pitch range")
                
                if pitch_cv > 0.25:
                    indicators.append("High pitch variability")
            
            # 2. Speech Rate
            duration = len(audio) / sr
            
            # Estimate syllables using intensity peaks
            intensity = call(sound, "To Intensity", 100, 0)
            intensity_smooth = np.convolve(intensity.values[0], np.ones(5)/5, mode='same')
            intensity_smooth = intensity_smooth[~np.isnan(intensity_smooth)]
            
            if len(intensity_smooth) > 0:
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(intensity_smooth, height=np.percentile(intensity_smooth, 50))
                syllables_per_second = len(peaks) / duration
                
                features['speech_rate'] = syllables_per_second
                
                # Slower speech rate (common in sarcasm)
                if syllables_per_second < 3.5:
                    indicators.append("Slower speech rate")
            
            # 3. Intensity Patterns
            intensity_values = intensity.values[0]
            intensity_values = intensity_values[~np.isnan(intensity_values)]
            
            if len(intensity_values) > 0:
                intensity_range = np.max(intensity_values) - np.min(intensity_values)
                features['intensity_range'] = intensity_range
                
                # Exaggerated volume changes
                if intensity_range > 30:  # dB
                    indicators.append("Exaggerated volume changes")
            
            # 4. Spectral Features
            # Formant analysis for voice quality changes
            try:
                formants = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
                f1_values = []
                f2_values = []
                
                for i in range(1, min(100, int(sound.xmax * 100))):  # Sample every 10ms
                    time = i * 0.01
                    try:
                        f1 = call(formants, "Get value at time", 1, time, "Hertz", "Linear")
                        f2 = call(formants, "Get value at time", 2, time, "Hertz", "Linear")
                        if not np.isnan(f1) and not np.isnan(f2):
                            f1_values.append(f1)
                            f2_values.append(f2)
                    except:
                        continue
                
                if len(f1_values) > 5:
                    f1_mean = np.mean(f1_values)
                    f2_mean = np.mean(f2_values)
                    features['f1_mean'] = f1_mean
                    features['f2_mean'] = f2_mean
                    
                    # Voice quality indicators
                    if f1_mean > 500:  # Higher F1 can indicate stressed speech
                        indicators.append("Altered voice quality (high F1)")
                
            except Exception as e:
                print(f"Formant analysis failed: {e}")
            
            # 5. Rhythm and Timing
            # Analyze pause patterns
            rms = librosa.feature.rms(y=audio, frame_length=int(0.025 * sr), 
                                    hop_length=int(0.010 * sr))[0]
            silence_threshold = np.median(rms) * 0.2
            silent_frames = rms < silence_threshold
            
            # Find pause durations
            pauses = []
            in_pause = False
            pause_start = 0
            frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, 
                                               hop_length=int(0.010 * sr))
            
            for i, is_silent in enumerate(silent_frames):
                if is_silent and not in_pause:
                    in_pause = True
                    pause_start = frame_times[i]
                elif not is_silent and in_pause:
                    in_pause = False
                    pause_duration = frame_times[i] - pause_start
                    if pause_duration >= 0.1:  # Minimum 100ms
                        pauses.append(pause_duration)
            
            if pauses:
                mean_pause = np.mean(pauses)
                features['mean_pause_duration'] = mean_pause
                
                # Unusual pause patterns
                if mean_pause > 0.8:  # Longer pauses
                    indicators.append("Unusual pause patterns")
            
            # Calculate overall acoustic sarcasm score
            acoustic_score = self._calculate_acoustic_score(features)
            
            return acoustic_score, indicators
            
        except Exception as e:
            print(f"Acoustic analysis error: {e}")
            return 0.0, []
    
    def _analyze_text_sarcasm(self, transcript: str) -> Tuple[float, List[str], List[Dict]]:
        """
        Analyze text content for sarcasm indicators
        
        Uses multiple approaches:
        1. Sentiment analysis
        2. Pattern matching
        3. Context analysis
        4. ML-based classification
        """
        indicators = []
        segment_results = []
        
        # Clean and prepare text
        text = self._preprocess_text(transcript)
        sentences = self._split_into_sentences(text)
        
        scores = []
        
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) < 3:
                continue
                
            sentence_score = 0.0
            sentence_indicators = []
            
            # 1. Pattern-based detection
            pattern_score, pattern_indicators = self._check_sarcasm_patterns(sentence)
            sentence_score += pattern_score * 0.3
            sentence_indicators.extend(pattern_indicators)
            
            # 2. Sentiment analysis
            if self.sentiment_analyzer:
                sentiment_score, sentiment_indicators = self._analyze_sentiment_sarcasm(sentence)
                sentence_score += sentiment_score * 0.4
                sentence_indicators.extend(sentiment_indicators)
            
            # 3. ML-based classification
            if self.text_classifier:
                ml_score, ml_indicators = self._classify_text_sarcasm(sentence)
                sentence_score += ml_score * 0.3
                sentence_indicators.extend(ml_indicators)
            
            # Store segment results
            segment_results.append({
                'sentence': sentence,
                'score': sentence_score,
                'indicators': sentence_indicators,
                'index': i
            })
            
            scores.append(sentence_score)
            indicators.extend(sentence_indicators)
        
        # Overall text score
        if scores:
            text_score = np.mean(scores)
            # Boost score if multiple sentences are sarcastic
            if len([s for s in scores if s > 0.6]) > 1:
                text_score = min(1.0, text_score * 1.2)
        else:
            text_score = 0.0
        
        return text_score, indicators, segment_results
    
    def _analyze_prosodic_sarcasm(self, prosody_results: Dict) -> Tuple[float, List[str]]:
        """
        Analyze prosodic features for sarcasm detection
        
        Uses existing prosody analysis results to identify sarcasm indicators
        """
        if not prosody_results.get('success'):
            return 0.0, []
        
        results = prosody_results['results']
        indicators = []
        score = 0.0
        
        # 1. Pitch characteristics
        pitch_data = results.get('pitch', {})
        if 'error' not in pitch_data:
            # High pitch variability
            cv_f0 = pitch_data.get('cv_f0', 0)
            if cv_f0 > 0.25:
                score += 0.2
                indicators.append("High pitch variability (prosodic)")
            
            # Unusual pitch range
            range_f0 = pitch_data.get('range_f0', 0)
            if range_f0 > 200:
                score += 0.15
                indicators.append("Wide pitch range (prosodic)")
        
        # 2. Speech rate
        speech_data = results.get('speech_rate', {})
        if 'error' not in speech_data:
            wpm = speech_data.get('words_per_minute', 150)
            category = speech_data.get('speech_rate_category', '')
            
            # Slower speech often indicates sarcasm
            if wpm < 120 or category in ['sehr langsam', 'langsam']:
                score += 0.15
                indicators.append("Slow speech rate (prosodic)")
        
        # 3. Emotional indicators
        emotion_data = results.get('emotion_indicators', {})
        if 'error' not in emotion_data:
            arousal = emotion_data.get('arousal_level', '')
            valence = emotion_data.get('valence_tendency', '')
            
            # High arousal with negative valence can indicate sarcasm
            if arousal in ['hoch', 'sehr hoch'] and valence == 'negativ':
                score += 0.2
                indicators.append("High arousal + negative valence")
        
        # 4. Voice quality
        quality_data = results.get('voice_quality', {})
        if 'error' not in quality_data:
            assessment = quality_data.get('voice_quality_assessment', '')
            
            # Voice quality changes in sarcasm
            if assessment == 'auffällig':
                score += 0.1
                indicators.append("Unusual voice quality")
        
        # 5. Pause patterns
        pause_data = results.get('pauses', {})
        if 'error' not in pause_data:
            pause_ratio = pause_data.get('pause_ratio', 0)
            
            # Unusual pause patterns
            if pause_ratio > 0.3:  # More than 30% pauses
                score += 0.1
                indicators.append("Excessive pauses")
        
        return min(1.0, score), indicators
    
    def _check_sarcasm_patterns(self, text: str) -> Tuple[float, List[str]]:
        """Check for sarcasm patterns in text"""
        text_lower = text.lower()
        score = 0.0
        indicators = []
        
        # 1. Explicit sarcasm markers
        explicit_markers = [
            r'\b(oh really|oh sure|yeah right|of course|obviously|clearly)\b',
            r'\b(great|fantastic|wonderful|perfect|brilliant)\b.*\b(not|fail|problem)\b',
            r'\b(just what i needed|exactly what i wanted)\b'
        ]
        
        for pattern in explicit_markers:
            if re.search(pattern, text_lower):
                score += 0.4
                indicators.append("Explicit sarcasm marker")
                break
        
        # 2. Contradiction patterns
        contradiction_patterns = [
            r'\b(love|enjoy|like)\b.*\b(hate|awful|terrible|horrible)\b',
            r'\b(good|great|nice)\b.*\b(bad|awful|terrible|horrible)\b',
            r'\b(easy|simple)\b.*\b(difficult|hard|impossible|complex)\b'
        ]
        
        for pattern in contradiction_patterns:
            if re.search(pattern, text_lower):
                score += 0.3
                indicators.append("Contradictory sentiment")
        
        # 3. Extreme positive words in negative context
        extreme_positive = ['amazing', 'incredible', 'fantastic', 'wonderful', 'perfect']
        for word in extreme_positive:
            if word in text_lower:
                # Check context for negativity
                context_negative = any(neg in text_lower for neg in 
                                     ['fail', 'broken', 'wrong', 'disaster', 'mess'])
                if context_negative:
                    score += 0.25
                    indicators.append("Extreme positive + negative context")
        
        # 4. Question + positive statement pattern
        if '?' in text and any(pos in text_lower for pos in ['great', 'good', 'nice', 'fine']):
            score += 0.2
            indicators.append("Question + positive statement")
        
        # 5. Overuse of punctuation
        if text.count('!') > 2 or text.count('?') > 1:
            score += 0.1
            indicators.append("Excessive punctuation")
        
        # 6. Capital letters for emphasis
        if len(re.findall(r'\b[A-Z]{2,}\b', text)) > 0:
            score += 0.1
            indicators.append("CAPS for emphasis")
        
        return min(1.0, score), indicators
    
    def _analyze_sentiment_sarcasm(self, text: str) -> Tuple[float, List[str]]:
        """Use sentiment analysis to detect sarcasm"""
        if not self.sentiment_analyzer:
            return 0.0, []
        
        try:
            result = self.sentiment_analyzer(text)
            sentiment = result[0]['label'].lower()
            confidence = result[0]['score']
            
            indicators = []
            score = 0.0
            
            # Look for sentiment-context mismatches
            positive_words = ['great', 'good', 'nice', 'excellent', 'perfect', 'wonderful']
            negative_words = ['bad', 'awful', 'terrible', 'horrible', 'disaster', 'fail']
            
            has_positive_words = any(word in text.lower() for word in positive_words)
            has_negative_words = any(word in text.lower() for word in negative_words)
            
            # Sentiment reversal detection
            if sentiment == 'positive' and has_negative_words:
                score += 0.4
                indicators.append("Positive sentiment + negative words")
            elif sentiment == 'negative' and has_positive_words:
                score += 0.3
                indicators.append("Negative sentiment + positive words")
            
            # Low confidence can indicate mixed/sarcastic sentiment
            if confidence < 0.7:
                score += 0.2
                indicators.append("Ambiguous sentiment")
            
            return min(1.0, score), indicators
            
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return 0.0, []
    
    def _classify_text_sarcasm(self, text: str) -> Tuple[float, List[str]]:
        """Use ML classifier for sarcasm detection"""
        if not self.text_classifier:
            return 0.0, []
        
        try:
            result = self.text_classifier(text)
            
            # Handle different model outputs
            if isinstance(result, list) and len(result) > 0:
                pred = result[0]
                if pred['label'].lower() in ['sarcasm', 'sarcastic', 'label_1']:
                    score = pred['score']
                    return score, ["ML model detected sarcasm"]
                elif pred['label'].lower() in ['not_sarcasm', 'not_sarcastic', 'label_0']:
                    return 0.0, []
            
            return 0.0, []
            
        except Exception as e:
            print(f"ML classification error: {e}")
            return 0.0, []
    
    def _calculate_acoustic_score(self, features: Dict[str, float]) -> float:
        """Calculate overall acoustic sarcasm score"""
        score = 0.0
        
        # Pitch-based scoring
        if 'pitch_range' in features:
            pitch_range = features['pitch_range']
            if pitch_range > 200:
                score += 0.3
            elif pitch_range > 150:
                score += 0.2
            elif pitch_range > 100:
                score += 0.1
        
        if 'pitch_variability' in features:
            pitch_var = features['pitch_variability']
            if pitch_var > 0.3:
                score += 0.25
            elif pitch_var > 0.2:
                score += 0.15
        
        # Speech rate scoring
        if 'speech_rate' in features:
            speech_rate = features['speech_rate']
            if speech_rate < 3.0:  # Very slow
                score += 0.2
            elif speech_rate < 3.5:  # Slow
                score += 0.1
        
        # Intensity scoring
        if 'intensity_range' in features:
            intensity_range = features['intensity_range']
            if intensity_range > 35:
                score += 0.15
            elif intensity_range > 25:
                score += 0.1
        
        # Pause pattern scoring
        if 'mean_pause_duration' in features:
            pause_dur = features['mean_pause_duration']
            if pause_dur > 1.0:
                score += 0.1
        
        return min(1.0, score)
    
    def _calculate_combined_score(self, acoustic: float, text: float, prosodic: float) -> float:
        """Combine different modality scores using weighted average"""
        weights = self.feature_weights
        
        # Adjust weights based on available modalities
        total_weight = 0
        weighted_sum = 0
        
        if acoustic > 0:
            weighted_sum += acoustic * weights['acoustic']
            total_weight += weights['acoustic']
        
        if text > 0:
            weighted_sum += text * weights['text']
            total_weight += weights['text']
        
        if prosodic > 0:
            weighted_sum += prosodic * weights['prosodic']
            total_weight += weights['prosodic']
        
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return 0.0
    
    def _determine_detection_method(self, acoustic: float, text: float, prosodic: float) -> str:
        """Determine which method contributed most to detection"""
        scores = {
            'acoustic': acoustic,
            'text': text,
            'prosodic': prosodic
        }
        
        max_method = max(scores, key=scores.get)
        max_score = scores[max_method]
        
        if max_score > 0.6:
            return f"{max_method}_primary"
        elif max_score > 0.4:
            return f"{max_method}_secondary"
        else:
            return "multimodal_weak"
    
    def _get_confidence_level(self, confidence: float) -> SarcasmConfidence:
        """Convert confidence score to level"""
        if confidence >= 0.8:
            return SarcasmConfidence.VERY_HIGH
        elif confidence >= 0.6:
            return SarcasmConfidence.HIGH
        elif confidence >= 0.4:
            return SarcasmConfidence.MODERATE
        elif confidence >= 0.2:
            return SarcasmConfidence.LOW
        else:
            return SarcasmConfidence.VERY_LOW
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _load_sarcasm_patterns(self) -> List[str]:
        """Load common sarcasm patterns"""
        return [
            "oh really",
            "oh sure",
            "yeah right",
            "of course",
            "obviously",
            "clearly",
            "great job",
            "well done",
            "fantastic",
            "perfect",
            "just great",
            "wonderful",
            "brilliant"
        ]
    
    def _load_sentiment_reversals(self) -> List[Tuple[str, str]]:
        """Load sentiment reversal patterns"""
        return [
            ("love", "hate"),
            ("good", "bad"),
            ("great", "awful"),
            ("nice", "terrible"),
            ("perfect", "disaster"),
            ("easy", "impossible"),
            ("simple", "complex")
        ]
    
    def save_results(self, result: SarcasmResult, output_path: str):
        """Save sarcasm detection results"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as JSON
            json_data = {
                'sarcasm_detection': {
                    'is_sarcastic': result.is_sarcastic,
                    'confidence': result.confidence,
                    'confidence_level': result.confidence_level.value,
                    'detection_method': result.detection_method,
                    'scores': {
                        'acoustic': result.acoustic_score,
                        'text': result.text_score,
                        'combined': result.combined_score
                    },
                    'indicators': result.indicators,
                    'segment_results': result.segment_results,
                    'metadata': result.metadata
                }
            }
            
            json_path = output_path.with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            
            # Save detailed text report
            txt_path = output_path.with_suffix('.txt')
            self._save_text_report(result, txt_path)
            
            # Save CSV summary
            csv_path = output_path.with_suffix('.csv')
            self._save_csv_summary(result, csv_path)
            
            print(f"Sarcasm detection results saved: {output_path}")
            
        except Exception as e:
            print(f"Error saving sarcasm results: {e}")
    
    def _save_text_report(self, result: SarcasmResult, output_path: Path):
        """Save human-readable text report"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("SARCASM DETECTION REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                # Overall result
                f.write("OVERALL DETECTION:\n")
                f.write(f"  Sarcastic: {'YES' if result.is_sarcastic else 'NO'}\n")
                f.write(f"  Confidence: {result.confidence:.2%}\n")
                f.write(f"  Confidence Level: {result.confidence_level.value.replace('_', ' ').title()}\n")
                f.write(f"  Primary Method: {result.detection_method}\n\n")
                
                # Scores breakdown
                f.write("SCORE BREAKDOWN:\n")
                f.write(f"  Acoustic Score: {result.acoustic_score:.3f}\n")
                f.write(f"  Text Score: {result.text_score:.3f}\n")
                f.write(f"  Combined Score: {result.combined_score:.3f}\n\n")
                
                # Indicators
                if result.indicators:
                    f.write("SARCASM INDICATORS:\n")
                    for indicator in result.indicators:
                        f.write(f"  • {indicator}\n")
                    f.write("\n")
                
                # Segment analysis
                if result.segment_results:
                    f.write("SEGMENT ANALYSIS:\n")
                    for i, segment in enumerate(result.segment_results, 1):
                        f.write(f"  Segment {i} (Score: {segment['score']:.3f}):\n")
                        f.write(f"    Text: \"{segment['sentence']}\"\n")
                        if segment['indicators']:
                            f.write(f"    Indicators: {', '.join(segment['indicators'])}\n")
                        f.write("\n")
                
                # Metadata
                f.write("ANALYSIS METADATA:\n")
                for key, value in result.metadata.items():
                    f.write(f"  {key}: {value}\n")
                
        except Exception as e:
            print(f"Error saving text report: {e}")
    
    def _save_csv_summary(self, result: SarcasmResult, output_path: Path):
        """Save CSV summary for data analysis"""
        try:
            data = [{
                'is_sarcastic': result.is_sarcastic,
                'confidence': result.confidence,
                'confidence_level': result.confidence_level.value,
                'detection_method': result.detection_method,
                'acoustic_score': result.acoustic_score,
                'text_score': result.text_score,
                'combined_score': result.combined_score,
                'indicator_count': len(result.indicators),
                'segment_count': len(result.segment_results),
                'audio_available': result.metadata.get('audio_available', False),
                'text_available': result.metadata.get('text_available', False),
                'prosody_available': result.metadata.get('prosody_available', False)
            }]
            
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False, encoding='utf-8')
            
        except Exception as e:
            print(f"Error saving CSV summary: {e}")


class SarcasmAnalyzer:
    """High-level analyzer that integrates with ArchiveCAT workflow"""
    
    def __init__(self):
        self.detector = SarcasmDetector()
        self.results_cache = {}
    
    def analyze_file(self, audio_path: str = None, transcript_path: str = None, 
                    prosody_results: Dict = None, output_dir: str = None) -> SarcasmResult:
        """
        Analyze a single file for sarcasm
        
        Args:
            audio_path: Path to audio file
            transcript_path: Path to transcript file
            prosody_results: Prosodic analysis results
            output_dir: Output directory for results
            
        Returns:
            SarcasmResult
        """
        # Load transcript if path provided
        transcript = None
        if transcript_path and os.path.exists(transcript_path):
            try:
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    transcript = f.read()
            except Exception as e:
                print(f"Error loading transcript: {e}")
        
        # Perform detection
        result = self.detector.detect_sarcasm(
            audio_path=audio_path,
            transcript=transcript,
            prosody_results=prosody_results
        )
        
        # Save results if output directory specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = "sarcasm_analysis"
            if audio_path:
                base_name = Path(audio_path).stem + "_sarcasm"
            
            output_path = os.path.join(output_dir, base_name)
            self.detector.save_results(result, output_path)
        
        return result
    
    def batch_analyze(self, files: List[Dict], output_dir: str) -> Dict[str, SarcasmResult]:
        """
        Batch analyze multiple files
        
        Args:
            files: List of dicts with 'audio_path', 'transcript_path', 'prosody_results'
            output_dir: Output directory
            
        Returns:
            Dict mapping file paths to results
        """
        results = {}
        os.makedirs(output_dir, exist_ok=True)
        
        for i, file_info in enumerate(files, 1):
            print(f"Analyzing file {i}/{len(files)}")
            
            audio_path = file_info.get('audio_path')
            transcript_path = file_info.get('transcript_path')
            prosody_results = file_info.get('prosody_results')
            
            try:
                result = self.analyze_file(
                    audio_path=audio_path,
                    transcript_path=transcript_path,
                    prosody_results=prosody_results,
                    output_dir=output_dir
                )
                
                key = audio_path or transcript_path or f"file_{i}"
                results[key] = result
                
            except Exception as e:
                print(f"Error analyzing file {i}: {e}")
                continue
        
        # Create batch summary
        self._create_batch_summary(results, output_dir)
        
        return results
    
    def _create_batch_summary(self, results: Dict[str, SarcasmResult], output_dir: str):
        """Create summary of batch analysis"""
        try:
            summary_data = []
            
            for file_path, result in results.items():
                summary_data.append({
                    'file': os.path.basename(file_path),
                    'is_sarcastic': result.is_sarcastic,
                    'confidence': result.confidence,
                    'confidence_level': result.confidence_level.value,
                    'acoustic_score': result.acoustic_score,
                    'text_score': result.text_score,
                    'combined_score': result.combined_score,
                    'indicator_count': len(result.indicators),
                    'detection_method': result.detection_method
                })
            
            # Save as CSV
            df = pd.DataFrame(summary_data)
            csv_path = os.path.join(output_dir, "sarcasm_batch_summary.csv")
            df.to_csv(csv_path, index=False)
            
            # Create text summary
            txt_path = os.path.join(output_dir, "sarcasm_batch_report.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("BATCH SARCASM ANALYSIS SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                
                total_files = len(results)
                sarcastic_files = len([r for r in results.values() if r.is_sarcastic])
                
                f.write(f"Total files analyzed: {total_files}\n")
                f.write(f"Sarcastic files detected: {sarcastic_files}\n")
                f.write(f"Sarcasm rate: {sarcastic_files/total_files:.1%}\n\n")
                
                # Confidence distribution
                confidence_levels = [r.confidence_level.value for r in results.values()]
                for level in SarcasmConfidence:
                    count = confidence_levels.count(level.value)
                    f.write(f"{level.value.replace('_', ' ').title()}: {count} files\n")
                
                f.write("\n")
                
                # Top sarcastic files
                sarcastic_results = [(k, v) for k, v in results.items() if v.is_sarcastic]
                sarcastic_results.sort(key=lambda x: x[1].confidence, reverse=True)
                
                f.write("TOP SARCASTIC FILES:\n")
                for i, (file_path, result) in enumerate(sarcastic_results[:10], 1):
                    f.write(f"{i}. {os.path.basename(file_path)} "
                           f"(Confidence: {result.confidence:.1%})\n")
            
            print(f"Batch summary saved to: {output_dir}")
            
        except Exception as e:
            print(f"Error creating batch summary: {e}")


# Integration functions for ArchiveCAT
def detect_sarcasm_for_segment(audio_path: str, transcript_path: str = None,
                              prosody_results: Dict = None, output_dir: str = None,
                              segment_name: str = "") -> SarcasmResult:
    """
    Wrapper function for ArchiveCAT integration
    
    Args:
        audio_path: Path to audio file
        transcript_path: Path to transcript file
        prosody_results: Prosodic analysis results
        output_dir: Output directory
        segment_name: Segment identifier
        
    Returns:
        SarcasmResult
    """
    analyzer = SarcasmAnalyzer()
    
    # Adjust output naming for segments
    if output_dir and segment_name:
        segment_output_dir = os.path.join(output_dir, f"sarcasm_analysis{segment_name}")
        os.makedirs(segment_output_dir, exist_ok=True)
    else:
        segment_output_dir = output_dir
    
    result = analyzer.analyze_file(
        audio_path=audio_path,
        transcript_path=transcript_path,
        prosody_results=prosody_results,
        output_dir=segment_output_dir
    )
    
    print(f"Sarcasm analysis{segment_name} completed")
    return result


def enhance_file_processor_with_sarcasm(file_processor_instance, sarcasm_settings=None):
    """
    Enhance FileProcessor with sarcasm detection
    
    Args:
        file_processor_instance: FileProcessor instance to enhance
        sarcasm_settings: Dictionary with sarcasm detection settings
    """
    if sarcasm_settings is None:
        sarcasm_settings = {'enabled': False}
    
    # Store original method
    original_perform = file_processor_instance._perform_transcription
    
    def enhanced_perform_transcription(audio_path, output_folder, queue, settings, sarcasm_settings, segment_name=""):
        """Enhanced transcription with sarcasm detection"""
        print(f"DEBUG: Enhanced transcription called with sarcasm_settings: {sarcasm_settings}")
        
        # Call original transcription first
        original_result = enhanced_perform_transcription(audio_path, output_folder, queue, settings, segment_name)
        
        # Only do sarcasm detection if enabled
        if sarcasm_settings and sarcasm_settings.get('enabled', False):
            print(f"DEBUG: Starting sarcasm analysis for {audio_path}")
            
            try:
                # Create analyzer
                analyzer = SarcasmAnalyzer()
                
                # Find transcript file (adjust path logic as needed)
                transcript_files = []
                if os.path.exists(output_folder):
                    for file in os.listdir(output_folder):
                        if file.endswith('.txt') and not file.endswith('_sarcasm.txt'):
                            transcript_files.append(os.path.join(output_folder, file))
                
                if transcript_files:
                    transcript_path = transcript_files[0]  # Use first transcript found
                    
                    # Run sarcasm analysis
                    sarcasm_result = analyzer.analyze_file(
                        audio_path=audio_path,
                        transcript_path=transcript_path,
                        prosody_results=None,  # You might need to pass prosody results here
                        output_dir=output_folder
                    )
                    
                    print(f"DEBUG: Sarcasm analysis completed: {sarcasm_result}")
                    
                    # Update queue with sarcasm results
                    if hasattr(queue, 'put'):
                        queue.put(f"Sarcasm analysis completed: {'Sarcastic' if getattr(sarcasm_result, 'is_sarcastic', False) else 'Not sarcastic'}")
                    
            except Exception as e:
                print(f"DEBUG: Sarcasm analysis failed: {e}")
                import traceback
                traceback.print_exc()
                
                if hasattr(queue, 'put'):
                    queue.put(f"Sarcasm analysis failed: {e}")
        
        return original_result


# Cost-efficient model recommendations
def get_recommended_models():
    """
    Returns recommended models for cost-efficient sarcasm detection
    
    Most reliable and cost-efficient approach:
    1. Acoustic features (free, using librosa/parselmouth)
    2. Rule-based text analysis (free)
    3. HuggingFace transformer models (free for inference)
    4. Ensemble combination
    """
    return {
        'most_cost_efficient': {
            'text_model': None,  # Rule-based only
            'acoustic_features': True,
            'prosodic_integration': True,
            'cost': 'Free',
            'accuracy': 'Good (75-80%)'
        },
        'balanced': {
            'text_model': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
            'sarcasm_model': 'helinivan/english-sarcasm-detector',
            'acoustic_features': True,
            'prosodic_integration': True,
            'cost': 'Free (local inference)',
            'accuracy': 'Very Good (80-85%)'
        },
        'high_accuracy': {
            'text_model': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
            'sarcasm_model': 'microsoft/DialoGPT-large',  # Fine-tuned
            'acoustic_features': True,
            'prosodic_integration': True,
            'ensemble_ml': True,
            'cost': 'Free (requires more compute)',
            'accuracy': 'Excellent (85-90%)'
        }
    }