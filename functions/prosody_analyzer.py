# prosody_analyzer.py - COMPLETE VERSION
"""
Modul für automatisierte Prosodieanalyse von Audio-Dateien
Analysiert Tonhöhe, Lautstärke, Sprechgeschwindigkeit, Pausen etc.
"""

import os
import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

class ProsodyAnalyzer:
    """Klasse für umfassende Prosodieanalyse"""
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialisiert den Prosody Analyzer
        
        Args:
            sample_rate: Sampling-Rate für Audio-Verarbeitung
        """
        self.sample_rate = sample_rate
        self.results = {}
        
    def analyze_audio(self, audio_path: str, gender: str = "unknown") -> Dict:
        """
        Führt komplette Prosodieanalyse durch
        
        Args:
            audio_path: Pfad zur Audio-Datei
            gender: Geschlecht des Sprechers ("male", "female", "unknown")
            
        Returns:
            Dictionary mit allen Analyseergebnissen
        """
        try:
            print(f"Starte Prosodieanalyse für: {audio_path}")
            
            # Lade Audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            print(f"Audio geladen: {len(audio)} samples, {sr} Hz")
            
            # Lade für Praat-Analyse
            sound = parselmouth.Sound(audio_path)
            
            # Führe alle Analysen durch
            results = {
                'file_info': self._get_file_info(audio_path, audio, sr),
                'pitch': {},
                'intensity': {},
                'speech_rate': {},
                'pauses': {},
                'voice_quality': {},
                'rhythm': {},
                'emotion_indicators': {},
                'overall_statistics': {}
            }
            
            # Einzelne Analysen mit Fehlerbehandlung
            try:
                print("Analysiere Tonhöhe...")
                results['pitch'] = self._analyze_pitch(sound, gender)
            except Exception as e:
                print(f"Fehler bei Tonhöhenanalyse: {e}")
                results['pitch'] = {'error': str(e)}
            
            try:
                print("Analysiere Intensität...")
                results['intensity'] = self._analyze_intensity(sound)
            except Exception as e:
                print(f"Fehler bei Intensitätsanalyse: {e}")
                results['intensity'] = {'error': str(e)}
            
            try:
                print("Analysiere Sprechgeschwindigkeit...")
                results['speech_rate'] = self._analyze_speech_rate(audio, sr, sound)
            except Exception as e:
                print(f"Fehler bei Sprechgeschwindigkeitsanalyse: {e}")
                results['speech_rate'] = {'error': str(e)}
            
            try:
                print("Analysiere Pausen...")
                results['pauses'] = self._analyze_pauses(audio, sr)
            except Exception as e:
                print(f"Fehler bei Pausenanalyse: {e}")
                results['pauses'] = {'error': str(e)}
            
            try:
                print("Analysiere Stimmqualität...")
                results['voice_quality'] = self._analyze_voice_quality(sound)
            except Exception as e:
                print(f"Fehler bei Stimmqualitätsanalyse: {e}")
                results['voice_quality'] = {'error': str(e)}
            
            try:
                print("Analysiere Rhythmus...")
                results['rhythm'] = self._analyze_rhythm(audio, sr)
            except Exception as e:
                print(f"Fehler bei Rhythmusanalyse: {e}")
                results['rhythm'] = {'error': str(e)}
            
            try:
                print("Analysiere emotionale Indikatoren...")
                results['emotion_indicators'] = self._analyze_emotion_indicators(sound)
            except Exception as e:
                print(f"Fehler bei Emotionsanalyse: {e}")
                results['emotion_indicators'] = {'error': str(e)}
            
            # Berechne Gesamt-Statistiken
            results['overall_statistics'] = self._calculate_overall_statistics(results)
            
            print("Prosodieanalyse abgeschlossen")
            
            return {
                'success': True,
                'results': results,
                'audio_path': audio_path
            }
            
        except Exception as e:
            print(f"Kritischer Fehler in Prosodieanalyse: {e}")
            return {
                'success': False,
                'error': str(e),
                'audio_path': audio_path
            }
    
    def _get_file_info(self, audio_path: str, audio: np.ndarray, sr: int) -> Dict:
        """Sammelt grundlegende Datei-Informationen"""
        duration = len(audio) / sr
        
        return {
            'filename': os.path.basename(audio_path),
            'duration_seconds': round(duration, 2),
            'sample_rate': sr,
            'total_samples': len(audio),
            'file_size_mb': round(os.path.getsize(audio_path) / (1024 * 1024), 2)
        }
    
    def _analyze_pitch(self, sound: parselmouth.Sound, gender: str) -> Dict:
        """Analysiert Tonhöhe (F0)"""
        try:
            # Pitch-Extraktion
            pitch = call(sound, "To Pitch", 0.0, 75, 600)
            
            # Extrahiere F0-Werte
            pitch_values = pitch.selected_array['frequency']
            pitch_values = pitch_values[pitch_values > 0]  # Entferne unvoiced Frames
            
            if len(pitch_values) == 0:
                return {'error': 'Keine stimmhaften Segmente gefunden'}
            
            # Geschlechtsspezifische Referenzwerte
            reference_ranges = {
                'male': {'low': 80, 'high': 180, 'mean': 120},
                'female': {'low': 150, 'high': 300, 'mean': 220},
                'unknown': {'low': 80, 'high': 300, 'mean': 170}
            }
            
            ref = reference_ranges.get(gender, reference_ranges['unknown'])
            
            # Statistiken
            mean_f0 = float(np.mean(pitch_values))
            
            # Sichere Konvertierung zu Python-Typen für JSON-Serialisierung
            result = {
                'mean_f0': round(float(mean_f0), 2),
                'median_f0': round(float(np.median(pitch_values)), 2),
                'std_f0': round(float(np.std(pitch_values)), 2),
                'min_f0': round(float(np.min(pitch_values)), 2),
                'max_f0': round(float(np.max(pitch_values)), 2),
                'range_f0': round(float(np.max(pitch_values) - np.min(pitch_values)), 2),
                'percentile_10': round(float(np.percentile(pitch_values, 10)), 2),
                'percentile_90': round(float(np.percentile(pitch_values, 90)), 2),
                'cv_f0': round(float(np.std(pitch_values) / np.mean(pitch_values)), 3),
                'voiced_frames_ratio': round(float(len(pitch_values)) / float(len(pitch.selected_array['frequency'])), 3),
                'gender_typical': bool(mean_f0 >= ref['low'] and mean_f0 <= ref['high']),
                'pitch_variability': self._classify_pitch_variability(float(np.std(pitch_values)))
            }
            
            return result
            
        except Exception as e:
            print(f"Fehler in _analyze_pitch: {e}")
            return {'error': f'Pitch-Analyse fehlgeschlagen: {str(e)}'}
    
    def _analyze_intensity(self, sound: parselmouth.Sound) -> Dict:
        """Analysiert Lautstärke/Intensität"""
        try:
            intensity = call(sound, "To Intensity", 100, 0)
            intensity_values = intensity.values[0]
            intensity_values = intensity_values[~np.isnan(intensity_values)]
            
            if len(intensity_values) == 0:
                return {'error': 'Keine Intensitätswerte gefunden'}
            
            return {
                'mean_intensity_db': round(float(np.mean(intensity_values)), 2),
                'median_intensity_db': round(float(np.median(intensity_values)), 2),
                'std_intensity_db': round(float(np.std(intensity_values)), 2),
                'min_intensity_db': round(float(np.min(intensity_values)), 2),
                'max_intensity_db': round(float(np.max(intensity_values)), 2),
                'range_intensity_db': round(float(np.max(intensity_values) - np.min(intensity_values)), 2),
                'dynamic_range_category': self._classify_dynamic_range(float(np.max(intensity_values) - np.min(intensity_values)))
            }
        except Exception as e:
            print(f"Fehler in _analyze_intensity: {e}")
            return {'error': f'Intensitäts-Analyse fehlgeschlagen: {str(e)}'}
    
    def _analyze_speech_rate(self, audio: np.ndarray, sr: int, sound: parselmouth.Sound) -> Dict:
        """Analysiert Sprechgeschwindigkeit"""
        try:
            # Silben-Detektion (vereinfacht über Intensitäts-Peaks)
            intensity = call(sound, "To Intensity", 100, 0)
            
            # Finde Peaks (potentielle Silben)
            from scipy.signal import find_peaks
            intensity_smooth = np.convolve(intensity.values[0], np.ones(5)/5, mode='same')
            
            # Entferne NaN-Werte
            intensity_smooth = intensity_smooth[~np.isnan(intensity_smooth)]
            
            if len(intensity_smooth) > 0:
                # Dynamische Höhe basierend auf Intensitätswerten
                height_threshold = np.percentile(intensity_smooth, 50)
                peaks, _ = find_peaks(intensity_smooth, height=height_threshold, distance=10)
                syllable_count = len(peaks)
            else:
                syllable_count = 0
            
            duration = float(len(audio)) / float(sr)
            
            # Schätzung der Wörter (durchschnittlich 1.5 Silben pro Wort im Deutschen)
            estimated_words = float(syllable_count) / 1.5
            
            return {
                'estimated_syllables': int(syllable_count),
                'estimated_words': round(estimated_words),
                'syllables_per_second': round(float(syllable_count) / duration, 2),
                'words_per_minute': round((estimated_words / duration) * 60, 1),
                'speech_rate_category': self._classify_speech_rate((estimated_words / duration) * 60),
                'articulation_rate': round(float(syllable_count) / (duration * 0.7), 2)  # Annahme: 70% Sprechzeit
            }
        except Exception as e:
            print(f"Fehler in _analyze_speech_rate: {e}")
            return {
                'estimated_syllables': 0,
                'estimated_words': 0,
                'syllables_per_second': 0,
                'words_per_minute': 0,
                'speech_rate_category': 'unbekannt',
                'articulation_rate': 0,
                'error': str(e)
            }
    
    def _analyze_pauses(self, audio: np.ndarray, sr: int) -> Dict:
        """Analysiert Pausen im Sprechen"""
        try:
            # Energie-basierte Pausenerkennung
            frame_length = int(0.025 * sr)  # 25ms Frames
            hop_length = int(0.010 * sr)    # 10ms Hop
            
            # Berechne RMS-Energie
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Threshold für Stille (20% des Medians)
            silence_threshold = np.median(rms) * 0.2
            
            # Finde stille Segmente
            silent_frames = rms < silence_threshold
            
            # Konvertiere zu Zeitstempeln
            frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
            
            # Finde Pausen (mindestens 200ms)
            pauses = []
            in_pause = False
            pause_start = 0
            
            for i, is_silent in enumerate(silent_frames):
                if is_silent and not in_pause:
                    in_pause = True
                    pause_start = frame_times[i]
                elif not is_silent and in_pause:
                    in_pause = False
                    pause_duration = frame_times[i] - pause_start
                    if pause_duration >= 0.2:  # Mindestens 200ms
                        pauses.append({
                            'start': float(pause_start),
                            'duration': float(pause_duration)
                        })
            
            # Pausenstatistiken
            pause_durations = [p['duration'] for p in pauses]
            total_pause_time = sum(pause_durations)
            total_duration = float(len(audio)) / float(sr)
            
            return {
                'pause_count': len(pauses),
                'total_pause_duration': round(float(total_pause_time), 2),
                'pause_ratio': round(float(total_pause_time) / float(total_duration), 3),
                'mean_pause_duration': round(float(np.mean(pause_durations)), 3) if pause_durations else 0,
                'pause_distribution': {
                    'short_pauses': len([p for p in pause_durations if p < 0.5]),
                    'medium_pauses': len([p for p in pause_durations if 0.5 <= p < 1.0]),
                    'long_pauses': len([p for p in pause_durations if p >= 1.0])
                }
            }
        except Exception as e:
            print(f"Fehler in _analyze_pauses: {e}")
            return {
                'pause_count': 0,
                'total_pause_duration': 0,
                'pause_ratio': 0,
                'mean_pause_duration': 0,
                'pause_distribution': {
                    'short_pauses': 0,
                    'medium_pauses': 0,
                    'long_pauses': 0
                },
                'error': str(e)
            }
    
    def _analyze_voice_quality(self, sound: parselmouth.Sound) -> Dict:
        """Analysiert Stimmqualität"""
        try:
            # Harmonics-to-Noise Ratio (HNR)
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr_values = harmonicity.values[0]
            hnr_values = hnr_values[~np.isnan(hnr_values)]
            
            # Jitter und Shimmer mit Fehlerbehandlung
            jitter_local = 0.0
            jitter_rap = 0.0
            shimmer_local = 0.0
            shimmer_apq = 0.0
            
            try:
                pitch = call(sound, "To Pitch", 0.0, 75, 600)
                point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
                
                # Prüfe ob genug Punkte vorhanden sind
                num_points = call(point_process, "Get number of points")
                
                if num_points > 2:
                    jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
                    jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
                    shimmer_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                    shimmer_apq = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            except Exception as e:
                print(f"Warnung: Jitter/Shimmer-Berechnung fehlgeschlagen: {e}")
            
            hnr_mean = float(np.mean(hnr_values)) if len(hnr_values) > 0 else 0.0
            
            return {
                'hnr_mean_db': round(hnr_mean, 2),
                'hnr_std_db': round(float(np.std(hnr_values)), 2) if len(hnr_values) > 0 else 0.0,
                'jitter_local_percent': round(float(jitter_local) * 100, 3),
                'jitter_rap_percent': round(float(jitter_rap) * 100, 3),
                'shimmer_local_percent': round(float(shimmer_local) * 100, 3),
                'shimmer_apq_percent': round(float(shimmer_apq) * 100, 3),
                'voice_quality_assessment': self._assess_voice_quality(float(jitter_local), float(shimmer_local), hnr_mean)
            }
        except Exception as e:
            print(f"Fehler in _analyze_voice_quality: {e}")
            return {
                'hnr_mean_db': 0.0,
                'hnr_std_db': 0.0,
                'jitter_local_percent': 0.0,
                'jitter_rap_percent': 0.0,
                'shimmer_local_percent': 0.0,
                'shimmer_apq_percent': 0.0,
                'voice_quality_assessment': 'unbekannt',
                'error': str(e)
            }
    
    def _analyze_rhythm(self, audio: np.ndarray, sr: int) -> Dict:
        """Analysiert Sprechrhythmus"""
        try:
            # Tempo-Erkennung über Onset-Detection
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            
            # Beat-Intervalle
            beat_times = librosa.frames_to_time(beats, sr=sr)
            beat_intervals = np.diff(beat_times)
            
            # Rhythmus-Regularität
            if len(beat_intervals) > 1:
                rhythm_regularity = 1 - (float(np.std(beat_intervals)) / float(np.mean(beat_intervals)))
            else:
                rhythm_regularity = 0
            
            return {
                'estimated_tempo_bpm': round(float(tempo), 1),
                'beat_count': int(len(beats)),
                'rhythm_regularity': round(float(rhythm_regularity), 3),
                'mean_beat_interval': round(float(np.mean(beat_intervals)), 3) if len(beat_intervals) > 0 else 0.0,
                'rhythm_variability': round(float(np.std(beat_intervals)), 3) if len(beat_intervals) > 1 else 0.0
            }
        except Exception as e:
            print(f"Fehler in _analyze_rhythm: {e}")
            return {
                'estimated_tempo_bpm': 0.0,
                'beat_count': 0,
                'rhythm_regularity': 0.0,
                'mean_beat_interval': 0.0,
                'rhythm_variability': 0.0,
                'error': str(e)
            }
    
    def _analyze_emotion_indicators(self, sound: parselmouth.Sound) -> Dict:
        """Analysiert emotionale Indikatoren"""
        try:
            # Diese sind vereinfachte Indikatoren basierend auf Prosodie-Features
            pitch = call(sound, "To Pitch", 0.0, 75, 600)
            intensity = call(sound, "To Intensity", 100, 0)
            
            pitch_values = pitch.selected_array['frequency']
            pitch_values = pitch_values[pitch_values > 0]
            
            intensity_values = intensity.values[0]
            intensity_values = intensity_values[~np.isnan(intensity_values)]
            
            # Emotionale Tendenzen basierend auf Prosodie
            indicators = {}
            
            if len(pitch_values) > 0 and len(intensity_values) > 0:
                # Hohe Variabilität + hohe Intensität -> Aufregung
                pitch_var = float(np.std(pitch_values)) / float(np.mean(pitch_values))
                intensity_mean = float(np.mean(intensity_values))
                
                indicators['arousal_level'] = self._calculate_arousal(pitch_var, intensity_mean)
                indicators['valence_tendency'] = self._calculate_valence(float(np.mean(pitch_values)), pitch_var)
                indicators['stress_indicators'] = {
                    'high_pitch_variation': bool(pitch_var > 0.2),
                    'elevated_intensity': bool(intensity_mean > 70),
                    'rapid_changes': bool(self._detect_rapid_changes(pitch_values))
                }
            else:
                indicators = {
                    'arousal_level': 'unbekannt',
                    'valence_tendency': 'unbekannt',
                    'stress_indicators': {
                        'high_pitch_variation': False,
                        'elevated_intensity': False,
                        'rapid_changes': False
                    }
                }
            
            return indicators
        except Exception as e:
            print(f"Fehler in _analyze_emotion_indicators: {e}")
            return {
                'arousal_level': 'unbekannt',
                'valence_tendency': 'unbekannt',
                'stress_indicators': {
                    'high_pitch_variation': False,
                    'elevated_intensity': False,
                    'rapid_changes': False
                },
                'error': str(e)
            }
    
    def _calculate_overall_statistics(self, results: Dict) -> Dict:
        """Berechnet zusammenfassende Statistiken"""
        stats = {
            'prosody_summary': [],
            'notable_features': [],
            'recommendations': []
        }
        
        # Zusammenfassung
        if 'pitch' in results and 'mean_f0' in results['pitch']:
            stats['prosody_summary'].append(f"Mittlere Tonhöhe: {results['pitch']['mean_f0']} Hz")
        
        if 'speech_rate' in results and 'words_per_minute' in results['speech_rate']:
            stats['prosody_summary'].append(f"Sprechgeschwindigkeit: {results['speech_rate']['words_per_minute']} Wörter/Min")
        
        # Notable Features
        if 'voice_quality' in results:
            vq = results['voice_quality']
            if vq.get('jitter_local_percent', 0) > 1.0:
                stats['notable_features'].append("Erhöhter Jitter - mögliche Stimminstabilität")
            if vq.get('hnr_mean_db', 20) < 15:
                stats['notable_features'].append("Niedriger HNR - raue oder heisere Stimme")
        
        return stats
    
    # Hilfsmethoden für Klassifikationen
    def _classify_pitch_variability(self, std_f0: float) -> str:
        if std_f0 < 20:
            return "monoton"
        elif std_f0 < 40:
            return "gering"
        elif std_f0 < 60:
            return "normal"
        elif std_f0 < 80:
            return "hoch"
        else:
            return "sehr hoch"
    
    def _classify_dynamic_range(self, range_db: float) -> str:
        if range_db < 20:
            return "sehr gering"
        elif range_db < 30:
            return "gering"
        elif range_db < 40:
            return "normal"
        elif range_db < 50:
            return "hoch"
        else:
            return "sehr hoch"
    
    def _classify_speech_rate(self, wpm: float) -> str:
        if wpm < 100:
            return "sehr langsam"
        elif wpm < 130:
            return "langsam"
        elif wpm < 170:
            return "normal"
        elif wpm < 210:
            return "schnell"
        else:
            return "sehr schnell"
    
    def _assess_voice_quality(self, jitter: float, shimmer: float, hnr: float) -> str:
        score = 0
        
        # Normale Werte: Jitter < 1%, Shimmer < 3%, HNR > 20dB
        if jitter < 0.01:
            score += 1
        if shimmer < 0.03:
            score += 1
        if hnr > 20:
            score += 1
        
        if score == 3:
            return "ausgezeichnet"
        elif score == 2:
            return "gut"
        elif score == 1:
            return "durchschnittlich"
        else:
            return "auffällig"
    
    def _calculate_arousal(self, pitch_var: float, intensity: float) -> str:
        arousal_score = (pitch_var * 100) + (intensity / 10)
        
        if arousal_score < 30:
            return "sehr niedrig"
        elif arousal_score < 50:
            return "niedrig"
        elif arousal_score < 70:
            return "mittel"
        elif arousal_score < 90:
            return "hoch"
        else:
            return "sehr hoch"
    
    def _calculate_valence(self, mean_pitch: float, pitch_var: float) -> str:
        # Vereinfachte Valenz-Schätzung
        if mean_pitch > 200 and pitch_var > 0.15:
            return "positiv"
        elif mean_pitch < 150 and pitch_var < 0.1:
            return "negativ"
        else:
            return "neutral"
    
    def _detect_rapid_changes(self, values: np.ndarray) -> bool:
        if len(values) < 10:
            return False
        
        # Berechne erste Ableitung
        diff = np.diff(values)
        rapid_changes = np.sum(np.abs(diff) > np.std(values))
        
        return bool(rapid_changes > len(values) * 0.3)
    
    def save_results(self, results: Dict, output_path: str):
        """Speichert Analyseergebnisse"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # JSON mit sicherer Serialisierung
            json_path = output_path.with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                # Konvertiere numpy-Typen zu Python-Typen
                json_safe_results = self._make_json_serializable(results)
                json.dump(json_safe_results, f, ensure_ascii=False, indent=2)
            
            # Readable Text Report
            txt_path = output_path.with_suffix('.txt')
            self._save_text_report(results, txt_path)
            
            # CSV für tabellarische Daten
            csv_path = output_path.with_suffix('.csv')
            self._save_csv_summary(results, csv_path)
            
            print(f"Ergebnisse gespeichert in: {output_path}")
            
        except Exception as e:
            print(f"Fehler beim Speichern der Ergebnisse: {e}")
    
    def _make_json_serializable(self, obj):
        """Konvertiert numpy-Typen zu Python-Typen für JSON-Serialisierung"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_json_serializable(item) for item in obj)
        else:
            return obj
    
    def _save_text_report(self, results: Dict, output_path: Path):
        """Erstellt lesbaren Text-Report"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("PROSODIEANALYSE-BERICHT\n")
                f.write("=" * 50 + "\n\n")
                
                if results.get('success'):
                    res = results['results']
                    
                    # Datei-Info
                    f.write("DATEI-INFORMATIONEN:\n")
                    for key, value in res['file_info'].items():
                        f.write(f"  {key}: {value}\n")
                    
                    # Tonhöhe
                    f.write("\n\nTONHÖHEN-ANALYSE (F0):\n")
                    if 'error' not in res['pitch']:
                        f.write(f"  Durchschnitt: {res['pitch']['mean_f0']} Hz\n")
                        f.write(f"  Bereich: {res['pitch']['min_f0']} - {res['pitch']['max_f0']} Hz\n")
                        f.write(f"  Variabilität: {res['pitch']['pitch_variability']}\n")
                    
                    # Sprechgeschwindigkeit
                    f.write("\n\nSPRECHGESCHWINDIGKEIT:\n")
                    if 'error' not in res['speech_rate']:
                        f.write(f"  Wörter pro Minute: {res['speech_rate']['words_per_minute']}\n")
                        f.write(f"  Kategorie: {res['speech_rate']['speech_rate_category']}\n")
                    
                    # Pausen
                    f.write("\n\nPAUSENANALYSE:\n")
                    if 'error' not in res['pauses']:
                        f.write(f"  Anzahl Pausen: {res['pauses']['pause_count']}\n")
                        f.write(f"  Pausenanteil: {res['pauses']['pause_ratio']*100:.1f}%\n")
                    
                    # Stimmqualität
                    f.write("\n\nSTIMMQUALITÄT:\n")
                    if 'error' not in res['voice_quality']:
                        f.write(f"  Bewertung: {res['voice_quality']['voice_quality_assessment']}\n")
                        f.write(f"  HNR: {res['voice_quality']['hnr_mean_db']} dB\n")
                        f.write(f"  Jitter: {res['voice_quality']['jitter_local_percent']}%\n")
                        f.write(f"  Shimmer: {res['voice_quality']['shimmer_local_percent']}%\n")
                    
                    # Emotionale Indikatoren
                    if res['emotion_indicators'] and 'error' not in res['emotion_indicators']:
                        f.write("\n\nEMOTIONALE INDIKATOREN:\n")
                        f.write(f"  Erregungsniveau: {res['emotion_indicators'].get('arousal_level', 'N/A')}\n")
                        f.write(f"  Valenz-Tendenz: {res['emotion_indicators'].get('valence_tendency', 'N/A')}\n")
                else:
                    f.write(f"FEHLER: {results.get('error', 'Unbekannter Fehler')}\n")
        except Exception as e:
            print(f"Fehler beim Speichern des Text-Reports: {e}")
    
    def _save_csv_summary(self, results: Dict, output_path: Path):
        """Speichert zusammenfassende CSV"""
        try:
            if not results.get('success'):
                return
            
            res = results['results']
            data = []
            
            # Sammle alle numerischen Werte
            row = {'filename': res['file_info']['filename']}
            
            # Füge alle numerischen Werte hinzu
            for category in ['pitch', 'intensity', 'speech_rate', 'pauses', 'voice_quality', 'rhythm']:
                if category in res and isinstance(res[category], dict) and 'error' not in res[category]:
                    for key, value in res[category].items():
                        if isinstance(value, (int, float)):
                            row[f"{category}_{key}"] = value
            
            data.append(row)
            
            # Speichere als CSV
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False, encoding='utf-8')
        except Exception as e:
            print(f"Fehler beim Speichern der CSV: {e}")
    
    def create_visualization(self, audio_path: str, results: Dict, output_path: str):
        """Erstellt erweiterte Visualisierungen der Prosodieanalyse mit Spektrogramm"""
        try:
            print(f"Erstelle Visualisierung für: {audio_path}")
            
            # Lade Audio für Visualisierung
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            sound = parselmouth.Sound(audio_path)
            
            # Setup matplotlib für bessere Qualität
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Erstelle Figure mit 6 Subplots für umfassende Analyse
            fig = plt.figure(figsize=(16, 20))
            
            # 1. Waveform + Intensität
            ax1 = plt.subplot(6, 1, 1)
            time = np.linspace(0, len(audio)/sr, len(audio))
            ax1.plot(time, audio, alpha=0.6, linewidth=0.5, color='steelblue', label='Waveform')
            ax1.set_ylabel('Amplitude', fontsize=10)
            ax1.set_title('Wellenform und Intensität', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Intensität überlagern
            try:
                intensity = call(sound, "To Intensity", 100, 0)
                intensity_time = np.linspace(0, sound.xmax, len(intensity.values[0]))
                intensity_values = intensity.values[0]
                
                # Entferne NaN-Werte
                valid_indices = ~np.isnan(intensity_values)
                intensity_time_clean = intensity_time[valid_indices]
                intensity_values_clean = intensity_values[valid_indices]
                
                ax1_twin = ax1.twinx()
                ax1_twin.plot(intensity_time_clean, intensity_values_clean, 'red', 
                             linewidth=2, alpha=0.8, label='Intensität')
                ax1_twin.set_ylabel('Intensität (dB)', color='red', fontsize=10)
                ax1_twin.tick_params(axis='y', labelcolor='red')
                
                # Legende
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax1_twin.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            except Exception as e:
                print(f"Warnung: Intensität konnte nicht dargestellt werden: {e}")
            
            # 2. Tonhöhenverlauf
            ax2 = plt.subplot(6, 1, 2)
            try:
                pitch = call(sound, "To Pitch", 0.0, 75, 600)
                pitch_values = pitch.selected_array['frequency']
                pitch_time = pitch.xs()
                
                # Entferne unvoiced frames für bessere Darstellung
                voiced_indices = pitch_values > 0
                pitch_time_voiced = pitch_time[voiced_indices]
                pitch_values_voiced = pitch_values[voiced_indices]
                
                if len(pitch_values_voiced) > 0:
                    ax2.plot(pitch_time_voiced, pitch_values_voiced, 'navy', linewidth=2, marker='o', 
                            markersize=1, alpha=0.8)
                    ax2.set_ylabel('F0 (Hz)', fontsize=10)
                    ax2.set_title('Tonhöhenverlauf (F0)', fontsize=12, fontweight='bold')
                    ax2.set_ylim(0, max(600, np.max(pitch_values_voiced) * 1.1))
                    ax2.grid(True, alpha=0.3)
                    
                    # Markiere Mittelwert
                    mean_f0 = np.mean(pitch_values_voiced)
                    ax2.axhline(y=mean_f0, color='red', linestyle='--', alpha=0.7, 
                               label=f'Mittelwert: {mean_f0:.1f} Hz')
                    ax2.legend()
                else:
                    ax2.text(0.5, 0.5, 'Keine stimmhaften Segmente gefunden', 
                            ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                    ax2.set_title('Tonhöhenverlauf (F0) - Keine Daten', fontsize=12, fontweight='bold')
            except Exception as e:
                print(f"Warnung: Tonhöhe konnte nicht dargestellt werden: {e}")
                ax2.text(0.5, 0.5, f'Fehler bei Tonhöhenanalyse: {str(e)[:50]}...', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=10)
            
            # 3. Spektrogramm (Mel-Scale)
            ax3 = plt.subplot(6, 1, 3)
            try:
                # Berechne Mel-Spektrogramm für bessere Darstellung
                S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
                S_dB = librosa.power_to_db(S, ref=np.max)
                
                img = librosa.display.specshow(S_dB, y_axis='mel', x_axis='time', ax=ax3, sr=sr, fmax=8000)
                ax3.set_title('Mel-Spektrogramm', fontsize=12, fontweight='bold')
                cbar = fig.colorbar(img, ax=ax3, format='%+2.0f dB')
                cbar.set_label('Amplitude (dB)', fontsize=10)
            except Exception as e:
                print(f"Warnung: Spektrogramm konnte nicht erstellt werden: {e}")
                ax3.text(0.5, 0.5, f'Spektrogramm-Fehler: {str(e)[:50]}...', 
                        ha='center', va='center', transform=ax3.transAxes, fontsize=10)
            
            # 4. Prosodische Variabilität über Zeit
            ax4 = plt.subplot(6, 1, 4)
            try:
                # Berechne gleitende Fenster-Features
                window_size = int(0.5 * sr)  # 500ms Fenster
                hop_size = int(0.1 * sr)     # 100ms Hop
                
                pitch_std = []
                intensity_std = []
                times = []
                
                for i in range(0, len(audio) - window_size, hop_size):
                    window_time = (i + window_size/2) / sr
                    times.append(window_time)
                    
                    # Pitch-Variabilität für dieses Fenster
                    try:
                        window_pitch = pitch_values[
                            (pitch_time >= i/sr) & (pitch_time < (i+window_size)/sr)
                        ]
                        window_pitch_voiced = window_pitch[window_pitch > 0]
                        
                        if len(window_pitch_voiced) > 1:
                            pitch_std.append(np.std(window_pitch_voiced))
                        else:
                            pitch_std.append(0)
                    except:
                        pitch_std.append(0)
                    
                    # Intensitäts-Variabilität
                    try:
                        window_audio = audio[i:i+window_size]
                        if len(window_audio) > 0:
                            rms = librosa.feature.rms(y=window_audio)[0]
                            intensity_std.append(np.std(rms))
                        else:
                            intensity_std.append(0)
                    except:
                        intensity_std.append(0)
                
                if len(times) > 0:
                    ax4.plot(times, pitch_std, 'green', linewidth=2, label='Tonhöhen-Variabilität', alpha=0.8)
                    ax4_twin = ax4.twinx()
                    ax4_twin.plot(times, intensity_std, 'orange', linewidth=2, label='Intensitäts-Variabilität', alpha=0.8)
                    
                    ax4.set_ylabel('Pitch Std (Hz)', color='green', fontsize=10)
                    ax4_twin.set_ylabel('Intensity Std', color='orange', fontsize=10)
                    ax4.set_title('Prosodische Variabilität über Zeit', fontsize=12, fontweight='bold')
                    ax4.grid(True, alpha=0.3)
                    
                    # Legende
                    lines1, labels1 = ax4.get_legend_handles_labels()
                    lines2, labels2 = ax4_twin.get_legend_handles_labels()
                    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
                else:
                    ax4.text(0.5, 0.5, 'Keine Variabilitätsdaten verfügbar', 
                            ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            except Exception as e:
                print(f"Warnung: Variabilitätsanalyse fehlgeschlagen: {e}")
                ax4.text(0.5, 0.5, f'Variabilitäts-Fehler: {str(e)[:50]}...', 
                        ha='center', va='center', transform=ax4.transAxes, fontsize=10)
            
            # 5. Pausenanalyse Visualisierung
            ax5 = plt.subplot(6, 1, 5)
            try:
                # Zeige Sprachaktivität vs. Pausen
                frame_length = int(0.025 * sr)  # 25ms
                hop_length = int(0.010 * sr)    # 10ms
                
                rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
                rms_time = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
                
                # Threshold für Sprachaktivität
                threshold = np.median(rms) * 0.2
                
                ax5.plot(rms_time, rms, 'blue', linewidth=1, alpha=0.7, label='RMS Energy')
                ax5.axhline(y=threshold, color='red', linestyle='--', alpha=0.8, label=f'Schwellwert ({threshold:.4f})')
                
                # Markiere Pausen
                speech_activity = rms > threshold
                ax5.fill_between(rms_time, 0, np.max(rms), where=~speech_activity, 
                               alpha=0.3, color='red', label='Pausen')
                
                ax5.set_ylabel('RMS Energy', fontsize=10)
                ax5.set_title('Sprachaktivität und Pausen', fontsize=12, fontweight='bold')
                ax5.grid(True, alpha=0.3)
                ax5.legend()
            except Exception as e:
                print(f"Warnung: Pausenvisualisierung fehlgeschlagen: {e}")
                ax5.text(0.5, 0.5, f'Pausen-Fehler: {str(e)[:50]}...', 
                        ha='center', va='center', transform=ax5.transAxes, fontsize=10)
            
            # 6. Zusammenfassung der Analyseergebnisse
            ax6 = plt.subplot(6, 1, 6)
            ax6.axis('off')
            
            # Erstelle Zusammenfassungstext
            if results.get('success') and 'results' in results:
                res = results['results']
                summary_text = "ANALYSEERGEBNISSE:\n\n"
                
                # Datei-Info
                file_info = res.get('file_info', {})
                summary_text += f"Datei: {file_info.get('filename', 'Unbekannt')}\n"
                summary_text += f"Dauer: {file_info.get('duration_seconds', 0):.1f} Sekunden\n\n"
                
                # Prosodische Features
                pitch_data = res.get('pitch', {})
                if 'error' not in pitch_data and 'mean_f0' in pitch_data:
                    summary_text += f"Mittlere Tonhöhe: {pitch_data['mean_f0']} Hz\n"
                    summary_text += f"Tonhöhenbereich: {pitch_data.get('min_f0', 0)} - {pitch_data.get('max_f0', 0)} Hz\n"
                    summary_text += f"Variabilität: {pitch_data.get('pitch_variability', 'unbekannt')}\n"
                
                speech_data = res.get('speech_rate', {})
                if 'error' not in speech_data and 'words_per_minute' in speech_data:
                    summary_text += f"\nSprechgeschwindigkeit: {speech_data['words_per_minute']} Wörter/Min\n"
                    summary_text += f"Kategorie: {speech_data.get('speech_rate_category', 'unbekannt')}\n"
                
                pause_data = res.get('pauses', {})
                if 'error' not in pause_data and 'pause_count' in pause_data:
                    summary_text += f"\nPausen: {pause_data['pause_count']} Stück\n"
                    summary_text += f"Pausenanteil: {pause_data.get('pause_ratio', 0)*100:.1f}%\n"
                
                voice_data = res.get('voice_quality', {})
                if 'error' not in voice_data and 'voice_quality_assessment' in voice_data:
                    summary_text += f"\nStimmqualität: {voice_data['voice_quality_assessment']}\n"
                
                emotion_data = res.get('emotion_indicators', {})
                if 'error' not in emotion_data and emotion_data:
                    summary_text += f"\nErregung: {emotion_data.get('arousal_level', 'unbekannt')}\n"
                    summary_text += f"Valenz: {emotion_data.get('valence_tendency', 'unbekannt')}\n"
            else:
                summary_text = f"FEHLER BEI DER ANALYSE:\n{results.get('error', 'Unbekannter Fehler')}"
            
            # Zeige Zusammenfassung
            ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=11, 
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
            
            # Layout anpassen
            plt.tight_layout(pad=2.0)
            
            # Speichere mit hoher Qualität
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"Visualisierung gespeichert: {output_path}")
            return True
            
        except Exception as e:
            print(f"Fehler bei Visualisierung: {e}")
            import traceback
            traceback.print_exc()
            return False


# Integrations-Funktionen für ArchiveCAT
def analyze_prosody_for_segment(audio_path: str, output_dir: str, 
                               segment_name: str = "", gender: str = "unknown") -> Dict:
    """
    Wrapper-Funktion für die Integration in ArchiveCAT
    
    Args:
        audio_path: Pfad zur Audio-Datei
        output_dir: Ausgabe-Verzeichnis
        segment_name: Name des Segments
        gender: Geschlecht des Sprechers
        
    Returns:
        Analyseergebnisse
    """
    analyzer = ProsodyAnalyzer()
    
    # Führe Analyse durch
    results = analyzer.analyze_audio(audio_path, gender)
    
    if results['success']:
        # Speichere Ergebnisse
        output_base = os.path.join(output_dir, f"prosody_analysis{segment_name}")
        analyzer.save_results(results, output_base)
        
        # Erstelle Visualisierung
        viz_path = output_base + "_visualization.png"
        analyzer.create_visualization(audio_path, results, viz_path)
        
        print(f"Prosodieanalyse abgeschlossen: {output_base}")
    else:
        print(f"Prosodieanalyse fehlgeschlagen: {results.get('error')}")
    
    return results


def batch_analyze_prosody(audio_files: List[str], output_dir: str, 
                         gender: str = "unknown") -> Dict[str, Dict]:
    """
    Analysiert mehrere Audio-Dateien
    
    Args:
        audio_files: Liste von Audio-Pfaden
        output_dir: Ausgabe-Verzeichnis
        gender: Geschlecht der Sprecher
        
    Returns:
        Dictionary mit Ergebnissen für jede Datei
    """
    analyzer = ProsodyAnalyzer()
    results = {}
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\nAnalysiere {i}/{len(audio_files)}: {os.path.basename(audio_file)}")
        
        result = analyzer.analyze_audio(audio_file, gender)
        results[audio_file] = result
        
        if result['success']:
            # Speichere individuelle Ergebnisse
            base_name = Path(audio_file).stem
            output_base = os.path.join(output_dir, base_name)
            analyzer.save_results(result, output_base)
            
            # Visualisierung
            viz_path = output_base + "_visualization.png"
            analyzer.create_visualization(audio_file, result, viz_path)
    
    # Erstelle Gesamt-Zusammenfassung
    create_prosody_summary(results, output_dir)
    
    return results


def create_prosody_summary(results: Dict[str, Dict], output_dir: str):
    """Erstellt Zusammenfassung aller Prosodieanalysen"""
    summary_data = []
    
    for audio_file, result in results.items():
        if result['success']:
            res = result['results']
            row = {
                'file': os.path.basename(audio_file),
                'duration': res['file_info']['duration_seconds'],
                'mean_pitch': res['pitch'].get('mean_f0', 'N/A') if 'error' not in res['pitch'] else 'Error',
                'pitch_range': res['pitch'].get('range_f0', 'N/A') if 'error' not in res['pitch'] else 'Error',
                'speech_rate_wpm': res['speech_rate'].get('words_per_minute', 'N/A') if 'error' not in res['speech_rate'] else 'Error',
                'pause_ratio': res['pauses'].get('pause_ratio', 'N/A') if 'error' not in res['pauses'] else 'Error',
                'voice_quality': res['voice_quality'].get('voice_quality_assessment', 'N/A') if 'error' not in res['voice_quality'] else 'Error',
                'arousal': res['emotion_indicators'].get('arousal_level', 'N/A') if 'error' not in res['emotion_indicators'] else 'Error'
            }
            summary_data.append(row)
    
    if summary_data:
        # Speichere als CSV
        df = pd.DataFrame(summary_data)
        csv_path = os.path.join(output_dir, "prosody_summary.csv")
        df.to_csv(csv_path, index=False)
        
        # Erstelle statistischen Bericht
        report_path = os.path.join(output_dir, "prosody_summary_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("ZUSAMMENFASSUNG PROSODIEANALYSE\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysierte Dateien: {len(summary_data)}\n\n")
            
            # Berechne Durchschnittswerte
            numeric_cols = ['duration', 'mean_pitch', 'pitch_range', 'speech_rate_wpm', 'pause_ratio']
            for col in numeric_cols:
                values = [row[col] for row in summary_data if isinstance(row[col], (int, float))]
                if values:
                    f.write(f"{col}:\n")
                    f.write(f"  Durchschnitt: {np.mean(values):.2f}\n")
                    f.write(f"  Min: {np.min(values):.2f}\n")
                    f.write(f"  Max: {np.max(values):.2f}\n")
                    f.write(f"  Std: {np.std(values):.2f}\n\n")