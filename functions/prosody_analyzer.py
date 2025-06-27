# prosody_analyzer.py
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
            # Lade Audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Lade für Praat-Analyse
            sound = parselmouth.Sound(audio_path)
            
            # Führe alle Analysen durch
            results = {
                'file_info': self._get_file_info(audio_path, audio, sr),
                'pitch': self._analyze_pitch(sound, gender),
                'intensity': self._analyze_intensity(sound),
                'speech_rate': self._analyze_speech_rate(audio, sr, sound),
                'pauses': self._analyze_pauses(audio, sr),
                'voice_quality': self._analyze_voice_quality(sound),
                'rhythm': self._analyze_rhythm(audio, sr),
                'emotion_indicators': self._analyze_emotion_indicators(sound),
                'overall_statistics': {}
            }
            
            # Berechne Gesamt-Statistiken
            results['overall_statistics'] = self._calculate_overall_statistics(results)
            
            return {
                'success': True,
                'results': results,
                'audio_path': audio_path
            }
            
        except Exception as e:
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
        mean_f0 = np.mean(pitch_values)
        
        return {
            'mean_f0': round(mean_f0, 2),
            'median_f0': round(np.median(pitch_values), 2),
            'std_f0': round(np.std(pitch_values), 2),
            'min_f0': round(np.min(pitch_values), 2),
            'max_f0': round(np.max(pitch_values), 2),
            'range_f0': round(np.max(pitch_values) - np.min(pitch_values), 2),
            'percentile_10': round(np.percentile(pitch_values, 10), 2),
            'percentile_90': round(np.percentile(pitch_values, 90), 2),
            'cv_f0': round(np.std(pitch_values) / np.mean(pitch_values), 3),  # Variationskoeffizient
            'voiced_frames_ratio': round(len(pitch_values) / len(pitch.selected_array['frequency']), 3),
            'gender_typical': mean_f0 >= ref['low'] and mean_f0 <= ref['high'],
            'pitch_variability': self._classify_pitch_variability(np.std(pitch_values))
        }
    
    def _analyze_intensity(self, sound: parselmouth.Sound) -> Dict:
        """Analysiert Lautstärke/Intensität"""
        intensity = call(sound, "To Intensity", 100, 0)
        intensity_values = intensity.values[0]
        intensity_values = intensity_values[~np.isnan(intensity_values)]
        
        if len(intensity_values) == 0:
            return {'error': 'Keine Intensitätswerte gefunden'}
        
        return {
            'mean_intensity_db': round(np.mean(intensity_values), 2),
            'median_intensity_db': round(np.median(intensity_values), 2),
            'std_intensity_db': round(np.std(intensity_values), 2),
            'min_intensity_db': round(np.min(intensity_values), 2),
            'max_intensity_db': round(np.max(intensity_values), 2),
            'range_intensity_db': round(np.max(intensity_values) - np.min(intensity_values), 2),
            'dynamic_range_category': self._classify_dynamic_range(np.max(intensity_values) - np.min(intensity_values))
        }
    
    def _analyze_speech_rate(self, audio: np.ndarray, sr: int, sound: parselmouth.Sound) -> Dict:
        """Analysiert Sprechgeschwindigkeit"""
        # Silben-Detektion (vereinfacht über Intensitäts-Peaks)
        intensity = call(sound, "To Intensity", 100, 0)
        
        # Finde Peaks (potentielle Silben)
        from scipy.signal import find_peaks
        intensity_smooth = np.convolve(intensity.values[0], np.ones(5)/5, mode='same')
        peaks, _ = find_peaks(intensity_smooth, height=50, distance=10)
        
        duration = len(audio) / sr
        syllable_count = len(peaks)
        
        # Schätzung der Wörter (durchschnittlich 1.5 Silben pro Wort im Deutschen)
        estimated_words = syllable_count / 1.5
        
        return {
            'estimated_syllables': syllable_count,
            'estimated_words': round(estimated_words),
            'syllables_per_second': round(syllable_count / duration, 2),
            'words_per_minute': round((estimated_words / duration) * 60, 1),
            'speech_rate_category': self._classify_speech_rate((estimated_words / duration) * 60),
            'articulation_rate': round(syllable_count / (duration * 0.7), 2)  # Annahme: 70% Sprechzeit
        }
    
    def _analyze_pauses(self, audio: np.ndarray, sr: int) -> Dict:
        """Analysiert Pausen im Sprechen"""
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
                        'start': pause_start,
                        'duration': pause_duration
                    })
        
        # Pausenstatistiken
        pause_durations = [p['duration'] for p in pauses]
        total_pause_time = sum(pause_durations)
        total_duration = len(audio) / sr
        
        return {
            'pause_count': len(pauses),
            'total_pause_duration': round(total_pause_time, 2),
            'pause_ratio': round(total_pause_time / total_duration, 3),
            'mean_pause_duration': round(np.mean(pause_durations), 3) if pause_durations else 0,
            'pause_distribution': {
                'short_pauses': len([p for p in pause_durations if p < 0.5]),
                'medium_pauses': len([p for p in pause_durations if 0.5 <= p < 1.0]),
                'long_pauses': len([p for p in pause_durations if p >= 1.0])
            }
        }
    
    def _analyze_voice_quality(self, sound: parselmouth.Sound) -> Dict:
        """Analysiert Stimmqualität"""
        # Harmonics-to-Noise Ratio (HNR)
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr_values = harmonicity.values[0]
        hnr_values = hnr_values[~np.isnan(hnr_values)]
        
        # Jitter (Periodenlängen-Variation)
        pitch = call(sound, "To Pitch", 0.0, 75, 600)
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
        
        jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        
        # Shimmer (Amplituden-Variation)
        shimmer_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        
        return {
            'hnr_mean_db': round(np.mean(hnr_values), 2) if len(hnr_values) > 0 else None,
            'hnr_std_db': round(np.std(hnr_values), 2) if len(hnr_values) > 0 else None,
            'jitter_local_percent': round(jitter_local * 100, 3),
            'jitter_rap_percent': round(jitter_rap * 100, 3),
            'shimmer_local_percent': round(shimmer_local * 100, 3),
            'shimmer_apq_percent': round(shimmer_apq * 100, 3),
            'voice_quality_assessment': self._assess_voice_quality(jitter_local, shimmer_local, np.mean(hnr_values) if len(hnr_values) > 0 else 0)
        }
    
    def _analyze_rhythm(self, audio: np.ndarray, sr: int) -> Dict:
        """Analysiert Sprechrhythmus"""
        # Tempo-Erkennung über Onset-Detection
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        
        # Beat-Intervalle
        beat_times = librosa.frames_to_time(beats, sr=sr)
        beat_intervals = np.diff(beat_times)
        
        # Rhythmus-Regularität
        if len(beat_intervals) > 1:
            rhythm_regularity = 1 - (np.std(beat_intervals) / np.mean(beat_intervals))
        else:
            rhythm_regularity = 0
        
        return {
            'estimated_tempo_bpm': round(float(tempo), 1),
            'beat_count': len(beats),
            'rhythm_regularity': round(rhythm_regularity, 3),
            'mean_beat_interval': round(np.mean(beat_intervals), 3) if len(beat_intervals) > 0 else None,
            'rhythm_variability': round(np.std(beat_intervals), 3) if len(beat_intervals) > 1 else None
        }
    
    def _analyze_emotion_indicators(self, sound: parselmouth.Sound) -> Dict:
        """Analysiert emotionale Indikatoren"""
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
            pitch_var = np.std(pitch_values) / np.mean(pitch_values)
            intensity_mean = np.mean(intensity_values)
            
            indicators['arousal_level'] = self._calculate_arousal(pitch_var, intensity_mean)
            indicators['valence_tendency'] = self._calculate_valence(np.mean(pitch_values), pitch_var)
            indicators['stress_indicators'] = {
                'high_pitch_variation': pitch_var > 0.2,
                'elevated_intensity': intensity_mean > 70,
                'rapid_changes': self._detect_rapid_changes(pitch_values)
            }
        
        return indicators
    
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
        
        return rapid_changes > len(values) * 0.3
    
    def save_results(self, results: Dict, output_path: str):
        """Speichert Analyseergebnisse"""
        output_path = Path(output_path)
        
        # JSON
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Readable Text Report
        txt_path = output_path.with_suffix('.txt')
        self._save_text_report(results, txt_path)
        
        # CSV für tabellarische Daten
        csv_path = output_path.with_suffix('.csv')
        self._save_csv_summary(results, csv_path)
    
    def _save_text_report(self, results: Dict, output_path: Path):
        """Erstellt lesbaren Text-Report"""
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
                f.write(f"  Wörter pro Minute: {res['speech_rate']['words_per_minute']}\n")
                f.write(f"  Kategorie: {res['speech_rate']['speech_rate_category']}\n")
                
                # Pausen
                f.write("\n\nPAUSENANALYSE:\n")
                f.write(f"  Anzahl Pausen: {res['pauses']['pause_count']}\n")
                f.write(f"  Pausenanteil: {res['pauses']['pause_ratio']*100:.1f}%\n")
                
                # Stimmqualität
                f.write("\n\nSTIMMQUALITÄT:\n")
                f.write(f"  Bewertung: {res['voice_quality']['voice_quality_assessment']}\n")
                f.write(f"  HNR: {res['voice_quality']['hnr_mean_db']} dB\n")
                f.write(f"  Jitter: {res['voice_quality']['jitter_local_percent']}%\n")
                f.write(f"  Shimmer: {res['voice_quality']['shimmer_local_percent']}%\n")
                
                # Emotionale Indikatoren
                if res['emotion_indicators']:
                    f.write("\n\nEMOTIONALE INDIKATOREN:\n")
                    f.write(f"  Erregungsniveau: {res['emotion_indicators'].get('arousal_level', 'N/A')}\n")
                    f.write(f"  Valenz-Tendenz: {res['emotion_indicators'].get('valence_tendency', 'N/A')}\n")
            else:
                f.write(f"FEHLER: {results.get('error', 'Unbekannter Fehler')}\n")
    
    def _save_csv_summary(self, results: Dict, output_path: Path):
        """Speichert zusammenfassende CSV"""
        if not results.get('success'):
            return
        
        res = results['results']
        data = []
        
        # Sammle alle numerischen Werte
        row = {'filename': res['file_info']['filename']}
        
        # Füge alle numerischen Werte hinzu
        for category in ['pitch', 'intensity', 'speech_rate', 'pauses', 'voice_quality', 'rhythm']:
            if category in res and isinstance(res[category], dict):
                for key, value in res[category].items():
                    if isinstance(value, (int, float)):
                        row[f"{category}_{key}"] = value
        
        data.append(row)
        
        # Speichere als CSV
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False, encoding='utf-8')
    
    def create_visualization(self, audio_path: str, results: Dict, output_path: str):
        """Erstellt Visualisierungen der Prosodieanalyse"""
        try:
            # Lade Audio für Visualisierung
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            sound = parselmouth.Sound(audio_path)
            
            # Erstelle Figure mit Subplots
            fig, axes = plt.subplots(4, 1, figsize=(12, 16))
            
            # 1. Waveform + Intensität
            ax1 = axes[0]
            time = np.linspace(0, len(audio)/sr, len(audio))
            ax1.plot(time, audio, alpha=0.5, linewidth=0.5)
            ax1.set_ylabel('Amplitude')
            ax1.set_title('Wellenform und Intensität')
            
            # Intensität überlagern
            intensity = call(sound, "To Intensity", 100, 0)
            intensity_time = np.linspace(0, sound.xmax, len(intensity.values[0]))
            ax1_twin = ax1.twinx()
            ax1_twin.plot(intensity_time, intensity.values[0], 'r', linewidth=2, alpha=0.7)
            ax1_twin.set_ylabel('Intensität (dB)', color='r')
            
            # 2. Tonhöhenverlauf
            ax2 = axes[1]
            pitch = call(sound, "To Pitch", 0.0, 75, 600)
            pitch_values = pitch.selected_array['frequency']
            pitch_time = pitch.xs()
            ax2.plot(pitch_time, pitch_values, 'b', linewidth=2)
            ax2.set_ylabel('F0 (Hz)')
            ax2.set_title('Tonhöhenverlauf')
            ax2.set_ylim(0, 500)
            
            # 3. Spektrogramm
            ax3 = axes[2]
            D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
            img = librosa.display.specshow(D, y_axis='hz', x_axis='time', ax=ax3, sr=sr)
            ax3.set_title('Spektrogramm')
            fig.colorbar(img, ax=ax3, format='%+2.0f dB')
            
            # 4. Prosodische Features über Zeit
            ax4 = axes[3]
            
            # Berechne gleitende Fenster-Features
            window_size = int(0.5 * sr)  # 500ms Fenster
            hop_size = int(0.1 * sr)     # 100ms Hop
            
            pitch_std = []
            times = []
            
            for i in range(0, len(audio) - window_size, hop_size):
                window = audio[i:i+window_size]
                times.append((i + window_size/2) / sr)
                
                # Berechne Pitch-Standardabweichung für dieses Fenster
                window_pitch = pitch_values[
                    (pitch_time >= i/sr) & (pitch_time < (i+window_size)/sr)
                ]
                if len(window_pitch) > 0:
                    pitch_std.append(np.std(window_pitch[window_pitch > 0]))
                else:
                    pitch_std.append(0)
            
            ax4.plot(times, pitch_std, 'g', linewidth=2)
            ax4.set_ylabel('Tonhöhen-Variabilität')
            ax4.set_xlabel('Zeit (s)')
            ax4.set_title('Prosodische Variabilität über Zeit')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return True
            
        except Exception as e:
            print(f"Fehler bei Visualisierung: {e}")
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
                'mean_pitch': res['pitch'].get('mean_f0', 'N/A'),
                'pitch_range': res['pitch'].get('range_f0', 'N/A'),
                'speech_rate_wpm': res['speech_rate'].get('words_per_minute', 'N/A'),
                'pause_ratio': res['pauses'].get('pause_ratio', 'N/A'),
                'voice_quality': res['voice_quality'].get('voice_quality_assessment', 'N/A'),
                'arousal': res['emotion_indicators'].get('arousal_level', 'N/A')
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


class ProsodyComparator:
    """Klasse zum Vergleich von Prosodieanalysen (z.B. verschiedene Sprecher)"""
    
    def __init__(self):
        self.analyzer = ProsodyAnalyzer()
    
    def compare_speakers(self, audio_files: Dict[str, str], output_dir: str) -> Dict:
        """
        Vergleicht Prosodie verschiedener Sprecher
        
        Args:
            audio_files: Dictionary {speaker_name: audio_path}
            output_dir: Ausgabe-Verzeichnis
            
        Returns:
            Vergleichsergebnisse
        """
        results = {}
        
        # Analysiere alle Sprecher
        for speaker, audio_path in audio_files.items():
            results[speaker] = self.analyzer.analyze_audio(audio_path)
        
        # Erstelle Vergleichsbericht
        comparison = self._create_comparison(results)
        
        # Speichere Vergleich
        self._save_comparison(comparison, output_dir)
        
        # Erstelle Vergleichsvisualisierung
        self._create_comparison_plot(results, output_dir)
        
        return comparison
    
    def _create_comparison(self, results: Dict) -> Dict:
        """Erstellt detaillierten Vergleich"""
        comparison = {
            'speakers': list(results.keys()),
            'feature_comparison': {},
            'significant_differences': []
        }
        
        # Vergleiche wichtige Features
        features_to_compare = [
            ('pitch', 'mean_f0', 'Mittlere Tonhöhe'),
            ('pitch', 'range_f0', 'Tonhöhenumfang'),
            ('speech_rate', 'words_per_minute', 'Sprechgeschwindigkeit'),
            ('pauses', 'pause_ratio', 'Pausenanteil'),
            ('voice_quality', 'jitter_local_percent', 'Jitter'),
            ('intensity', 'mean_intensity_db', 'Mittlere Lautstärke')
        ]
        
        for category, feature, label in features_to_compare:
            values = {}
            for speaker, result in results.items():
                if result['success'] and category in result['results']:
                    value = result['results'][category].get(feature)
                    if value is not None:
                        values[speaker] = value
            
            if len(values) > 1:
                comparison['feature_comparison'][label] = values
                
                # Prüfe auf signifikante Unterschiede
                vals = list(values.values())
                if max(vals) / min(vals) > 1.5:  # 50% Unterschied
                    comparison['significant_differences'].append(
                        f"{label}: Großer Unterschied zwischen Sprechern"
                    )
        
        return comparison
    
    def _save_comparison(self, comparison: Dict, output_dir: str):
        """Speichert Vergleichsergebnisse"""
        # JSON
        json_path = os.path.join(output_dir, "speaker_comparison.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        
        # Lesbarer Bericht
        report_path = os.path.join(output_dir, "speaker_comparison_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("SPRECHER-VERGLEICH\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Verglichene Sprecher: {', '.join(comparison['speakers'])}\n\n")
            
            f.write("FEATURE-VERGLEICH:\n")
            for feature, values in comparison['feature_comparison'].items():
                f.write(f"\n{feature}:\n")
                for speaker, value in values.items():
                    f.write(f"  {speaker}: {value:.2f}\n")
            
            if comparison['significant_differences']:
                f.write("\n\nSIGNIFIKANTE UNTERSCHIEDE:\n")
                for diff in comparison['significant_differences']:
                    f.write(f"  • {diff}\n")
    
    def _create_comparison_plot(self, results: Dict, output_dir: str):
        """Erstellt Vergleichsvisualisierung"""
        try:
            # Sammle Daten für Radar-Plot
            features = ['Tonhöhe', 'Variabilität', 'Tempo', 'Pausen', 'Qualität']
            speakers = []
            data = []
            
            for speaker, result in results.items():
                if result['success']:
                    res = result['results']
                    speakers.append(speaker)
                    
                    # Normalisiere Werte auf 0-1 Skala
                    values = [
                        min(res['pitch'].get('mean_f0', 100) / 300, 1),  # Tonhöhe
                        min(res['pitch'].get('cv_f0', 0.1) / 0.3, 1),    # Variabilität
                        min(res['speech_rate'].get('words_per_minute', 150) / 250, 1),  # Tempo
                        1 - res['pauses'].get('pause_ratio', 0.2),       # Pausen (invertiert)
                        1 if res['voice_quality'].get('voice_quality_assessment') in ['ausgezeichnet', 'gut'] else 0.5
                    ]
                    data.append(values)
            
            if len(speakers) > 0:
                # Erstelle Radar-Plot
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='polar')
                
                # Winkel für Features
                angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
                angles += angles[:1]  # Schließe den Kreis
                
                # Plot für jeden Sprecher
                for i, (speaker, values) in enumerate(zip(speakers, data)):
                    values += values[:1]  # Schließe den Kreis
                    ax.plot(angles, values, 'o-', linewidth=2, label=speaker)
                    ax.fill(angles, values, alpha=0.25)
                
                ax.set_theta_offset(np.pi / 2)
                ax.set_theta_direction(-1)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(features)
                ax.set_ylim(0, 1)
                ax.set_title('Prosodischer Sprecher-Vergleich', size=20, y=1.1)
                ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
                ax.grid(True)
                
                plt.tight_layout()
                plot_path = os.path.join(output_dir, "speaker_comparison_radar.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            print(f"Fehler bei Vergleichsplot: {e}")


# Beispiel-Integration für ArchiveCAT
if __name__ == "__main__":
    # Test mit einzelner Datei
    analyzer = ProsodyAnalyzer()
    
    # Beispiel-Analyse
    audio_file = "path/to/audio.wav"
    results = analyzer.analyze_audio(audio_file, gender="male")
    
    if results['success']:
        print("\nProsodieanalyse erfolgreich!")
        print(f"Mittlere Tonhöhe: {results['results']['pitch']['mean_f0']} Hz")
        print(f"Sprechgeschwindigkeit: {results['results']['speech_rate']['words_per_minute']} WPM")
        
        # Speichere Ergebnisse
        analyzer.save_results(results, "output/prosody_analysis")
        
        # Erstelle Visualisierung
        analyzer.create_visualization(audio_file, results, "output/prosody_viz.png")
    
    # Beispiel: Batch-Analyse
    audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
    batch_results = batch_analyze_prosody(audio_files, "output/batch_prosody")
    
    # Beispiel: Sprecher-Vergleich
    comparator = ProsodyComparator()
    speaker_files = {
        "Sprecher_1": "speaker1.wav",
        "Sprecher_2": "speaker2.wav"
    }
    comparison = comparator.compare_speakers(speaker_files, "output/comparison")