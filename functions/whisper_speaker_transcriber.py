# whisper_speaker_transcriber.py
"""
Modul zur Audio-Transkription mit Whisper und Speaker Diarization mit pyannote
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from openai import OpenAI
from pyannote.audio import Pipeline
from pydub import AudioSegment
import torch

class WhisperSpeakerTranscriber:
    """
    Kombiniert OpenAI Whisper Transkription mit pyannote Speaker Diarization
    """
    
    def __init__(self, openai_api_key: str, hf_auth_token: str, model: str = "whisper-1"):
        """
        Initialisiert den Transcriber
        
        Args:
            openai_api_key (str): OpenAI API Schlüssel
            hf_auth_token (str): Hugging Face Token für pyannote
            model (str): Whisper Modell (Standard: whisper-1)
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.whisper_model = model
        
        # Initialisiere pyannote Pipeline
        print("Lade Speaker Diarization Modell...")
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_auth_token
        )
        
        # Nutze GPU wenn verfügbar
        if torch.cuda.is_available():
            self.diarization_pipeline.to(torch.device("cuda"))
            print("Nutze GPU für Speaker Diarization")
        else:
            print("Nutze CPU für Speaker Diarization")
    
    def transcribe_with_speakers(self, audio_path: str, language: str = None, 
                                num_speakers: int = None, min_speakers: int = None, 
                                max_speakers: int = None) -> Dict:
        """
        Transkribiert Audio mit Speaker-Identifikation
        
        Args:
            audio_path (str): Pfad zur Audio-Datei
            language (str, optional): Sprache des Audios
            num_speakers (int, optional): Exakte Anzahl der Sprecher (wenn bekannt)
            min_speakers (int, optional): Minimale Anzahl Sprecher
            max_speakers (int, optional): Maximale Anzahl Sprecher
            
        Returns:
            dict: Transkription mit Speaker-Information
        """
        try:
            audio_path = Path(audio_path)
            
            if not audio_path.exists():
                return {'success': False, 'error': f'Audio-Datei nicht gefunden: {audio_path}'}
            
            print(f"Verarbeite: {audio_path.name}")
            
            # Schritt 1: Speaker Diarization mit pyannote
            print("Schritt 1: Erkenne Sprecher...")
            diarization = self._perform_diarization(
                audio_path, num_speakers, min_speakers, max_speakers
            )
            
            if not diarization:
                return {'success': False, 'error': 'Speaker Diarization fehlgeschlagen'}
            
            # Schritt 2: Whisper Transkription
            print("Schritt 2: Transkribiere Audio...")
            whisper_result = self._transcribe_with_whisper(audio_path, language)
            
            if not whisper_result['success']:
                return whisper_result
            
            # Schritt 3: Kombiniere Diarization mit Transkription
            print("Schritt 3: Kombiniere Sprecher mit Transkription...")
            combined_result = self._combine_diarization_with_transcription(
                diarization, whisper_result, audio_path
            )
            
            return combined_result
            
        except Exception as e:
            return {'success': False, 'error': f'Unerwarteter Fehler: {str(e)}'}
    
    def _perform_diarization(self, audio_path: Path, num_speakers: int = None,
                           min_speakers: int = None, max_speakers: int = None):
        """Führt Speaker Diarization durch"""
        try:
            # Konfiguriere Diarization Parameter
            diarization_params = {}
            if num_speakers:
                diarization_params['num_speakers'] = num_speakers
            elif min_speakers or max_speakers:
                diarization_params['min_speakers'] = min_speakers or 1
                diarization_params['max_speakers'] = max_speakers or 10
            
            # Führe Diarization durch
            diarization = self.diarization_pipeline(
                str(audio_path), 
                **diarization_params
            )
            
            return diarization
            
        except Exception as e:
            print(f"Diarization Fehler: {e}")
            return None
    
    def _transcribe_with_whisper(self, audio_path: Path, language: str = None) -> Dict:
        """Transkribiert mit Whisper API"""
        try:
            # Prüfe Dateigröße
            file_size_mb = audio_path.stat().st_size / (1024 * 1024)
            if file_size_mb > 25:
                return {'success': False, 'error': f'Datei zu groß: {file_size_mb:.1f} MB (Max: 25 MB)'}
            
            with open(audio_path, 'rb') as audio_file:
                params = {
                    "model": self.whisper_model,
                    "file": audio_file,
                    "response_format": "verbose_json",
                    "timestamp_granularities": ["segment", "word"]  # Wenn verfügbar
                }
                
                if language:
                    params["language"] = language
                
                response = self.client.audio.transcriptions.create(**params)
            
            return {
                'success': True,
                'text': response.text,
                'segments': [
                    {
                        'start': seg.start,
                        'end': seg.end,
                        'text': seg.text
                    }
                    for seg in response.segments
                ],
                'language': getattr(response, 'language', language)
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Whisper Fehler: {str(e)}'}
    
    def _combine_diarization_with_transcription(self, diarization, whisper_result, audio_path):
        """Kombiniert Speaker Diarization mit Whisper Transkription"""
        try:
            segments_with_speakers = []
            
            # Erstelle Speaker-Mapping
            speaker_mapping = {}
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if speaker not in speaker_mapping:
                    speaker_mapping[speaker] = f"Sprecher_{len(speaker_mapping) + 1}"
            
            # Für jeden Whisper-Segment, finde den zugehörigen Sprecher
            for segment in whisper_result['segments']:
                segment_mid = (segment['start'] + segment['end']) / 2
                
                # Finde Sprecher für diesen Zeitpunkt
                current_speaker = None
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    if turn.start <= segment_mid <= turn.end:
                        current_speaker = speaker_mapping[speaker]
                        break
                
                if not current_speaker:
                    # Fallback: Finde nächsten Sprecher
                    min_distance = float('inf')
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        distance = min(
                            abs(turn.start - segment_mid),
                            abs(turn.end - segment_mid)
                        )
                        if distance < min_distance:
                            min_distance = distance
                            current_speaker = speaker_mapping[speaker]
                
                segments_with_speakers.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'speaker': current_speaker or 'Unbekannt',
                    'text': segment['text']
                })
            
            # Erstelle finales Ergebnis
            result = {
                'success': True,
                'text': whisper_result['text'],
                'segments_with_speakers': segments_with_speakers,
                'speakers': list(speaker_mapping.values()),
                'speaker_count': len(speaker_mapping),
                'language': whisper_result['language'],
                'duration': self._get_audio_duration(audio_path)
            }
            
            # Erstelle auch eine nach Sprecher gruppierte Ansicht
            result['by_speaker'] = self._group_by_speaker(segments_with_speakers)
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Fehler beim Kombinieren: {str(e)}',
                'whisper_result': whisper_result  # Gebe wenigstens Whisper-Ergebnis zurück
            }
    
    def _group_by_speaker(self, segments):
        """Gruppiert Segmente nach Sprecher"""
        by_speaker = {}
        for segment in segments:
            speaker = segment['speaker']
            if speaker not in by_speaker:
                by_speaker[speaker] = []
            by_speaker[speaker].append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text']
            })
        return by_speaker
    
    def _get_audio_duration(self, audio_path):
        """Ermittelt Audio-Dauer"""
        try:
            audio = AudioSegment.from_file(audio_path)
            return len(audio) / 1000.0  # Konvertiere zu Sekunden
        except:
            return None
    
    def save_transcription_with_speakers(self, result: Dict, output_path: str) -> bool:
        """
        Speichert die Transkription mit Speaker-Information
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 1. Speichere als formatierter Text mit Sprechern
            text_path = output_path.with_suffix('.txt')
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(f"Transkription mit {result['speaker_count']} Sprechern\n")
                f.write("=" * 50 + "\n\n")
                
                for segment in result['segments_with_speakers']:
                    timestamp = f"[{self._seconds_to_time(segment['start'])} - {self._seconds_to_time(segment['end'])}]"
                    f.write(f"{timestamp} {segment['speaker']}: {segment['text']}\n\n")
            
            # 2. Speichere als JSON
            json_path = output_path.with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            # 3. Erstelle SRT mit Speaker-Information
            srt_path = output_path.with_suffix('.srt')
            self._create_srt_with_speakers(result['segments_with_speakers'], srt_path)
            
            # 4. Erstelle Speaker-Summary
            summary_path = output_path.with_name(f"{output_path.stem}_speaker_summary.txt")
            self._create_speaker_summary(result, summary_path)
            
            print(f"Transkription mit Sprechern gespeichert: {text_path}")
            return True
            
        except Exception as e:
            print(f"Fehler beim Speichern: {e}")
            return False
    
    def _create_srt_with_speakers(self, segments, output_path):
        """Erstellt SRT-Datei mit Speaker-Labels"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, seg in enumerate(segments, 1):
                start_time = self._seconds_to_srt_time(seg['start'])
                end_time = self._seconds_to_srt_time(seg['end'])
                text_with_speaker = f"[{seg['speaker']}] {seg['text'].strip()}"
                f.write(f"{i}\n{start_time} --> {end_time}\n{text_with_speaker}\n\n")
    
    def _create_speaker_summary(self, result, output_path):
        """Erstellt Zusammenfassung pro Sprecher"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("SPEAKER SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            for speaker, segments in result['by_speaker'].items():
                total_time = sum(seg['end'] - seg['start'] for seg in segments)
                word_count = sum(len(seg['text'].split()) for seg in segments)
                
                f.write(f"{speaker}:\n")
                f.write(f"  - Sprechzeit: {self._seconds_to_time(total_time)}\n")
                f.write(f"  - Anzahl Segmente: {len(segments)}\n")
                f.write(f"  - Geschätzte Wörter: {word_count}\n")
                f.write(f"  - Anteil an Gesamtzeit: {(total_time / result['duration'] * 100):.1f}%\n")
                f.write("\n")
    
    def _seconds_to_time(self, seconds: float) -> str:
        """Konvertiert Sekunden zu HH:MM:SS Format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Konvertiert Sekunden zu SRT-Zeitformat"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')
    
    def batch_transcribe_with_speakers(self, audio_paths: List[str], output_dir: str,
                                      language: str = None, **speaker_params) -> Dict:
        """
        Transkribiert mehrere Dateien mit Speaker-Erkennung
        """
        results = {}
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, audio_path in enumerate(audio_paths, 1):
            print(f"\n{'='*50}")
            print(f"Verarbeite {i}/{len(audio_paths)}: {Path(audio_path).name}")
            print(f"{'='*50}")
            
            result = self.transcribe_with_speakers(audio_path, language, **speaker_params)
            
            if result['success']:
                audio_name = Path(audio_path).stem
                output_path = output_dir / audio_name
                self.save_transcription_with_speakers(result, str(output_path))
            
            results[audio_path] = result
            
            if i < len(audio_paths):
                time.sleep(1)
        
        return results


# Beispiel-Verwendung
if __name__ == "__main__":
    # API Keys
    OPENAI_API_KEY = "dein-openai-api-key"
    HF_AUTH_TOKEN = "dein-huggingface-token"  # Für pyannote
    
    # Initialisiere Transcriber
    transcriber = WhisperSpeakerTranscriber(OPENAI_API_KEY, HF_AUTH_TOKEN)
    
    # Transkribiere mit Speaker-Erkennung
    result = transcriber.transcribe_with_speakers(
        "interview.wav",
        language="de",
        min_speakers=2,  # Mindestens 2 Sprecher
        max_speakers=4   # Maximal 4 Sprecher
    )
    
    if result['success']:
        print(f"\nErkannte Sprecher: {result['speaker_count']}")
        print(f"Sprecher: {', '.join(result['speakers'])}")
        
        # Zeige erste Segmente
        print("\nErste Segmente mit Sprechern:")
        for seg in result['segments_with_speakers'][:5]:
            print(f"[{seg['speaker']}] {seg['text']}")
        
        # Speichere Ergebnis
        transcriber.save_transcription_with_speakers(result, "output/interview_transcript")