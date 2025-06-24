# whisper_transcriber.py
"""
Modul zur automatisierten Audio-Transkription mit OpenAI's Whisper API
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import openai
from openai import OpenAI

class WhisperTranscriber:
    """
    Klasse zur Verwaltung von Audio-Transkriptionen mit OpenAI's Whisper
    """
    
    def __init__(self, api_key: str, model: str = "whisper-1"):
        """
        Initialisiert den Transcriber
        
        Args:
            api_key (str): OpenAI API Schlüssel
            model (str): Whisper Modell (Standard: whisper-1)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
    def transcribe_audio(self, audio_path: str, language: str = None, prompt: str = None) -> Dict:
        """
        Transkribiert eine einzelne Audio-Datei
        
        Args:
            audio_path (str): Pfad zur Audio-Datei
            language (str, optional): Sprache des Audios (z.B. "de", "en")
            prompt (str, optional): Kontext-Prompt für bessere Erkennung
            
        Returns:
            dict: {
                'success': bool,
                'text': str,
                'segments': list (optional),
                'error': str (bei Fehler)
            }
        """
        try:
            audio_path = Path(audio_path)
            
            if not audio_path.exists():
                return {'success': False, 'error': f'Audio-Datei nicht gefunden: {audio_path}'}
            
            # Prüfe Dateigröße (OpenAI Limit: 25 MB)
            file_size_mb = audio_path.stat().st_size / (1024 * 1024)
            if file_size_mb > 25:
                return {'success': False, 'error': f'Datei zu groß: {file_size_mb:.1f} MB (Max: 25 MB)'}
            
            print(f"Transkribiere: {audio_path.name}")
            
            # Öffne Audio-Datei
            with open(audio_path, 'rb') as audio_file:
                # Basis-Parameter
                params = {
                    "model": self.model,
                    "file": audio_file,
                    "response_format": "verbose_json"  # Gibt auch Timestamps zurück
                }
                
                # Optionale Parameter
                if language:
                    params["language"] = language
                if prompt:
                    params["prompt"] = prompt
                
                # API-Aufruf
                response = self.client.audio.transcriptions.create(**params)
            
            # Ergebnis verarbeiten
            result = {
                'success': True,
                'text': response.text,
                'language': getattr(response, 'language', language),
                'duration': getattr(response, 'duration', None)
            }
            
            # Segments mit Timestamps wenn verfügbar
            if hasattr(response, 'segments'):
                result['segments'] = [
                    {
                        'start': seg.start,
                        'end': seg.end,
                        'text': seg.text
                    }
                    for seg in response.segments
                ]
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': f'Transkriptionsfehler: {str(e)}'}
    
    def save_transcription(self, transcription: Dict, output_path: str) -> bool:
        """
        Speichert Transkription in verschiedenen Formaten
        
        Args:
            transcription (dict): Transkriptionsergebnis
            output_path (str): Ausgabepfad (ohne Erweiterung)
            
        Returns:
            bool: Erfolg
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Speichere als Text
            text_path = output_path.with_suffix('.txt')
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(transcription['text'])
            
            # Speichere als JSON mit allen Details
            json_path = output_path.with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(transcription, f, ensure_ascii=False, indent=2)
            
            # Wenn Segments vorhanden, erstelle auch SRT-Untertitel
            if 'segments' in transcription and transcription['segments']:
                srt_path = output_path.with_suffix('.srt')
                self._create_srt(transcription['segments'], srt_path)
            
            print(f"Transkription gespeichert: {text_path}")
            return True
            
        except Exception as e:
            print(f"Fehler beim Speichern: {e}")
            return False
    
    def _create_srt(self, segments: List[Dict], output_path: Path):
        """Erstellt SRT-Untertiteldatei aus Segments"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, seg in enumerate(segments, 1):
                start_time = self._seconds_to_srt_time(seg['start'])
                end_time = self._seconds_to_srt_time(seg['end'])
                f.write(f"{i}\n{start_time} --> {end_time}\n{seg['text'].strip()}\n\n")
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Konvertiert Sekunden in SRT-Zeitformat"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')
    
    def batch_transcribe(self, audio_paths: List[str], output_dir: str, 
                        language: str = None, prompt: str = None) -> Dict[str, Dict]:
        """
        Transkribiert mehrere Audio-Dateien
        
        Args:
            audio_paths (list): Liste von Audio-Pfaden
            output_dir (str): Ausgabe-Verzeichnis
            language (str, optional): Sprache für alle Audios
            prompt (str, optional): Kontext-Prompt
            
        Returns:
            dict: Ergebnisse für jede Datei
        """
        results = {}
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, audio_path in enumerate(audio_paths, 1):
            print(f"\nVerarbeite {i}/{len(audio_paths)}: {Path(audio_path).name}")
            
            # Transkribiere
            result = self.transcribe_audio(audio_path, language, prompt)
            
            if result['success']:
                # Speichere Transkription
                audio_name = Path(audio_path).stem
                output_path = output_dir / audio_name
                self.save_transcription(result, str(output_path))
            
            results[audio_path] = result
            
            # Rate limiting (um API-Limits zu respektieren)
            if i < len(audio_paths):
                time.sleep(1)
        
        return results
    
    def transcribe_video_segments(self, video_folder: str, language: str = "de") -> Dict:
        """
        Speziell für deine Ordnerstruktur: Transkribiert alle WAV-Dateien in Video-Segment-Ordnern
        
        Args:
            video_folder (str): Pfad zum Hauptordner des Videos
            language (str): Sprache der Audios
            
        Returns:
            dict: Transkriptionsergebnisse
        """
        video_folder = Path(video_folder)
        results = {}
        
        # Finde alle WAV-Dateien in Unterordnern
        wav_files = list(video_folder.rglob("*_audio.wav"))
        
        if not wav_files:
            print(f"Keine Audio-Dateien gefunden in: {video_folder}")
            return results
        
        print(f"Gefunden: {len(wav_files)} Audio-Dateien")
        
        for wav_path in wav_files:
            # Transkribiere
            result = self.transcribe_audio(str(wav_path), language=language)
            
            if result['success']:
                # Speichere im gleichen Ordner wie die WAV-Datei
                output_path = wav_path.parent / wav_path.stem
                self.save_transcription(result, str(output_path))
            
            results[str(wav_path)] = result
            time.sleep(1)  # Rate limiting
        
        # Erstelle Gesamt-Transkript
        self._create_combined_transcript(results, video_folder)
        
        return results
    
    def _create_combined_transcript(self, results: Dict, output_folder: Path):
        """Erstellt ein kombiniertes Transkript aller Segmente"""
        combined_text = []
        
        # Sortiere nach Segment-Nummer
        sorted_results = sorted(
            [(path, res) for path, res in results.items() if res['success']],
            key=lambda x: self._extract_segment_number(x[0])
        )
        
        for path, result in sorted_results:
            segment_name = Path(path).parent.name
            combined_text.append(f"=== {segment_name} ===")
            combined_text.append(result['text'])
            combined_text.append("")
        
        # Speichere kombiniertes Transkript
        combined_path = output_folder / "Gesamttranskript.txt"
        with open(combined_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(combined_text))
        
        print(f"Gesamttranskript erstellt: {combined_path}")
    
    def _extract_segment_number(self, path: str) -> int:
        """Extrahiert Segment-Nummer aus Pfad für Sortierung"""
        import re
        match = re.search(r'Segment_(\d+)', path)
        return int(match.group(1)) if match else 0


# Beispiel-Integration in dein bestehendes System
def integrate_with_archivecat(api_key: str):
    """
    Beispiel-Funktion zur Integration mit ArchiveCAT
    """
    # Initialisiere Transcriber
    transcriber = WhisperTranscriber(api_key)
    
    # Beispiel: Transkribiere alle Videos in einem Verzeichnis
    videos_dir = r"C:\Users\rodtv\OneDrive\Desktop\Desktop\ArchiveCAT\data\videos"
    
    # Finde alle Video-Ordner
    video_folders = [f for f in Path(videos_dir).iterdir() if f.is_dir()]
    
    for video_folder in video_folders:
        print(f"\n{'='*50}")
        print(f"Verarbeite Video: {video_folder.name}")
        print(f"{'='*50}")
        
        # Transkribiere alle Segmente
        results = transcriber.transcribe_video_segments(str(video_folder))
        
        # Zeige Zusammenfassung
        successful = sum(1 for r in results.values() if r['success'])
        print(f"\nErgebnis: {successful}/{len(results)} erfolgreich transkribiert")


# Standalone-Verwendung
if __name__ == "__main__":
    # Beispiel-Verwendung
    API_KEY = "dein-openai-api-key-hier"
    
    transcriber = WhisperTranscriber(API_KEY)
    
    # Einzelne Datei transkribieren
    result = transcriber.transcribe_audio(
        "pfad/zur/audio.wav",
        language="de",
        prompt="Dies ist ein wissenschaftliches Interview über..."
    )
    
    if result['success']:
        print(f"Transkription: {result['text'][:200]}...")
        transcriber.save_transcription(result, "ausgabe/transkript")
    
    # Batch-Verarbeitung
    audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
    results = transcriber.batch_transcribe(
        audio_files,
        output_dir="transkriptionen",
        language="de"
    )