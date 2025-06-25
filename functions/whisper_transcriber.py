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
        Transkribiert eine einzelne Audio-Datei mit erweiterten Features
        
        Args:
            audio_path (str): Pfad zur Audio-Datei
            language (str, optional): Sprache des Audios (z.B. "de", "en")
            prompt (str, optional): Kontext-Prompt für bessere Erkennung
            
        Returns:
            dict: {
                'success': bool,
                'text': str,
                'segments': list (optional),
                'words': list (optional),
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
                # Erweiterte Parameter für bessere Ergebnisse
                params = {
                    "model": self.model,
                    "file": audio_file,
                    "response_format": "verbose_json",  # Gibt Timestamps zurück
                    "temperature": 0.0,  # Deterministischere Ergebnisse
                }
                
                # Optionale Parameter
                if language:
                    params["language"] = language
                if prompt:
                    params["prompt"] = prompt
                
                # Versuche erweiterte Features (nicht alle API-Versionen unterstützen diese)
                try:
                    # Timestamp Granularität für Wort-Level Timestamps
                    params["timestamp_granularities"] = ["segment", "word"]
                except:
                    # Fallback für ältere API-Versionen
                    pass
                
                # API-Aufruf mit Retry-Logik
                max_retries = 3
                last_error = None
                
                for attempt in range(max_retries):
                    try:
                        response = self.client.audio.transcriptions.create(**params)
                        break  # Erfolgreich, beende Schleife
                    except Exception as api_error:
                        last_error = api_error
                        error_msg = str(api_error)
                        
                        # Spezifische Fehlerbehandlung
                        if 'Incorrect API key' in error_msg:
                            return {'success': False, 'error': 'Ungültiger API Key'}
                        elif 'quota' in error_msg.lower() or 'rate limit' in error_msg.lower():
                            return {'success': False, 'error': 'API-Kontingent überschritten oder Rate-Limit erreicht'}
                        elif 'timeout' in error_msg.lower():
                            if attempt < max_retries - 1:
                                print(f"Timeout bei Versuch {attempt + 1}/{max_retries}, versuche erneut...")
                                time.sleep(2 ** attempt)  # Exponentielles Backoff
                                continue
                        
                        # Bei anderen Fehlern: Retry mit Backoff
                        if attempt < max_retries - 1:
                            print(f"API-Fehler bei Versuch {attempt + 1}/{max_retries}: {error_msg}")
                            time.sleep(2 ** attempt)
                        else:
                            # Letzter Versuch fehlgeschlagen
                            raise last_error
            
            # Verarbeite erfolgreiche Response
            result = {
                'success': True,
                'text': response.text,
                'language': getattr(response, 'language', language),
                'duration': getattr(response, 'duration', None),
                'model': self.model
            }
            
            # Segments mit erweiterten Informationen
            if hasattr(response, 'segments'):
                result['segments'] = []
                for idx, seg in enumerate(response.segments):
                    segment_data = {
                        'id': idx,
                        'start': seg.start,
                        'end': seg.end,
                        'text': seg.text
                    }
                    
                    # Füge optionale Felder hinzu wenn verfügbar
                    if hasattr(seg, 'avg_logprob'):
                        segment_data['avg_logprob'] = seg.avg_logprob
                    if hasattr(seg, 'compression_ratio'):
                        segment_data['compression_ratio'] = seg.compression_ratio
                    if hasattr(seg, 'no_speech_prob'):
                        segment_data['no_speech_prob'] = seg.no_speech_prob
                    if hasattr(seg, 'tokens'):
                        segment_data['tokens'] = seg.tokens
                    
                    result['segments'].append(segment_data)
            
            # Words mit Timestamps (wenn verfügbar)
            if hasattr(response, 'words'):
                result['words'] = [
                    {
                        'word': word.word,
                        'start': word.start,
                        'end': word.end,
                        'probability': getattr(word, 'probability', None)
                    }
                    for word in response.words
                ]
            
            # Zusätzliche Metriken wenn verfügbar
            if hasattr(response, 'avg_logprob'):
                result['avg_logprob'] = response.avg_logprob
            if hasattr(response, 'compression_ratio'):
                result['compression_ratio'] = response.compression_ratio
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            
            # Detaillierte Fehlerbehandlung
            if 'Incorrect API key' in error_msg:
                return {'success': False, 'error': 'Ungültiger API Key'}
            elif 'quota' in error_msg.lower():
                return {'success': False, 'error': 'API-Kontingent überschritten'}
            elif 'Invalid audio file' in error_msg:
                return {'success': False, 'error': 'Ungültige Audio-Datei'}
            elif 'timeout' in error_msg.lower():
                return {'success': False, 'error': 'Zeitüberschreitung bei der API-Anfrage'}
            else:
                return {'success': False, 'error': f'Transkriptionsfehler: {error_msg}'}


    # === ZUSÄTZLICH: Erweitere die save_transcription Methode ===
    # Um die erweiterten Daten zu nutzen:

    def save_transcription(self, transcription: Dict, output_path: str) -> bool:
        """
        Speichert Transkription in verschiedenen Formaten (erweiterte Version)
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Speichere als Text
            text_path = output_path.with_suffix('.txt')
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(transcription['text'])
            
            # Speichere als JSON mit allen Details (inklusive Words wenn vorhanden)
            json_path = output_path.with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(transcription, f, ensure_ascii=False, indent=2)
            
            # Wenn Segments vorhanden, erstelle auch SRT-Untertitel
            if 'segments' in transcription and transcription['segments']:
                srt_path = output_path.with_suffix('.srt')
                self._create_srt(transcription['segments'], srt_path)
            
            # NEU: Wenn Words vorhanden, erstelle auch eine Wort-Level Datei
            if 'words' in transcription and transcription['words']:
                words_path = output_path.with_suffix('.words.json')
                with open(words_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'words': transcription['words'],
                        'total_words': len(transcription['words']),
                        'duration': transcription.get('duration', 0)
                    }, f, ensure_ascii=False, indent=2)
            
            # NEU: Erstelle erweiterte Statistik-Datei
            if any(key in transcription for key in ['avg_logprob', 'compression_ratio', 'segments']):
                stats_path = output_path.with_name(f"{output_path.stem}_stats.json")
                stats = {
                    'file': output_path.stem,
                    'language': transcription.get('language', 'unknown'),
                    'duration': transcription.get('duration', 0),
                    'word_count': len(transcription.get('text', '').split()),
                    'segment_count': len(transcription.get('segments', [])),
                    'avg_logprob': transcription.get('avg_logprob'),
                    'compression_ratio': transcription.get('compression_ratio')
                }
                
                # Berechne durchschnittliche Konfidenz wenn Segments vorhanden
                if 'segments' in transcription and transcription['segments']:
                    no_speech_probs = [seg.get('no_speech_prob', 0) for seg in transcription['segments'] if 'no_speech_prob' in seg]
                    if no_speech_probs:
                        stats['avg_speech_confidence'] = 1 - (sum(no_speech_probs) / len(no_speech_probs))
                
                with open(stats_path, 'w', encoding='utf-8') as f:
                    json.dump(stats, f, ensure_ascii=False, indent=2)
            
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