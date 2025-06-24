# audio_video_splitter.py
"""
Modul zum Aufsplitten von Videos in separate Audio (.wav) und stumme Video-Dateien
"""

import os
import subprocess
import re
from pathlib import Path

def extract_audio_to_wav(input_video_path, output_audio_path=None):
    """
    Extrahiert Audio aus Video und speichert als WAV-Datei
    
    Args:
        input_video_path (str): Pfad zur Eingabe-Videodatei
        output_audio_path (str, optional): Pfad für Audio-Ausgabe. Falls None, wird automatisch generiert.
    
    Returns:
        tuple: (success: bool, output_path: str, error_message: str)
    """
    try:
        input_path = Path(input_video_path)
        
        if not input_path.exists():
            return False, "", f"Eingabedatei nicht gefunden: {input_video_path}"
        
        # Automatische Pfad-Generierung falls nicht angegeben
        if output_audio_path is None:
            output_audio_path = input_path.parent / f"{input_path.stem}_audio.wav"
        
        output_path = Path(output_audio_path)
        
        # FFmpeg Command für Audio-Extraktion
        command = [
            'ffmpeg',
            '-i', str(input_path),
            '-vn',  # Kein Video
            '-acodec', 'pcm_s16le',  # WAV Format
            '-ar', '44100',  # Sample Rate
            '-ac', '2',  # Stereo
            '-y',  # Überschreiben falls vorhanden
            str(output_path)
        ]
        
        print(f"Extrahiere Audio: {input_path.name} -> {output_path.name}")
        
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if process.returncode == 0:
            if output_path.exists() and output_path.stat().st_size > 0:
                print(f"Audio erfolgreich extrahiert: {output_path}")
                return True, str(output_path), ""
            else:
                return False, "", "Audio-Datei wurde nicht erstellt oder ist leer"
        else:
            return False, "", f"FFmpeg Fehler: {process.stderr}"
            
    except Exception as e:
        return False, "", f"Unerwarteter Fehler: {str(e)}"

def create_silent_video(input_video_path, output_video_path=None):
    """
    Erstellt stumme Version des Videos (entfernt Audio-Spur)
    
    Args:
        input_video_path (str): Pfad zur Eingabe-Videodatei
        output_video_path (str, optional): Pfad für Video-Ausgabe. Falls None, wird automatisch generiert.
    
    Returns:
        tuple: (success: bool, output_path: str, error_message: str)
    """
    try:
        input_path = Path(input_video_path)
        
        if not input_path.exists():
            return False, "", f"Eingabedatei nicht gefunden: {input_video_path}"
        
        # Automatische Pfad-Generierung falls nicht angegeben
        if output_video_path is None:
            output_video_path = input_path.parent / f"{input_path.stem}_silent{input_path.suffix}"
        
        output_path = Path(output_video_path)
        
        # FFmpeg Command für stummes Video
        command = [
            'ffmpeg',
            '-i', str(input_path),
            '-an',  # Kein Audio
            '-c:v', 'libx264',  # Video Codec beibehalten
            '-preset', 'fast',  # Schnellere Encoding
            '-y',  # Überschreiben falls vorhanden
            str(output_path)
        ]
        
        print(f"Erstelle stummes Video: {input_path.name} -> {output_path.name}")
        
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if process.returncode == 0:
            if output_path.exists() and output_path.stat().st_size > 0:
                print(f"Stummes Video erfolgreich erstellt: {output_path}")
                return True, str(output_path), ""
            else:
                return False, "", "Video-Datei wurde nicht erstellt oder ist leer"
        else:
            return False, "", f"FFmpeg Fehler: {process.stderr}"
            
    except Exception as e:
        return False, "", f"Unerwarteter Fehler: {str(e)}"

def split_video_audio(input_video_path, output_dir=None, keep_original=True):
    """
    Splittet Video in Audio (.wav) und stumme Video-Datei
    
    Args:
        input_video_path (str): Pfad zur Eingabe-Videodatei
        output_dir (str, optional): Ausgabe-Verzeichnis. Falls None, wird das Verzeichnis der Eingabedatei verwendet.
        keep_original (bool): Ob die ursprüngliche Datei behalten werden soll
    
    Returns:
        dict: {
            'success': bool,
            'audio_path': str,
            'video_path': str,
            'errors': list
        }
    """
    try:
        input_path = Path(input_video_path)
        
        if not input_path.exists():
            return {
                'success': False,
                'audio_path': '',
                'video_path': '',
                'errors': [f"Eingabedatei nicht gefunden: {input_video_path}"]
            }
        
        # Ausgabe-Verzeichnis bestimmen
        if output_dir is None:
            output_dir = input_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Ausgabe-Pfade generieren
        base_name = input_path.stem
        audio_path = output_dir / f"{base_name}_audio.wav"
        video_path = output_dir / f"{base_name}_silent{input_path.suffix}"
        
        errors = []
        
        # Audio extrahieren
        print(f"Starte Audio-Extraktion für: {input_path.name}")
        audio_success, audio_output, audio_error = extract_audio_to_wav(input_video_path, str(audio_path))
        if not audio_success:
            errors.append(f"Audio-Extraktion fehlgeschlagen: {audio_error}")
        
        # Stummes Video erstellen
        print(f"Starte Erstellung des stummen Videos für: {input_path.name}")
        video_success, video_output, video_error = create_silent_video(input_video_path, str(video_path))
        if not video_success:
            errors.append(f"Stummes Video fehlgeschlagen: {video_error}")
        
        # Original löschen falls gewünscht und beide Operationen erfolgreich
        if not keep_original and audio_success and video_success:
            try:
                input_path.unlink()
                print(f"Original-Datei gelöscht: {input_path}")
            except Exception as e:
                errors.append(f"Fehler beim Löschen der Original-Datei: {str(e)}")
        
        return {
            'success': audio_success and video_success,
            'audio_path': audio_output if audio_success else '',
            'video_path': video_output if video_success else '',
            'errors': errors
        }
        
    except Exception as e:
        return {
            'success': False,
            'audio_path': '',
            'video_path': '',
            'errors': [f"Unerwarteter Fehler: {str(e)}"]
        }

def batch_split_videos(video_paths, output_dir=None, keep_originals=True):
    """
    Splittet mehrere Videos gleichzeitig
    
    Args:
        video_paths (list): Liste von Video-Pfaden
        output_dir (str, optional): Ausgabe-Verzeichnis für alle Videos
        keep_originals (bool): Ob die ursprünglichen Dateien behalten werden sollen
    
    Returns:
        dict: Ergebnisse für jede Datei
    """
    results = {}
    
    for video_path in video_paths:
        print(f"\n--- Verarbeite: {Path(video_path).name} ---")
        result = split_video_audio(video_path, output_dir, keep_originals)
        results[video_path] = result
        
        if result['success']:
            print(f"✓ Erfolgreich aufgesplitten: {Path(video_path).name}")
        else:
            print(f"✗ Fehler bei {Path(video_path).name}: {result['errors']}")
    
    return results

def has_audio_stream(video_path):
    """
    Prüft ob eine Video-Datei eine Audio-Spur hat
    
    Args:
        video_path (str): Pfad zur Video-Datei
    
    Returns:
        bool: True wenn Audio-Spur vorhanden
    """
    try:
        command = [
            'ffprobe',
            '-v', 'quiet',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_type',
            '-of', 'csv=p=0',
            video_path
        ]
        
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return 'audio' in result.stdout.lower()
        
    except Exception:
        return False

# Beispiel für die Verwendung
if __name__ == "__main__":
    # Beispiel 1: Einzelne Datei splitten
    video_file = "example_video.mp4"
    result = split_video_audio(video_file, keep_original=True)
    
    if result['success']:
        print(f"Audio gespeichert: {result['audio_path']}")
        print(f"Stummes Video gespeichert: {result['video_path']}")
    else:
        print(f"Fehler: {result['errors']}")
    
    # Beispiel 2: Mehrere Dateien gleichzeitig
    video_files = ["video1.mp4", "video2.mp4", "video3.mp4"]
    results = batch_split_videos(video_files, output_dir="split_output", keep_originals=True)