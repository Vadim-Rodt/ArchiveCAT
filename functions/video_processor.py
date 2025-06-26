# video_processor.py
"""
Modul für Video-Verarbeitungsfunktionen wie Segmentierung und Dauer-Ermittlung
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Tuple, Optional

class VideoProcessor:
    """Klasse für Video-Verarbeitungsoperationen"""
    
    @staticmethod
    def get_video_duration(file_path: str) -> float:
        """
        Ermittelt die Dauer eines Videos in Sekunden
        
        Args:
            file_path: Pfad zur Videodatei
            
        Returns:
            Dauer in Sekunden oder 0.0 bei Fehler
        """
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
                 "-of", "default=noprint_wrappers=1:nokey=1", file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            duration = float(result.stdout.strip())
            return duration
        except Exception as e:
            print(f"Fehler beim Ermitteln der Videolänge: {e}")
            return 0.0
    
    @staticmethod
    def cut_video_segment(input_path: str, output_path: str, 
                         start_time: str, end_time: str) -> bool:
        """
        Schneidet ein Segment aus einem Video
        
        Args:
            input_path: Pfad zur Eingabedatei
            output_path: Pfad zur Ausgabedatei
            start_time: Startzeit im Format HH:MM:SS
            end_time: Endzeit im Format HH:MM:SS
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            # Berechne Dauer
            start_seconds = VideoProcessor.time_to_seconds(start_time)
            end_seconds = VideoProcessor.time_to_seconds(end_time)
            duration = str(end_seconds - start_seconds)
            
            command = [
                'ffmpeg',
                '-ss', start_time,
                '-i', input_path,
                '-t', duration,
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-strict', 'experimental',
                '-y',
                output_path
            ]
            
            print(f"Schneide Segment: {start_time} bis {end_time}")
            
            process = subprocess.Popen(
                command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                print(f"FFmpeg Fehler: {stderr}")
                return False
            
            # Prüfen ob Datei erfolgreich erstellt wurde
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"Segment erfolgreich erstellt: {output_path}")
                return True
            else:
                print(f"Segment konnte nicht erstellt werden: {output_path}")
                return False
                
        except Exception as e:
            print(f"Fehler beim Schneiden: {e}")
            return False
    
    @staticmethod
    def time_to_seconds(time_str: str) -> int:
        """Konvertiert Zeit-String (HH:MM:SS) zu Sekunden"""
        try:
            h, m, s = map(int, time_str.split(':'))
            return h * 3600 + m * 60 + s
        except:
            return 0
    
    @staticmethod
    def seconds_to_time(seconds: int) -> str:
        """Konvertiert Sekunden zu Zeit-String (HH:MM:SS)"""
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
    
    @staticmethod
    def validate_video_file(file_path: str) -> Tuple[bool, str]:
        """
        Validiert eine Video-Datei
        
        Returns:
            (is_valid, error_message)
        """
        if not os.path.exists(file_path):
            return False, "Datei nicht gefunden"
        
        if not os.path.isfile(file_path):
            return False, "Pfad ist keine Datei"
        
        # Prüfe Dateigröße
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return False, "Datei ist leer"
        
        # Prüfe ob Video lesbar ist
        duration = VideoProcessor.get_video_duration(file_path)
        if duration == 0:
            return False, "Video-Dauer konnte nicht ermittelt werden"
        
        return True, ""
    
    @staticmethod
    def get_video_info(file_path: str) -> dict:
        """
        Holt detaillierte Informationen über ein Video
        
        Returns:
            Dictionary mit Video-Informationen
        """
        info = {
            'duration': 0,
            'width': 0,
            'height': 0,
            'fps': 0,
            'codec': '',
            'bitrate': 0,
            'size': 0,
            'format': ''
        }
        
        try:
            # Hole Dateiinfo
            info['size'] = os.path.getsize(file_path)
            info['format'] = Path(file_path).suffix[1:]
            
            # Hole Video-Stream-Info
            cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,avg_frame_rate,codec_name,bit_rate',
                '-show_entries', 'format=duration',
                '-of', 'json',
                file_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                
                if 'streams' in data and data['streams']:
                    stream = data['streams'][0]
                    info['width'] = stream.get('width', 0)
                    info['height'] = stream.get('height', 0)
                    info['codec'] = stream.get('codec_name', '')
                    info['bitrate'] = int(stream.get('bit_rate', 0))
                    
                    # Parse FPS
                    fps_str = stream.get('avg_frame_rate', '0/1')
                    if '/' in fps_str:
                        num, den = map(int, fps_str.split('/'))
                        info['fps'] = num / den if den > 0 else 0
                
                if 'format' in data:
                    info['duration'] = float(data['format'].get('duration', 0))
        
        except Exception as e:
            print(f"Fehler beim Abrufen der Video-Info: {e}")
        
        return info