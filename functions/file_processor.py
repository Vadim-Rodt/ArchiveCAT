# file_processor.py
"""
Modul zur Verarbeitung lokaler Dateien und Verwaltung von Segmenten
"""

import os
import shutil
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from queue import Queue
from video_processor import VideoProcessor
from audio_video_splitter import split_video_audio
from transcription_manager import TranscriptionManager

class FileProcessor:
    """Klasse zur Verarbeitung lokaler Video-Dateien"""
    
    def __init__(self, download_dir: str, transcription_manager: Optional[TranscriptionManager] = None):
        self.download_dir = download_dir
        self.transcription_manager = transcription_manager
        self.video_processor = VideoProcessor()
    
    def process_local_file(self, file_path: str, segment_times: List[Tuple[str, str]], 
                          delete_source: bool, queue: Optional[Queue] = None,
                          transcription_settings: Optional[Dict] = None) -> bool:
        """
        Verarbeitet eine lokale Video-Datei
        
        Args:
            file_path: Pfad zur Video-Datei
            segment_times: Liste von (start_time, end_time) Tupeln
            delete_source: Ob die Quelldatei gelöscht werden soll
            queue: Queue für Status-Updates
            transcription_settings: Einstellungen für Transkription
            
        Returns:
            Erfolg
        """
        try:
            # Extrahiere Dateiinfos
            filename = os.path.basename(file_path)
            title_text = self._sanitize_filename(os.path.splitext(filename)[0])
            ext = os.path.splitext(filename)[1][1:]
            
            # Erstelle Hauptordner
            video_folder = os.path.join(self.download_dir, title_text)
            os.makedirs(video_folder, exist_ok=True)
            
            if queue:
                queue.put("status:Verarbeite lokale Datei...")
            
            # Hole Video-Dauer
            video_duration = self.video_processor.get_video_duration(file_path)
            video_duration_seconds = int(video_duration)
            
            # Segmentierung
            total_segments = len(segment_times)
            
            # Prüfe ob Gesamtvideo
            is_full_video = (total_segments == 1 and 
                           segment_times[0][0] == "00:00:00" and 
                           VideoProcessor.time_to_seconds(segment_times[0][1]) == video_duration_seconds)
            
            if is_full_video:
                success = self._process_full_video(
                    file_path, video_folder, filename, queue, transcription_settings,
                )
            else:
                success = self._process_segments(
                    file_path, video_folder, title_text, ext, segment_times, 
                    queue, transcription_settings
                )
            
            if success:
                if queue:
                    queue.put("done")
                return True
            else:
                if queue:
                    queue.put("error:Verarbeitung fehlgeschlagen")
                return False
                
        except Exception as e:
            if queue:
                queue.put(f"error:{str(e)}")
            return False
    
    def _process_full_video(self, file_path: str, video_folder: str, filename: str,
                           queue: Optional[Queue], transcription_settings: Optional[Dict]) -> bool:
        """Verarbeitet das gesamte Video ohne Segmentierung"""
        segment_folder = os.path.join(video_folder, "Gesamtvideo")
        os.makedirs(segment_folder, exist_ok=True)
        
        # Kopiere Video
        final_video_path = os.path.join(segment_folder, filename)
        
        if queue:
            queue.put("status:Kopiere Video...")
        
        shutil.copy2(file_path, final_video_path)
        
        if queue:
            queue.put("status:Extrahiere Audio...")
        
        # Audio extrahieren
        result = split_video_audio(final_video_path, output_dir=segment_folder, keep_original=True)
        
        if result['success']:
            if queue:
                queue.put("status:Audio erfolgreich extrahiert")
            
            # Transkription wenn aktiviert
            if result['audio_path'] and transcription_settings and transcription_settings.get('enabled'):
                self._perform_transcription(
                    result['audio_path'], segment_folder, queue, 
                    transcription_settings, ""
                )
            
            return True
        else:
            print(f"Fehler bei Audio-Extraktion: {result['errors']}")
            if queue:
                queue.put("status:Audio-Extraktion fehlgeschlagen")
            return False
    
    def _process_segments(self, file_path: str, video_folder: str, title_text: str, 
                         ext: str, segment_times: List[Tuple[str, str]], 
                         queue: Optional[Queue], transcription_settings: Optional[Dict]) -> bool:
        """Verarbeitet Video in Segmenten"""
        successful_segments = 0
        total_segments = len(segment_times)
        all_audio_paths = []
        
        for i, (start_time, end_time) in enumerate(segment_times, 1):
            if queue:
                queue.put(f"status:Erstelle Segment {i}/{total_segments}...")
            
            # Erstelle Segment-Ordner
            segment_folder = os.path.join(video_folder, f"Segment_{i}")
            os.makedirs(segment_folder, exist_ok=True)
            
            # Segment-Dateiname
            segment_filename = f"{title_text}_Segment_{i}.{ext}"
            segment_path = os.path.join(segment_folder, segment_filename)
            
            # Schneide Segment
            if self.video_processor.cut_video_segment(file_path, segment_path, start_time, end_time):
                if queue:
                    queue.put(f"status:Extrahiere Audio für Segment {i}...")
                
                # Audio extrahieren
                result = split_video_audio(segment_path, output_dir=segment_folder, keep_original=True)
                
                if result['success']:
                    successful_segments += 1
                    if queue:
                        queue.put(f"status:Segment {i} mit Audio erfolgreich erstellt")
                    
                    if result['audio_path']:
                        all_audio_paths.append((i, result['audio_path'], segment_folder))
                        
                        # Transkription wenn aktiviert
                        if transcription_settings and transcription_settings.get('enabled'):
                            self._perform_transcription(
                                result['audio_path'], segment_folder, queue,
                                transcription_settings, f"_segment_{i}"
                            )
                else:
                    print(f"Fehler bei Audio-Extraktion für Segment {i}")
                    successful_segments += 1
            else:
                print(f"Segment {i} konnte nicht erstellt werden")
        
        # Erstelle Gesamttranskript wenn mehrere Segmente
        if len(all_audio_paths) > 1 and transcription_settings and transcription_settings.get('enabled'):
            if queue:
                queue.put("status:Erstelle Gesamttranskript...")
            
            if self.transcription_manager:
                self.transcription_manager.create_combined_transcript(
                    video_folder, transcription_settings.get('use_speakers', False)
                )
        
        if queue:
            queue.put(f"status:Fertig! {successful_segments}/{total_segments} Segmente erstellt")
        
        return successful_segments == total_segments
    
    def _perform_transcription(self, audio_path: str, output_folder: str, 
                              queue: Optional[Queue], settings: Dict, segment_name: str = ""):
        """Führt Transkription durch"""
        if not self.transcription_manager or not self.transcription_manager.is_initialized:
            return
        
        if queue:
            queue.put(f"status:Transkribiere{' Segment' if segment_name else ''}...")
        
        try:
            # Transkribiere
            result = self.transcription_manager.transcribe_audio(
                audio_path,
                language=settings.get('language'),
                use_speakers=settings.get('use_speakers', False),
                speaker_params=settings.get('speaker_params')
            )
            
            if result['success']:
                # Speichere Transkription
                output_path = os.path.join(output_folder, f"transkript{segment_name}")
                self.transcription_manager.save_transcription(
                    result, output_path,
                    use_speakers=settings.get('use_speakers', False),
                    export_formats=settings.get('export_formats')
                )
                
                if queue:
                    if settings.get('use_speakers') and 'speaker_count' in result:
                        queue.put(f"status:Transkription mit {result['speaker_count']} Sprechern erstellt")
                    else:
                        queue.put(f"status:Transkription{segment_name} erstellt")

            else:
                if queue:
                    queue.put(f"status:Transkription fehlgeschlagen: {result.get('error', 'Unbekannter Fehler')}")
                    
        except Exception as e:
            print(f"Transkriptionsfehler: {e}")
            if queue:
                queue.put(f"status:Transkription fehlgeschlagen: {str(e)}")

    
    def _sanitize_filename(self, name: str, max_length: int = 50) -> str:
        """Bereinigt Dateinamen"""
        import re
        name = re.sub(r'[\\/*?:"<>|]', '', name).strip()
        return name[:max_length]
    
    def validate_segments(self, segment_times: List[Tuple[str, str]], 
                         video_duration: int) -> Tuple[bool, str]:
        """
        Validiert Segment-Zeiten
        
        Returns:
            (is_valid, error_message)
        """
        for i, (start_time, end_time) in enumerate(segment_times, 1):
            start_seconds = VideoProcessor.time_to_seconds(start_time)
            end_seconds = VideoProcessor.time_to_seconds(end_time)
            
            if start_seconds >= end_seconds:
                return False, f"Segment {i}: Startzeit muss vor Endzeit liegen"
            
            if end_seconds > video_duration:
                return False, f"Segment {i}: Endzeit überschreitet Videolänge"
            
            # Prüfe Überlappungen
            for j, (other_start, other_end) in enumerate(segment_times[:i-1], 1):
                other_start_sec = VideoProcessor.time_to_seconds(other_start)
                other_end_sec = VideoProcessor.time_to_seconds(other_end)
                
                if (start_seconds < other_end_sec and end_seconds > other_start_sec):
                    return False, f"Segment {i} überlappt mit Segment {j}"
        
        return True, ""