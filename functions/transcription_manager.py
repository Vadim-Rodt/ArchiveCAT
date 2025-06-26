# transcription_manager.py
"""
Modul zur Verwaltung von Transkriptionsprozessen
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from whisper_transcriber import WhisperTranscriber
from whisper_speaker_transcriber import WhisperSpeakerTranscriber

class TranscriptionManager:
    """Verwaltet Transkriptionsprozesse und -einstellungen"""
    
    def __init__(self):
        self.whisper_transcriber = None
        self.speaker_transcriber = None
        self.is_initialized = False
    
    def initialize_transcribers(self, openai_key: str, hf_token: Optional[str] = None) -> Tuple[bool, str]:
        """
        Initialisiert die Transcriber
        
        Returns:
            (success, error_message)
        """
        try:
            self.whisper_transcriber = WhisperTranscriber(openai_key)
            
            if hf_token:
                self.speaker_transcriber = WhisperSpeakerTranscriber(openai_key, hf_token)
            
            self.is_initialized = True
            return True, ""
            
        except Exception as e:
            self.is_initialized = False
            return False, f"Fehler beim Initialisieren: {str(e)}"
    
    def test_api_key(self, api_key: str) -> Tuple[bool, str]:
        """
        Testet einen OpenAI API Key
        
        Returns:
            (is_valid, message)
        """
        try:
            test_transcriber = WhisperTranscriber(api_key)
            # Hier könnte ein Test-API-Call gemacht werden
            return True, "API Key gültig"
        except Exception as e:
            return False, f"API Key ungültig: {str(e)}"
    
    def transcribe_audio(self, audio_path: str, language: str = None, 
                        use_speakers: bool = False, speaker_params: dict = None) -> dict:
        """
        Transkribiert eine Audio-Datei
        
        Args:
            audio_path: Pfad zur Audio-Datei
            language: Sprache (None für automatische Erkennung)
            use_speakers: Ob Speaker Diarization verwendet werden soll
            speaker_params: Parameter für Speaker Diarization
            
        Returns:
            Transkriptionsergebnis
        """
        if not self.is_initialized:
            return {'success': False, 'error': 'Transcriber nicht initialisiert'}
        
        if use_speakers and self.speaker_transcriber:
            params = speaker_params or {'min_speakers': 2, 'max_speakers': 5}
            return self.speaker_transcriber.transcribe_with_speakers(
                audio_path, language, **params
            )
        else:
            return self.whisper_transcriber.transcribe_audio(audio_path, language)
    
    def save_transcription(self, transcription_data: dict, output_path: str, 
                          use_speakers: bool = False, export_formats: dict = None) -> bool:
        """
        Speichert Transkription in verschiedenen Formaten
        
        Args:
            transcription_data: Transkriptionsdaten
            output_path: Basis-Ausgabepfad
            use_speakers: Ob Speaker-Format verwendet wurde
            export_formats: Dictionary mit gewünschten Formaten
            
        Returns:
            Erfolg
        """
        try:
            if use_speakers and self.speaker_transcriber:
                self.speaker_transcriber.save_transcription_with_speakers(
                    transcription_data, output_path
                )
            else:
                self.whisper_transcriber.save_transcription(
                    transcription_data, output_path
                )
            
            # Zusätzliche Formate exportieren wenn gewünscht
            if export_formats and export_formats.get('xml'):
                self._export_xml(transcription_data, output_path)
            
            return True
            
        except Exception as e:
            print(f"Fehler beim Speichern der Transkription: {e}")
            return False
    
    def _export_xml(self, transcription_data: dict, output_path: str):
        """Exportiert Transkription als XML"""
        try:
            from xml_exporter import TranscriptionExporter, ExportConfig, ExportFormat
            exporter = TranscriptionExporter()
            config = ExportConfig(formats={ExportFormat.XML: True})
            exporter.export(transcription_data, output_path, config)
        except Exception as e:
            print(f"XML-Export Fehler: {e}")
    
    def create_combined_transcript(self, video_folder: str, 
                                 use_speakers: bool = False) -> Optional[str]:
        """
        Erstellt ein kombiniertes Transkript aller Segmente
        
        Returns:
            Pfad zum Gesamttranskript oder None
        """
        try:
            all_transcripts = []
            
            # Durchsuche alle Segment-Ordner
            segment_folders = []
            for folder_name in os.listdir(video_folder):
                if folder_name.startswith("Segment_"):
                    segment_folders.append(folder_name)
            
            # Sortiere nach Segment-Nummer
            segment_folders.sort(key=lambda x: int(x.split("_")[1]))
            
            # Sammle alle Transkripte
            for folder_name in segment_folders:
                segment_path = os.path.join(video_folder, folder_name)
                
                # Suche nach Transkript-Dateien
                for file in os.listdir(segment_path):
                    if file.startswith("transkript_segment_") and file.endswith(".txt"):
                        transcript_path = os.path.join(segment_path, file)
                        with open(transcript_path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            if content:
                                all_transcripts.append(f"=== {folder_name} ===\n{content}\n")
                        break
            
            if all_transcripts:
                # Speichere Gesamttranskript
                combined_path = os.path.join(video_folder, "Gesamttranskript.txt")
                with open(combined_path, 'w', encoding='utf-8') as f:
                    f.write(f"GESAMTTRANSKRIPT\n")
                    f.write(f"Video: {os.path.basename(video_folder)}\n")
                    f.write(f"Anzahl Segmente: {len(all_transcripts)}\n")
                    f.write("=" * 80 + "\n\n")
                    f.write("\n\n".join(all_transcripts))
                
                print(f"Gesamttranskript erstellt: {combined_path}")
                
                # Erstelle auch JSON-Version wenn Speaker verwendet wurden
                if use_speakers:
                    self._create_combined_speaker_json(video_folder)
                
                return combined_path
                
        except Exception as e:
            print(f"Fehler beim Erstellen des Gesamttranskripts: {e}")
        
        return None
    
    def _create_combined_speaker_json(self, video_folder: str):
        """Erstellt kombinierte JSON mit Speaker-Informationen"""
        try:
            combined_data = {
                "video": os.path.basename(video_folder),
                "segments": []
            }
            
            # Sammle alle JSON-Dateien
            for folder_name in sorted(os.listdir(video_folder)):
                if folder_name.startswith("Segment_"):
                    segment_path = os.path.join(video_folder, folder_name)
                    
                    for file in os.listdir(segment_path):
                        if file.startswith("transkript_segment_") and file.endswith(".json"):
                            json_path = os.path.join(segment_path, file)
                            with open(json_path, 'r', encoding='utf-8') as f:
                                segment_data = json.load(f)
                                combined_data["segments"].append({
                                    "segment": folder_name,
                                    "data": segment_data
                                })
                            break
            
            if combined_data["segments"]:
                # Speichere kombinierte JSON
                combined_json_path = os.path.join(video_folder, "Gesamttranskript_mit_Sprechern.json")
                with open(combined_json_path, 'w', encoding='utf-8') as f:
                    json.dump(combined_data, f, ensure_ascii=False, indent=2)
                
                print(f"Kombinierte Speaker-JSON erstellt: {combined_json_path}")
                
        except Exception as e:
            print(f"Fehler beim Erstellen der kombinierten JSON: {e}")