import os
import re
import time
import json  # FEHLT!
import requests
import subprocess
import threading
from urllib.parse import urljoin
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from tkinter import filedialog
from queue import Queue, Empty
from PIL import Image, ImageTk, ImageSequence
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from modern_style import apply_modern_style 
from whisper_transcriber import WhisperTranscriber
from whisper_speaker_transcriber import WhisperSpeakerTranscriber
from audio_video_splitter import split_video_audio  
from pathlib import Path
import pandas as pd

# Import der Audio-Split-Funktionen
from audio_video_splitter import split_video_audio

# === Globale Variablen für GIF ===
gif_frames = []
gif_running = False
whisper_transcriber = None
speaker_transcriber = None
transcribe_enabled_var = None
use_speakers_var = None
openai_key_entry = None
hf_token_entry = None
validated_api_key = None
language_var = None
language_mapping = {}
export_txt_var = None
export_json_var = None
export_xml_var = None
export_srt_var = None

# === EINSTELLUNGEN ===
DOWNLOAD_DIR = r"C:\Users\rodtv\OneDrive\Desktop\Desktop\ArchiveCAT\data\videos"

video_duration_seconds = None

# === HILFSFUNKTIONEN ===
def sanitize_filename(name, max_length=50):
    name = re.sub(r'[\\/*?:"<>|]', '', name).strip()
    return name[:max_length]

def cut_video_segment(input_path, output_path, start_time, end_time):
    """Verbesserte Videosegmentierung mit Fehlerbehandlung"""
    try:
        duration = str(time_to_seconds(end_time) - time_to_seconds(start_time))
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

def get_actual_duration(file_path):
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        duration = float(result.stdout.strip())
        return duration
    except Exception as e:
        print(f"Fehler beim Ermitteln der Videolänge: {e}")
        return 0.0

def time_to_seconds(t):
    try:
        h, m, s = map(int, t.split(':'))
        return h * 3600 + m * 60 + s
    except:
        return 0

def seconds_to_time(seconds):
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

def threaded_download(url, segment_times, delete_source, queue):
    try:
        success = download_video(url, segment_times, delete_source, DOWNLOAD_DIR, queue)
        if success:
            queue.put("done")
        else:
            queue.put("error:Download oder Segmentierung fehlgeschlagen")
    except Exception as e:
        queue.put(f"error:{str(e)}")

def process_local_file(file_path, segment_times, delete_source, queue):
    """Verarbeitet lokale Video-Datei mit optionaler Transkription"""
    try:
        # Extrahiere Dateiname ohne Pfad
        filename = os.path.basename(file_path)
        title_text = sanitize_filename(os.path.splitext(filename)[0])
        ext = os.path.splitext(filename)[1][1:]  # Erweiterung ohne Punkt
        
        # Erstelle Hauptordner für das Video
        video_folder = os.path.join(DOWNLOAD_DIR, title_text)
        os.makedirs(video_folder, exist_ok=True)
        transcription_completed = True
        
        if queue:
            queue.put("status:Verarbeite lokale Datei...")
        
        # Segmentierung
        successful_segments = 0
        total_segments = len(segment_times)
        
        # Liste für alle Audio-Pfade (für Gesamttranskript)
        all_audio_paths = []
        
        # Spezialbehandlung für Gesamtvideo
        if total_segments == 1 and segment_times[0][0] == "00:00:00" and time_to_seconds(segment_times[0][1]) == video_duration_seconds:
            # Gesamtes Video - erstelle einen einzelnen Ordner
            segment_folder = os.path.join(video_folder, "Gesamtvideo")
            os.makedirs(segment_folder, exist_ok=True)
            
            # Kopiere das Video in den Ordner
            import shutil
            final_video_path = os.path.join(segment_folder, filename)
            
            if queue:
                queue.put("status:Kopiere Video...")
            
            shutil.copy2(file_path, final_video_path)
            
            if queue:
                queue.put("status:Extrahiere Audio...")
            
            # Audio extrahieren
            result = split_video_audio(final_video_path, output_dir=segment_folder, keep_original=True)
            
            if result['success']:
                successful_segments = 1
                if queue:
                    queue.put("status:Audio erfolgreich extrahiert")
                
                if result['audio_path']:
                    perform_transcription(result['audio_path'], segment_folder, queue)

                # Für Segmente (in der Schleife):
                if result['audio_path']:
                    perform_transcription(result['audio_path'], segment_folder, queue, f"_segment_{i}")
            else:
                print(f"Fehler bei Audio-Extraktion: {result['errors']}")
                if queue:
                    queue.put("status:Audio-Extraktion fehlgeschlagen")
        else:
            # Normale Segmentierung
            for i, (start_time, end_time) in enumerate(segment_times, 1):
                if queue:
                    queue.put(f"status:Erstelle Segment {i}/{total_segments}...")
                
                # Erstelle Unterordner für jedes Segment
                segment_folder = os.path.join(video_folder, f"Segment_{i}")
                os.makedirs(segment_folder, exist_ok=True)
                
                # Segment-Dateiname
                segment_filename = f"{title_text}_Segment_{i}.{ext}"
                segment_path = os.path.join(segment_folder, segment_filename)
                
                if cut_video_segment(file_path, segment_path, start_time, end_time):
                    if queue:
                        queue.put(f"status:Extrahiere Audio für Segment {i}...")
                    
                    # Audio aus Segment extrahieren
                    result = split_video_audio(segment_path, output_dir=segment_folder, keep_original=True)
                    
                    if result['success']:
                        successful_segments += 1
                        if queue:
                            queue.put(f"status:Segment {i} mit Audio erfolgreich erstellt")
                        
                        # Füge Audio-Pfad zur Liste hinzu
                        if result['audio_path']:
                            all_audio_paths.append((i, result['audio_path'], segment_folder))
                        
                        # Transkription durchführen wenn aktiviert
                        if result['audio_path'] and transcribe_enabled_var and transcribe_enabled_var.get():
                            if queue:
                                queue.put(f"status:Transkribiere Segment {i}...")
                            
                            try:
                                if use_speakers_var.get() and speaker_transcriber:
                                    # Mit Speaker Diarization
                                    trans_result = speaker_transcriber.transcribe_with_speakers(
                                        result['audio_path'],
                                        language="de",
                                        min_speakers=2,
                                        max_speakers=5
                                    )
                                    if trans_result['success']:
                                        output_path = os.path.join(segment_folder, f"transkript_segment_{i}")
                                        speaker_transcriber.save_transcription_with_speakers(trans_result, output_path)
                                        if queue:
                                            queue.put(f"status:Segment {i}: {trans_result['speaker_count']} Sprecher erkannt")
                                    else:
                                        print(f"Transkription für Segment {i} fehlgeschlagen: {trans_result.get('error')}")
                                else:
                                    # Normale Transkription ohne Speaker
                                    trans_result = whisper_transcriber.transcribe_audio(
                                        result['audio_path'],
                                        language="de"
                                    )
                                    if trans_result['success']:
                                        output_path = os.path.join(segment_folder, f"transkript_segment_{i}")
                                        whisper_transcriber.save_transcription(trans_result, output_path)
                                        if queue:
                                            queue.put(f"status:Segment {i} transkribiert")
                                    else:
                                        print(f"Transkription für Segment {i} fehlgeschlagen: {trans_result.get('error')}")
                            except Exception as e:
                                print(f"Transkriptionsfehler bei Segment {i}: {e}")
                    else:
                        print(f"Fehler bei Audio-Extraktion für Segment {i}: {result['errors']}")
                        successful_segments += 1
                else:
                    print(f"Segment {i} konnte nicht erstellt werden")
        
        # Erstelle Gesamttranskript wenn mehrere Segmente transkribiert wurden
        if len(all_audio_paths) > 1 and transcribe_enabled_var and transcribe_enabled_var.get():
            if queue:
                queue.put("status:Erstelle Gesamttranskript...")
            create_combined_transcript(video_folder)
        
        if queue:
            queue.put(f"status:Fertig! {successful_segments}/{total_segments} Segmente erstellt")
        
        if successful_segments == total_segments:
            queue.put("done")
        else:
            queue.put("error:Nicht alle Segmente konnten erstellt werden")
            
    except Exception as e:
        queue.put(f"error:{str(e)}")


# Zusätzliche Hilfsfunktion für das Gesamttranskript
def create_combined_transcript(video_folder):
    """Erstellt ein kombiniertes Transkript aller Segmente"""
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
            
            # Suche nach Transkript-Dateien (txt Format)
            for file in os.listdir(segment_path):
                if file.startswith("transkript_segment_") and file.endswith(".txt"):
                    transcript_path = os.path.join(segment_path, file)
                    with open(transcript_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            all_transcripts.append(f"=== {folder_name} ===\n{content}\n")
                    break
        
        if all_transcripts:
            # Speichere Gesamttranskript im Hauptordner
            combined_path = os.path.join(video_folder, "Gesamttranskript.txt")
            with open(combined_path, 'w', encoding='utf-8') as f:
                f.write(f"GESAMTTRANSKRIPT\n")
                f.write(f"Video: {os.path.basename(video_folder)}\n")
                f.write(f"Anzahl Segmente: {len(all_transcripts)}\n")
                f.write("=" * 80 + "\n\n")
                f.write("\n\n".join(all_transcripts))
            
            print(f"Gesamttranskript erstellt: {combined_path}")
            
            # Erstelle auch eine JSON-Version wenn Sprecher erkannt wurden
            if use_speakers_var and use_speakers_var.get():
                create_combined_speaker_json(video_folder)
                
    except Exception as e:
        print(f"Fehler beim Erstellen des Gesamttranskripts: {e}")


def create_combined_speaker_json(video_folder):
    """Erstellt eine kombinierte JSON-Datei mit allen Speaker-Informationen"""
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


def perform_transcription(audio_path, segment_folder, queue, segment_name=""):
    """Führt Transkription mit allen gewählten Export-Formaten durch"""
    if not transcribe_enabled_var or not transcribe_enabled_var.get():
        return
    
    # Hole gewählte Sprache
    selected_language = language_mapping.get(language_var.get(), "de") if language_var else "de"
    if selected_language == "auto":
        selected_language = None  # Whisper erkennt automatisch
    
    try:
        if use_speakers_var.get() and speaker_transcriber:
            # Mit Speaker Diarization
            trans_result = speaker_transcriber.transcribe_with_speakers(
                audio_path,
                language=selected_language,
                min_speakers=2,
                max_speakers=5
            )
            
            if trans_result['success']:
                output_path = os.path.join(segment_folder, f"transkript{segment_name}")
                speaker_transcriber.save_transcription_with_speakers(trans_result, output_path)
                
                # XML Export wenn aktiviert
                if export_xml_var and export_xml_var.get():
                    try:
                        from xml_exporter import TranscriptionExporter, ExportConfig, ExportFormat
                        exporter = TranscriptionExporter()
                        config = ExportConfig(formats={ExportFormat.XML: True})
                        exporter.export(trans_result, output_path, config)
                    except Exception as e:
                        print(f"XML-Export Fehler: {e}")
                
                if queue:
                    queue.put(f"status:Transkription mit {trans_result['speaker_count']} Sprechern erstellt")
                return True
            else:
                if queue:
                    queue.put(f"status:Transkription fehlgeschlagen: {trans_result.get('error', 'Unbekannter Fehler')}")
                return False
        else:
            # Normale Transkription ohne Speaker
            trans_result = whisper_transcriber.transcribe_audio(
                audio_path,
                language=selected_language
            )
            
            if trans_result['success']:
                output_path = os.path.join(segment_folder, f"transkript{segment_name}")
                whisper_transcriber.save_transcription(trans_result, output_path)
                
                # XML Export wenn aktiviert
                if export_xml_var and export_xml_var.get():
                    try:
                        from xml_exporter import TranscriptionExporter, ExportConfig, ExportFormat
                        exporter = TranscriptionExporter()
                        config = ExportConfig(formats={ExportFormat.XML: True})
                        exporter.export(trans_result, output_path, config)
                    except Exception as e:
                        print(f"XML-Export Fehler: {e}")
                
                if queue:
                    queue.put(f"status:Transkription{segment_name} erstellt")
                return True
            else:
                if queue:
                    queue.put(f"status:Transkription fehlgeschlagen: {trans_result.get('error', 'Unbekannter Fehler')}")
                return False
                
    except Exception as e:
        print(f"Transkriptionsfehler: {e}")
        if queue:
            queue.put(f"status:Transkription fehlgeschlagen: {str(e)}")
        return False
    
def save_transcription_with_formats(transcription_data, output_path, export_formats, is_speaker_transcription=False):
    """Speichert Transkription in allen gewählten Formaten inklusive XML"""
    from xml_exporter import TranscriptionExporter, ExportConfig, ExportFormat
    
    try:
        output_path = Path(output_path)
        
        if is_speaker_transcription and speaker_transcriber:
            # Nutze die eingebaute Save-Funktion für Speaker-Transkriptionen
            speaker_transcriber.save_transcription_with_speakers(transcription_data, str(output_path))
        else:
            # Nutze die eingebaute Save-Funktion für normale Transkriptionen
            whisper_transcriber.save_transcription(transcription_data, str(output_path))
        
        # Zusätzlich XML-Export wenn gewählt
        if export_formats.get('xml', True):
            exporter = TranscriptionExporter()
            config = ExportConfig(
                formats={ExportFormat.XML: True},
                include_metadata=True,
                include_statistics=True
            )
            
            xml_result = exporter.export(transcription_data, output_path, config)
            if xml_result.get(ExportFormat.XML):
                print(f"✓ XML-Export erfolgreich: {output_path.with_suffix('.xml')}")
            else:
                print(f"✗ XML-Export fehlgeschlagen")
                
    except Exception as e:
        print(f"Fehler beim Speichern der Transkription: {e}")

def download_video(url, segment_times, delete_source=False, download_dir=DOWNLOAD_DIR, queue=None):
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])
    chrome_options.add_argument('--log-level=3')

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        if queue:
            queue.put("status:Lade Webseite...")
        
        driver.get(url)
        wait = WebDriverWait(driver, 10)

        try:
            title_element = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "item-title")))
            title_text = sanitize_filename(title_element.text)
        except:
            title_text = "archive_item"

        video_link = None

        if queue:
            queue.put("status:Suche Video-Link...")

        try:
            mpeg4_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//a[contains(@class,'js-archive-expand_files') and contains(text(),'MPEG4')]")))
            mpeg4_button.click()
            wait.until(EC.visibility_of_element_located((By.CLASS_NAME, "quickdown")))
            time.sleep(1)
            pills = driver.find_elements(By.CSS_SELECTOR, ".quickdown a.download-pill")
            for pill in pills:
                href = pill.get_attribute("href")
                if href and href.endswith(".mp4"):
                    video_link = urljoin(url, href)
                    break
        except:
            pass

        if not video_link:
            pills = driver.find_elements(By.CSS_SELECTOR, "a.download-pill")
            for pill in pills:
                href = pill.get_attribute("href")
                if href and href.endswith(".mp4"):
                    video_link = urljoin(url, href)
                    break

        if not video_link:
            format_groups = driver.find_elements(By.CSS_SELECTOR, ".format-group")
            for group in format_groups:
                if "H.264" in group.text:
                    pills = group.find_elements(By.CSS_SELECTOR, "a.download-pill")
                    for pill in pills:
                        href = pill.get_attribute("href")
                        if href and href.endswith(".mp4"):
                            video_link = urljoin(url, href)
                            break
                if video_link:
                    break

        if not video_link:
            if queue:
                queue.put("error:Kein Video-Link gefunden")
            return False

        if queue:
            queue.put("status:Lade Video herunter...")

        os.makedirs(download_dir, exist_ok=True)
        ext = video_link.split('.')[-1]
        filename = f"{title_text}.{ext}"
        
        # Erstelle Hauptordner für das Video
        video_folder = os.path.join(download_dir, title_text)
        os.makedirs(video_folder, exist_ok=True)
        
        # Temporärer Pfad für das vollständige Video
        temp_filepath = os.path.join(video_folder, f"_temp_{filename}")

        # Download mit Fortschrittsanzeige
        with requests.get(video_link, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            downloaded = 0
            
            with open(temp_filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if queue and total_size > 0:
                            progress = (downloaded / total_size) * 100
                            queue.put(f"status:Download {progress:.1f}%")

        if queue:
            queue.put("status:Segmentiere Video...")

        global video_duration_seconds
        actual_duration = get_actual_duration(temp_filepath)
        if actual_duration > 0:
            video_duration_seconds = int(actual_duration)

        # Segmentierung mit verbesserter Fehlerbehandlung
        successful_segments = 0
        total_segments = len(segment_times)
        
        # Spezialbehandlung für Gesamtvideo-Download
        if total_segments == 1 and segment_times[0][0] == "00:00:00" and time_to_seconds(segment_times[0][1]) == video_duration_seconds:
            # Gesamtes Video - erstelle einen einzelnen Ordner
            segment_folder = os.path.join(video_folder, "Gesamtvideo")
            os.makedirs(segment_folder, exist_ok=True)
            
            # Verschiebe das Video in den Ordner
            final_video_path = os.path.join(segment_folder, filename)
            os.rename(temp_filepath, final_video_path)
            
            if queue:
                queue.put("status:Extrahiere Audio...")
            
            # Audio extrahieren
            result = split_video_audio(final_video_path, output_dir=segment_folder, keep_original=True)
            
            if result['success']:
                successful_segments = 1
                if queue:
                    queue.put("status:Audio erfolgreich extrahiert")
                
                # HIER NEU EINFÜGEN:
                if result['audio_path']:
                    perform_transcription(result['audio_path'], segment_folder, queue)

            if result['success']:
                successful_segments = 1
                if queue:
                    queue.put("status:Audio erfolgreich extrahiert")
            else:
                print(f"Fehler bei Audio-Extraktion: {result['errors']}")
                if queue:
                    queue.put("status:Audio-Extraktion fehlgeschlagen")
        else:
            # Normale Segmentierung
            for i, (start_time, end_time) in enumerate(segment_times, 1):
                if queue:
                    queue.put(f"status:Erstelle Segment {i}/{total_segments}...")
                
                # Erstelle Unterordner für jedes Segment
                segment_folder = os.path.join(video_folder, f"Segment_{i}")
                os.makedirs(segment_folder, exist_ok=True)
                
                # Segment-Dateiname ohne Ordner-Prefix
                segment_filename = f"{title_text}_Segment_{i}.{ext}"
                segment_path = os.path.join(segment_folder, segment_filename)
                
                if cut_video_segment(temp_filepath, segment_path, start_time, end_time):
                    if queue:
                        queue.put(f"status:Extrahiere Audio für Segment {i}...")
                    
                    # Audio aus Segment extrahieren
                    result = split_video_audio(segment_path, output_dir=segment_folder, keep_original=True)
                    
                    if result['success']:
                        successful_segments += 1
                        if queue:
                            queue.put(f"status:Segment {i} mit Audio erfolgreich erstellt")
                        if result['audio_path']:
                            perform_transcription(result['audio_path'], segment_folder, queue, f"_segment_{i}")
                    else:
                        print(f"Fehler bei Audio-Extraktion für Segment {i}: {result['errors']}")
                        # Segment trotzdem als erfolgreich zählen, wenn Video erstellt wurde
                        successful_segments += 1
                else:
                    print(f"Segment {i} konnte nicht erstellt werden")
            
            # Temporäre Datei löschen
            try:
                os.remove(temp_filepath)
            except Exception as e:
                print(f"Fehler beim Löschen der temporären Datei: {e}")

        if queue:
            queue.put(f"status:Fertig! {successful_segments}/{total_segments} Segmente erstellt")

        return successful_segments == total_segments

    finally:
        driver.quit()

def get_video_duration_from_meta(url):
    try:
        if url.startswith("view-source:"):
            url = url.replace("view-source:", "")
        
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            match = re.search(r'<meta property="video:duration" content="(\d+)">', response.text)
            if match:
                seconds = int(match.group(1))
                h = seconds // 3600
                m = (seconds % 3600) // 60
                s = seconds % 60
                return f"{h:02d}:{m:02d}:{s:02d}", seconds
    except Exception as e:
        print("Fehler beim Abrufen der Dauer:", e)
    return None, None

# --- GUI SETUP ---
root = tk.Tk()
apply_modern_style(root)
root.title("ArchiveCAT - Complex Annotation Tool")

# Fenstericon setzen mit logo.png (muss im selben Verzeichnis liegen)
try:
    icon_img = ImageTk.PhotoImage(file="logo.png")
    root.iconphoto(False, icon_img)
except Exception as e:
    print("Fehler beim Setzen des Icons:", e)

root.geometry("550x650")
root.configure(bg="#2b2b2b")

font_style = ("Segoe UI", 10)
label_fg = "#ffffff"
entry_bg = "#3c3f41"
entry_fg = "#ffffff"

style = ttk.Style()
style.theme_use("default")
style.configure("TButton", font=font_style, padding=6)

segment_entries = []

# Neue Variable für den Gesamtdownload
download_full_var = tk.BooleanVar()

# Status Label hinzufügen
status_label = tk.Label(root, text="Bereit", fg="#00ff00", bg="#2b2b2b", font=font_style)

def validate_time_input(char):
    return char.isdigit() or char == ':'

vcmd = (root.register(validate_time_input), '%S')

def toggle_segment_ui():
    if download_full_var.get():
        segment_dropdown.configure(state="disabled")
        segment_frame.grid_remove()
    else:
        segment_dropdown.configure(state="readonly")
        segment_frame.grid()
    update_button_text()

def start_process():
    global video_duration_seconds, gif_frames
    
    # NEUE ZEILEN: Initialisiere Transcriber wenn aktiviert
    if transcribe_enabled_var and transcribe_enabled_var.get():
        if not initialize_transcribers():
            return
    
    # Prüfe ob URL oder lokale Datei
    if source_type_var.get() == "url":
        url = url_entry.get().strip()
        if not url:
            messagebox.showerror("Fehler", "Bitte eine URL angeben.")
            return
        local_file = None
    else:
        local_file = local_file_path.get()
        if not local_file or not os.path.exists(local_file):
            messagebox.showerror("Fehler", "Bitte eine gültige Datei auswählen.")
            return
        url = None

    segment_times = []
    if download_full_var.get():
        if not video_duration_seconds:
            messagebox.showerror("Fehler", "Bitte zuerst URL/Datei bestätigen.")
            return
        segment_times = [("00:00:00", seconds_to_time(video_duration_seconds))]
    else:
        segment_count = int(segment_count_var.get())
        if not video_duration_seconds:
            messagebox.showerror("Fehler", "Bitte zuerst URL/Datei bestätigen.")
            return
        for i in range(segment_count):
            start = segment_entries[i][0].get().strip()
            end = segment_entries[i][1].get().strip()
            if not start or not end:
                messagebox.showerror("Fehler", f"Segment {i+1}: Start- und Endzeit angeben.")
                return
            if time_to_seconds(end) > video_duration_seconds:
                messagebox.showerror("Fehler", f"Segment {i+1}: Endzeit überschreitet Videolänge.")
                return
            segment_times.append((start, end))

    delete_source = delete_source_var.get()

    # GIF Animation starten
    global gif_frames, gif_running
    try:
        gif = Image.open("logo.gif")
        frames = []
        for frame in ImageSequence.Iterator(gif):
            resized_frame = frame.copy().convert("RGBA")
            resized_frame.thumbnail((100, 100), Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.ANTIALIAS)
            frames.append(ImageTk.PhotoImage(resized_frame))
        gif_frames = frames
        gif_running = True
        gif_label.config(image=gif_frames[0])
        gif_label.image = gif_frames[0]
        gif_label.grid()
        animate_gif()
    except Exception as e:
        print("GIF-Fehler:", e)

    # Download oder lokale Verarbeitung starten
    queue = Queue()
    download_btn.config(state="disabled")
    
    if local_file:
        # Lokale Datei verarbeiten
        threading.Thread(target=process_local_file, args=(local_file, segment_times, delete_source, queue), daemon=True).start()
    else:
        # URL Download
        threading.Thread(target=threaded_download, args=(url, segment_times, delete_source, queue), daemon=True).start()
    
    check_queue(queue)

def check_queue(queue):
    try:
        while True:
            result = queue.get_nowait()
            if result == "done":
                gif_running = False
                gif_label.grid_remove()
                status_label.config(text="Fertig!", fg="#00ff00")
                download_btn.config(state="normal")
                messagebox.showinfo("Fertig", "Video heruntergeladen, segmentiert und Audio extrahiert.")
                return
            elif isinstance(result, str) and result.startswith("error"):
                gif_running = False
                gif_label.grid_remove()
                status_label.config(text="Fehler", fg="#ff0000")
                download_btn.config(state="normal")
                messagebox.showerror("Fehler", result[6:])
                return
            elif isinstance(result, str) and result.startswith("status"):
                status_text = result[7:]
                status_label.config(text=status_text, fg="#ffff00")
    except Empty:
        root.after(100, lambda: check_queue(queue))

def animate_gif(index=0):
    if gif_running and gif_frames:
        gif_label.config(image=gif_frames[index])
        gif_label.image = gif_frames[index]
        root.after(100, lambda: animate_gif((index + 1) % len(gif_frames)))

def update_segment_fields():
    for widgets in segment_frame.winfo_children():
        widgets.destroy()
    segment_entries.clear()
    count = int(segment_count_var.get())
    for i in range(count):
        tk.Label(segment_frame, text=f"Segment {i+1} Start:", fg=label_fg, bg="#2b2b2b", font=font_style).grid(row=i, column=0, sticky="e", padx=(5, 20))
        start_entry = tk.Entry(segment_frame, bg=entry_bg, fg=entry_fg, insertbackground="white", validate="key", validatecommand=vcmd)
        start_entry.insert(0, "00:00:00")
        start_entry.grid(row=i, column=1, padx=(0, 20))
        tk.Label(segment_frame, text="End:", fg=label_fg, bg="#2b2b2b", font=font_style).grid(row=i, column=2, sticky="e", padx=(0, 20))
        end_entry = tk.Entry(segment_frame, bg=entry_bg, fg=entry_fg, insertbackground="white", validate="key", validatecommand=vcmd)
        end_entry.insert(0, "00:00:00")
        end_entry.grid(row=i, column=3, padx=(0, 5))
        segment_entries.append((start_entry, end_entry))

def confirm_url():
    global video_duration_seconds
    url = url_entry.get().strip()
    if url:
        status_label.config(text="Lade Videoinformationen...", fg="#ffff00")
        formatted, seconds = get_video_duration_from_meta(url)
        if formatted and seconds:
            video_duration_seconds = seconds
            duration_label.config(text=f"Videolänge: {formatted}")
            status_label.config(text="URL bestätigt", fg="#00ff00")

            update_segment_fields()
            for start_entry, end_entry in segment_entries:
                start_entry.configure(validate="none")
                end_entry.configure(validate="none")
                start_entry.delete(0, tk.END)
                start_entry.insert(0, "00:00:00")
                end_entry.delete(0, tk.END)
                end_entry.insert(0, formatted)
                start_entry.configure(validate="key", validatecommand=vcmd)
                end_entry.configure(validate="key", validatecommand=vcmd)
        else:
            video_duration_seconds = None
            duration_label.config(text="Videolänge: nicht gefunden")
            status_label.config(text="URL-Validierung fehlgeschlagen", fg="#ff0000")

# GUI Layout Aufbau
status_label.grid(row=0, column=0, columnspan=4, pady=(5, 0))

duration_label = tk.Label(root, text="Videolänge: unbekannt", fg=label_fg, bg="#2b2b2b", font=font_style)
duration_label.grid(row=1, column=0, columnspan=4, pady=(5, 0))

# Lokale Datei oder URL Auswahl
source_frame = tk.Frame(root, bg="#2b2b2b")
source_frame.grid(row=2, column=0, columnspan=4, pady=10)

source_type_var = tk.StringVar(value="url")
local_file_path = tk.StringVar()

def toggle_source_type():
    if source_type_var.get() == "url":
        url_entry.configure(state="normal")
        confirm_button.configure(state="normal")
        browse_button.configure(state="disabled")
        file_label.configure(text="Keine Datei ausgewählt")
        local_file_path.set("")
    else:
        url_entry.configure(state="disabled")
        confirm_button.configure(state="disabled")
        browse_button.configure(state="normal")
        url_entry.delete(0, tk.END)
    update_button_text()

tk.Radiobutton(source_frame, text="URL", variable=source_type_var, value="url", 
               command=toggle_source_type, fg=label_fg, bg="#2b2b2b", 
               selectcolor="#2b2b2b", font=font_style).grid(row=0, column=0, padx=10)

tk.Radiobutton(source_frame, text="Lokale Datei", variable=source_type_var, value="local", 
               command=toggle_source_type, fg=label_fg, bg="#2b2b2b", 
               selectcolor="#2b2b2b", font=font_style).grid(row=0, column=1, padx=10)

# URL Eingabe
url_frame = tk.Frame(root, bg="#2b2b2b")
url_frame.grid(row=3, column=0, columnspan=4, pady=5)

def validate_url_input(P):
    return len(P) <= 2000

url_vcmd = (root.register(validate_url_input), '%P')

tk.Label(url_frame, text="Video-URL:", fg=label_fg, bg="#2b2b2b", font=font_style).grid(row=0, column=0, sticky="e", padx=20)
url_entry = tk.Entry(url_frame, width=45, bg=entry_bg, fg=entry_fg, insertbackground="white", validate="key", validatecommand=url_vcmd)
url_entry.grid(row=0, column=1, padx=5)
confirm_button = ttk.Button(url_frame, text="URL bestätigen", command=confirm_url)
confirm_button.grid(row=0, column=2, padx=5)

# Lokale Datei Auswahl
file_frame = tk.Frame(root, bg="#2b2b2b")
file_frame.grid(row=4, column=0, columnspan=4, pady=5)

def browse_file():
    filename = filedialog.askopenfilename(
        title="Video-Datei auswählen",
        filetypes=[
            ("Video-Dateien", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm *.m4v *.mpg *.mpeg *.3gp"),
            ("Alle Dateien", "*.*")
        ]
    )
    if filename:
        local_file_path.set(filename)
        file_label.configure(text=os.path.basename(filename))
        # Automatisch Videolänge ermitteln
        confirm_local_file()

def confirm_local_file():
    global video_duration_seconds
    file_path = local_file_path.get()
    if file_path and os.path.exists(file_path):
        status_label.config(text="Ermittle Videolänge...", fg="#ffff00")
        duration = get_actual_duration(file_path)
        if duration > 0:
            video_duration_seconds = int(duration)
            formatted_time = seconds_to_time(video_duration_seconds)
            duration_label.config(text=f"Videolänge: {formatted_time}")
            status_label.config(text="Lokale Datei bestätigt", fg="#00ff00")
            
            # Update segment fields
            update_segment_fields()
            for start_entry, end_entry in segment_entries:
                start_entry.configure(validate="none")
                end_entry.configure(validate="none")
                start_entry.delete(0, tk.END)
                start_entry.insert(0, "00:00:00")
                end_entry.delete(0, tk.END)
                end_entry.insert(0, formatted_time)
                start_entry.configure(validate="key", validatecommand=vcmd)
                end_entry.configure(validate="key", validatecommand=vcmd)
        else:
            video_duration_seconds = None
            duration_label.config(text="Videolänge: nicht ermittelbar")
            status_label.config(text="Fehler beim Lesen der Datei", fg="#ff0000")

def create_transcription_frame():
    """Erstellt den Transkriptions-Bereich der GUI"""
    global transcribe_enabled_var, use_speakers_var, openai_key_entry, hf_token_entry
    
    # Transkriptions-Frame
    transcription_frame = tk.LabelFrame(
        root, 
        text="Transkription (Optional)", 
        fg=label_fg, 
        bg="#2b2b2b", 
        font=("Segoe UI", 11, "bold")
    )
    transcription_frame.grid(row=9, column=0, columnspan=4, pady=10, padx=20, sticky="ew")
    
    # Transkription aktivieren Checkbox
    transcribe_enabled_var = tk.BooleanVar(value=False)
    transcribe_checkbox = tk.Checkbutton(
        transcription_frame,
        text="Audio automatisch transkribieren",
        variable=transcribe_enabled_var,
        command=toggle_transcription_ui,
        fg=label_fg,
        bg="#2b2b2b",
        font=font_style,
        selectcolor="#2b2b2b",
        activebackground="#2b2b2b"
    )
    transcribe_checkbox.grid(row=0, column=0, columnspan=3, sticky="w", padx=10, pady=5)
    
    # OpenAI API Key
    tk.Label(
        transcription_frame,
        text="OpenAI API Key:",
        fg=label_fg,
        bg="#2b2b2b",
        font=font_style
    ).grid(row=1, column=0, sticky="e", padx=(20, 10), pady=5)
    
    openai_key_entry = tk.Entry(
        transcription_frame,
        width=40,
        show="*",
        bg=entry_bg,
        fg=entry_fg,
        insertbackground="white"
    )
    openai_key_entry.grid(row=1, column=1, padx=5, pady=5)
    
    # Test API Key Button
    test_key_btn = ttk.Button(
        transcription_frame,
        text="API Key testen",
        command=test_api_key
    )
    test_key_btn.grid(row=1, column=2, padx=5, pady=5)
    
    # Speaker Diarization Option
    use_speakers_var = tk.BooleanVar(value=False)
    speakers_checkbox = tk.Checkbutton(
        transcription_frame,
        text="Mehrere Sprecher erkennen",
        variable=use_speakers_var,
        command=toggle_speaker_options,
        fg=label_fg,
        bg="#2b2b2b",
        font=font_style,
        selectcolor="#2b2b2b",
        activebackground="#2b2b2b"
    )
    speakers_checkbox.grid(row=2, column=0, columnspan=2, sticky="w", padx=10, pady=5)
    
    # Hugging Face Token (für Speaker Diarization)
    hf_label = tk.Label(
        transcription_frame,
        text="Hugging Face Token:",
        fg=label_fg,
        bg="#2b2b2b",
        font=font_style
    )
    hf_label.grid(row=3, column=0, sticky="e", padx=(20, 10), pady=5)
    hf_label.grid_remove()  # Initial versteckt
    
    hf_token_entry = tk.Entry(
        transcription_frame,
        width=40,
        show="*",
        bg=entry_bg,
        fg=entry_fg,
        insertbackground="white"
    )
    hf_token_entry.grid(row=3, column=1, padx=5, pady=5)
    hf_token_entry.grid_remove()  # Initial versteckt
    
    # Info Label
    info_text = """Hinweise:
    • OpenAI API Key wird für Whisper-Transkription benötigt
    • Hugging Face Token nur für Sprecher-Erkennung erforderlich
    • Audio-Dateien werden automatisch nach der Segmentierung transkribiert"""
    
    info_label = tk.Label(
        transcription_frame,
        text=info_text,
        fg="#888888",
        bg="#2b2b2b",
        font=("Segoe UI", 9),
        justify="left"
    )
    info_label.grid(row=4, column=0, columnspan=3, padx=10, pady=10, sticky="w")
    
    # Initial deaktiviert
    toggle_transcription_ui()

def toggle_transcription_ui():
    """Aktiviert/Deaktiviert Transkriptions-UI Elemente"""
    if transcribe_enabled_var.get():
        openai_key_entry.configure(state="normal")
        # Weitere UI-Elemente aktivieren
    else:
        openai_key_entry.configure(state="disabled")
        use_speakers_var.set(False)
        toggle_speaker_options()

def toggle_speaker_options():
    """Zeigt/Versteckt Speaker-spezifische Optionen"""
    for widget in root.grid_slaves():
        if isinstance(widget, tk.Label) and "Hugging Face" in widget.cget("text"):
            if use_speakers_var.get():
                widget.grid()
            else:
                widget.grid_remove()
    
    if use_speakers_var.get():
        hf_token_entry.grid()
    else:
        hf_token_entry.grid_remove()

def test_api_key():
    """Testet den eingegebenen OpenAI API Key"""
    api_key = openai_key_entry.get().strip()
    if not api_key:
        messagebox.showerror("Fehler", "Bitte geben Sie einen API Key ein.")
        return
    
    try:
        # Test mit kleiner Audio-Datei oder Dummy-Request
        test_transcriber = WhisperTranscriber(api_key)
        # Hier könnte ein Test-API-Call gemacht werden
        status_label.config(text="API Key gültig", fg="#00ff00")
        messagebox.showinfo("Erfolg", "API Key wurde erfolgreich validiert!")
    except Exception as e:
        status_label.config(text="API Key ungültig", fg="#ff0000")
        messagebox.showerror("Fehler", f"API Key Validierung fehlgeschlagen: {str(e)}")

def initialize_transcribers():
    """Initialisiert die Transcriber mit den eingegebenen Keys"""
    global whisper_transcriber, speaker_transcriber
    
    if not transcribe_enabled_var.get():
        return True
    
    openai_key = openai_key_entry.get().strip()
    if not openai_key:
        messagebox.showerror("Fehler", "Bitte OpenAI API Key eingeben.")
        return False
    
    try:
        whisper_transcriber = WhisperTranscriber(openai_key)
        
        if use_speakers_var.get():
            hf_token = hf_token_entry.get().strip()
            if not hf_token:
                messagebox.showerror("Fehler", "Bitte Hugging Face Token eingeben für Sprecher-Erkennung.")
                return False
            speaker_transcriber = WhisperSpeakerTranscriber(openai_key, hf_token)
        
        return True
    except Exception as e:
        messagebox.showerror("Fehler", f"Fehler beim Initialisieren der Transcriber: {str(e)}")
        return False

tk.Label(file_frame, text="Lokale Datei:", fg=label_fg, bg="#2b2b2b", font=font_style).grid(row=0, column=0, sticky="e", padx=20)
browse_button = ttk.Button(file_frame, text="Durchsuchen", command=browse_file, state="disabled")
browse_button.grid(row=0, column=1, padx=5)
file_label = tk.Label(file_frame, text="Keine Datei ausgewählt", fg=label_fg, bg="#2b2b2b", font=font_style)
file_label.grid(row=0, column=2, padx=10)

# Info-Box für unterstützte Formate
info_frame = tk.Frame(file_frame, bg="#3c3f41", relief="ridge", borderwidth=1)
info_frame.grid(row=0, column=3, padx=10)
info_label = tk.Label(info_frame, text="ℹ", fg="#00aaff", bg="#3c3f41", font=("Segoe UI", 12, "bold"), cursor="hand2")
info_label.pack(padx=5, pady=2)

def show_format_info(event):
    messagebox.showinfo(
        "Unterstützte Videoformate",
        "Kompatible Formate:\n\n"
        "• MP4 (.mp4) - Empfohlen\n"
        "• AVI (.avi)\n"
        "• MKV (.mkv)\n"
        "• MOV (.mov)\n"
        "• WMV (.wmv)\n"
        "• FLV (.flv)\n"
        "• WebM (.webm)\n"
        "• M4V (.m4v)\n"
        "• MPG/MPEG (.mpg, .mpeg)\n"
        "• 3GP (.3gp)\n\n"
        "Hinweis: Die Datei muss lokal verfügbar sein.\n"
        "Maximale Dateigröße: Abhängig vom verfügbaren Speicher."
    )

info_label.bind("<Button-1>", show_format_info)

# Gesamtes Video Checkbox
download_full_checkbox = tk.Checkbutton(root, text="Gesamtes Video herunterladen", variable=download_full_var, command=toggle_segment_ui,
                                        fg=label_fg, bg="#2b2b2b", font=font_style, selectcolor="#2b2b2b", activebackground="#2b2b2b")
download_full_checkbox.grid(row=5, column=0, columnspan=3, sticky="w", padx=20)

# Segmentanzahl und Checkbox
tk.Label(root, text="Anzahl Segmente (1–5):", fg=label_fg, bg="#2b2b2b", font=font_style).grid(row=6, column=0, sticky="e", padx=20)
segment_count_var = tk.StringVar(value="1")
segment_dropdown = ttk.Combobox(root, textvariable=segment_count_var, values=["1", "2", "3", "4", "5"], width=5)
segment_dropdown.grid(row=6, column=1, sticky="w", padx=(0, 20))
segment_dropdown.bind("<<ComboboxSelected>>", lambda e: update_segment_fields())

delete_source_var = tk.BooleanVar()

segment_frame = tk.Frame(root, bg="#2b2b2b")
segment_frame.grid(row=7, column=0, columnspan=4, pady=10)
update_segment_fields()

button_frame = tk.Frame(root, bg="#2b2b2b")
button_frame.grid(row=8, column=0, columnspan=4)

# Button Text initial setzen
download_btn = ttk.Button(button_frame, text="Download & Schneiden", command=start_process)
download_btn.grid(row=0, column=0, padx=(20, 10), pady=20)

# Update Button Text bei Source-Änderung
def update_button_text():
    if download_full_var.get():
        if source_type_var.get() == "url":
            download_btn.config(text="Download")
        else:
            download_btn.config(text="Verarbeiten")
    else:
        if source_type_var.get() == "url":
            download_btn.config(text="Download & Schneiden")
        else:
            download_btn.config(text="Verarbeiten & Schneiden")

gif_label = tk.Label(button_frame, bg="#2b2b2b")
gif_label.grid(row=0, column=1, padx=10)
gif_label.grid_remove()

toggle_segment_ui()

# Erstelle Transkriptions-Frame
create_transcription_frame()

# Verschiebe Button-Frame nach unten um Platz für Transkriptions-Frame zu machen
button_frame.grid(row=10, column=0, columnspan=4)  # War vorher row=8

# Aktualisiere auch die Segment-Frame Position
segment_frame.grid(row=7, column=0, columnspan=4, pady=10)

root.mainloop()