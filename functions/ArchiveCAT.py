import os
import re
import time
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
import pandas as pd

# Import der Audio-Split-Funktionen
from audio_video_splitter import split_video_audio

# === Globale Variablen für GIF ===
gif_frames = []
gif_running = False

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
root.title("ArchiveCAT - Complex Annotation Tool")

# Fenstericon setzen mit logo.png (muss im selben Verzeichnis liegen)
try:
    icon_img = ImageTk.PhotoImage(file="logo.png")
    root.iconphoto(False, icon_img)
except Exception as e:
    print("Fehler beim Setzen des Icons:", e)

root.geometry("700x500")
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
        delete_checkbox.grid_remove()
        segment_frame.grid_remove()
        download_btn.config(text="Download")
    else:
        segment_dropdown.configure(state="readonly")
        delete_checkbox.grid()
        segment_frame.grid()
        download_btn.config(text="Download & Schneiden")

def start_process():
    global video_duration_seconds, gif_frames
    url = url_entry.get().strip()

    if not url:
        messagebox.showerror("Fehler", "Bitte eine URL angeben.")
        return

    segment_times = []
    if download_full_var.get():
        if not video_duration_seconds:
            messagebox.showerror("Fehler", "Bitte zuerst URL bestätigen.")
            return
        segment_times = [("00:00:00", seconds_to_time(video_duration_seconds))]
    else:
        segment_count = int(segment_count_var.get())
        if not video_duration_seconds:
            messagebox.showerror("Fehler", "Bitte zuerst URL bestätigen.")
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

    # Download starten
    queue = Queue()
    download_btn.config(state="disabled")
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

def validate_url_input(P):
    return len(P) <= 2000

url_vcmd = (root.register(validate_url_input), '%P')

tk.Label(root, text="Video-URL:", fg=label_fg, bg="#2b2b2b", font=font_style).grid(row=2, column=0, sticky="e", padx=20, pady=10)
url_entry = tk.Entry(root, width=45, bg=entry_bg, fg=entry_fg, insertbackground="white", validate="key", validatecommand=url_vcmd)
url_entry.grid(row=2, column=1, padx=5)
confirm_button = ttk.Button(root, text="URL bestätigen", command=confirm_url)
confirm_button.grid(row=2, column=2, padx=5)

# Gesamtes Video Checkbox
download_full_checkbox = tk.Checkbutton(root, text="Gesamtes Video herunterladen", variable=download_full_var, command=toggle_segment_ui,
                                        fg=label_fg, bg="#2b2b2b", font=font_style, selectcolor="#2b2b2b", activebackground="#2b2b2b")
download_full_checkbox.grid(row=3, column=0, columnspan=3, sticky="w", padx=20)

# Segmentanzahl und Checkbox
tk.Label(root, text="Anzahl Segmente (1–5):", fg=label_fg, bg="#2b2b2b", font=font_style).grid(row=4, column=0, sticky="e", padx=20)
segment_count_var = tk.StringVar(value="1")
segment_dropdown = ttk.Combobox(root, textvariable=segment_count_var, values=["1", "2", "3", "4", "5"], width=5)
segment_dropdown.grid(row=4, column=1, sticky="w", padx=(0, 20))
segment_dropdown.bind("<<ComboboxSelected>>", lambda e: update_segment_fields())

delete_source_var = tk.BooleanVar()
delete_checkbox = tk.Checkbutton(root, text="Quellvideo nach Segmentierung löschen?", variable=delete_source_var,
                                  fg=label_fg, bg="#2b2b2b", font=font_style, selectcolor="#2b2b2b", activebackground="#2b2b2b")
delete_checkbox.grid(row=4, column=2, columnspan=2, sticky="w")

segment_frame = tk.Frame(root, bg="#2b2b2b")
segment_frame.grid(row=5, column=0, columnspan=4, pady=10)
update_segment_fields()

button_frame = tk.Frame(root, bg="#2b2b2b")
button_frame.grid(row=6, column=0, columnspan=4)

download_btn = ttk.Button(button_frame, text="Download & Schneiden", command=start_process)
download_btn.grid(row=0, column=0, padx=(20, 10), pady=20)

gif_label = tk.Label(button_frame, bg="#2b2b2b")
gif_label.grid(row=0, column=1, padx=10)
gif_label.grid_remove()

toggle_segment_ui()

root.mainloop()