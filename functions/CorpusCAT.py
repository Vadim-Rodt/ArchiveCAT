# ArchiveCAT.py - Modularisierte Version (FIXED)
"""
Hauptmodul für ArchiveCAT - Complex Annotation Tool
"""

import sys
import os
# Füge den aktuellen Ordner zum Python-Path hinzu
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from tkinter import messagebox, ttk, filedialog
from queue import Queue, Empty
from PIL import Image, ImageTk, ImageSequence
import threading
from pathlib import Path



# Eigene Module
from modern_style import apply_modern_style
from config_manager import ConfigManager
from video_processor import VideoProcessor
from download_manager import ArchiveDownloader
from transcription_manager import TranscriptionManager
from file_processor import FileProcessor
from audio_video_splitter import split_video_audio
from prosody_analyzer import ProsodyAnalyzer

from prosody_integration import (
    ProsodyIntegration, 
    enhance_file_processor_with_prosody,
    add_prosody_menu_items
)

# === Globale Variablen ===

gif_frames = []
gif_running = False
config_manager = None
video_processor = None
archive_downloader = None
transcription_manager = None
file_processor = None
prosody_integration = None
script_dir = os.path.dirname(os.path.abspath(__file__))
icon_path = os.path.join(script_dir, "logo.png")

"""analyzer = ProsodyAnalyzer()
results = analyzer.analyze_audio("audio.wav", gender="male")
analyzer.save_results(results, "output/prosody")
analyzer.create_visualization("audio.wav", results, "output/viz.png")"""

# UI Variablen
transcribe_enabled_var = None
use_speakers_var = None
openai_key_entry = None
hf_token_entry = None
language_var = None
language_mapping = {
    "Deutsch": "de",
    "Englisch": "en",
    "Französisch": "fr",
    "Spanisch": "es",
    "Italienisch": "it",
    "Portugiesisch": "pt",
    "Niederländisch": "nl",
    "Polnisch": "pl",
    "Russisch": "ru",
    "Chinesisch": "zh",
    "Japanisch": "ja",
    "Koreanisch": "ko",
    "Arabisch": "ar",
    "Automatisch": "auto"
}
export_txt_var = None
export_json_var = None
export_xml_var = None
export_srt_var = None
export_frame = None  # Globale Variable für Export-Frame

video_duration_seconds = None
segment_entries = []

# === HAUPTFUNKTIONEN ===

def initialize_managers():
    """Initialisiert alle Manager-Klassen"""
    global config_manager, video_processor, archive_downloader, transcription_manager, file_processor, prosody_integration

    config_manager = ConfigManager()
    video_processor = VideoProcessor()
    archive_downloader = ArchiveDownloader()
    transcription_manager = TranscriptionManager()
    prosody_integration = ProsodyIntegration()
    
    download_dir = config_manager.get_download_dir()
    if download_dir:
        file_processor = FileProcessor(download_dir, transcription_manager)
        prosody_integration.add_to_file_processor(file_processor)


def select_download_directory():
    """Öffnet Dialog zur Auswahl des Download-Verzeichnisses"""
    current_dir = config_manager.get_download_dir()
    initial_dir = current_dir if current_dir and os.path.exists(current_dir) else os.path.expanduser("~/Desktop")
    
    selected_dir = filedialog.askdirectory(
        title="Download-Verzeichnis auswählen",
        initialdir=initial_dir,
        mustexist=True
    )
    
    if selected_dir:
        if config_manager.set_download_dir(selected_dir):
            # Update FileProcessor
            global file_processor
            file_processor = FileProcessor(selected_dir, transcription_manager)
            
            update_download_dir_label()
            messagebox.showinfo("Erfolg", f"Download-Verzeichnis gesetzt:\n{selected_dir}")

def update_download_dir_label():
    """Aktualisiert das Label mit dem aktuellen Download-Verzeichnis"""
    download_dir = config_manager.get_download_dir()
    if download_dir:
        display_path = download_dir
        if len(display_path) > 50:
            display_path = "..." + display_path[-47:]
        download_dir_label.config(text=display_path, fg="#00ff00")
    else:
        download_dir_label.config(text="Kein Verzeichnis ausgewählt", fg="#ff9900")

def threaded_download(url, segment_times, delete_source, queue):
    """Thread-Funktion für Downloads"""
    try:
        download_dir = config_manager.get_download_dir()
        if not download_dir:
            queue.put("error:Kein Download-Verzeichnis gesetzt")
            return
        
        # Download von archive.org
        result = archive_downloader.download_video_from_archive(url, download_dir, queue)
        
        if not result['success']:
            queue.put(f"error:{result.get('error', 'Download fehlgeschlagen')}")
            return
        
        # Video-Segmentierung
        if queue:
            queue.put("status:Segmentiere Video...")
        
        # Update globale Video-Dauer
        global video_duration_seconds
        video_duration = video_processor.get_video_duration(result['video_path'])
        video_duration_seconds = int(video_duration)
        
        # Erstelle Transcription-Settings
        transcription_settings = None
        if transcribe_enabled_var and transcribe_enabled_var.get():
            transcription_settings = {
                'enabled': True,
                'language': language_mapping.get(language_var.get(), "de"),
                'use_speakers': use_speakers_var.get(),
                'export_formats': {
                    'txt': export_txt_var.get() if export_txt_var else True,
                    'json': export_json_var.get() if export_json_var else True,
                    'xml': export_xml_var.get() if export_xml_var else False,
                    'srt': export_srt_var.get() if export_srt_var else True
                }
            }
    

        # Verarbeite mit FileProcessor
        success = file_processor.process_local_file(
            result['video_path'], segment_times, delete_source, 
            queue, transcription_settings
        )
        
        if success:
            queue.put("done")
        else:
            queue.put("error:Segmentierung fehlgeschlagen")
            
    except Exception as e:
        queue.put(f"error:{str(e)}")

def process_local_file_thread(file_path, segment_times, delete_source, queue):
    """Thread-Funktion für lokale Dateiverarbeitung"""
    try:
        download_dir = config_manager.get_download_dir()
        if not download_dir:
            queue.put("error:Kein Download-Verzeichnis gesetzt")
            return
        
        # Erstelle Transcription-Settings
        transcription_settings = None
        if transcribe_enabled_var and transcribe_enabled_var.get():
            transcription_settings = {
                'enabled': True,
                'language': language_mapping.get(language_var.get(), "de"),
                'use_speakers': use_speakers_var.get(),
                'export_formats': {
                    'txt': export_txt_var.get() if export_txt_var else True,
                    'json': export_json_var.get() if export_json_var else True,
                    'xml': export_xml_var.get() if export_xml_var else False,
                    'srt': export_srt_var.get() if export_srt_var else True
                }
            }
        
        # Verarbeite Datei
        success = file_processor.process_local_file(
            file_path, segment_times, delete_source, 
            queue, transcription_settings
        )
        
        if not success:
            queue.put("error:Verarbeitung fehlgeschlagen")
            
    except Exception as e:
        queue.put(f"error:{str(e)}")

def start_process():
    """Process files with current settings"""
    global video_duration_seconds, gif_frames
    
    # Prüfe Download-Verzeichnis
    if not config_manager.get_download_dir():
        messagebox.showerror("Fehler", "Bitte wählen Sie zuerst ein Download-Verzeichnis aus.")
        return
    
    # Initialisiere Transcriber wenn aktiviert
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
    
    # Sammle Segment-Zeiten
    segment_times = []
    if download_full_var.get():
        if not video_duration_seconds:
            messagebox.showerror("Fehler", "Bitte zuerst URL/Datei bestätigen.")
            return
        segment_times = [("00:00:00", video_processor.seconds_to_time(video_duration_seconds))]
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
            segment_times.append((start, end))
        
        # Validiere Segmente
        is_valid, error_msg = file_processor.validate_segments(segment_times, video_duration_seconds)
        if not is_valid:
            messagebox.showerror("Fehler", error_msg)
            return
    
    delete_source = delete_source_var.get()
    
    # Starte GIF-Animation
    start_gif_animation()
    
    # Starte Verarbeitung
    queue = Queue()
    download_btn.config(state="disabled")
    
    if local_file:
        threading.Thread(
            target=process_local_file_thread, 
            args=(local_file, segment_times, delete_source, queue), 
            daemon=True
        ).start()
    else:
        threading.Thread(
            target=threaded_download, 
            args=(url, segment_times, delete_source, queue), 
            daemon=True
        ).start()
    
    check_queue(queue)

def check_queue(queue):
    """Überprüft die Queue auf Status-Updates"""
    global gif_running
    
    try:
        while True:
            result = queue.get_nowait()
            if result == "done":
                gif_running = False
                gif_label.grid_remove()
                status_label.config(text="Fertig!", fg="#00ff00")
                download_btn.config(state="normal")
                messagebox.showinfo("Fertig", "Verarbeitung erfolgreich abgeschlossen.")
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

def start_gif_animation():
    """Startet die GIF-Animation"""
    global gif_frames, gif_running
    
    try:
        
        gif_path = os.path.join(script_dir,"logo.gif")
        
        if os.path.exists(gif_path):
            gif = Image.open(gif_path)
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
        else:
            print(f"GIF nicht gefunden: {gif_path}")
    except Exception as e:
        print(f"GIF-Fehler: {e}")

def animate_gif(index=0):
    """Animiert das GIF"""
    if gif_running and gif_frames:
        gif_label.config(image=gif_frames[index])
        gif_label.image = gif_frames[index]
        root.after(100, lambda: animate_gif((index + 1) % len(gif_frames)))

def confirm_url():
    """Bestätigt URL und holt Video-Metadaten"""
    global video_duration_seconds
    url = url_entry.get().strip()
    
    if url:
        status_label.config(text="Lade Videoinformationen...", fg="#ffff00")
        formatted, seconds = archive_downloader.get_video_metadata(url)
        
        if formatted and seconds:
            video_duration_seconds = seconds
            duration_label.config(text=f"Videolänge: {formatted}")
            status_label.config(text="URL bestätigt", fg="#00ff00")
            update_segment_fields_with_duration(formatted)
        else:
            video_duration_seconds = None
            duration_label.config(text="Videolänge: nicht gefunden")
            status_label.config(text="URL-Validierung fehlgeschlagen", fg="#ff0000")

def confirm_local_file():
    """Bestätigt lokale Datei und ermittelt Video-Dauer"""
    global video_duration_seconds
    file_path = local_file_path.get()
    
    if file_path and os.path.exists(file_path):
        status_label.config(text="Ermittle Videolänge...", fg="#ffff00")
        
        # Validiere Video-Datei
        is_valid, error_msg = video_processor.validate_video_file(file_path)
        if not is_valid:
            messagebox.showerror("Fehler", error_msg)
            return
        
        duration = video_processor.get_video_duration(file_path)
        if duration > 0:
            video_duration_seconds = int(duration)
            formatted_time = video_processor.seconds_to_time(video_duration_seconds)
            duration_label.config(text=f"Videolänge: {formatted_time}")
            status_label.config(text="Lokale Datei bestätigt", fg="#00ff00")
            update_segment_fields_with_duration(formatted_time)
        else:
            video_duration_seconds = None
            duration_label.config(text="Videolänge: nicht ermittelbar")
            status_label.config(text="Fehler beim Lesen der Datei", fg="#ff0000")

def update_segment_fields_with_duration(duration_str):
    """Aktualisiert Segment-Felder mit Video-Dauer"""
    update_segment_fields()
    for start_entry, end_entry in segment_entries:
        start_entry.configure(validate="none")
        end_entry.configure(validate="none")
        start_entry.delete(0, tk.END)
        start_entry.insert(0, "00:00:00")
        end_entry.delete(0, tk.END)
        end_entry.insert(0, duration_str)
        start_entry.configure(validate="key", validatecommand=vcmd)
        end_entry.configure(validate="key", validatecommand=vcmd)

def initialize_transcribers():
    """Initialisiert die Transcriber mit den eingegebenen Keys"""
    if not transcribe_enabled_var.get():
        return True
    
    openai_key = openai_key_entry.get().strip()
    if not openai_key:
        messagebox.showerror("Fehler", "Bitte OpenAI API Key eingeben.")
        return False
    
    hf_token = None
    if use_speakers_var.get():
        hf_token = hf_token_entry.get().strip()
        if not hf_token:
            messagebox.showerror("Fehler", "Bitte Hugging Face Token eingeben für Sprecher-Erkennung.")
            return False
    
    success, error_msg = transcription_manager.initialize_transcribers(openai_key, hf_token)
    if not success:
        messagebox.showerror("Fehler", error_msg)
        return False
    
    return True

# === GUI FUNKTIONEN ===

def create_transcription_frame():
    """Erstellt den Transkriptions-Bereich der GUI"""
    global transcribe_enabled_var, use_speakers_var, openai_key_entry, hf_token_entry
    global language_var, export_txt_var, export_json_var, export_xml_var, export_srt_var
    global export_frame
    
    # Transkriptions-Frame
    transcription_frame = tk.LabelFrame(
        root, 
        text="Transkription (Optional)", 
        fg=label_fg, 
        bg="#2b2b2b", 
        font=("Segoe UI", 11, "bold")
    )
    transcription_frame.grid(row=10, column=0, columnspan=4, pady=10, padx=20, sticky="ew")
    
    # Transkription aktivieren
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
    
    # Sprache
    tk.Label(
        transcription_frame,
        text="Sprache:",
        fg=label_fg,
        bg="#2b2b2b",
        font=font_style
    ).grid(row=1, column=0, sticky="e", padx=(20, 10), pady=5)
    
    language_var = tk.StringVar(value="Deutsch")
    language_menu = ttk.Combobox(
        transcription_frame,
        textvariable=language_var,
        values=list(language_mapping.keys()),
        state="readonly",
        width=15
    )
    language_menu.grid(row=1, column=1, sticky="w", padx=5, pady=5)
    
    # OpenAI API Key
    tk.Label(
        transcription_frame,
        text="OpenAI API Key:",
        fg=label_fg,
        bg="#2b2b2b",
        font=font_style
    ).grid(row=2, column=0, sticky="e", padx=(20, 10), pady=5)
    
    openai_key_entry = tk.Entry(
        transcription_frame,
        width=40,
        show="*",
        bg=entry_bg,
        fg=entry_fg,
        insertbackground="white"
    )
    openai_key_entry.grid(row=2, column=1, padx=5, pady=5)
    
    # Test API Key Button
    test_key_btn = ttk.Button(
        transcription_frame,
        text="API Key testen",
        command=test_api_key
    )
    test_key_btn.grid(row=2, column=2, padx=5, pady=5)
    
    # Speaker Diarization
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
    speakers_checkbox.grid(row=3, column=0, columnspan=2, sticky="w", padx=10, pady=5)
    
    # Hugging Face Token
    hf_label = tk.Label(
        transcription_frame,
        text="Hugging Face Token:",
        fg=label_fg,
        bg="#2b2b2b",
        font=font_style
    )
    hf_label.grid(row=4, column=0, sticky="e", padx=(20, 10), pady=5)
    hf_label.grid_remove()
    
    hf_token_entry = tk.Entry(
        transcription_frame,
        width=40,
        show="*",
        bg=entry_bg,
        fg=entry_fg,
        insertbackground="white"
    )
    hf_token_entry.grid(row=4, column=1, padx=5, pady=5)
    hf_token_entry.grid_remove()
    
    # Export-Formate
    export_frame = tk.LabelFrame(
        transcription_frame,
        text="Export-Formate",
        fg=label_fg,
        bg="#2b2b2b",
        font=font_style
    )
    export_frame.grid(row=5, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
    
    export_txt_var = tk.BooleanVar(value=True)
    export_json_var = tk.BooleanVar(value=True)
    export_xml_var = tk.BooleanVar(value=False)
    export_srt_var = tk.BooleanVar(value=True)
    
    tk.Checkbutton(export_frame, text="TXT", variable=export_txt_var,
                   fg=label_fg, bg="#2b2b2b", selectcolor="#2b2b2b").grid(row=0, column=0, padx=5)
    tk.Checkbutton(export_frame, text="JSON", variable=export_json_var,
                   fg=label_fg, bg="#2b2b2b", selectcolor="#2b2b2b").grid(row=0, column=1, padx=5)
    tk.Checkbutton(export_frame, text="XML", variable=export_xml_var,
                   fg=label_fg, bg="#2b2b2b", selectcolor="#2b2b2b").grid(row=0, column=2, padx=5)
    tk.Checkbutton(export_frame, text="SRT", variable=export_srt_var,
                   fg=label_fg, bg="#2b2b2b", selectcolor="#2b2b2b").grid(row=0, column=3, padx=5)
    
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
    info_label.grid(row=6, column=0, columnspan=3, padx=10, pady=10, sticky="w")
    
    # Initial deaktiviert
    toggle_transcription_ui()


def toggle_transcription_ui():
    """Aktiviert/Deaktiviert Transkriptions-UI Elemente"""
    global export_frame
    
    if transcribe_enabled_var.get():
        openai_key_entry.configure(state="normal")
        # Aktiviere Export-Format Checkboxen
        if export_frame:
            for child in export_frame.winfo_children():
                if isinstance(child, tk.Checkbutton):
                    child.configure(state="normal")
    else:
        openai_key_entry.configure(state="disabled")
        use_speakers_var.set(False)
        toggle_speaker_options()
        # Deaktiviere Export-Format Checkboxen
        if export_frame:
            for child in export_frame.winfo_children():
                if isinstance(child, tk.Checkbutton):
                    child.configure(state="disabled")

def toggle_speaker_options():
    """Zeigt/Versteckt Speaker-spezifische Optionen"""
    for widget in root.grid_slaves():
        if isinstance(widget, tk.Label) and widget.cget("text") == "Hugging Face Token:":
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
    
    is_valid, message = transcription_manager.test_api_key(api_key)
    status_label.config(text=message, fg="#00ff00" if is_valid else "#ff0000")
    
    if is_valid:
        messagebox.showinfo("Erfolg", message)
    else:
        messagebox.showerror("Fehler", message)

def toggle_segment_ui():
    """Zeigt/Versteckt Segment-UI basierend auf Checkbox"""
    if download_full_var.get():
        segment_dropdown.configure(state="disabled")
        segment_frame.grid_remove()
    else:
        segment_dropdown.configure(state="readonly")
        segment_frame.grid()
    update_button_text()

def update_button_text():
    """Aktualisiert Button-Text basierend auf Auswahl"""
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

def update_segment_fields():
    """Aktualisiert Segment-Eingabefelder"""
    for widgets in segment_frame.winfo_children():
        widgets.destroy()
    segment_entries.clear()
    
    count = int(segment_count_var.get())
    for i in range(count):
        tk.Label(segment_frame, text=f"Segment {i+1} Start:", 
                 fg=label_fg, bg="#2b2b2b", font=font_style).grid(
                     row=i, column=0, sticky="e", padx=(5, 20))
        
        start_entry = tk.Entry(segment_frame, bg=entry_bg, fg=entry_fg, 
                              insertbackground="white", validate="key", 
                              validatecommand=vcmd)
        start_entry.insert(0, "00:00:00")
        start_entry.grid(row=i, column=1, padx=(0, 20))
        
        tk.Label(segment_frame, text="End:", fg=label_fg, bg="#2b2b2b", 
                 font=font_style).grid(row=i, column=2, sticky="e", padx=(0, 20))
        
        end_entry = tk.Entry(segment_frame, bg=entry_bg, fg=entry_fg, 
                            insertbackground="white", validate="key", 
                            validatecommand=vcmd)
        end_entry.insert(0, "00:00:00")
        end_entry.grid(row=i, column=3, padx=(0, 5))
        
        segment_entries.append((start_entry, end_entry))

def toggle_source_type():
    """Wechselt zwischen URL und lokaler Datei"""
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

def browse_file():
    """Öffnet Dateiauswahl-Dialog"""
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
        confirm_local_file()

def show_format_info(event):
    """Zeigt Info über unterstützte Formate"""
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

def validate_time_input(char):
    """Validiert Zeiteingabe"""
    return char.isdigit() or char == ':'

# === HAUPTPROGRAMM ===

# Initialisiere
initialize_managers()

prosody_integration = ProsodyIntegration()
if file_processor:
    prosody_integration.add_to_file_processor(file_processor)

# GUI Start
root = tk.Tk() 
apply_modern_style(root)
root.title("CorpusCAT - Complex Annotation Tool")

# Fenstericon
try:
    icon_img = ImageTk.PhotoImage(file="logo.png")
    root.iconphoto(False, icon_img)
except Exception as e:
    print("Fehler beim Setzen des Icons:", e)

root.geometry("680x1200")
root.configure(bg="#2b2b2b")

# Style-Variablen
font_style = ("Segoe UI", 10)
label_fg = "#ffffff"
entry_bg = "#3c3f41"
entry_fg = "#ffffff"

style = ttk.Style()
style.theme_use("default")
style.configure("TButton", font=font_style, padding=6)

# Variablen
download_full_var = tk.BooleanVar()
source_type_var = tk.StringVar(value="url")
local_file_path = tk.StringVar()
segment_count_var = tk.StringVar(value="1")
delete_source_var = tk.BooleanVar()

# Validierung
vcmd = (root.register(validate_time_input), '%S')

# === GUI LAYOUT ===

# Status Label
status_label = tk.Label(root, text="Bereit", fg="#00ff00", bg="#2b2b2b", font=font_style)
status_label.grid(row=0, column=0, columnspan=4, pady=(5, 0))

# Download-Verzeichnis
dir_frame = tk.LabelFrame(root, text="Download-Verzeichnis", fg=label_fg, bg="#2b2b2b", 
                         font=("Segoe UI", 11, "bold"))
dir_frame.grid(row=1, column=0, columnspan=4, pady=10, padx=20, sticky="ew")

dir_button_frame = tk.Frame(dir_frame, bg="#2b2b2b")
dir_button_frame.grid(row=0, column=0, columnspan=2, pady=10)

select_dir_btn = ttk.Button(dir_button_frame, text="📁 Verzeichnis wählen", 
                           command=select_download_directory)
select_dir_btn.grid(row=0, column=0, padx=10)

download_dir_label = tk.Label(dir_button_frame, text="Kein Verzeichnis ausgewählt", 
                             fg="#ff9900", bg="#2b2b2b", font=("Segoe UI", 9))
download_dir_label.grid(row=0, column=1, padx=10)

update_download_dir_label()

# Videolänge
duration_label = tk.Label(root, text="Videolänge: unbekannt", fg=label_fg, 
                         bg="#2b2b2b", font=font_style)
duration_label.grid(row=2, column=0, columnspan=4, pady=(5, 0))

# Quelle auswählen
source_frame = tk.Frame(root, bg="#2b2b2b")
source_frame.grid(row=3, column=0, columnspan=4, pady=10)

tk.Radiobutton(source_frame, text="URL", variable=source_type_var, value="url", 
               command=toggle_source_type, fg=label_fg, bg="#2b2b2b", 
               selectcolor="#2b2b2b", font=font_style).grid(row=0, column=0, padx=10)

tk.Radiobutton(source_frame, text="Lokale Datei", variable=source_type_var, value="local", 
               command=toggle_source_type, fg=label_fg, bg="#2b2b2b", 
               selectcolor="#2b2b2b", font=font_style).grid(row=0, column=1, padx=10)

# URL Eingabe
url_frame = tk.Frame(root, bg="#2b2b2b")
url_frame.grid(row=4, column=0, columnspan=4, pady=5)

url_vcmd = (root.register(lambda P: len(P) <= 2000), '%P')

tk.Label(url_frame, text="Video-URL:", fg=label_fg, bg="#2b2b2b", 
         font=font_style).grid(row=0, column=0, sticky="e", padx=20)
url_entry = tk.Entry(url_frame, width=45, bg=entry_bg, fg=entry_fg, 
                    insertbackground="white", validate="key", validatecommand=url_vcmd)
url_entry.grid(row=0, column=1, padx=5)
confirm_button = ttk.Button(url_frame, text="URL bestätigen", command=confirm_url)
confirm_button.grid(row=0, column=2, padx=5)

# Lokale Datei
file_frame = tk.Frame(root, bg="#2b2b2b")
file_frame.grid(row=5, column=0, columnspan=4, pady=5)

tk.Label(file_frame, text="Lokale Datei:", fg=label_fg, bg="#2b2b2b", 
         font=font_style).grid(row=0, column=0, sticky="e", padx=20)
browse_button = ttk.Button(file_frame, text="Durchsuchen", command=browse_file, state="disabled")
browse_button.grid(row=0, column=1, padx=5)
file_label = tk.Label(file_frame, text="Keine Datei ausgewählt", fg=label_fg, 
                     bg="#2b2b2b", font=font_style)
file_label.grid(row=0, column=2, padx=10)

# Info-Box
info_frame = tk.Frame(file_frame, bg="#3c3f41", relief="ridge", borderwidth=1)
info_frame.grid(row=0, column=3, padx=10)
info_label = tk.Label(info_frame, text="ℹ", fg="#00aaff", bg="#3c3f41", 
                     font=("Segoe UI", 12, "bold"), cursor="hand2")
info_label.pack(padx=5, pady=2)
info_label.bind("<Button-1>", show_format_info)

# Gesamtes Video Checkbox
download_full_checkbox = tk.Checkbutton(root, text="Gesamtes Video herunterladen", 
                                       variable=download_full_var, command=toggle_segment_ui,
                                       fg=label_fg, bg="#2b2b2b", font=font_style, 
                                       selectcolor="#2b2b2b", activebackground="#2b2b2b")
download_full_checkbox.grid(row=6, column=0, columnspan=3, sticky="w", padx=20)

# Segmentanzahl
tk.Label(root, text="Anzahl Segmente (1–5):", fg=label_fg, bg="#2b2b2b", 
         font=font_style).grid(row=7, column=0, sticky="e", padx=20)
segment_dropdown = ttk.Combobox(root, textvariable=segment_count_var, 
                               values=["1", "2", "3", "4", "5"], width=5)
segment_dropdown.grid(row=7, column=1, sticky="w", padx=(0, 20))
segment_dropdown.bind("<<ComboboxSelected>>", lambda e: update_segment_fields())

# Segment Frame
segment_frame = tk.Frame(root, bg="#2b2b2b")
segment_frame.grid(row=8, column=0, columnspan=4, pady=10)
update_segment_fields()

# Button Frame
button_frame = tk.Frame(root, bg="#2b2b2b")
button_frame.grid(row=9, column=0, columnspan=4)

download_btn = ttk.Button(button_frame, text="Download & Schneiden", command=start_process)
download_btn.grid(row=0, column=0, padx=(20, 10), pady=20)

gif_label = tk.Label(button_frame, bg="#2b2b2b")
gif_label.grid(row=0, column=1, padx=10)
gif_label.grid_remove()

# Transkriptions-Frame
create_transcription_frame()
prosody_frame = prosody_integration.create_gui_elements(root, row_start=11)

menu_bar = tk.Menu(root)
root.config(menu=menu_bar)
add_prosody_menu_items(menu_bar, root)
        
prosody_integration.apply_settings()

# Initial UI-Status
toggle_segment_ui()

# Hauptschleife
root.mainloop()