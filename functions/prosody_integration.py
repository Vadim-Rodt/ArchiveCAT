# prosody_integration.py
"""
Integration der Prosodieanalyse in ArchiveCAT
"""

import os
from pathlib import Path
from typing import Dict, Optional
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from prosody_analyzer import ProsodyAnalyzer, analyze_prosody_for_segment, batch_analyze_prosody, create_prosody_summary

class ProsodyIntegration:
    """Integration der Prosodieanalyse in ArchiveCAT"""
    
    def __init__(self):
        self.analyzer = ProsodyAnalyzer()
        self.prosody_enabled = False
        self.gender_selection = "unknown"
        self.create_visualizations = True
        
    def add_to_file_processor(self, file_processor):
        """Fügt Prosodieanalyse zum FileProcessor hinzu"""
        # Monkey-patch die perform_transcription Methode
        original_perform_transcription = file_processor._perform_transcription
        
        def enhanced_perform_transcription(audio_path, output_folder, queue, settings, segment_name=""):
            # Originale Transkription
            original_perform_transcription(audio_path, output_folder, queue, settings, segment_name)
            
            # Prosodieanalyse wenn aktiviert
            if self.prosody_enabled:
                if queue:
                    queue.put(f"status:Analysiere Prosodie{' für Segment' if segment_name else ''}...")
                
                try:
                    results = analyze_prosody_for_segment(
                        audio_path, 
                        output_folder, 
                        segment_name,
                        self.gender_selection
                    )
                    
                    if results['success'] and queue:
                        queue.put(f"status:Prosodieanalyse{segment_name} abgeschlossen")
                except Exception as e:
                    print(f"Prosodieanalyse-Fehler: {e}")
                    if queue:
                        queue.put(f"status:Prosodieanalyse fehlgeschlagen")
        
        file_processor._perform_transcription = enhanced_perform_transcription
    
    def create_gui_elements(self, parent_frame, row_start=0):
        """Erstellt GUI-Elemente für Prosodieanalyse"""
        # Prosodieanalyse Frame
        prosody_frame = tk.LabelFrame(
            parent_frame, 
            text="Prosodieanalyse (Optional)", 
            fg="#ffffff", 
            bg="#2b2b2b", 
            font=("Segoe UI", 11, "bold")
        )
        prosody_frame.grid(row=row_start, column=0, columnspan=4, pady=10, padx=20, sticky="ew")
        
        # Aktivieren Checkbox
        self.prosody_var = tk.BooleanVar(value=False)
        prosody_checkbox = tk.Checkbutton(
            prosody_frame,
            text="Prosodieanalyse durchführen",
            variable=self.prosody_var,
            command=self._toggle_prosody_ui,
            fg="#ffffff",
            bg="#2b2b2b",
            font=("Segoe UI", 10),
            selectcolor="#2b2b2b",
            activebackground="#2b2b2b"
        )
        prosody_checkbox.grid(row=0, column=0, columnspan=3, sticky="w", padx=10, pady=5)
        
        # Geschlecht-Auswahl
        tk.Label(
            prosody_frame,
            text="Sprecher-Geschlecht:",
            fg="#ffffff",
            bg="#2b2b2b",
            font=("Segoe UI", 10)
        ).grid(row=1, column=0, sticky="e", padx=(20, 10), pady=5)
        
        self.gender_var = tk.StringVar(value="unknown")
        self.gender_menu = ttk.Combobox(
            prosody_frame,
            textvariable=self.gender_var,
            values=["unknown", "male", "female"],
            state="readonly",
            width=15
        )
        self.gender_menu.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        self.gender_menu.set("unknown")
        
        # Visualisierungen
        self.viz_var = tk.BooleanVar(value=True)
        self.viz_checkbox = tk.Checkbutton(
            prosody_frame,
            text="Visualisierungen erstellen",
            variable=self.viz_var,
            fg="#ffffff",
            bg="#2b2b2b",
            font=("Segoe UI", 10),
            selectcolor="#2b2b2b",
            activebackground="#2b2b2b"
        )
        self.viz_checkbox.grid(row=2, column=0, columnspan=2, sticky="w", padx=10, pady=5)
        
        # Analyse-Features
        features_frame = tk.LabelFrame(
            prosody_frame,
            text="Analyse-Features",
            fg="#ffffff",
            bg="#2b2b2b",
            font=("Segoe UI", 9)
        )
        features_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        
        # Feature-Liste
        features = [
            "✓ Tonhöhe (F0) - Mittelwert, Bereich, Variabilität",
            "✓ Sprechgeschwindigkeit - Wörter/Minute, Silben/Sekunde",
            "✓ Pausen - Anzahl, Dauer, Verteilung",
            "✓ Stimmqualität - Jitter, Shimmer, HNR",
            "✓ Emotionale Indikatoren - Erregung, Valenz",
            "✓ Sprechrhythmus - Tempo, Regularität"
        ]
        
        for i, feature in enumerate(features):
            tk.Label(
                features_frame,
                text=feature,
                fg="#888888",
                bg="#2b2b2b",
                font=("Segoe UI", 8),
                justify="left"
            ).grid(row=i//2, column=i%2, sticky="w", padx=5, pady=2)
        
        # Info
        info_text = """Hinweise:
        • Prosodieanalyse erfordert librosa, parselmouth und numpy
        • Geschlechtsangabe verbessert Tonhöhenanalyse
        • Visualisierungen zeigen Tonhöhe, Intensität und Spektrogramm
        • Ergebnisse werden als JSON, TXT und CSV gespeichert"""
        
        info_label = tk.Label(
            prosody_frame,
            text=info_text,
            fg="#888888",
            bg="#2b2b2b",
            font=("Segoe UI", 9),
            justify="left"
        )
        info_label.grid(row=4, column=0, columnspan=3, padx=10, pady=10, sticky="w")
        
        # Initial deaktiviert
        self._toggle_prosody_ui()
        
        return prosody_frame
    
    def _toggle_prosody_ui(self):
        """Aktiviert/Deaktiviert Prosody-UI Elemente"""
        self.prosody_enabled = self.prosody_var.get()
        
        if self.prosody_enabled:
            self.gender_menu.configure(state="readonly")
            self.viz_checkbox.configure(state="normal")
        else:
            self.gender_menu.configure(state="disabled")
            self.viz_checkbox.configure(state="disabled")
    
    def get_settings(self) -> Dict:
        """Gibt aktuelle Einstellungen zurück"""
        return {
            'enabled': self.prosody_enabled,
            'gender': self.gender_var.get() if hasattr(self, 'gender_var') else "unknown",
            'create_visualizations': self.viz_var.get() if hasattr(self, 'viz_var') else True
        }
    
    def apply_settings(self):
        """Wendet die GUI-Einstellungen an"""
        self.prosody_enabled = self.prosody_var.get() if hasattr(self, 'prosody_var') else False
        self.gender_selection = self.gender_var.get() if hasattr(self, 'gender_var') else "unknown"
        self.create_visualizations = self.viz_var.get() if hasattr(self, 'viz_var') else True


# Standalone Prosody Analysis Window
class ProsodyAnalysisWindow:
    """Eigenständiges Fenster für Prosodieanalyse"""
    
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("Prosodieanalyse")
        self.window.geometry("600x500")
        self.window.configure(bg="#2b2b2b")
        
        self.analyzer = ProsodyAnalyzer()
        self.setup_ui()
    
    def setup_ui(self):
        """Erstellt die UI"""
        # Titel
        title_label = tk.Label(
            self.window,
            text="Prosodieanalyse für Audio-Dateien",
            fg="#ffffff",
            bg="#2b2b2b",
            font=("Segoe UI", 14, "bold")
        )
        title_label.pack(pady=10)
        
        # Dateiauswahl
        file_frame = tk.Frame(self.window, bg="#2b2b2b")
        file_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Label(
            file_frame,
            text="Audio-Datei:",
            fg="#ffffff",
            bg="#2b2b2b",
            font=("Segoe UI", 10)
        ).pack(side="left", padx=5)
        
        self.file_path_var = tk.StringVar()
        file_entry = tk.Entry(
            file_frame,
            textvariable=self.file_path_var,
            width=40,
            bg="#3c3f41",
            fg="#ffffff"
        )
        file_entry.pack(side="left", padx=5, fill="x", expand=True)
        
        browse_btn = ttk.Button(
            file_frame,
            text="Durchsuchen",
            command=self.browse_file
        )
        browse_btn.pack(side="left", padx=5)
        
        # Einstellungen
        settings_frame = tk.LabelFrame(
            self.window,
            text="Einstellungen",
            fg="#ffffff",
            bg="#2b2b2b",
            font=("Segoe UI", 11, "bold")
        )
        settings_frame.pack(pady=10, padx=20, fill="x")
        
        # Geschlecht
        gender_frame = tk.Frame(settings_frame, bg="#2b2b2b")
        gender_frame.pack(pady=5, padx=10, fill="x")
        
        tk.Label(
            gender_frame,
            text="Geschlecht:",
            fg="#ffffff",
            bg="#2b2b2b",
            font=("Segoe UI", 10)
        ).pack(side="left", padx=5)
        
        self.gender_var = tk.StringVar(value="unknown")
        gender_menu = ttk.Combobox(
            gender_frame,
            textvariable=self.gender_var,
            values=["unknown", "male", "female"],
            state="readonly",
            width=15
        )
        gender_menu.pack(side="left", padx=5)
        
        # Visualisierung
        self.viz_var = tk.BooleanVar(value=True)
        viz_check = tk.Checkbutton(
            settings_frame,
            text="Visualisierung erstellen",
            variable=self.viz_var,
            fg="#ffffff",
            bg="#2b2b2b",
            selectcolor="#2b2b2b"
        )
        viz_check.pack(pady=5, padx=10, anchor="w")
        
        # Ausgabe
        output_frame = tk.Frame(self.window, bg="#2b2b2b")
        output_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Label(
            output_frame,
            text="Ausgabe-Ordner:",
            fg="#ffffff",
            bg="#2b2b2b",
            font=("Segoe UI", 10)
        ).pack(side="left", padx=5)
        
        self.output_path_var = tk.StringVar()
        output_entry = tk.Entry(
            output_frame,
            textvariable=self.output_path_var,
            width=40,
            bg="#3c3f41",
            fg="#ffffff"
        )
        output_entry.pack(side="left", padx=5, fill="x", expand=True)
        
        output_browse_btn = ttk.Button(
            output_frame,
            text="Durchsuchen",
            command=self.browse_output
        )
        output_browse_btn.pack(side="left", padx=5)
        
        # Analyse-Button
        analyze_btn = ttk.Button(
            self.window,
            text="Analyse starten",
            command=self.run_analysis,
            style="Accent.TButton"
        )
        analyze_btn.pack(pady=20)
        
        # Ergebnis-Anzeige
        result_frame = tk.LabelFrame(
            self.window,
            text="Ergebnisse",
            fg="#ffffff",
            bg="#2b2b2b",
            font=("Segoe UI", 11, "bold")
        )
        result_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        self.result_text = tk.Text(
            result_frame,
            bg="#3c3f41",
            fg="#ffffff",
            font=("Consolas", 9),
            wrap="word",
            height=10
        )
        self.result_text.pack(pady=10, padx=10, fill="both", expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(self.result_text)
        scrollbar.pack(side="right", fill="y")
        self.result_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.result_text.yview)
    
    def browse_file(self):
        """Dateiauswahl-Dialog"""
        filename = filedialog.askopenfilename(
            title="Audio-Datei auswählen",
            filetypes=[
                ("Audio-Dateien", "*.wav *.mp3 *.flac *.m4a *.aac"),
                ("Alle Dateien", "*.*")
            ]
        )
        if filename:
            self.file_path_var.set(filename)
            # Setze automatisch Ausgabe-Ordner
            output_dir = os.path.join(os.path.dirname(filename), "prosody_analysis")
            self.output_path_var.set(output_dir)
    
    def browse_output(self):
        """Ausgabe-Ordner auswählen"""
        folder = filedialog.askdirectory(title="Ausgabe-Ordner auswählen")
        if folder:
            self.output_path_var.set(folder)
    
    def run_analysis(self):
        """Führt die Analyse durch"""
        audio_file = self.file_path_var.get()
        output_dir = self.output_path_var.get()
        
        if not audio_file or not os.path.exists(audio_file):
            messagebox.showerror("Fehler", "Bitte wählen Sie eine gültige Audio-Datei.")
            return
        
        if not output_dir:
            messagebox.showerror("Fehler", "Bitte wählen Sie einen Ausgabe-Ordner.")
            return
        
        # Zeige Fortschritt
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Analyse läuft...\n")
        self.window.update()
        
        try:
            # Führe Analyse durch
            results = self.analyzer.analyze_audio(audio_file, self.gender_var.get())
            
            if results['success']:
                # Speichere Ergebnisse
                os.makedirs(output_dir, exist_ok=True)
                base_name = Path(audio_file).stem
                output_base = os.path.join(output_dir, base_name)
                
                self.analyzer.save_results(results, output_base)
                
                # Erstelle Visualisierung wenn gewünscht
                if self.viz_var.get():
                    viz_path = output_base + "_visualization.png"
                    self.analyzer.create_visualization(audio_file, results, viz_path)
                
                # Zeige Ergebnisse
                self.display_results(results)
                
                messagebox.showinfo("Erfolg", f"Analyse abgeschlossen!\nErgebnisse gespeichert in:\n{output_dir}")
            else:
                self.result_text.insert(tk.END, f"\nFehler: {results.get('error')}\n")
                messagebox.showerror("Fehler", f"Analyse fehlgeschlagen: {results.get('error')}")
                
        except Exception as e:
            self.result_text.insert(tk.END, f"\nFehler: {str(e)}\n")
            messagebox.showerror("Fehler", f"Unerwarteter Fehler: {str(e)}")
    
    def display_results(self, results):
        """Zeigt Analyseergebnisse im Textfeld"""
        self.result_text.delete(1.0, tk.END)
        
        res = results['results']
        
        # Übersicht
        self.result_text.insert(tk.END, "PROSODIEANALYSE ERGEBNISSE\n", "heading")
        self.result_text.insert(tk.END, "=" * 50 + "\n\n")
        
        # Datei-Info
        info = res['file_info']
        self.result_text.insert(tk.END, f"Datei: {info['filename']}\n")
        self.result_text.insert(tk.END, f"Dauer: {info['duration_seconds']} Sekunden\n\n")
        
        # Tonhöhe
        if 'error' not in res['pitch']:
            pitch = res['pitch']
            self.result_text.insert(tk.END, "TONHÖHE (F0):\n", "section")
            self.result_text.insert(tk.END, f"  Durchschnitt: {pitch['mean_f0']} Hz\n")
            self.result_text.insert(tk.END, f"  Bereich: {pitch['min_f0']} - {pitch['max_f0']} Hz\n")
            self.result_text.insert(tk.END, f"  Variabilität: {pitch['pitch_variability']}\n")
            self.result_text.insert(tk.END, f"  Geschlechtstypisch: {'Ja' if pitch['gender_typical'] else 'Nein'}\n\n")
        
        # Sprechgeschwindigkeit
        speech = res['speech_rate']
        self.result_text.insert(tk.END, "SPRECHGESCHWINDIGKEIT:\n", "section")
        self.result_text.insert(tk.END, f"  Wörter/Minute: {speech['words_per_minute']}\n")
        self.result_text.insert(tk.END, f"  Silben/Sekunde: {speech['syllables_per_second']}\n")
        self.result_text.insert(tk.END, f"  Kategorie: {speech['speech_rate_category']}\n\n")
        
        # Pausen
        pauses = res['pauses']
        self.result_text.insert(tk.END, "PAUSEN:\n", "section")
        self.result_text.insert(tk.END, f"  Anzahl: {pauses['pause_count']}\n")
        self.result_text.insert(tk.END, f"  Gesamtdauer: {pauses['total_pause_duration']} Sekunden\n")
        self.result_text.insert(tk.END, f"  Pausenanteil: {pauses['pause_ratio']*100:.1f}%\n\n")
        
        # Stimmqualität
        quality = res['voice_quality']
        self.result_text.insert(tk.END, "STIMMQUALITÄT:\n", "section")
        self.result_text.insert(tk.END, f"  Bewertung: {quality['voice_quality_assessment']}\n")
        self.result_text.insert(tk.END, f"  HNR: {quality['hnr_mean_db']} dB\n")
        self.result_text.insert(tk.END, f"  Jitter: {quality['jitter_local_percent']}%\n")
        self.result_text.insert(tk.END, f"  Shimmer: {quality['shimmer_local_percent']}%\n\n")
        
        # Emotionale Indikatoren
        if res['emotion_indicators']:
            emotion = res['emotion_indicators']
            self.result_text.insert(tk.END, "EMOTIONALE INDIKATOREN:\n", "section")
            self.result_text.insert(tk.END, f"  Erregung: {emotion.get('arousal_level', 'N/A')}\n")
            self.result_text.insert(tk.END, f"  Valenz: {emotion.get('valence_tendency', 'N/A')}\n")
        
        # Formatierung
        self.result_text.tag_config("heading", font=("Segoe UI", 12, "bold"))
        self.result_text.tag_config("section", font=("Segoe UI", 10, "bold"))


# Modifizierte file_processor.py Integration
def enhance_file_processor_with_prosody(file_processor_instance, prosody_settings=None):
    """
    Erweitert eine FileProcessor-Instanz um Prosodieanalyse
    
    Args:
        file_processor_instance: FileProcessor Instanz
        prosody_settings: Dictionary mit Prosody-Einstellungen
    """
    if prosody_settings is None:
        prosody_settings = {
            'enabled': False,
            'gender': 'unknown',
            'create_visualizations': True
        }
    
    # Speichere Original-Methode
    original_perform = file_processor_instance._perform_transcription
    
    def enhanced_perform_transcription(audio_path, output_folder, queue, settings, segment_name=""):
        # Originale Transkription
        original_perform(audio_path, output_folder, queue, settings, segment_name)
        
        # Prosodieanalyse wenn aktiviert
        if prosody_settings.get('enabled', False):
            if queue:
                queue.put(f"status:Analysiere Prosodie{' für Segment' if segment_name else ''}...")
            
            try:
                results = analyze_prosody_for_segment(
                    audio_path,
                    output_folder,
                    segment_name,
                    prosody_settings.get('gender', 'unknown')
                )
                
                if results['success']:
                    if queue:
                        queue.put(f"status:Prosodieanalyse{segment_name} abgeschlossen")
                else:
                    if queue:
                        queue.put(f"status:Prosodieanalyse fehlgeschlagen: {results.get('error')}")
                        
            except Exception as e:
                print(f"Prosodieanalyse-Fehler: {e}")
                if queue:
                    queue.put(f"status:Prosodieanalyse fehlgeschlagen")
    
    # Ersetze Methode
    file_processor_instance._perform_transcription = enhanced_perform_transcription
    
    return file_processor_instance


# Batch-Prosodieanalyse für ArchiveCAT
class BatchProsodyAnalyzer:
    """Batch-Prosodieanalyse für mehrere Dateien"""
    
    def __init__(self, parent_window):
        self.window = tk.Toplevel(parent_window)
        self.window.title("Batch-Prosodieanalyse")
        self.window.geometry("700x600")
        self.window.configure(bg="#2b2b2b")
        
        self.analyzer = ProsodyAnalyzer()
        self.audio_files = []
        self.setup_ui()
    
    def setup_ui(self):
        """Erstellt die UI für Batch-Analyse"""
        # Titel
        title_label = tk.Label(
            self.window,
            text="Batch-Prosodieanalyse",
            fg="#ffffff",
            bg="#2b2b2b",
            font=("Segoe UI", 14, "bold")
        )
        title_label.pack(pady=10)
        
        # Dateiliste
        list_frame = tk.LabelFrame(
            self.window,
            text="Audio-Dateien",
            fg="#ffffff",
            bg="#2b2b2b",
            font=("Segoe UI", 11, "bold")
        )
        list_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        # Listbox mit Scrollbar
        list_container = tk.Frame(list_frame, bg="#2b2b2b")
        list_container.pack(pady=10, padx=10, fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(list_container)
        scrollbar.pack(side="right", fill="y")
        
        self.file_listbox = tk.Listbox(
            list_container,
            bg="#3c3f41",
            fg="#ffffff",
            selectmode="extended",
            yscrollcommand=scrollbar.set
        )
        self.file_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.file_listbox.yview)
        
        # Buttons für Dateimanagement
        button_frame = tk.Frame(list_frame, bg="#2b2b2b")
        button_frame.pack(pady=5)
        
        ttk.Button(button_frame, text="Dateien hinzufügen", command=self.add_files).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Ausgewählte entfernen", command=self.remove_files).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Liste leeren", command=self.clear_files).pack(side="left", padx=5)
        
        # Einstellungen
        settings_frame = tk.LabelFrame(
            self.window,
            text="Einstellungen",
            fg="#ffffff",
            bg="#2b2b2b",
            font=("Segoe UI", 11, "bold")
        )
        settings_frame.pack(pady=10, padx=20, fill="x")
        
        # Geschlecht
        gender_frame = tk.Frame(settings_frame, bg="#2b2b2b")
        gender_frame.pack(pady=5, padx=10, anchor="w")
        
        tk.Label(
            gender_frame,
            text="Geschlecht (für alle):",
            fg="#ffffff",
            bg="#2b2b2b"
        ).pack(side="left", padx=5)
        
        self.gender_var = tk.StringVar(value="unknown")
        ttk.Combobox(
            gender_frame,
            textvariable=self.gender_var,
            values=["unknown", "male", "female"],
            state="readonly",
            width=15
        ).pack(side="left", padx=5)
        
        # Ausgabe
        output_frame = tk.Frame(settings_frame, bg="#2b2b2b")
        output_frame.pack(pady=5, padx=10, fill="x")
        
        tk.Label(
            output_frame,
            text="Ausgabe-Ordner:",
            fg="#ffffff",
            bg="#2b2b2b"
        ).pack(side="left", padx=5)
        
        self.output_var = tk.StringVar()
        tk.Entry(
            output_frame,
            textvariable=self.output_var,
            bg="#3c3f41",
            fg="#ffffff",
            width=40
        ).pack(side="left", padx=5, fill="x", expand=True)
        
        ttk.Button(
            output_frame,
            text="Durchsuchen",
            command=self.browse_output
        ).pack(side="left", padx=5)
        
        # Optionen
        self.viz_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            settings_frame,
            text="Visualisierungen erstellen",
            variable=self.viz_var,
            fg="#ffffff",
            bg="#2b2b2b",
            selectcolor="#2b2b2b"
        ).pack(pady=5, padx=10, anchor="w")
        
        self.summary_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            settings_frame,
            text="Zusammenfassungsbericht erstellen",
            variable=self.summary_var,
            fg="#ffffff",
            bg="#2b2b2b",
            selectcolor="#2b2b2b"
        ).pack(pady=5, padx=10, anchor="w")
        
        # Analyse starten
        ttk.Button(
            self.window,
            text="Batch-Analyse starten",
            command=self.run_batch_analysis,
            style="Accent.TButton"
        ).pack(pady=20)
        
        # Fortschrittsanzeige
        self.progress_var = tk.StringVar(value="Bereit")
        progress_label = tk.Label(
            self.window,
            textvariable=self.progress_var,
            fg="#00ff00",
            bg="#2b2b2b",
            font=("Segoe UI", 10)
        )
        progress_label.pack(pady=5)
        
        self.progress_bar = ttk.Progressbar(
            self.window,
            length=400,
            mode='determinate'
        )
        self.progress_bar.pack(pady=10)
    
    def add_files(self):
        """Fügt Dateien zur Liste hinzu"""
        files = filedialog.askopenfilenames(
            title="Audio-Dateien auswählen",
            filetypes=[
                ("Audio-Dateien", "*.wav *.mp3 *.flac *.m4a *.aac"),
                ("Alle Dateien", "*.*")
            ]
        )
        
        for file in files:
            if file not in self.audio_files:
                self.audio_files.append(file)
                self.file_listbox.insert(tk.END, os.path.basename(file))
        
        # Setze Ausgabe-Ordner wenn noch nicht gesetzt
        if files and not self.output_var.get():
            self.output_var.set(os.path.join(os.path.dirname(files[0]), "batch_prosody_analysis"))
    
    def remove_files(self):
        """Entfernt ausgewählte Dateien"""
        selections = self.file_listbox.curselection()
        for index in reversed(selections):
            self.file_listbox.delete(index)
            del self.audio_files[index]
    
    def clear_files(self):
        """Leert die Dateiliste"""
        self.file_listbox.delete(0, tk.END)
        self.audio_files = []
    
    def browse_output(self):
        """Wählt Ausgabe-Ordner"""
        folder = filedialog.askdirectory(title="Ausgabe-Ordner wählen")
        if folder:
            self.output_var.set(folder)
    
    def run_batch_analysis(self):
        """Führt Batch-Analyse durch"""
        if not self.audio_files:
            messagebox.showerror("Fehler", "Keine Dateien ausgewählt.")
            return
        
        output_dir = self.output_var.get()
        if not output_dir:
            messagebox.showerror("Fehler", "Kein Ausgabe-Ordner angegeben.")
            return
        
        # Erstelle Ausgabe-Ordner
        os.makedirs(output_dir, exist_ok=True)
        
        # Deaktiviere UI während Analyse
        self.window.update()
        
        # Führe Analyse durch
        total_files = len(self.audio_files)
        self.progress_bar['maximum'] = total_files
        
        results = {}
        for i, audio_file in enumerate(self.audio_files):
            self.progress_var.set(f"Analysiere {i+1}/{total_files}: {os.path.basename(audio_file)}")
            self.progress_bar['value'] = i
            self.window.update()
            
            try:
                result = self.analyzer.analyze_audio(audio_file, self.gender_var.get())
                results[audio_file] = result
                
                if result['success']:
                    # Speichere Ergebnisse
                    base_name = Path(audio_file).stem
                    output_base = os.path.join(output_dir, base_name)
                    self.analyzer.save_results(result, output_base)
                    
                    # Visualisierung
                    if self.viz_var.get():
                        viz_path = output_base + "_visualization.png"
                        self.analyzer.create_visualization(audio_file, result, viz_path)
                        
            except Exception as e:
                print(f"Fehler bei {audio_file}: {e}")
                results[audio_file] = {'success': False, 'error': str(e)}
        
        # Erstelle Zusammenfassung
        if self.summary_var.get():
            create_prosody_summary(results, output_dir)
        
        self.progress_var.set("Analyse abgeschlossen!")
        self.progress_bar['value'] = total_files
        
        # Zeige Ergebnis
        successful = len([r for r in results.values() if r['success']])
        messagebox.showinfo(
            "Erfolg", 
            f"Batch-Analyse abgeschlossen!\n"
            f"{successful} von {total_files} Dateien erfolgreich analysiert.\n\n"
            f"Ergebnisse in: {output_dir}"
        )


# Menü-Integration für ArchiveCAT
def add_prosody_menu_items(menu_bar, root):
    """Fügt Prosodieanalyse-Menüpunkte hinzu"""
    # Erstelle Analyse-Menü wenn noch nicht vorhanden
    analysis_menu = None
    for i in range(menu_bar.index("end") + 1):
        try:
            if menu_bar.entryconfig(i)['label'][4] == "Analyse":
                analysis_menu = menu_bar.nametowidget(menu_bar.entryconfig(i)['menu'][4])
                break
        except:
            continue
    
    if not analysis_menu:
        analysis_menu = tk.Menu(menu_bar, tearoff=0, bg="#2b2b2b", fg="#ffffff")
        menu_bar.add_cascade(label="Analyse", menu=analysis_menu)
    
    # Füge Separator hinzu wenn Menü bereits Einträge hat
    try:
        if analysis_menu.index("end") is not None:
            analysis_menu.add_separator()
    except:
        pass
    
    # Füge Prosodieanalyse-Einträge hinzu
    analysis_menu.add_command(
        label="Prosodieanalyse (Einzeldatei)",
        command=lambda: ProsodyAnalysisWindow(root)
    )
    
    analysis_menu.add_command(
        label="Batch-Prosodieanalyse",
        command=lambda: BatchProsodyAnalyzer(root)
    )
    
    # Sprecher-Vergleich
    analysis_menu.add_command(
        label="Sprecher-Vergleich",
        command=lambda: SpeakerComparisonWindow(root)
    )


class SpeakerComparisonWindow:
    """Fenster für Sprecher-Vergleich"""
    
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("Sprecher-Vergleich")
        self.window.geometry("700x500")
        self.window.configure(bg="#2b2b2b")
        
        self.speakers = {}
        self.comparator = None
        self.setup_ui()
    
    def setup_ui(self):
        """Erstellt UI für Sprecher-Vergleich"""
        # Titel
        title_label = tk.Label(
            self.window,
            text="Prosodischer Sprecher-Vergleich",
            fg="#ffffff",
            bg="#2b2b2b",
            font=("Segoe UI", 14, "bold")
        )
        title_label.pack(pady=10)
        
        # Sprecher-Frame
        speaker_frame = tk.LabelFrame(
            self.window,
            text="Sprecher",
            fg="#ffffff",
            bg="#2b2b2b",
            font=("Segoe UI", 11, "bold")
        )
        speaker_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        # Sprecher-Liste
        list_container = tk.Frame(speaker_frame, bg="#2b2b2b")
        list_container.pack(pady=10, padx=10, fill="both", expand=True)
        
        # Header
        header_frame = tk.Frame(list_container, bg="#2b2b2b")
        header_frame.pack(fill="x")
        
        tk.Label(header_frame, text="Name", fg="#ffffff", bg="#2b2b2b", width=20, anchor="w").pack(side="left", padx=5)
        tk.Label(header_frame, text="Audio-Datei", fg="#ffffff", bg="#2b2b2b", anchor="w").pack(side="left", fill="x", expand=True)
        
        # Listbox für Sprecher
        self.speaker_listbox = tk.Listbox(
            list_container,
            bg="#3c3f41",
            fg="#ffffff",
            selectmode="single",
            height=6
        )
        self.speaker_listbox.pack(fill="both", expand=True)
        
        # Buttons
        button_frame = tk.Frame(speaker_frame, bg="#2b2b2b")
        button_frame.pack(pady=5)
        
        ttk.Button(button_frame, text="Sprecher hinzufügen", command=self.add_speaker).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Entfernen", command=self.remove_speaker).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Alle entfernen", command=self.clear_speakers).pack(side="left", padx=5)
        
        # Ausgabe
        output_frame = tk.Frame(self.window, bg="#2b2b2b")
        output_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Label(
            output_frame,
            text="Ausgabe-Ordner:",
            fg="#ffffff",
            bg="#2b2b2b"
        ).pack(side="left", padx=5)
        
        self.output_var = tk.StringVar()
        tk.Entry(
            output_frame,
            textvariable=self.output_var,
            bg="#3c3f41",
            fg="#ffffff",
            width=40
        ).pack(side="left", padx=5, fill="x", expand=True)
        
        ttk.Button(
            output_frame,
            text="Durchsuchen",
            command=self.browse_output
        ).pack(side="left", padx=5)
        
        # Vergleich starten
        ttk.Button(
            self.window,
            text="Vergleich starten",
            command=self.run_comparison,
            style="Accent.TButton"
        ).pack(pady=20)
        
        # Status
        self.status_var = tk.StringVar(value="Bereit")
        status_label = tk.Label(
            self.window,
            textvariable=self.status_var,
            fg="#00ff00",
            bg="#2b2b2b"
        )
        status_label.pack()
    
    def add_speaker(self):
        """Fügt einen Sprecher hinzu"""
        dialog = tk.Toplevel(self.window)
        dialog.title("Sprecher hinzufügen")
        dialog.geometry("400x150")
        dialog.configure(bg="#2b2b2b")
        
        # Name
        name_frame = tk.Frame(dialog, bg="#2b2b2b")
        name_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Label(name_frame, text="Name:", fg="#ffffff", bg="#2b2b2b", width=10).pack(side="left")
        name_var = tk.StringVar()
        name_entry = tk.Entry(name_frame, textvariable=name_var, bg="#3c3f41", fg="#ffffff")
        name_entry.pack(side="left", fill="x", expand=True)
        
        # Datei
        file_frame = tk.Frame(dialog, bg="#2b2b2b")
        file_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Label(file_frame, text="Audio:", fg="#ffffff", bg="#2b2b2b", width=10).pack(side="left")
        file_var = tk.StringVar()
        file_entry = tk.Entry(file_frame, textvariable=file_var, bg="#3c3f41", fg="#ffffff")
        file_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        def browse_audio():
            filename = filedialog.askopenfilename(
                title="Audio-Datei wählen",
                filetypes=[("Audio", "*.wav *.mp3 *.flac *.m4a"), ("Alle", "*.*")]
            )
            if filename:
                file_var.set(filename)
        
        ttk.Button(file_frame, text="...", command=browse_audio, width=3).pack(side="left")
        
        # Buttons
        button_frame = tk.Frame(dialog, bg="#2b2b2b")
        button_frame.pack(pady=20)
        
        def add():
            name = name_var.get().strip()
            file = file_var.get().strip()
            
            if name and file and os.path.exists(file):
                self.speakers[name] = file
                self.speaker_listbox.insert(tk.END, f"{name} - {os.path.basename(file)}")
                dialog.destroy()
            else:
                messagebox.showerror("Fehler", "Bitte Name und gültige Datei angeben.")
        
        ttk.Button(button_frame, text="Hinzufügen", command=add).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Abbrechen", command=dialog.destroy).pack(side="left", padx=5)
    
    def remove_speaker(self):
        """Entfernt ausgewählten Sprecher"""
        selection = self.speaker_listbox.curselection()
        if selection:
            index = selection[0]
            item = self.speaker_listbox.get(index)
            name = item.split(" - ")[0]
            del self.speakers[name]
            self.speaker_listbox.delete(index)
    
    def clear_speakers(self):
        """Entfernt alle Sprecher"""
        self.speakers = {}
        self.speaker_listbox.delete(0, tk.END)
    
    def browse_output(self):
        """Wählt Ausgabe-Ordner"""
        folder = filedialog.askdirectory(title="Ausgabe-Ordner wählen")
        if folder:
            self.output_var.set(folder)
    
    def run_comparison(self):
        """Führt Sprecher-Vergleich durch"""
        if len(self.speakers) < 2:
            messagebox.showerror("Fehler", "Mindestens 2 Sprecher für Vergleich nötig.")
            return
        
        output_dir = self.output_var.get()
        if not output_dir:
            messagebox.showerror("Fehler", "Bitte Ausgabe-Ordner wählen.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.status_var.set("Analysiere Sprecher...")
        self.window.update()
        
        try:
            from prosody_analyzer import ProsodyComparator
            self.comparator = ProsodyComparator()
            
            comparison = self.comparator.compare_speakers(self.speakers, output_dir)
            
            self.status_var.set("Vergleich abgeschlossen!")
            
            # Zeige Ergebnisse
            result_text = "SPRECHER-VERGLEICH ERGEBNISSE\n\n"
            
            for feature, values in comparison['feature_comparison'].items():
                result_text += f"{feature}:\n"
                for speaker, value in values.items():
                    result_text += f"  {speaker}: {value:.2f}\n"
                result_text += "\n"
            
            if comparison['significant_differences']:
                result_text += "Signifikante Unterschiede:\n"
                for diff in comparison['significant_differences']:
                    result_text += f"  • {diff}\n"
            
            # Zeige in neuem Fenster
            result_window = tk.Toplevel(self.window)
            result_window.title("Vergleichsergebnisse")
            result_window.geometry("500x400")
            result_window.configure(bg="#2b2b2b")
            
            text_widget = tk.Text(
                result_window,
                bg="#3c3f41",
                fg="#ffffff",
                font=("Consolas", 10),
                wrap="word"
            )
            text_widget.pack(pady=10, padx=10, fill="both", expand=True)
            text_widget.insert(1.0, result_text)
            text_widget.config(state="disabled")
            
            messagebox.showinfo("Erfolg", f"Vergleich abgeschlossen!\nErgebnisse in: {output_dir}")
            
        except Exception as e:
            self.status_var.set("Fehler beim Vergleich")
            messagebox.showerror("Fehler", f"Vergleich fehlgeschlagen: {str(e)}")


# Beispiel-Integration in ArchiveCAT
if __name__ == "__main__":
    # Test-Window
    root = tk.Tk()
    root.withdraw()
    
    # Teste einzelne Komponenten
    # window = ProsodyAnalysisWindow(root)
    # window = BatchProsodyAnalyzer(root)
    window = SpeakerComparisonWindow(root)
    
    root.mainloop()