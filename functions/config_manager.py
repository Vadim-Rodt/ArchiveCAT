# config_manager.py
"""
Modul zur Verwaltung von Konfigurationen und Einstellungen
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigManager:
    """Verwaltet Anwendungskonfigurationen"""
    
    def __init__(self, config_file: str = "archivecat_config.json"):
        self.config_file = config_file
        self.config = {}
        self.download_dir = None
        self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Lädt die gespeicherte Konfiguration"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                    self.download_dir = self.config.get('download_dir', None)
                    return self.config
        except Exception as e:
            print(f"Fehler beim Laden der Konfiguration: {e}")
        return {}
    
    def save_config(self) -> bool:
        """Speichert die aktuelle Konfiguration"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Fehler beim Speichern der Konfiguration: {e}")
            return False
    
    def get_download_dir(self) -> Optional[str]:
        """Gibt das Download-Verzeichnis zurück"""
        return self.download_dir
    
    def set_download_dir(self, directory: str) -> bool:
        """Setzt das Download-Verzeichnis"""
        if os.path.exists(directory):
            self.download_dir = directory
            self.config['download_dir'] = directory
            return self.save_config()
        return False
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Holt eine spezifische Einstellung"""
        return self.config.get(key, default)
    
    def set_setting(self, key: str, value: Any) -> bool:
        """Setzt eine spezifische Einstellung"""
        self.config[key] = value
        return self.save_config()
    
    def get_last_used_settings(self) -> Dict[str, Any]:
        """Gibt die zuletzt verwendeten Einstellungen zurück"""
        return {
            'language': self.get_setting('last_language', 'de'),
            'export_formats': self.get_setting('last_export_formats', {
                'txt': True,
                'json': True,
                'xml': False,
                'srt': True
            }),
            'use_speakers': self.get_setting('last_use_speakers', False)
        }
    
    def save_last_used_settings(self, settings: Dict[str, Any]) -> bool:
        """Speichert die zuletzt verwendeten Einstellungen"""
        for key, value in settings.items():
            self.set_setting(f'last_{key}', value)
        return self.save_config()