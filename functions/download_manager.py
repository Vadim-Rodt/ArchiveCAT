# download_manager.py
"""
Modul für Download-Funktionen von archive.org
"""

import os
import re
import time
import requests
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from typing import Optional, Tuple, List
from pathlib import Path

class ArchiveDownloader:
    """Klasse zum Herunterladen von Videos von archive.org"""
    
    def __init__(self):
        self.chrome_options = self._setup_chrome_options()
    
    def _setup_chrome_options(self) -> Options:
        """Konfiguriert Chrome-Optionen"""
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--window-size=1920,1080')
        options.add_experimental_option("excludeSwitches", ["enable-logging"])
        options.add_argument('--log-level=3')
        return options
    
    def get_video_metadata(self, url: str) -> Tuple[Optional[str], Optional[int]]:
        """
        Holt Video-Metadaten von archive.org
        
        Returns:
            (duration_string, duration_seconds) oder (None, None) bei Fehler
        """
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
            print(f"Fehler beim Abrufen der Metadaten: {e}")
        return None, None
    
    def find_video_link(self, driver, url: str) -> Optional[str]:
        """
        Findet den direkten Video-Download-Link
        
        Args:
            driver: Selenium WebDriver
            url: Archive.org URL
            
        Returns:
            Direkter Video-Link oder None
        """
        try:
            driver.get(url)
            wait = WebDriverWait(driver, 10)
            
            # Versuche verschiedene Methoden, den Video-Link zu finden
            
            # Methode 1: MPEG4 Button
            try:
                mpeg4_button = wait.until(
                    EC.element_to_be_clickable(
                        (By.XPATH, "//a[contains(@class,'js-archive-expand_files') and contains(text(),'MPEG4')]")
                    )
                )
                mpeg4_button.click()
                wait.until(EC.visibility_of_element_located((By.CLASS_NAME, "quickdown")))
                time.sleep(1)
                
                pills = driver.find_elements(By.CSS_SELECTOR, ".quickdown a.download-pill")
                for pill in pills:
                    href = pill.get_attribute("href")
                    if href and href.endswith(".mp4"):
                        return urljoin(url, href)
            except:
                pass
            
            # Methode 2: Direkte Download-Pills
            pills = driver.find_elements(By.CSS_SELECTOR, "a.download-pill")
            for pill in pills:
                href = pill.get_attribute("href")
                if href and href.endswith(".mp4"):
                    return urljoin(url, href)
            
            # Methode 3: H.264 Format-Gruppe
            format_groups = driver.find_elements(By.CSS_SELECTOR, ".format-group")
            for group in format_groups:
                if "H.264" in group.text:
                    pills = group.find_elements(By.CSS_SELECTOR, "a.download-pill")
                    for pill in pills:
                        href = pill.get_attribute("href")
                        if href and href.endswith(".mp4"):
                            return urljoin(url, href)
            
            return None
            
        except Exception as e:
            print(f"Fehler beim Finden des Video-Links: {e}")
            return None
    
    def get_video_title(self, driver) -> str:
        """Extrahiert den Video-Titel"""
        try:
            wait = WebDriverWait(driver, 10)
            title_element = wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, "item-title"))
            )
            return title_element.text.strip()  # Return raw title, don't sanitize here
        except:
            return "archive_item"
    
    def _sanitize_filename(self, name: str, max_length: int = 50) -> str:
        """Bereinigt Dateinamen"""
        name = re.sub(r'[\\/*?:"<>|]', '', name).strip()
        return name[:max_length]
    
    def download_file(self, url: str, output_path: str, queue=None) -> bool:
        """
        Lädt eine Datei herunter mit Fortschrittsanzeige
        
        Args:
            url: Download-URL
            output_path: Ausgabepfad
            queue: Optional Queue für Statusupdates
            
        Returns:
            True bei Erfolg
        """
        try:
            with requests.get(url, stream=True) as response:
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if queue and total_size > 0:
                                progress = (downloaded / total_size) * 100
                                queue.put(f"status:Download {progress:.1f}%")
                
                return True
                
        except Exception as e:
            print(f"Download-Fehler: {e}")
            if queue:
                queue.put(f"error:Download fehlgeschlagen: {str(e)}")
            return False
    
    def download_video_from_archive(self, url: str, download_dir: str, queue=None) -> dict:
        """
        Haupt-Download-Funktion für archive.org Videos
        
        Returns:
            Dictionary mit Ergebnis-Informationen
        """
        result = {
            'success': False,
            'video_path': None,
            'video_folder': None,
            'title': None,
            'duration': None,
            'error': None
        }
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=self.chrome_options)
        
        try:
            if queue:
                queue.put("status:Lade Webseite...")
            
            # Hole Video-Titel
            result['title'] = self.get_video_title(driver)
            
            # NEUE LOGIK: Erstelle Ordnername aus ersten 20 Zeichen des Titels
            if result['title']:
                # Bereinige den Titel und nehme nur die ersten 20 Zeichen
                folder_name = self._create_folder_name_from_title(result['title'])
            else:
                # Fallback wenn kein Titel gefunden
                folder_name = "archive_item"
            
            print(f"DEBUG: Original title: '{result['title']}'")
            print(f"DEBUG: Folder name: '{folder_name}'")
            
            # Finde Video-Link
            if queue:
                queue.put("status:Suche Video-Link...")
            
            video_link = self.find_video_link(driver, url)
            if not video_link:
                result['error'] = "Kein Video-Link gefunden"
                return result
            
            # Erstelle Ausgabepfad mit neuem Ordnernamen
            ext = video_link.split('.')[-1]
            filename = f"{folder_name}.{ext}"  # Verwende folder_name statt result['title']
            video_folder = os.path.join(download_dir, folder_name)
            os.makedirs(video_folder, exist_ok=True)
            
            temp_filepath = os.path.join(video_folder, f"_temp_{filename}")
            
            # Download
            if queue:
                queue.put("status:Lade Video herunter...")
            
            if self.download_file(video_link, temp_filepath, queue):
                # Benenne temporäre Datei um
                final_path = os.path.join(video_folder, filename)
                if os.path.exists(temp_filepath):
                    if os.path.exists(final_path):
                        os.remove(final_path)
                    os.rename(temp_filepath, final_path)
                    
                    result['success'] = True
                    result['video_path'] = final_path
                    result['video_folder'] = video_folder
                else:
                    result['error'] = "Download-Datei nicht gefunden"
            else:
                result['error'] = "Download fehlgeschlagen"
            
        except Exception as e:
            result['error'] = str(e)
            
        finally:
            driver.quit()
        
        return result
    
    def _create_folder_name_from_title(self, title: str) -> str:
        """Creates folder name from first 20 characters of title"""
        if not title or title.strip() == "":
            return "archive_item"
        
        # Clean invalid characters
        sanitized = re.sub(r'[\\/*?:"<>|]', '', title)
        sanitized = re.sub(r'[^\w\s\-_.]', '', sanitized)
        sanitized = re.sub(r'\s+', '_', sanitized)
        sanitized = sanitized.strip('_-.')
        
        # Take only first 20 characters
        folder_name = sanitized[:20]
        
        return folder_name if folder_name else "archive_item"