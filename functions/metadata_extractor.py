"""
Metadata Extractor Module
Extracts metadata from Archive.org HTML pages and creates CSV files with video information.
"""

import re
import csv
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import logging
from typing import Dict, Optional, List, Tuple

class MetadataExtractor:
    """Extractor for Archive.org video metadata"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def fetch_html_content(self, url: str) -> Optional[str]:
        """
        Fetch HTML content from the given URL
        
        Args:
            url: The URL to fetch content from
            
        Returns:
            HTML content as string, or None if failed
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.text
        except Exception as e:
            self.logger.error(f"Failed to fetch HTML from {url}: {e}")
            return None
    
    def extract_title(self, soup: BeautifulSoup) -> str:
        """
        Extract title from HTML
        
        Args:
            soup: BeautifulSoup object of the HTML
            
        Returns:
            Title string or empty string if not found
        """
        try:
            # Look for <h1 class="item-title"><span class="breaker-breaker" itemprop="name">
            title_element = soup.find('h1', class_='item-title')
            if title_element:
                span_element = title_element.find('span', class_='breaker-breaker')
                if span_element and span_element.get('itemprop') == 'name':
                    return span_element.get_text(strip=True)
                    
            # Fallback: try to find any h1 with item-title class
            if title_element:
                return title_element.get_text(strip=True)
                
            return ""
        except Exception as e:
            self.logger.error(f"Error extracting title: {e}")
            return ""
    
    def extract_creator(self, soup: BeautifulSoup) -> str:
        """
        Extract creator from HTML metadata
        
        Args:
            soup: BeautifulSoup object of the HTML
            
        Returns:
            Creator name or empty string if not found
        """
        try:
            # Look for <dl class="metadata-definition"><dt>by</dt><dd>...
            metadata_sections = soup.find_all('dl', class_='metadata-definition')
            
            for section in metadata_sections:
                dt_element = section.find('dt')
                if dt_element and dt_element.get_text(strip=True).lower() == 'by':
                    dd_element = section.find('dd')
                    if dd_element:
                        # Look for link with creator name
                        link = dd_element.find('a')
                        if link:
                            return link.get_text(strip=True)
                        else:
                            return dd_element.get_text(strip=True)
            return ""
        except Exception as e:
            self.logger.error(f"Error extracting creator: {e}")
            return ""
    
    def extract_uploader(self, soup: BeautifulSoup) -> str:
        """
        Extract uploader from HTML
        
        Args:
            soup: BeautifulSoup object of the HTML
            
        Returns:
            Uploader name or empty string if not found
        """
        try:
            # Look for <a href="/details/@username" class="item-upload-info__uploader-name">
            uploader_element = soup.find('a', class_='item-upload-info__uploader-name')
            if uploader_element:
                return uploader_element.get_text(strip=True)
            return ""
        except Exception as e:
            self.logger.error(f"Error extracting uploader: {e}")
            return ""
    
    def extract_upload_date(self, soup: BeautifulSoup) -> str:
        """
        Extract upload date from HTML
        
        Args:
            soup: BeautifulSoup object of the HTML
            
        Returns:
            Upload date string or empty string if not found
        """
        try:
            # Look for <time itemprop="uploadDate">
            time_element = soup.find('time', itemprop='uploadDate')
            if time_element:
                return time_element.get_text(strip=True)
            return ""
        except Exception as e:
            self.logger.error(f"Error extracting upload date: {e}")
            return ""
    
    def extract_topics(self, soup: BeautifulSoup) -> str:
        """
        Extract topics/keywords from HTML
        
        Args:
            soup: BeautifulSoup object of the HTML
            
        Returns:
            Comma-separated topics string or empty string if not found
        """
        try:
            # Look for <dd class="" itemprop="keywords">
            keywords_element = soup.find('dd', itemprop='keywords')
            if keywords_element:
                # Extract all link texts
                links = keywords_element.find_all('a')
                topics = []
                for link in links:
                    topic = link.get_text(strip=True)
                    if topic:  # Only add non-empty topics
                        topics.append(topic)
                
                return ', '.join(topics)
            return ""
        except Exception as e:
            self.logger.error(f"Error extracting topics: {e}")
            return ""
    
    def extract_original_url(self, soup: BeautifulSoup) -> str:
        """
        Extract original URL from HTML metadata
        
        Args:
            soup: BeautifulSoup object of the HTML
            
        Returns:
            Original URL string or empty string if not found
        """
        try:
            # Look for <dl class="metadata-definition"><dt>Originalurl</dt>
            metadata_sections = soup.find_all('dl', class_='metadata-definition')
            
            for section in metadata_sections:
                dt_element = section.find('dt')
                if dt_element and dt_element.get_text(strip=True).lower() == 'originalurl':
                    dd_element = section.find('dd')
                    if dd_element:
                        # Look for link with original URL
                        link = dd_element.find('a')
                        if link:
                            return link.get('href', '') or link.get_text(strip=True)
                        else:
                            return dd_element.get_text(strip=True)
            return ""
        except Exception as e:
            self.logger.error(f"Error extracting original URL: {e}")
            return ""
    
    def read_transcript(self, transcript_path: str) -> str:
        """
        Read transcript from file
        
        Args:
            transcript_path: Path to transcript file
            
        Returns:
            Transcript content or empty string if not found
        """
        try:
            if os.path.exists(transcript_path):
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            return ""
        except Exception as e:
            self.logger.error(f"Error reading transcript from {transcript_path}: {e}")
            return ""
    
    def extract_prosody_values(self, prosody_path: str) -> Tuple[str, str]:
        try:
            if not os.path.exists(prosody_path):
                return "", ""
                
            with open(prosody_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Mehrere Suchmuster für Arousal
            arousal_patterns = [
                r'Arousal\s*Level[:\s]+([^\n\r]+)',
                r'Overall\s+Arousal[:\s]+([^\n\r]+)',
                r'Erregung[:\s]+([^\n\r]+)'
            ]
            
            arousal_level = ""
            for pattern in arousal_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    arousal_level = match.group(1).strip()
                    break
            
            # Mehrere Suchmuster für Valence
            valence_patterns = [
                r'Valence[:\s]+([^\n\r]+)',
                r'Overall\s+Valence[:\s]+([^\n\r]+)',
                r'Valenz[:\s]+([^\n\r]+)'
            ]
            
            valence = ""
            for pattern in valence_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    valence = match.group(1).strip()
                    break
            
            return arousal_level, valence
            
        except Exception as e:
            return "", ""
    
    def create_metadata_record(self, 
                             source_url: str,
                             html_content: str = None,
                             transcript_path: str = None,
                             prosody_path: str = None,
                             timestamp_start: str = "",
                             timestamp_end: str = "") -> Dict[str, str]:
        """
        Create a metadata record from various sources
        
        Args:
            source_url: The original URL/link
            html_content: HTML content string (if None, will try to fetch from URL)
            transcript_path: Path to transcript file
            prosody_path: Path to prosody analysis file
            timestamp_start: Start timestamp
            timestamp_end: End timestamp
            
        Returns:
            Dictionary with metadata fields
        """
        # Initialize record with empty values
        record = {
            'source': source_url,
            'title': '',
            'creator': '',
            'uploader': '',
            'upload_date': '',
            'topics': '',
            'original_url': '',
            'timestamp_start': timestamp_start,
            'timestamp_end': timestamp_end,
            'transcript': '',
            'arousal_level': '',
            'valence': ''
        }
        
        # Fetch HTML content if not provided
        if html_content is None:
            html_content = self.fetch_html_content(source_url)
        
        # Extract metadata from HTML
        if html_content:
            try:
                soup = BeautifulSoup(html_content, 'html.parser')
                
                record['title'] = self.extract_title(soup)
                record['creator'] = self.extract_creator(soup)
                record['uploader'] = self.extract_uploader(soup)
                record['upload_date'] = self.extract_upload_date(soup)
                record['topics'] = self.extract_topics(soup)
                record['original_url'] = self.extract_original_url(soup)
                
            except Exception as e:
                self.logger.error(f"Error parsing HTML content: {e}")
        
        # Read transcript
        if transcript_path:
            record['transcript'] = self.read_transcript(transcript_path)
        
        # Extract prosody values
        if prosody_path:
            arousal, valence = self.extract_prosody_values(prosody_path)
            record['arousal_level'] = arousal
            record['valence'] = valence
        
        return record
    
    def save_metadata_csv(self, record: Dict[str, str], output_path: str) -> bool:
        """
        Save metadata record to CSV file
        
        Args:
            record: Metadata record dictionary
            output_path: Path to output CSV file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Define CSV columns in the required order
            columns = [
                'source', 'title', 'creator', 'uploader', 'upload_date',
                'topics', 'original_url', 'timestamp_start', 'timestamp_end',
                'transcript', 'arousal_level', 'valence'
            ]
            
            # Check if file exists to determine if we need to write headers
            file_exists = os.path.exists(output_path)
            
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=columns)
                
                # Write header if file is new
                if not file_exists:
                    writer.writeheader()
                
                # Write the record
                writer.writerow(record)
            
            self.logger.info(f"Metadata saved to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving metadata to {output_path}: {e}")
            return False
    
    def generate_metadata_filename(self, base_filename: str) -> str:
        """
        Generate metadata CSV filename from base filename
        
        Args:
            base_filename: Base filename (e.g., "video.mp4")
            
        Returns:
            Metadata CSV filename (e.g., "video_metadata.csv")
        """
        # Remove extension and add _metadata.csv
        name_without_ext = os.path.splitext(base_filename)[0]
        return f"{name_without_ext}_metadata.csv"


class MetadataProcessor:
    """Main processor for handling metadata extraction workflow"""
    
    def __init__(self):
        self.extractor = MetadataExtractor()
        self.logger = logging.getLogger(__name__)
    
    def process_video_metadata(self,
                            source_url: str,
                            output_dir: str,
                            base_filename: str,
                            timestamp_start: str = "",
                            timestamp_end: str = "",
                            html_content: str = None) -> Optional[str]:
        """
        Process metadata for a video and save to CSV
        """
        try:
            print(f"DEBUG: Looking for files in: {output_dir}")
            
            # Find transcript files - look for multiple possible patterns
            transcript_files = []
            prosody_files = []
            
            # Search recursively in output directory and subdirectories
            for root_dir, dirs, files in os.walk(output_dir):
                print(f"DEBUG: Searching in directory: {root_dir}")
                print(f"DEBUG: Files found: {files}")
                
                for file in files:
                    file_path = os.path.join(root_dir, file)
                    
                    # Look for transcript files
                    if file.endswith('.txt') and not file.endswith('_metadata.csv'):
                        # Common transcript patterns
                        if any(pattern in file.lower() for pattern in ['transkript', 'transcript', 'transcription']):
                            transcript_files.append(file_path)
                            print(f"DEBUG: Found transcript file: {file_path}")
                        elif file == f"{os.path.splitext(base_filename)[0]}.txt":
                            transcript_files.append(file_path)
                            print(f"DEBUG: Found transcript file by name: {file_path}")
                    
                    # Look for prosody files
                    if file.lower() in ['prosody_analysis.txt', 'prosody.txt', 'prosody_results.txt']:
                        prosody_files.append(file_path)
                        print(f"DEBUG: Found prosody file: {file_path}")
            
            # Use the first transcript and prosody files found
            transcript_path = transcript_files[0] if transcript_files else None
            prosody_path = prosody_files[0] if prosody_files else None
            
            print(f"DEBUG: Using transcript: {transcript_path}")
            print(f"DEBUG: Using prosody: {prosody_path}")
            
            # Create metadata record
            record = self.extractor.create_metadata_record(
                source_url=source_url,
                html_content=html_content,
                transcript_path=transcript_path,
                prosody_path=prosody_path,
                timestamp_start=timestamp_start,
                timestamp_end=timestamp_end
            )
            
            # Generate output CSV path
            csv_filename = self.extractor.generate_metadata_filename(base_filename)
            csv_path = os.path.join(output_dir, csv_filename)
            
            # Save to CSV
            if self.extractor.save_metadata_csv(record, csv_path):
                return csv_path
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error processing video metadata: {e}")
            print(f"DEBUG: Error in process_video_metadata: {e}")
            import traceback
            traceback.print_exc()
            return None