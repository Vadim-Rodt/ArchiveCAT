# functions/xml_exporter.py
"""
Modul für den Export von Transkriptionen in verschiedene Formate.
Unterstützt XML, CSV, WebVTT und weitere Formate.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
import html
import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging

# Logging konfigurieren
logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Unterstützte Export-Formate"""
    TXT = "txt"
    JSON = "json"
    XML = "xml"
    SRT = "srt"
    CSV = "csv"
    WEBVTT = "vtt"
    TTML = "ttml"  # Timed Text Markup Language
    

@dataclass
class ExportConfig:
    """Konfiguration für Export-Optionen"""
    formats: Dict[ExportFormat, bool]
    include_metadata: bool = True
    include_statistics: bool = True
    pretty_print: bool = True
    encoding: str = "utf-8"
    csv_delimiter: str = ","
    timestamp_format: str = "seconds"  # "seconds", "milliseconds", "timecode"
    

class TranscriptionExporter:
    """
    Universeller Exporter für Transkriptionsdaten.
    Unterstützt multiple Formate und ist erweiterbar.
    """
    
    def __init__(self, format_version: str = "1.0"):
        self.format_version = format_version
        self._exporters = {
            ExportFormat.XML: self._export_xml,
            ExportFormat.CSV: self._export_csv,
            ExportFormat.WEBVTT: self._export_webvtt,
            ExportFormat.TTML: self._export_ttml,
        }
    
    def export(self, 
               transcription_data: Dict[str, Any], 
               output_path: Union[str, Path], 
               config: Optional[ExportConfig] = None) -> Dict[ExportFormat, bool]:
        """
        Exportiert Transkription in konfigurierte Formate.
        
        Args:
            transcription_data: Transkriptionsdaten
            output_path: Basis-Ausgabepfad (ohne Erweiterung)
            config: Export-Konfiguration
            
        Returns:
            Dict mit Format -> Erfolg
        """
        if config is None:
            config = ExportConfig(formats={
                ExportFormat.XML: True,
                ExportFormat.JSON: True,
                ExportFormat.TXT: True,
                ExportFormat.SRT: True,
                ExportFormat.CSV: False,
                ExportFormat.WEBVTT: False,
                ExportFormat.TTML: False
            })
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results = {}
        
        # Validiere Daten
        if not self._validate_data(transcription_data):
            logger.error("Ungültige Transkriptionsdaten")
            return {fmt: False for fmt in config.formats}
        
        # Exportiere in gewünschte Formate
        for format_type, enabled in config.formats.items():
            if enabled:
                try:
                    if format_type in self._exporters:
                        success = self._exporters[format_type](
                            transcription_data, 
                            output_path.with_suffix(f".{format_type.value}"),
                            config
                        )
                    else:
                        # Basis-Formate (TXT, JSON, SRT) werden extern gehandhabt
                        success = True
                    
                    results[format_type] = success
                    if success:
                        logger.info(f"✓ {format_type.value.upper()} exportiert: {output_path.stem}.{format_type.value}")
                except Exception as e:
                    logger.error(f"✗ Fehler beim {format_type.value.upper()}-Export: {e}")
                    results[format_type] = False
        
        return results
    
    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """Validiert Transkriptionsdaten"""
        required_fields = ['text']
        return all(field in data for field in required_fields)
    
    def _export_xml(self, data: Dict[str, Any], output_path: Path, config: ExportConfig) -> bool:
        """Optimierter XML-Export mit besserer Struktur"""
        try:
            # Root mit Namespaces für bessere Kompatibilität
            root = ET.Element("transcription", {
                "version": self.format_version,
                "xmlns": "http://archivecat.org/transcription/1.0",
                "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            })
            
            # Metadaten
            if config.include_metadata:
                self._add_xml_metadata(root, data, config)
            
            # Inhalt
            content = ET.SubElement(root, "content")
            
            # Volltext mit CDATA für bessere Lesbarkeit
            fulltext = ET.SubElement(content, "fulltext")
            fulltext.text = data.get('text', '')
            
            # Segmente
            segments_elem = ET.SubElement(content, "segments")
            self._add_xml_segments(segments_elem, data, config)
            
            # Speaker-Analyse
            if 'by_speaker' in data:
                analysis = ET.SubElement(root, "analysis")
                self._add_xml_speaker_analysis(analysis, data, config)
            
            # XML schreiben
            if config.pretty_print:
                xml_str = self._prettify_xml(ET.tostring(root, encoding='unicode'))
            else:
                xml_str = ET.tostring(root, encoding='unicode')
            
            output_path.write_text(xml_str, encoding=config.encoding)
            return True
            
        except Exception as e:
            logger.error(f"XML-Export Fehler: {e}")
            return False
    
    def _add_xml_metadata(self, root: ET.Element, data: Dict[str, Any], config: ExportConfig):
        """Fügt strukturierte Metadaten hinzu"""
        metadata = ET.SubElement(root, "metadata")
        
        # Basis-Informationen
        info = ET.SubElement(metadata, "info")
        ET.SubElement(info, "language").text = data.get('language', 'unknown')
        ET.SubElement(info, "duration").text = self._format_duration(data.get('duration', 0))
        ET.SubElement(info, "tool").text = "ArchiveCAT with OpenAI Whisper"
        ET.SubElement(info, "model").text = data.get('model', 'whisper-1')
        
        # Statistiken
        if config.include_statistics:
            stats = ET.SubElement(metadata, "statistics")
            text = data.get('text', '')
            ET.SubElement(stats, "words").text = str(len(text.split()))
            ET.SubElement(stats, "characters").text = str(len(text))
            
            # Segment-Statistiken
            segments = data.get('segments_with_speakers', data.get('segments', []))
            ET.SubElement(stats, "segments").text = str(len(segments))
            
            if segments:
                avg_duration = sum(s['end'] - s['start'] for s in segments) / len(segments)
                ET.SubElement(stats, "average_segment_duration").text = f"{avg_duration:.2f}"
        
        # Speaker-Informationen
        if 'speakers' in data:
            speakers = ET.SubElement(metadata, "speakers", {"count": str(data.get('speaker_count', 0))})
            for speaker_id in data['speakers']:
                ET.SubElement(speakers, "speaker", {"id": speaker_id}).text = speaker_id
    
    def _add_xml_segments(self, parent: ET.Element, data: Dict[str, Any], config: ExportConfig):
        """Fügt Segmente mit optimierter Struktur hinzu"""
        segments = data.get('segments_with_speakers', data.get('segments', []))
        
        for i, seg in enumerate(segments, 1):
            segment = ET.SubElement(parent, "segment", {"id": str(i)})
            
            # Zeitstempel
            timing = ET.SubElement(segment, "timing")
            ET.SubElement(timing, "start").text = self._format_timestamp(seg['start'], config.timestamp_format)
            ET.SubElement(timing, "end").text = self._format_timestamp(seg['end'], config.timestamp_format)
            ET.SubElement(timing, "duration").text = f"{seg['end'] - seg['start']:.3f}"
            
            # Speaker (falls vorhanden)
            if 'speaker' in seg:
                ET.SubElement(segment, "speaker").text = seg['speaker']
            
            # Text
            text_elem = ET.SubElement(segment, "text")
            text_elem.text = seg['text'].strip()
            
            # Optionale Konfidenz-Werte (falls vorhanden)
            if 'confidence' in seg:
                ET.SubElement(segment, "confidence").text = f"{seg['confidence']:.3f}"
    
    def _add_xml_speaker_analysis(self, parent: ET.Element, data: Dict[str, Any], config: ExportConfig):
        """Fügt detaillierte Speaker-Analyse hinzu"""
        speaker_analysis = ET.SubElement(parent, "speaker_analysis")
        total_duration = data.get('duration', 0)
        
        for speaker_id, utterances in data['by_speaker'].items():
            speaker = ET.SubElement(speaker_analysis, "speaker", {"id": speaker_id})
            
            # Metriken berechnen
            metrics = self._calculate_speaker_metrics(utterances, total_duration)
            
            # Statistiken
            stats = ET.SubElement(speaker, "statistics")
            for key, value in metrics.items():
                ET.SubElement(stats, key).text = str(value)
            
            # Utterances (optional, kann bei großen Dateien weggelassen werden)
            if len(utterances) <= 100:  # Limit für Performance
                utterances_elem = ET.SubElement(speaker, "utterances")
                for i, utt in enumerate(utterances, 1):
                    self._add_xml_utterance(utterances_elem, i, utt, config)
    
    def _add_xml_utterance(self, parent: ET.Element, index: int, utterance: Dict, config: ExportConfig):
        """Fügt einzelne Äußerung hinzu"""
        utt_elem = ET.SubElement(parent, "utterance", {"id": str(index)})
        
        timing = ET.SubElement(utt_elem, "timing")
        ET.SubElement(timing, "start").text = self._format_timestamp(utterance['start'], config.timestamp_format)
        ET.SubElement(timing, "end").text = self._format_timestamp(utterance['end'], config.timestamp_format)
        
        ET.SubElement(utt_elem, "text").text = utterance['text'].strip()
    
    def _export_csv(self, data: Dict[str, Any], output_path: Path, config: ExportConfig) -> bool:
        """Optimierter CSV-Export mit konfigurierbarem Delimiter"""
        try:
            segments = data.get('segments_with_speakers', data.get('segments', []))
            
            with output_path.open('w', newline='', encoding=config.encoding) as f:
                # Dynamische Feldnamen basierend auf verfügbaren Daten
                fieldnames = ['id', 'start', 'end', 'duration']
                
                if any('speaker' in s for s in segments):
                    fieldnames.append('speaker')
                
                fieldnames.extend(['text', 'word_count'])
                
                if any('confidence' in s for s in segments):
                    fieldnames.append('confidence')
                
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=config.csv_delimiter)
                writer.writeheader()
                
                for i, seg in enumerate(segments, 1):
                    row = {
                        'id': i,
                        'start': self._format_timestamp(seg['start'], config.timestamp_format),
                        'end': self._format_timestamp(seg['end'], config.timestamp_format),
                        'duration': f"{seg['end'] - seg['start']:.3f}",
                        'text': seg['text'].strip(),
                        'word_count': len(seg['text'].split())
                    }
                    
                    # Optionale Felder
                    if 'speaker' in seg:
                        row['speaker'] = seg['speaker']
                    if 'confidence' in seg:
                        row['confidence'] = f"{seg['confidence']:.3f}"
                    
                    writer.writerow(row)
            
            return True
            
        except Exception as e:
            logger.error(f"CSV-Export Fehler: {e}")
            return False
    
    def _export_webvtt(self, data: Dict[str, Any], output_path: Path, config: ExportConfig) -> bool:
        """WebVTT-Export mit erweiterten Features"""
        try:
            with output_path.open('w', encoding=config.encoding) as f:
                # WebVTT Header
                f.write("WEBVTT\n")
                f.write(f"NOTE Created by ArchiveCAT on {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                segments = data.get('segments_with_speakers', data.get('segments', []))
                
                for i, seg in enumerate(segments, 1):
                    # Cue identifier
                    f.write(f"{i}\n")
                    
                    # Timings
                    start = self._format_webvtt_time(seg['start'])
                    end = self._format_webvtt_time(seg['end'])
                    f.write(f"{start} --> {end}")
                    
                    # Cue settings (Position, Größe, etc.)
                    if 'speaker' in seg:
                        # Verschiedene Positionen für verschiedene Sprecher
                        speaker_num = int(seg['speaker'].split('_')[-1]) if '_' in seg['speaker'] else 1
                        position = 50 + (speaker_num - 1) * 10
                        f.write(f" position:{position}%")
                    
                    f.write("\n")
                    
                    # Text mit Speaker-Tag
                    if 'speaker' in seg:
                        f.write(f"<v {seg['speaker']}>{seg['text'].strip()}</v>\n\n")
                    else:
                        f.write(f"{seg['text'].strip()}\n\n")
            
            return True
            
        except Exception as e:
            logger.error(f"WebVTT-Export Fehler: {e}")
            return False
    
    def _export_ttml(self, data: Dict[str, Any], output_path: Path, config: ExportConfig) -> bool:
        """TTML (Timed Text Markup Language) Export für professionelle Untertitel"""
        try:
            # TTML Namespace
            ttml_ns = "http://www.w3.org/ns/ttml"
            root = ET.Element("{%s}tt" % ttml_ns, {
                "xml:lang": data.get('language', 'de'),
                "{http://www.w3.org/XML/1998/namespace}lang": data.get('language', 'de')
            })
            
            # Head
            head = ET.SubElement(root, "{%s}head" % ttml_ns)
            
            # Styling
            styling = ET.SubElement(head, "{%s}styling" % ttml_ns)
            style = ET.SubElement(styling, "{%s}style" % ttml_ns, {
                "xml:id": "defaultStyle",
                "tts:fontFamily": "Arial",
                "tts:fontSize": "100%",
                "tts:textAlign": "center"
            })
            
            # Layout
            layout = ET.SubElement(head, "{%s}layout" % ttml_ns)
            region = ET.SubElement(layout, "{%s}region" % ttml_ns, {
                "xml:id": "defaultRegion",
                "tts:origin": "10% 80%",
                "tts:extent": "80% 20%"
            })
            
            # Body
            body = ET.SubElement(root, "{%s}body" % ttml_ns)
            div = ET.SubElement(body, "{%s}div" % ttml_ns)
            
            segments = data.get('segments_with_speakers', data.get('segments', []))
            
            for seg in segments:
                p = ET.SubElement(div, "{%s}p" % ttml_ns, {
                    "begin": self._format_ttml_time(seg['start']),
                    "end": self._format_ttml_time(seg['end']),
                    "region": "defaultRegion"
                })
                
                if 'speaker' in seg:
                    span = ET.SubElement(p, "{%s}span" % ttml_ns, {
                        "tts:fontStyle": "italic"
                    })
                    span.text = f"[{seg['speaker']}] "
                    span.tail = seg['text'].strip()
                else:
                    p.text = seg['text'].strip()
            
            # Schreibe TTML
            xml_str = self._prettify_xml(ET.tostring(root, encoding='unicode'))
            output_path.write_text(xml_str, encoding=config.encoding)
            
            return True
            
        except Exception as e:
            logger.error(f"TTML-Export Fehler: {e}")
            return False
    
    # Hilfsmethoden
    def _calculate_speaker_metrics(self, utterances: List[Dict], total_duration: float) -> Dict[str, Any]:
        """Berechnet detaillierte Speaker-Metriken"""
        total_time = sum(utt['end'] - utt['start'] for utt in utterances)
        words = sum(len(utt['text'].split()) for utt in utterances)
        
        # Durchschnittliche Sprechgeschwindigkeit
        wpm = (words / (total_time / 60)) if total_time > 0 else 0
        
        # Pausen zwischen Äußerungen
        pauses = []
        for i in range(1, len(utterances)):
            pause = utterances[i]['start'] - utterances[i-1]['end']
            if pause > 0:
                pauses.append(pause)
        
        avg_pause = sum(pauses) / len(pauses) if pauses else 0
        
        return {
            "utterance_count": len(utterances),
            "total_speaking_time": f"{total_time:.1f}",
            "percentage_of_total": f"{(total_time / total_duration * 100):.1f}" if total_duration > 0 else "0",
            "word_count": words,
            "words_per_minute": f"{wpm:.1f}",
            "average_utterance_duration": f"{total_time / len(utterances):.1f}" if utterances else "0",
            "average_pause_duration": f"{avg_pause:.1f}",
            "longest_utterance": f"{max((u['end'] - u['start']) for u in utterances):.1f}" if utterances else "0"
        }
    
    def _format_timestamp(self, seconds: float, format_type: str) -> str:
        """Formatiert Zeitstempel je nach gewünschtem Format"""
        if format_type == "milliseconds":
            return f"{int(seconds * 1000)}"
        elif format_type == "timecode":
            return self._seconds_to_timecode(seconds)
        else:  # seconds
            return f"{seconds:.3f}"
    
    def _format_duration(self, seconds: float) -> str:
        """Formatiert Dauer in lesbares Format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs:.1f}s"
        elif minutes > 0:
            return f"{minutes}m {secs:.1f}s"
        else:
            return f"{secs:.1f}s"
    
    def _seconds_to_timecode(self, seconds: float) -> str:
        """Konvertiert Sekunden zu Timecode HH:MM:SS.mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    
    def _format_webvtt_time(self, seconds: float) -> str:
        """WebVTT Zeitformat"""
        return self._seconds_to_timecode(seconds).replace(',', '.')
    
    def _format_ttml_time(self, seconds: float) -> str:
        """TTML Zeitformat"""
        return f"{seconds:.3f}s"
    
    def _prettify_xml(self, xml_string: str) -> str:
        """Formatiert XML mit Einrückung"""
        try:
            parsed = minidom.parseString(xml_string)
            pretty = parsed.toprettyxml(indent="  ", encoding=None)
            # Entferne leere Zeilen
            lines = [line for line in pretty.split('\n') if line.strip()]
            return '\n'.join(lines)
        except:
            return xml_string


# Convenience-Funktionen für direkten Export
def export_transcription_to_xml(transcription_data: Dict[str, Any], 
                               output_path: Union[str, Path]) -> bool:
    """Direkter XML-Export"""
    exporter = TranscriptionExporter()
    config = ExportConfig(formats={ExportFormat.XML: True})
    results = exporter.export(transcription_data, output_path, config)
    return results.get(ExportFormat.XML, False)


def export_transcription_to_csv(transcription_data: Dict[str, Any], 
                               output_path: Union[str, Path]) -> bool:
    """Direkter CSV-Export"""
    exporter = TranscriptionExporter()
    config = ExportConfig(formats={ExportFormat.CSV: True})
    results = exporter.export(transcription_data, output_path, config)
    return results.get(ExportFormat.CSV, False)


def export_transcription_to_all_formats(transcription_data: Dict[str, Any], 
                                       output_path: Union[str, Path],
                                       include_extended: bool = False) -> Dict[ExportFormat, bool]:
    """Exportiert in alle (oder Standard-) Formate"""
    exporter = TranscriptionExporter()
    
    formats = {
        ExportFormat.XML: True,
        ExportFormat.JSON: True,
        ExportFormat.TXT: True,
        ExportFormat.SRT: True,
        ExportFormat.CSV: include_extended,
        ExportFormat.WEBVTT: include_extended,
        ExportFormat.TTML: include_extended
    }
    
    config = ExportConfig(formats=formats)
    return exporter.export(transcription_data, output_path, config)