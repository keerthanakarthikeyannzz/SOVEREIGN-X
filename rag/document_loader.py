"""
SOVEREIGN-X — RAG Document Loader
Loads PDF, TXT, CSV files and extracts text content.
Uses PyMuPDF. Falls back to OCR for scanned PDFs.
"""

import fitz  # PyMuPDF
import csv
import io
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class DocumentChunk:
    text: str
    source_file: str
    source_type: str  # MRPL_PUBLIC | SYNTHETIC_DEMO | USER_UPLOAD
    document_type: str
    page_number: Optional[int] = None
    section: Optional[str] = None
    equipment_id: Optional[str] = None
    equipment_type: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class DocumentLoader:
    def __init__(self, ocr_enabled: bool = True):
        self.ocr_enabled = ocr_enabled
        self._tesseract_available = self._check_tesseract()

    def _check_tesseract(self) -> bool:
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def load_pdf(
        self,
        file_path: str,
        source_type: str = "SYNTHETIC_DEMO",
        document_type: str = "Document",
        equipment_id: Optional[str] = None,
        equipment_type: Optional[str] = None,
    ) -> list[DocumentChunk]:
        """Load a PDF file, extracting text per page. Uses OCR if needed."""
        chunks = []
        path = Path(file_path)

        try:
            doc = fitz.open(str(path))
        except Exception as e:
            return [DocumentChunk(
                text=f"[ERROR: Could not open PDF: {e}]",
                source_file=path.name,
                source_type=source_type,
                document_type="Error",
            )]

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()

            # If page has no text, try OCR
            if not text and self.ocr_enabled and self._tesseract_available:
                text = self._ocr_page(page)

            if text:
                chunks.append(DocumentChunk(
                    text=text,
                    source_file=path.name,
                    source_type=source_type,
                    document_type=document_type,
                    page_number=page_num + 1,
                    equipment_id=equipment_id,
                    equipment_type=equipment_type,
                    metadata={
                        "file_path": str(path),
                        "total_pages": len(doc),
                        "ocr_applied": not page.get_text("text").strip(),
                    },
                ))

        doc.close()
        return chunks

    def _ocr_page(self, page: fitz.Page) -> str:
        """Apply OCR to a fitz page."""
        try:
            import pytesseract
            from PIL import Image
            import io

            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            return pytesseract.image_to_string(img, lang="eng").strip()
        except Exception as e:
            return f"[OCR failed: {e}]"

    def load_text(
        self,
        file_path: str,
        source_type: str = "SYNTHETIC_DEMO",
        document_type: str = "Text Document",
    ) -> list[DocumentChunk]:
        path = Path(file_path)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            return [DocumentChunk(
                text=text,
                source_file=path.name,
                source_type=source_type,
                document_type=document_type,
                metadata={"file_path": str(path)},
            )]
        except Exception as e:
            return [DocumentChunk(
                text=f"[ERROR: Could not read text file: {e}]",
                source_file=path.name,
                source_type=source_type,
                document_type="Error",
            )]

    def load_csv(
        self,
        file_path: str,
        source_type: str = "SYNTHETIC_DEMO",
        document_type: str = "CSV Data",
    ) -> list[DocumentChunk]:
        """Load CSV as a text representation."""
        path = Path(file_path)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if not rows:
                return []

            # Create a text summary
            headers = list(rows[0].keys())
            text_parts = [f"CSV file: {path.name}", f"Columns: {', '.join(headers)}", f"Rows: {len(rows)}", ""]

            # Include first 100 rows as text
            for i, row in enumerate(rows[:100]):
                row_text = " | ".join(f"{k}: {v}" for k, v in row.items())
                text_parts.append(row_text)

            return [DocumentChunk(
                text="\n".join(text_parts),
                source_file=path.name,
                source_type=source_type,
                document_type=document_type,
                metadata={
                    "file_path": str(path),
                    "columns": headers,
                    "row_count": len(rows),
                },
            )]
        except Exception as e:
            return [DocumentChunk(
                text=f"[ERROR loading CSV: {e}]",
                source_file=path.name,
                source_type=source_type,
                document_type="Error",
            )]

    def load_file(
        self,
        file_path: str,
        source_type: str = "SYNTHETIC_DEMO",
        equipment_id: Optional[str] = None,
        equipment_type: Optional[str] = None,
    ) -> list[DocumentChunk]:
        """Auto-detect file type and load accordingly."""
        path = Path(file_path)
        ext = path.suffix.lower()

        doc_type = self._detect_doc_type(path.name)

        if ext == ".pdf":
            return self.load_pdf(
                file_path, source_type, doc_type, equipment_id, equipment_type
            )
        elif ext in (".txt", ".md"):
            return self.load_text(file_path, source_type, doc_type)
        elif ext == ".csv":
            return self.load_csv(file_path, source_type, doc_type)
        else:
            return []

    @staticmethod
    def _detect_doc_type(filename: str) -> str:
        fn = filename.lower()
        if "sop" in fn or "procedure" in fn:
            return "SOP"
        if "history" in fn or "historical" in fn:
            return "Historical Record"
        if "inspection" in fn:
            return "Inspection Report"
        if "limit" in fn or "standard" in fn or "threshold" in fn:
            return "Operating Limits"
        if "profile" in fn or "equipment" in fn:
            return "Equipment Profile"
        if "failure" in fn:
            return "Failure Mode Reference"
        if "sensor" in fn:
            return "Sensor Data"
        if "work_order" in fn:
            return "Work Order"
        return "Document"
