import os
from pathlib import Path
import pymupdf as fitz
import docx
from PIL import Image
from io import BytesIO


class DocumentProcessor:
    """
    Handles text extraction from PDF, DOCX, and TXT resume files with fallback OCR.
    """

    def __init__(self, tesseract_cmd: str = None):
        self.tesseract_available = False
        tesseract_path = tesseract_cmd or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        
        if os.path.exists(tesseract_path):
            try:
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                self.pytesseract = pytesseract
                self.tesseract_available = True
            except ImportError:
                self.tesseract_available = False
        else:
            try:
                import pytesseract
                self.pytesseract = pytesseract
                self.tesseract_available = True
            except Exception:
                self.tesseract_available = False

    def extract_text(self, file_path: str) -> str:
        """
        Extract raw text from a PDF, DOCX, or TXT document.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()

        if ext == ".pdf":
            return self._extract_from_pdf(path)
        elif ext in [".docx", ".doc"]:
            return self._extract_from_docx(path)
        elif ext == ".txt":
            return self._extract_from_txt(path)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Allowed formats: .pdf, .docx, .txt")

    def _extract_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from PDF using PyMuPDF, falling back to OCR if page has no selectable text.
        """
        doc = fitz.open(pdf_path)
        all_text = []

        for page_number, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                all_text.append(text)
            elif self.tesseract_available:
                # Page has no selectable text, attempt OCR
                try:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                    image_bytes = pix.tobytes("png")
                    image = Image.open(BytesIO(image_bytes))
                    ocr_text = self.pytesseract.image_to_string(image, lang="eng").strip()
                    if ocr_text:
                        all_text.append(ocr_text)
                except Exception:
                    pass

        doc.close()
        return "\n\n".join(all_text).strip()

    def _extract_from_docx(self, docx_path: Path) -> str:
        """
        Extract text from DOCX file.
        """
        doc = docx.Document(docx_path)
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text.strip())
        for table in doc.tables:
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_text:
                    full_text.append(" | ".join(row_text))
        return "\n".join(full_text).strip()

    def _extract_from_txt(self, txt_path: Path) -> str:
        """
        Extract text from TXT file.
        """
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()