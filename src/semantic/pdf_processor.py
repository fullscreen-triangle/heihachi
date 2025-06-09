import os
import json
from typing import Dict, List, Optional
import fitz  # PyMuPDF
from pathlib import Path
import re

class PDFProcessor:
    """Process PDFs and extract structured content"""
    
    def __init__(self):
        self.section_patterns = [
            r"^[0-9]+\.[0-9]*\s+[A-Z]",  # Numbered sections
            r"^[A-Z][A-Za-z\s]{2,}$",     # All caps or title case headers
            r"^[0-9]+\s+[A-Z]",           # Simple numbered sections
        ]
    
    def process_pdf(self, pdf_path: str) -> Dict:
        """Process a PDF file and extract structured content"""
        doc = fitz.open(pdf_path)
        
        # Extract metadata
        metadata = self._extract_metadata(doc)
        
        # Extract structured content
        sections = self._extract_sections(doc)
        
        # Extract references
        references = self._extract_references(doc)
        
        # Extract figures and tables
        figures = self._extract_figures(doc)
        tables = self._extract_tables(doc)
        
        return {
            "metadata": metadata,
            "sections": sections,
            "references": references,
            "figures": figures,
            "tables": tables
        }
    
    def _extract_metadata(self, doc: fitz.Document) -> Dict:
        """Extract metadata from PDF"""
        metadata = doc.metadata
        
        # Try to extract title from first page
        first_page = doc[0]
        title_candidates = []
        
        # Look for text that might be the title
        for block in first_page.get_text("blocks"):
            text = block[4]  # Block text is in index 4
            if len(text.split()) > 3 and len(text.split()) < 20:  # Reasonable title length
                title_candidates.append(text)
        
        if title_candidates:
            metadata["title"] = title_candidates[0]
        
        return metadata
    
    def _extract_sections(self, doc: fitz.Document) -> List[Dict]:
        """Extract structured sections from PDF"""
        sections = []
        current_section = None
        
        for page in doc:
            blocks = page.get_text("blocks")
            
            for block in blocks:
                text = block[4]  # Block text
                
                # Check if this is a section header
                if self._is_section_header(text):
                    # Save previous section if it exists
                    if current_section:
                        sections.append(current_section)
                    
                    # Start new section
                    current_section = {
                        "title": text.strip(),
                        "content": "",
                        "subsections": []
                    }
                elif current_section:
                    # Add content to current section
                    current_section["content"] += text + "\n"
        
        # Add the last section
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def _is_section_header(self, text: str) -> bool:
        """Check if text matches section header patterns"""
        text = text.strip()
        return any(re.match(pattern, text) for pattern in self.section_patterns)
    
    def _extract_references(self, doc: fitz.Document) -> List[Dict]:
        """Extract references from PDF"""
        references = []
        reference_section = None
        
        for page in doc:
            blocks = page.get_text("blocks")
            
            for block in blocks:
                text = block[4]
                
                # Look for reference section
                if "references" in text.lower():
                    reference_section = []
                elif reference_section is not None:
                    # Check if this is a reference entry
                    if re.match(r"^\[\d+\]", text):
                        reference_section.append({
                            "citation": text.split("\n")[0],
                            "content": text
                        })
                elif reference_section is not None and text.strip():
                    # End of references section
                    references = reference_section
                    reference_section = None
        
        return references
    
    def _extract_figures(self, doc: fitz.Document) -> List[Dict]:
        """Extract figures from PDF"""
        figures = []
        
        for page_num, page in enumerate(doc):
            # Extract images
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Try to find figure caption
                caption = self._find_figure_caption(page, img_index)
                
                figures.append({
                    "page": page_num + 1,
                    "index": img_index,
                    "caption": caption,
                    "image_data": image_bytes
                })
        
        return figures
    
    def _extract_tables(self, doc: fitz.Document) -> List[Dict]:
        """Extract tables from PDF"""
        tables = []
        
        for page_num, page in enumerate(doc):
            # Extract tables using PyMuPDF's table detection
            tables_on_page = page.get_tables()
            
            for table_index, table in enumerate(tables_on_page):
                # Try to find table caption
                caption = self._find_table_caption(page, table_index)
                
                tables.append({
                    "page": page_num + 1,
                    "index": table_index,
                    "caption": caption,
                    "data": table
                })
        
        return tables
    
    def _find_figure_caption(self, page: fitz.Page, figure_index: int) -> Optional[str]:
        """Find caption for a figure"""
        blocks = page.get_text("blocks")
        
        for block in blocks:
            text = block[4].lower()
            if f"figure {figure_index + 1}" in text or f"fig. {figure_index + 1}" in text:
                return block[4].strip()
        
        return None
    
    def _find_table_caption(self, page: fitz.Page, table_index: int) -> Optional[str]:
        """Find caption for a table"""
        blocks = page.get_text("blocks")
        
        for block in blocks:
            text = block[4].lower()
            if f"table {table_index + 1}" in text:
                return block[4].strip()
        
        return None 