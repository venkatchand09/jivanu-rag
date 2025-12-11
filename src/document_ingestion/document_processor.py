# src/document_ingestion/document_processor.py
"""
Enhanced document processing module with image and table extraction
Optimized for scientific PDFs with complex layouts
DOCKER-READY: Uses environment variable for Tesseract path
"""

from typing import List, Union
from pathlib import Path
import fitz  # PyMuPDF
import pandas as pd
from PIL import Image
import pytesseract
import io
import os
from tqdm import tqdm

# SET TESSERACT PATH FROM ENVIRONMENT VARIABLE (Docker-compatible)
if os.getenv('TESSERACT_CMD'):
    pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_CMD')
elif os.name == 'nt':  # Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:  # Linux/Mac
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader
)

class DocumentProcessor:
    """Handles comprehensive document loading and processing"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document processor
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks for context preservation
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.assets_dir = Path("extracted_assets")
        self.assets_dir.mkdir(exist_ok=True)
    
    def extract_images_from_pdf(self, pdf_path: Union[str, Path], output_dir: Path) -> List[str]:
        """Extract images from PDF and perform OCR"""
        pdf_path = Path(pdf_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_paths = []
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            images = page.get_images()
            
            for img_index, img in enumerate(images):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Save image
                    img_filename = f"page_{page_num+1}_img_{img_index+1}.png"
                    img_path = output_dir / img_filename
                    
                    with open(img_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    image_paths.append(str(img_path))
                    
                except Exception as e:
                    print(f"Error extracting image from {pdf_path}, page {page_num}: {e}")
        
        doc.close()
        return image_paths
    
    def ocr_image(self, image_path: str) -> str:
        """Perform OCR on extracted image"""
        try:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img)
            return text.strip()
        except Exception as e:
            print(f"OCR error on {image_path}: {e}")
            return ""
    
    def extract_tables_from_pdf(self, pdf_path: Union[str, Path], output_dir: Path) -> List[str]:
        """Extract tables from PDF using PyMuPDF"""
        pdf_path = Path(pdf_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        table_paths = []
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            try:
                # Find tables using PyMuPDF's table detection
                tables = page.find_tables()
                
                for table_index, table in enumerate(tables):
                    try:
                        # Extract table as pandas DataFrame
                        df = table.to_pandas()
                        
                        if not df.empty:
                            table_filename = f"page_{page_num+1}_table_{table_index+1}.csv"
                            table_path = output_dir / table_filename
                            df.to_csv(table_path, index=False)
                            table_paths.append(str(table_path))
                    except Exception as e:
                        print(f"Error processing table: {e}")
                        
            except Exception as e:
                print(f"Error finding tables on page {page_num}: {e}")
        
        doc.close()
        return table_paths
    
    def process_pdf_with_assets(
        self, 
        pdf_path: Union[str, Path],
        include_images: bool = True,
        include_tables: bool = True
    ) -> List[Document]:
        """
        Process a single PDF with text, images, and tables
        
        Returns:
            List of Document objects with enhanced metadata
        """
        pdf_path = Path(pdf_path)
        pdf_stem = pdf_path.stem
        
        # Create asset directories
        pdf_assets_dir = self.assets_dir / pdf_stem
        images_dir = pdf_assets_dir / "images"
        tables_dir = pdf_assets_dir / "tables"
        
        documents = []
        
        # Extract main text using PyPDFLoader
        try:
            loader = PyPDFLoader(str(pdf_path))
            text_docs = loader.load()
            
            for doc in text_docs:
                doc.metadata["source"] = str(pdf_path)
                doc.metadata["type"] = "page_text"
                doc.metadata["pdf_name"] = pdf_path.name
            
            documents.extend(text_docs)
        except Exception as e:
            print(f"Error loading text from {pdf_path}: {e}")
        
        # Extract and process images
        if include_images:
            try:
                image_paths = self.extract_images_from_pdf(pdf_path, images_dir)
                
                for img_path in image_paths:
                    ocr_text = self.ocr_image(img_path)
                    
                    if ocr_text and len(ocr_text) > 50:  # Only include meaningful text
                        img_doc = Document(
                            page_content=f"[IMAGE OCR TEXT]\n{ocr_text}",
                            metadata={
                                "source": str(pdf_path),
                                "type": "image_ocr",
                                "image_path": img_path,
                                "pdf_name": pdf_path.name
                            }
                        )
                        documents.append(img_doc)
            except Exception as e:
                print(f"Error processing images from {pdf_path}: {e}")
        
        # Extract and process tables
        if include_tables:
            try:
                table_paths = self.extract_tables_from_pdf(pdf_path, tables_dir)
                
                for tbl_path in table_paths:
                    try:
                        df = pd.read_csv(tbl_path)
                        table_text = f"[TABLE DATA]\n{df.to_string()}"
                        
                        tbl_doc = Document(
                            page_content=table_text,
                            metadata={
                                "source": str(pdf_path),
                                "type": "table",
                                "table_path": tbl_path,
                                "pdf_name": pdf_path.name
                            }
                        )
                        documents.append(tbl_doc)
                    except Exception as e:
                        print(f"Error processing table {tbl_path}: {e}")
            except Exception as e:
                print(f"Error extracting tables from {pdf_path}: {e}")
        
        return documents
    
    def process_pdf_dir(
        self, 
        directory: Union[str, Path],
        include_images: bool = True,
        include_tables: bool = True
    ) -> List[Document]:
        """
        Process all PDFs in a directory
        
        Args:
            directory: Path to directory containing PDFs
            include_images: Whether to extract and OCR images
            include_tables: Whether to extract tables
            
        Returns:
            List of all processed documents
        """
        directory = Path(directory)
        pdf_files = list(directory.glob("*.pdf"))
        
        print(f"Found {len(pdf_files)} PDF files in {directory}")
        
        all_documents = []
        
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            docs = self.process_pdf_with_assets(
                pdf_file,
                include_images=include_images,
                include_tables=include_tables
            )
            all_documents.extend(docs)
        
        return all_documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks while preserving metadata"""
        return self.splitter.split_documents(documents)
    
    def process_sources(
        self, 
        sources: List[str],
        include_images: bool = True,
        include_tables: bool = True
    ) -> List[Document]:
        """
        Complete pipeline to load, process, and split documents
        
        Args:
            sources: List of URLs or directory paths
            include_images: Extract images
            include_tables: Extract tables
            
        Returns:
            List of processed and split document chunks
        """
        all_docs = []
        
        for source in sources:
            if source.startswith("http://") or source.startswith("https://"):
                # Web URLs
                try:
                    loader = WebBaseLoader(source)
                    docs = loader.load()
                    all_docs.extend(docs)
                except Exception as e:
                    print(f"Error loading URL {source}: {e}")
            else:
                # Local directories
                path = Path(source)
                if path.is_dir():
                    docs = self.process_pdf_dir(
                        path, 
                        include_images=include_images,
                        include_tables=include_tables
                    )
                    all_docs.extend(docs)
                elif path.suffix.lower() == ".pdf":
                    docs = self.process_pdf_with_assets(
                        path,
                        include_images=include_images,
                        include_tables=include_tables
                    )
                    all_docs.extend(docs)
                else:
                    print(f"Unsupported source: {source}")
        
        # Split documents into chunks
        print(f"Splitting {len(all_docs)} documents into chunks...")
        split_docs = self.split_documents(all_docs)
        print(f"Created {len(split_docs)} document chunks")
        
        return split_docs