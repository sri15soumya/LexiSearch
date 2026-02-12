"""
Extract text from CUAD-style PDFs (text-based only)
Uses pdfplumber + PyPDF2 fallback
"""

import pdfplumber
import PyPDF2
import os
from pathlib import Path
import time


# --------------------------------------------------
# pdfplumber extraction
# --------------------------------------------------
def extract_with_pdfplumber(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        return text
    except Exception as e:
        print(f"      pdfplumber error: {e}")
        return None


# --------------------------------------------------
# PyPDF2 extraction (fallback)
# --------------------------------------------------
def extract_with_pypdf2(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        return text
    except Exception as e:
        print(f"      PyPDF2 error: {e}")
        return None


# --------------------------------------------------
# Main pipeline
# --------------------------------------------------
def extract_all_pdfs(
    pdf_folder="sample/Part_II/Agency Agreements",
    output_folder="data/extracted_text",
    min_text_length=50
):
    print("=" * 70)
    print("PDF TEXT EXTRACTION (pdfplumber + PyPDF2)")
    print("=" * 70 + "\n")

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF files\n")

    successful = 0
    failed = []
    start_time = time.time()

    for i, pdf_file in enumerate(pdf_files, 1):
        pdf_path = os.path.join(pdf_folder, pdf_file)
        print(f"[{i}/{len(pdf_files)}] {pdf_file}")

        # Try pdfplumber first
        text = extract_with_pdfplumber(pdf_path)
        method = "pdfplumber"

        # Fallback to PyPDF2
        if not text or len(text.strip()) < min_text_length:
            print("   Fallback to PyPDF2...")
            text = extract_with_pypdf2(pdf_path)
            method = "pypdf2"

        if text and len(text.strip()) >= min_text_length:
            output_path = os.path.join(
                output_folder, pdf_file.replace(".pdf", ".txt")
            )
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"  Extracted {len(text):,} chars ({method})")
            successful += 1
        else:
            print("Extraction failed")
            failed.append(pdf_file)

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Successful: {successful}/{len(pdf_files)}")
    print(f"Failed: {len(failed)}")
    print(f"Time: {elapsed:.1f} seconds")
    print(f" Output: {output_folder}")

    if failed:
        print("\n Failed files:")
        for f in failed[:10]:
            print(f"   - {f}")


# --------------------------------------------------
# Run
# --------------------------------------------------
if __name__ == "__main__":
    extract_all_pdfs()
