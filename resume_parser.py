"""
resume_parser.py
----------------
Handles text extraction from uploaded resume files.

Supported formats:
- PDF files (.pdf)  → Extracted using PyMuPDF (fitz)
- Text files (.txt) → Read directly as plain text

Each resume's filename (without extension) is used as the Candidate Name.
"""

import os
import fitz  # PyMuPDF library for PDF parsing


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extracts raw text from a PDF file provided as bytes.

    Parameters:
        file_bytes (bytes): Raw bytes of the uploaded PDF file.

    Returns:
        str: Concatenated text from all pages of the PDF.
    """
    # Open the PDF from in-memory bytes (no need to save to disk)
    pdf_document = fitz.open(stream=file_bytes, filetype="pdf")

    extracted_text = ""

    # Iterate through each page of the PDF and extract text
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        extracted_text += page.get_text()  # Extract text from this page

    pdf_document.close()
    return extracted_text


def extract_text_from_txt(file_bytes: bytes) -> str:
    """
    Decodes and returns text from a plain text (.txt) file provided as bytes.

    Parameters:
        file_bytes (bytes): Raw bytes of the uploaded TXT file.

    Returns:
        str: Decoded text content of the file.
    """
    # Decode bytes to string using UTF-8 encoding (with fallback for special chars)
    return file_bytes.decode('utf-8', errors='ignore')


def parse_resume(uploaded_file) -> tuple[str, str]:
    """
    Determines the file type and extracts text accordingly.

    Parameters:
        uploaded_file: A Streamlit UploadedFile object.

    Returns:
        tuple: (candidate_name, extracted_text)
            - candidate_name (str): Filename without extension.
            - extracted_text (str): Raw text content of the resume.
    """
    # Derive candidate name from filename (remove extension)
    filename = uploaded_file.name
    candidate_name = os.path.splitext(filename)[0]

    # Read the uploaded file bytes
    file_bytes = uploaded_file.read()

    # Determine file type and extract text accordingly
    if filename.lower().endswith('.pdf'):
        extracted_text = extract_text_from_pdf(file_bytes)
    elif filename.lower().endswith('.txt'):
        extracted_text = extract_text_from_txt(file_bytes)
    else:
        # Return empty string for unsupported file types
        extracted_text = ""

    return candidate_name, extracted_text


def parse_all_resumes(uploaded_files: list) -> list[dict]:
    """
    Parses a list of uploaded resume files and returns structured data.

    Parameters:
        uploaded_files (list): List of Streamlit UploadedFile objects.

    Returns:
        list[dict]: A list of dictionaries, each containing:
            - 'name'  : Candidate name (derived from filename)
            - 'text'  : Raw extracted text from the resume
    """
    resumes = []

    for uploaded_file in uploaded_files:
        candidate_name, extracted_text = parse_resume(uploaded_file)

        # Only include resumes where text was successfully extracted
        if extracted_text.strip():
            resumes.append({
                'name': candidate_name,
                'text': extracted_text
            })

    return resumes
