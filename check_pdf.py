"""
Quick check: how much readable text did we actually get from the PDF?
Run this from inside your study_assistant folder with venv activated:
    python check_pdf.py
"""
from langchain_community.document_loaders import PyPDFLoader

path = "data/B.E.CSE.pdf"
docs = PyPDFLoader(path).load()

total_chars = sum(len(d.page_content) for d in docs)
print(f"Pages loaded: {len(docs)}")
print(f"Total characters extracted: {total_chars}")
print("\n--- First 500 characters of extracted text ---\n")
print(docs[0].page_content[:500] if docs else "(nothing extracted)")