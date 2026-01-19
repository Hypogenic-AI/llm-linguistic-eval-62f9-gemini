import pdfplumber
import os

papers = [
    "evaluating_linguistic_discrimination.pdf",
    "how_llms_handle_multilingualism.pdf"
]

for filename in papers:
    path = os.path.join("papers", filename)
    print(f"=== {filename} ===")
    try:
        with pdfplumber.open(path) as pdf:
            text = ""
            for i in range(min(3, len(pdf.pages))):
                text += pdf.pages[i].extract_text() + "\n"
            print(text[:2000]) # Print first 2000 chars of extracted text
            print("\n... (truncated)\n")
    except Exception as e:
        print(f"Error reading {filename}: {e}")
