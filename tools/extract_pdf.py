#!/usr/bin/env python3
"""Extract text from a PDF file, one page at a time.

DESIGN DECISION (intentionally a thin wrapper):
    This script is deliberately kept as a minimal wrapper around PyPDF2's
    page.extract_text(). All cleaning of the output — page numbers, captions,
    bullet glyphs, layout artifacts, etc. — lives in C, inside
    src/data/text_parser.c. The split is one responsibility per tool:
        Python  : "get raw bytes out of the PDF"
        C       : "clean those bytes into training-ready text"

    Why not do more in Python? The seminar's learning goal is C / CUDA / MPI,
    not Python data engineering. Also, PyPDF2 only exposes flat text — it has
    no layout / coordinate / font information — so real layout-aware
    extraction would require switching libraries (e.g. PyMuPDF, pdfplumber)
    and is out of scope here.

    Contract for the C side: the output is a UTF-8 text file with literal
    "---PAGE N---" separator lines between pages. text_parser.c can rely on
    that marker format.
"""

import sys
import PyPDF2


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.pdf> <output.txt>")
        sys.exit(1)

    pdf_path, out_path = sys.argv[1], sys.argv[2]
    reader = PyPDF2.PdfReader(pdf_path)
    total_chars = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            f.write(text)
            f.write(f"\n---PAGE {i + 1}---\n")
            total_chars += len(text)

    print(f"Pages: {len(reader.pages)}")
    print(f"Characters: {total_chars}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
