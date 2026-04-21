#!/usr/bin/env python3
"""Extract text from a PDF file, one page at a time."""

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
