#!/usr/bin/env python3
"""
Render PDF pages as images to capture vector graphics figures.
"""
import fitz  # PyMuPDF
import os
from pathlib import Path

def render_pdf_pages(pdf_path, output_dir="page_renders", dpi=200):
    """Render PDF pages as PNG images."""
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Open PDF
    doc = fitz.open(pdf_path)

    # Pages with likely figures based on paper structure
    # Main figures typically on pages 2-10, appendix figures later
    important_pages = list(range(1, 15))  # Pages 2-15 (main body)
    important_pages.extend(range(15, 30))  # Pages 16-30 (appendix)

    rendered_pages = []

    for page_num in important_pages:
        if page_num >= len(doc):
            break

        page = doc[page_num]

        # Set zoom factor for resolution (dpi/72)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)

        # Render page to pixmap
        pix = page.get_pixmap(matrix=mat)

        # Create filename
        image_filename = f"page_{page_num + 1:03d}.png"
        image_path = os.path.join(output_dir, image_filename)

        # Save image
        pix.save(image_path)

        rendered_pages.append({
            "page": page_num + 1,
            "filename": image_filename,
            "size": os.path.getsize(image_path)
        })

        print(f"Rendered page {page_num + 1} -> {image_filename} ({os.path.getsize(image_path)} bytes)")

    doc.close()

    print(f"\nTotal pages rendered: {len(rendered_pages)}")
    return rendered_pages

if __name__ == "__main__":
    pdf_file = "incentivize.pdf"
    pages = render_pdf_pages(pdf_file, dpi=150)  # Lower DPI for file size

    print("\n=== Complete ===")
