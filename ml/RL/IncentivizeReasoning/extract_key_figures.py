#!/usr/bin/env python3
"""
Extract specific figure regions from PDF pages.
"""
import fitz  # PyMuPDF
import os
from pathlib import Path

def render_key_pages(pdf_path, output_dir="figures", dpi=150):
    """Render key pages that contain figures."""
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Open PDF
    doc = fitz.open(pdf_path)

    # Key pages with figures (0-indexed)
    # Based on typical paper structure
    key_pages = {
        2: "figure_1_conceptual",
        4: "figure_2_pass_k_curves_page1",
        5: "figure_2_pass_k_curves_page2",
        6: "figure_3_4_code_vision",
        7: "figure_5_accuracy_distribution",
        8: "figure_6_7_perplexity_distillation",
        9: "figure_8_algorithm_comparison",
        16: "appendix_figure_10",
        17: "appendix_figure_11",
        18: "appendix_figure_12",
        19: "appendix_figure_13",
        20: "appendix_figure_14_perplexity",  # Page 21, where we found the perplexity image
    }

    rendered = []

    for page_num, name in key_pages.items():
        if page_num >= len(doc):
            print(f"Skipping page {page_num + 1} (out of range)")
            continue

        page = doc[page_num]

        # Set zoom factor for resolution
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)

        # Render page to pixmap
        pix = page.get_pixmap(matrix=mat)

        # Create filename
        image_filename = f"{name}_p{page_num + 1}.png"
        image_path = os.path.join(output_dir, image_filename)

        # Save image
        pix.save(image_path)

        size_mb = os.path.getsize(image_path) / (1024 * 1024)
        rendered.append({
            "page": page_num + 1,
            "name": name,
            "filename": image_filename,
            "size_mb": size_mb
        })

        print(f"Rendered page {page_num + 1:2d} -> {image_filename} ({size_mb:.2f} MB)")

    doc.close()

    print(f"\nTotal pages rendered: {len(rendered)}")
    return rendered

if __name__ == "__main__":
    pdf_file = "incentivize.pdf"
    pages = render_key_pages(pdf_file, dpi=150)

    print("\n=== Complete ===")
    print("Figures saved to ./figures/ directory")
