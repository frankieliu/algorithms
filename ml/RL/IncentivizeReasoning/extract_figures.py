#!/usr/bin/env python3
"""
Extract all figures/images from the PDF file.
"""
import fitz  # PyMuPDF
import os
from pathlib import Path

def extract_images_from_pdf(pdf_path, output_dir="figures"):
    """Extract all images from PDF and save them."""
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Open PDF
    doc = fitz.open(pdf_path)

    image_list = []

    # Iterate through pages
    for page_num in range(len(doc)):
        page = doc[page_num]

        # Get images on the page
        images = page.get_images()

        print(f"Page {page_num + 1}: Found {len(images)} images")

        # Extract each image
        for img_index, img in enumerate(images):
            xref = img[0]  # xref is the image reference

            # Extract image data
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # Create filename
            image_filename = f"page_{page_num + 1:03d}_img_{img_index + 1:02d}.{image_ext}"
            image_path = os.path.join(output_dir, image_filename)

            # Save image
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)

            # Store info
            image_list.append({
                "page": page_num + 1,
                "index": img_index + 1,
                "filename": image_filename,
                "ext": image_ext,
                "size": len(image_bytes)
            })

            print(f"  Saved: {image_filename} ({len(image_bytes)} bytes)")

    doc.close()

    print(f"\nTotal images extracted: {len(image_list)}")
    return image_list

if __name__ == "__main__":
    pdf_file = "incentivize.pdf"
    images = extract_images_from_pdf(pdf_file)

    # Print summary
    print("\n=== Summary ===")
    for img in images:
        print(f"Page {img['page']}: {img['filename']}")
