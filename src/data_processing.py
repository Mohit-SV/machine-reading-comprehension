"""
Preprocessing common to all datasets
"""

import fitz  # PyMuPDF
import os
import json
from typing import Optional
from PIL import Image
import shutil


##################################################
# Generating images, words & their bounding boxes
##################################################


def save_images_words_bboxes(
    pdf_path: str,
    pages_dir: str,
    width: Optional[int] = None,
    height: Optional[int] = None
):
    """
    Saves words and bounding boxes of each page in json file in the format as
        shown below:
        {
            'words': [...], # list of all detectable words (doesn't contain words
                from images in document)
            'bboxes': [...] # list of bounding boxes corresponding to words saved
                above
        }

    :param pdf_path: path to document pdf
    :param pages_dir: path of directory in which each page information (full page
      images, texts) is to be saved as separate directory named by the page number
    :param size_factor: how many times of original page has to be scaled
    # :return: page count of the pdf
    """
    doc = fitz.open(pdf_path)
    bboxes = []

    for page_nr, page in enumerate(doc):
        page_dir = os.path.join(pages_dir, str(page_nr))
        if os.path.exists(page_dir):
            shutil.rmtree(page_dir)
        os.makedirs(page_dir)

        # Get the text, words and their bounding boxes on the page
        words = page.get_text("words")
        text = page.get_text()

        # Save page text
        text_file_path = os.path.join(page_dir, "text.txt")
        with open(text_file_path, "w", encoding="utf-8") as file:
            file.write(text)

        # Create a matrix to scale the page to the desired size
        size_factor = 1
        mat = fitz.Matrix(size_factor, size_factor)
        pixmap = page.get_pixmap(matrix=mat)

        image_path = os.path.join(page_dir, "image.png")

        if width and height:
            pixmap_width, pixmap_height = pixmap.width, pixmap.height

            # Calculate the scaling factors
            scale_x = width / pixmap_width
            scale_y = height / pixmap_height

            # Convert the pixmap to a PIL image
            pil_image = Image.frombytes("RGB", (pixmap_width, pixmap_height), pixmap.samples)

            # Resize the PIL image to the specified size
            pil_image = pil_image.resize((width, height))

            # Save image
            pil_image.save(image_path)

            # Scale bounding boxes
            bboxes = [
                [
                    int(scale_x * word[0]),
                    int(scale_y * word[1]),
                    int(scale_x * word[2]),
                    int(scale_y * word[3]),
                    ] 
                for word in words
            ]
        else:
            pixmap.save(image_path, "png")
            bboxes = [
                [int(coordinate) for coordinate in word[:4]] for word in words
            ]

        tokens = [word[4] for word in words]

        # Save bounding boxes
        bboxes_path = os.path.join(page_dir, "bboxes.json")
        with open(bboxes_path, "w", encoding="utf-8") as f:
            json.dump({"bboxes": bboxes, "words": tokens}, f)
