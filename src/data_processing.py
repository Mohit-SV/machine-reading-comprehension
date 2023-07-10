"""
Preprocessing common to all datasets
"""

import fitz  # PyMuPDF
import os
import json


def save_images_words_bboxes(
    pdf_path: str, image_dir: str, bboxes_path: str, size_factor: int = 1
):
    """
    Saves words and bounding boxes of each page in json file in the format as shown below:
        [
            'words': [...], # list of all detectable words (doesn't contain words from images in document)
            'bboxes': [...] # list of bounding boxes corresponding to words saved above
        ],... # one such list for each page

    :param pdf_path: path to document pdf
    :param image_dir: path directory in which document images are to be saved
    :param size_factor: how many times of original page has to be scaled
    :return: page count of the pdf
    """
    doc = fitz.open(pdf_path)
    if os.path.exists(bboxes_path):
        os.remove(bboxes_path)

    bboxes = []

    for page_nr, page in enumerate(doc):
        # Get the words and their bounding boxes on the page
        words = page.get_text("words")

        # Create a matrix to scale the page to the desired size
        mat = fitz.Matrix(size_factor, size_factor)
        pixmap = page.get_pixmap(matrix=mat)
        pixmap.save(os.path.join(image_dir, f"{page_nr}.png"), "png")

        tokens = [word[4] for word in words]
        bboxes = [
            [int(size_factor * coordinate) for coordinate in word[:4]] for word in words
        ]

        if page_nr == 0:
            with open(bboxes_path, "a") as f:
                f.write("[")
                json.dump({"bboxes": bboxes, "words": tokens}, f)
                f.write(",")
        elif page_nr == len(doc) - 1:
            with open(bboxes_path, "a") as f:
                json.dump({"bboxes": bboxes, "words": tokens}, f)
                f.write("]")
        else:
            with open(bboxes_path, "a") as f:
                json.dump({"bboxes": bboxes, "words": tokens}, f)
                f.write(",")

    return len(doc)
