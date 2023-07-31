"""
External tools that can be reused in visualize results preprocessed data
"""

from PIL import Image, ImageDraw
import os
from typing import List
import json


def show_bboxes(page_dir: str):
    """
    Displays image with word boxes.

    :param page_dir: path of directory in which page image, bboxes are saved
    """
    image_path = os.path.join(page_dir, "image.png")
    word_bboxes_path = os.path.join(page_dir, "bboxes.json")
    with open(word_bboxes_path, "r", encoding="utf-8") as f:
        word_bboxes = json.load(f)
    bboxes = word_bboxes["bboxes"]
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    for bbox in bboxes:
        draw.rectangle(bbox[:4], outline="blue", width=2)

    image.show()