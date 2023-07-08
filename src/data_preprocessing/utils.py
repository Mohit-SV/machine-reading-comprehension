# external tools that can be reused in preprocessing of any dataset
from PIL import Image, ImageDraw
import os
from typing import List


def show_bboxes(bboxes: List, image_dir: str, pg_nr: str):
    """
    Displays image of a page with word boxes.

    :param word_bboxes: list of bounding boxes
    :param image_dir: path of directory in which document images are saved
    :param pg_nr: page number of document
    """
    image_path = os.path.join(image_dir, f"{pg_nr}.png")
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    for bbox in bboxes:
        draw.rectangle(bbox[:4], outline="blue", width=2)

    image.show()
