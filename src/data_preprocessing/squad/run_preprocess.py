"""
Builds Visual SQuAD dataset
"""

import os
import sys

FILE_DIRECTORY = os.path.dirname(__file__)
sys.path.append(FILE_DIRECTORY)

from typing import Union
from preprocess_squad import (
    collect_docs,
    create_squad_like_json,
    generate_pdfs_images_bboxes,
    create_preliminary_modelling_data,
)
import zipfile
import logging
import shutil

# Init logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s: %(message)s"
)


def create_visual_squad(
    data: str,
    output_dir: str,
    doc_limit: Union[int, None] = None,
    zip_path: Union[str, None] = None,
    html_version: str = "2017",
    threshold_min: int = 1,
    threshold_max: int = 1,
    image_width: Union[int, None] = None,
    image_height: Union[int, None] = None,
    include_references: bool = False,
):
    """
    Creates Visual SQuAD dataset based on original SQuAD dataset
    The output is a directory of given output_dir path that comprises:
        - htmls folder: containing webpage htmls of all wiki articles in the dataset
        - unsuccessful_logs folder: if errors occur in validating question-answer pairs
            or processing documents (webpages), respective json files are created that
            contain information about the errors
        - visual_squad folder: the main visual SQuAD dataset folder having the following
            file structure:
            visual_squad\
                <doc_id>\
                    pdf.pdf,
                    pages\
                        <page_nr>\
                            bboxes.json,
                            image.png,
                            text.txt
        - doc_paras.json: list of dicts. 1 dict for each document(webpage) of following structure:
            {
                "title": ..., 
                "paras": [...,...,...], 
                "doc_id": ...
            }
        - qas.json: list of dicts. 1 dict for each answer of following structure:
            {
                "is_validated": True/False,
                "percent_match": ...,
                "idx_para_matched": ...,
                "answer_start_char_idx": ...,
                "answer_end_char_idx": ...,
                "question": ...,
                "answer_text": ...,
                "doc_id": ...,
                "doc_title": ...,
                "qas_id": ...,
                "is_impossible": True/False,
                "plausible_answers": [...,...,...]
            }
        - squad_like_json.json: Visual SQuAD data that has similar structure as the input json
        - modelling_data.json: (formatted data to run QA models) list of dicts. 
            1 dict for each question of following structure:
            {
                "answer_text": "...",
                "question": "...",
                "image_path": "...",
                "answer_start_char_idx": ...,
                "answer_end_char_idx": ...,
                "qas_id": ...
            }

    :param data: original dataset as list of dicts. 1 dict per document(webpage)
    :param output_dir: directory in which the output files/folders are to be saved
    :param doc_limit: how many of first few documents/webpages are to be considered
        for producing the visual squad data
    :param zip_path: path of zipfile into which all the necessary data files for
        modelling are to be packed
    :param html_version: "online" or "2017" wiki article version
    :param threshold_min: match percentage threshold (0 to 1) above which
        question-answer pair is said to be valid
    :param threshold_max: match percentage threshold (0 to 1) below which
        question-answer pair is said to be valid
    :param image_size: size of images (image_size x image_size) to be saved
    :param include_references: whether to include citation tags like '[5]' in all files
    """
    if doc_limit:
        data = data[:doc_limit]

    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

    logger.info("Validating webpages and collecting HTMLs....")
    collect_docs(
        data,
        output_dir,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        html_version=html_version,
        include_references=include_references,
    )

    logger.info("Creating SQuAD-like JSON....")
    create_squad_like_json(output_dir)

    logger.info("Generating PDFs, page images, their words & bounding boxes....")
    generate_pdfs_images_bboxes(
        output_dir, image_width=image_width, image_height=image_height
    )

    logger.info("Generating jsons that are needed for modelling....")
    create_preliminary_modelling_data(output_dir)

    logger.info(
        f"Completed generating Visual Squad data! The files are ready in {output_dir}"
    )

    if zip_path:
        zip_output_data(output_dir, zip_path)
        logger.info(f"Zipped Visual Squad data into {zip_path}")


def zip_output_data(output_dir: str, out_file: str):
    """
    Wraps Visual SQuAD dataset with files that are needed just for modelling.
    It is useful to run experiments on colab.
    """
    zf = zipfile.ZipFile(out_file, mode="w")

    for r, d, f in os.walk(output_dir):
        for file in f:
            if file in [
                "bboxes.json",
                "image.png",
                "text.txt",
                "modelling_data.json",
            ]:
                original_file = os.path.join(r, file)
                output_file = os.path.relpath(os.path.join(r, file))[18:]
                zf.write(original_file, output_file)

    zf.close()