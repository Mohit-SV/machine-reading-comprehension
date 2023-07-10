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
    generate_para_images,
    create_modelling_data,
    squad_load
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
    train_data_path: str,
    dev_data_path: str,
    output_dir: str,
    doc_limit: Union[int, None] = None,
    zip_path: Union[str, None] = None,
    html_version: str = "2017",
    threshold_min: int = 1,
    threshold_max: int = 1,
    size_factor: int = 1,
    max_token_length: int = 512,
    ignore_impossible: bool = True,
    para_asset_type: str = "para_box"
):
    """
    Creates Visual SQuAD dataset using path of original SQuAD dataset file
    The output is a directory of given output_dir path that comprises:
        - htmls folder: containing webpage htmls of all wiki articles in the dataset
        - unsuccessful_logs folder: if errors occur in validating question-answer pairs
            or processing documents (webpages), respective json files are created that
            contain information about the errors
        - visual_squad folder: the main visual SQuAD dataset folder having the following
            file structure:
            visual_squad\
                <doc_id>\
                    ["images" folder, "para_images" folder, bboxes.json, pdf.pdf]
        - doc_data.json: list of dicts. 1 dict for each document(webpage) of following structure:
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
                "answer_start": ...,
                "answer_end": ...,
                "question": ...,
                "answer_text": ...,
                "doc_id": ...,
                "doc_title": ...,
                "qas_id": ...,
                "is_impossible": True/False,
                "plausible_answers": [...,...,...]
            }
        - squad_like_json.json: Visual SQuAD data that has similar structure as the input json
        - textual_data.json: formatted data for running textual QA models
        - visual_data.json: formatted data for running multi-modal QA models

    :param data: original dataset as list of dicts. 1 dict per document(webpage)
    :param output_dir: directory in which the output files/folders are to be saved
    :param html_version: "online" or "2017" wiki article version
    :param threshold_min: match percentage threshold (0 to 1) above which
        question-answer pair is said to be valid
    :param threshold_max: match percentage threshold (0 to 1) below which
        question-answer pair is said to be valid
    :param size_factor: factor by which the HTMLs are to be zoomed while
        generating images of the webpages
    :param para_asset_type: Type of para asset to generate "page_width_fit_para_box", 
        "para_box", or "whole_para_page"
    """

    train_data = squad_load(train_data_path)
    dev_data = squad_load(dev_data_path)

    data = train_data + dev_data
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
    )

    logger.info("Creating SQuAD-like JSON....")
    create_squad_like_json(output_dir)

    logger.info("Generating PDFs, page images, their words & bounding boxes....")
    # Assets: PDF, page images, words, bounding boxes
    generate_pdfs_images_bboxes(output_dir, size_factor=size_factor)

    logger.info("Generating para images, their words & bounding boxes....")
    # Asset: para images
    generate_para_images(output_dir, size_factor=size_factor)

    logger.info("Generating jsons that are needed for modelling....")
    # Assets: textual and multi-modal model data
    create_modelling_data(
        output_dir,
        max_token_length=max_token_length,
        ignore_impossible=ignore_impossible,
        para_asset_type=para_asset_type
    )

    logger.info(f"Completed generating Visual Squad data! The files are ready in {output_dir}")

    if zip_path:
        zip_output_data(output_dir, zip_path)
        logger.info(f"Zipped Visual Squad data into {zip_path}")


def zip_output_data(output_dir, out_file):
    """
    Wraps Visual SQuAD dataset with files that are needed just for modelling
    """

    zf = zipfile.ZipFile(out_file, mode="w")

    for r, d, f in os.walk(output_dir):
        for file in f:
            if file in [
                "para_box.json",
                "para_box.png",
                "doc_data.json",
                "squad_like_json.json",
                "modelling_data.json"
            ]:
                original_file= os.path.join(r, file)
                output_file = os.path.relpath(os.path.join(r, file))[18:]
                zf.write(original_file, output_file)

    zf.close()
