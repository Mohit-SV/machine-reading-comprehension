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
    create_preliminary_modelling_data,
    group_answers_as_lists,
)
import zipfile
import logging
import shutil

SRC_DIRECTORY = os.path.split(os.path.split(os.path.dirname(__file__))[0])[0]
sys.path.append(SRC_DIRECTORY)
sys.path.append(os.path.join(SRC_DIRECTORY, "data_processing"))
from data_processing import offset_start_end_positions

# Init logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s: %(message)s"
)


def create_visual_squad(
    data: str,
    output_dir: str,
    tokenizer_max_length: int,
    tokenizer_stride: int,
    batch_size: int,
    doc_limit: Union[int, None] = None,
    zip_path: Union[str, None] = None,
    html_version: str = "2017",
    threshold_min: int = 1,
    threshold_max: int = 1,
    size_factor: int = 1,
    max_token_length_limit: int = 512,
    ignore_impossible: bool = True,
    para_asset_type: str = "para_box",
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
                "idx_para_matched": ...,
                "doc_id": ...,
                "answer_text": "...",
                "question": "...",
                "image_path": "...",
                "answer_start_token_idx": ...,
                "answer_end_token_idx": ...
            }

    :param data: original dataset as list of dicts. 1 dict per document(webpage)
    :param output_dir: directory in which the output files/folders are to be saved
    :param tokenizer_max_length: max_length used while tokenizing each question-answer
    :param tokenizer_stride: stride used while tokenizing each question-answer
    :param batch_size: batch size for offsetting answer positions
    :param doc_limit: how many of first few documents/webpages are to be considered
        for producing the visual squad data
    :param zip_path: path of zipfile into which all the necessary data files for
        modelling are to be packed
    :param html_version: "online" or "2017" wiki article version
    :param threshold_min: match percentage threshold (0 to 1) above which
        question-answer pair is said to be valid
    :param threshold_max: match percentage threshold (0 to 1) below which
        question-answer pair is said to be valid
    :param size_factor: factor by which the HTMLs are to be zoomed while
        generating images of the webpages
    max_token_length
    ignore_impossible: 
    :param para_asset_type: Type of para asset to generate "page_width_fit_para_box", 
        "para_box", or "whole_para_page"
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
    )

    logger.info("Creating SQuAD-like JSON....")
    create_squad_like_json(output_dir)

    logger.info("Generating PDFs, page images, their words & bounding boxes....")
    generate_pdfs_images_bboxes(output_dir, size_factor=size_factor)

    logger.info("Generating para images, their words & bounding boxes....")
    generate_para_images(output_dir, size_factor=size_factor)

    logger.info("Generating jsons that are needed for modelling....")
    create_preliminary_modelling_data(
        output_dir,
        max_token_length=max_token_length_limit,
        ignore_impossible=ignore_impossible,
        para_asset_type=para_asset_type,
    )

    dataset_json_path = os.path.join(output_dir, "modelling_data.json")
    para_texts_path = os.path.join(output_dir, "doc_data.json")

    logger.info(
        "Offsetting RoBERTa and/or LLMv3 start and end positions in the modelling data...."
    )
    offset_start_end_positions(
        dataset_json_path,
        para_texts_path,
        tokenizer_max_length=tokenizer_max_length,
        tokenizer_stride=tokenizer_stride,
        batch_size=batch_size,
    )

    # logger.info("Group answers corresponding to each question as list in the modelling data....")
    # group_answers_as_lists(dataset_json_path)

    logger.info(
        f"Completed generating Visual Squad data! The files are ready in {output_dir}"
    )

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
                "modelling_data.json",
            ]:
                original_file = os.path.join(r, file)
                output_file = os.path.relpath(os.path.join(r, file))[18:]
                zf.write(original_file, output_file)

    zf.close()
