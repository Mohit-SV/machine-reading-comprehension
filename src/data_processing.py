"""
Preprocessing common to all datasets
"""

import fitz  # PyMuPDF
import os
import json
from transformers import RobertaTokenizerFast
from constants import RoBERTa_BACKBONE
import pandas as pd
from datasets import Dataset
import logging
from typing import Union, Dict, Tuple, List

# Init logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s: %(message)s"
)


##################################################
# Generating images, words & their bounding boxes
##################################################

# FIXED LATER: Fails for 2 questions out of 124052 questions
def save_images_words_bboxes(
    pdf_path: str, image_dir: str, bboxes_path: str, size_factor: int = 1
):
    """
    Saves words and bounding boxes of each page in json file in the format as
        shown below:
        [
            'words': [...], # list of all detectable words (doesn't contain words
                from images in document)
            'bboxes': [...] # list of bounding boxes corresponding to words saved
                above
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


##################################################
# Offsetting answer positions
##################################################


def offset_start_end_positions(
    dataset_json_path: str,
    para_texts_path: str,
    tokenizer_max_length: int,
    tokenizer_stride: int,
    batch_size: int,
):
    """
    Reads modelling data json and updates it by offsetting the answer start and end
    token indices for each question in it.

    :param dataset_json_path: path of json containing modelling data
    :param para_texts_path: path of json containing para texts of every document/website
    :param tokenizer_max_length: max_length used while tokenizing each question-answer
    :param tokenizer_stride: stride used while tokenizing each question-answer
    :param batch_size: number of question-answers to be processed as a batch
    """
    dataset = Dataset.from_json(dataset_json_path)
    para_texts_df = pd.read_json(para_texts_path, orient="records")
    roberta_tokenizer = RobertaTokenizerFast.from_pretrained(RoBERTa_BACKBONE)

    def add_positions_for_roberta_llmv3(examples):

        questions = [question.strip() for question in examples["question"]]

        contexts = [
            para_texts_df[para_texts_df["doc_id"] == doc_id]["paras"].values[0][
                idx_para_matched
            ]
            for doc_id, idx_para_matched in zip(
                examples["doc_id"], examples["idx_para_matched"]
            )
        ]

        answers = {
            "text": [answer_text.strip() for answer_text in examples["answer_text"]]
        }
        answers["start_positions"] = examples["answer_start_char_idx"]
        answers["end_positions"] = examples["answer_end_char_idx"]

        # Tokenize our examples with truncation and padding, but keep the overflows
        # using a stride. This results in one example possible giving several features
        # when a context is long, each of those features having a context that overlaps
        # a bit the context of the previous feature.

        tokenized_by_roberta = roberta_tokenizer(
            questions,
            contexts,
            truncation="only_second",
            max_length=tokenizer_max_length,
            stride=tokenizer_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        (
            examples["answer_start_token_idx"],
            examples["answer_end_token_idx"],
        ) = positions_from_tokenized_input(tokenized_by_roberta, answers)

        return examples

    dataset = dataset.map(
        add_positions_for_roberta_llmv3,
        batched=True,
        remove_columns=["answer_start_char_idx", "answer_end_char_idx"],
        batch_size=batch_size,
        load_from_cache_file=False,
    )

    dataset.to_json(dataset_json_path)


def positions_from_tokenized_input(
    tokenized_examples: Dict, answers: Dict[str, Union[str, int]]
) -> Tuple[List[int]]:
    """
    Returns offsetted answer start and end token indices in model input sequence.

    :param tokenized_examples: huggingface tokenizer output of a batch
    :param answers: dictonary containing list of answer text, start_char_idx,
        end_char_idx in a batch
    :returns:
        - list of answer start token indices in given batch
        - list of answer end token indices in given batch
    """
    sample_mapping = tokenized_examples["overflow_to_sample_mapping"]
    offset_mapping = tokenized_examples["offset_mapping"]
    cls_index = 0
    batch_size = max(sample_mapping) + 1
    start_positions = {sample_index: cls_index for sample_index in range(batch_size)}
    end_positions = {sample_index: cls_index for sample_index in range(batch_size)}

    for i, offsets in enumerate(offset_mapping):

        input_ids = tokenized_examples["input_ids"][i]
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]

        # If no answers are given, set the cls_index as answer.
        if len(answers["start_positions"]) != 0:
            # Start/end character index of the answer in the text.
            start_char = answers["start_positions"][sample_index]
            end_char = answers["end_positions"][sample_index]

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            if (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                try:
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    start_positions[sample_index] = token_start_index - 1

                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    end_positions[sample_index] = token_end_index + 1
                except:
                    logger.info(
                        "Failed to properly offset position of sample_index:{sample_index}"
                    )

    return list(start_positions.values()), list(end_positions.values())
