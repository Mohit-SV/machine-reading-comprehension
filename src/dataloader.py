"""
Creates and/or loads data as batches
"""

import torch
from datasets import Dataset
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ImageProcessor,
    LayoutLMv3TokenizerFast,
    RobertaTokenizer,
)
import json
from PIL import Image
from torch.utils.data import DataLoader
from constants import *
import pandas as pd
import numpy as np
import os
import pytorch_lightning as pl
from typing import Union
from data_preprocessing.squad.run_preprocess import create_visual_squad
from typing import Dict, Union, List
import configargparse


class SquadDataset(pl.LightningDataModule):
    def __init__(
        self,
        model_name: str,
        batch_size: int,
        dev_ratio: float,
        test_ratio: float,
        seed: int,
        llmv3_max_length: int,
        roberta_max_length: int,
        num_workers: int,
        dataset_dir: str = V_SQUAD_DIR,
        llmv3_checkpoint: str = LLMv3_BACKBONE,
        roberta_checkpoint: str = RoBERTa_BACKBONE,
    ):
        """
        :param model_name: LLMv3 or RoBERTa
        :param batch_size: number of samples per each batch
        :param dev_ratio: Validation set to total dataset size ratio
        :param test_ratio: Test set to total dataset size ratio
        :param seed: seed from randomness in splitting the dataset
        :param llmv3_max_length: max length for tokenizing for LLMv3
        :param roberta_max_length: max length for tokenizing for RoBERTa
        :param dataset_dir: directory in which dataset files live
        :param llmv3_checkpoint: LLMv3 checkpoint for tokenizing and processing
        :param roberta_checkpoint: RoBERTa checkpoint for tokenizing
        """

        super().__init__()

        self.batch_size = batch_size
        self.dev_ratio = dev_ratio
        self.test_ratio = test_ratio

        self.num_workers = num_workers

        self.dataset_dir = dataset_dir
        self.para_texts_path = os.path.join(self.dataset_dir, "doc_data.json")
        self.dataset_json_path = os.path.join(self.dataset_dir, "modelling_data.json")

        if not os.path.exists(self.para_texts_path) or not os.path.exists(self.dataset_json_path):
            self.create_dataset()

        self.para_texts_df = self.load_para_texts()
        self.dataset = self.load_dataset()

        self.seed = seed
        self.splitted_dataset = self.split_data()

        self.model_name = model_name
        assert self.model_name in ["LLMv3", "RoBERTa"]

        self.llmv3_max_length = llmv3_max_length
        self.roberta_max_length = roberta_max_length

        self.llmv3_tokenizer = LayoutLMv3TokenizerFast.from_pretrained(llmv3_checkpoint)
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained(roberta_checkpoint)

        self.feature_extractor = LayoutLMv3ImageProcessor(
            apply_ocr=False, ocr_lang="eng"
        )
        self.processor = LayoutLMv3Processor(
            self.feature_extractor, self.llmv3_tokenizer
        )

        self._prepare_examples = {
            "LLMv3": self.prepare_LLMv3_examples,
            "RoBERTa": self.prepare_RoBERTa_examples,
        }

    @staticmethod
    def create_dataset(
        train_data_path: str = TRAINSET_PATH,
        dev_data_path: str = DEVSET_PATH,
        output_dir: str = V_SQUAD_DIR,
        doc_limit: Union[int, None] = DOC_LIMIT,
        zip_path: Union[str, None] = ZIP_PATH,
        html_version: str = "2017",
        threshold_min: int = 1,
        threshold_max: int = 1,
        size_factor: int = 1,
        max_token_length: int = 512,
        ignore_impossible: bool = True,
        para_asset_type: str = "para_box",
    ):
        """
        Calls create_visual_squad from src\data_preprocessing\squad\run_preprocess.py to create
        visual squad dataset.
        """
        create_visual_squad(
            train_data_path=train_data_path,
            dev_data_path=dev_data_path,
            output_dir=output_dir,
            doc_limit=doc_limit,
            zip_path=zip_path,
            html_version=html_version,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            size_factor=size_factor,
            max_token_length=max_token_length,
            ignore_impossible=ignore_impossible,
            para_asset_type=para_asset_type,
        )

    def load_dataset(self) -> Dataset:
        """
        Loads dataset from input dataset json file.
        """
        return Dataset.from_json(self.dataset_json_path)

    def load_para_texts(self) -> pd.DataFrame:
        """
        Returns pandas dataframe containing doc_ids and list paras in each document.
        """
        return pd.read_json(self.para_texts_path, orient="records")

    def __getitem__(self, index: int) -> Dict:
        """
        Gets one item from dataset using its index.
        """
        return self.dataset[index]

    def __len__(self) -> int:
        """
        Number of documents in the dataset.
        """
        return len(self.dataset)

    def split_data(self) -> Dict[str, Dataset]:
        """
        Splits input dataset and returns dictionary of Dataset objects for train, dev, test.
        """
        rest_and_test = self.dataset.train_test_split(
            test_size=self.test_ratio, seed=self.seed
        )
        train_dev = rest_and_test["train"].train_test_split(
            test_size=self.dev_ratio / (1 - self.test_ratio), seed=self.seed
        )
        return {
            "train": train_dev["train"],
            "dev": train_dev["test"],
            "test": rest_and_test["test"],
        }

    def prepare_RoBERTa_examples(
        self, examples: List[Dict[str, Union[str, int]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Encodes the batch data (context + questions).

        :return: dictionary containing encoded input_ids, attention_mask, start_positions and
            end_positions.
        """
        contexts = [
            self.para_texts_df[self.para_texts_df["doc_id"] == example["doc_id"]][
                "paras"
            ].values[0][example["idx_para_matched"]]
            for example in examples
        ]

        questions = [example["question"].strip() for example in examples]

        encoding = self.roberta_tokenizer(
            questions,
            contexts,
            max_length=self.roberta_max_length,
            truncation="only_second",
            return_overflowing_tokens=False,
            return_offsets_mapping=False,
            padding="max_length",
        )

        encoding["start_positions"] = [
            example["answer_start"] + len(example["question"]) for example in examples
        ]
        encoding["end_positions"] = [
            example["answer_end"] + len(example["question"]) for example in examples
        ]

        encoding = {key: torch.LongTensor(value) for key, value in encoding.items()}

        return encoding

    def prepare_LLMv3_examples(
        self, examples: List[Dict[str, Union[str, int]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Encodes the combined data from batch information and image data got from paths in batch.

        :return: dictionary containing encoded input_ids, attention_mask, bounding box data,
            encoded pixel values, start_positions and end_positions.
        """
        images = [
            Image.open(os.path.join(os.getcwd(), example["image_path"]))
            for example in examples
        ]
        questions = [example["question"].strip() for example in examples]

        words, bboxes = list(), list()
        for example in examples:
            word_bboxes_path = os.path.join(
                os.getcwd(), example["image_path"].replace(".png", ".json")
            )
            with open(word_bboxes_path, "r", encoding="utf-8") as f:
                word_bboxes = json.load(f)
            words.append(word_bboxes["words"])
            bboxes.append(word_bboxes["bboxes"])

        encoding = self.processor(
            images,
            questions,
            words,
            boxes=bboxes,
            max_length=self.llmv3_max_length,
            truncation=True,
            padding="max_length",
        )

        encoding["start_positions"] = [
            example["answer_start"] + len(example["question"]) + 2
            for example in examples
        ]
        encoding["end_positions"] = [
            example["answer_end"] + len(example["question"]) + 2 for example in examples
        ]
        # FIXED LATER: Why +2 is needed? This trick doesn't work always

        for key, value in encoding.items():
            if key == "pixel_values":
                encoding[key] = torch.FloatTensor(
                    np.array(value)
                )  # need to use torch.cuda.FloatTensor?
            else:
                encoding[key] = torch.LongTensor(
                    np.array(value)
                )  # need to use torch.cuda.LongTensor?

        return encoding

    def dataloader(self, split_name: str) -> DataLoader:
        """
        Returns data as torch DataLoader object for given split (train/dev/test).
        """
        return DataLoader(
            self.splitted_dataset[split_name],
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers, # no.of cpu cores
            collate_fn=self._prepare_examples[self.model_name],
        )
        # set num_workers>0 and pin_memory=True (only for GPUs).

    @staticmethod
    def add_argparse_args(parent_parser) -> configargparse.ArgParser:
        """
        Imports args needed for running SquadDataset instance in src\main.py
        """
        parser = parent_parser.add_argument_group("SquadDataset")
        parser.add_argument(
            "--include_references",
            type=bool,
            default=False,
            help="To include citation numbers in text or not.",
        )
        parser.add_argument(
            "--html_version",
            type=str,
            default="2017",
            help="Version of wiki articles to scrape: 2017 or online.",
        )
        parser.add_argument(
            "--scale_factor",
            type=int,
            default=1,
            help="Zoom level for saving images from webpage PDFs.",
        )
        parser.add_argument(
            "--threshold_min",
            type=int,
            default=1,
            help="Match percentage (0 to 1) above which QA pair is said to be valid.",
        )
        parser.add_argument(
            "--threshold_max",
            type=int,
            default=1,
            help="Match percentage (0 to 1) below which QA pair is said to be valid.",
        )
        parser.add_argument(
            "--max_token_length",
            type=int,
            default=512,
            help="Limit on context + question length.",
        )
        parser.add_argument(
            "--ignore_impossible",
            type=bool,
            default=True,
            help="To ignore impossible QA pairs or not.",
        )
        parser.add_argument(
            "--para_asset_type",
            type=str,
            default="para_box",
            help="Type of para asset to generate: page_width_fit_para_box, para_box, or whole_para_page.",
        )
        parser.add_argument(
            "--dev_split",
            type=float,
            default=0.1,
            help="Dev set to whole dataset ratio.",
        )
        parser.add_argument(
            "--train_split",
            type=float,
            default=0.2,
            help="Test set to whole dataset ratio.",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=10,
            help="Train/validation/test batch size.",
        )
        parser.add_argument(
            "--image_size_llmv3",
            type=int,
            default=224,
            help="Image size to which LayoutLMv3ImageProcessor should resize.",
        )
        parser.add_argument(
            "--llmv3_max_length",
            type=int,
            default=512,
            help="Question + context tokens length for LLMv3.",
        )
        parser.add_argument(
            "--roberta_max_length",
            type=int,
            default=512,
            help="Question + context tokens length for RoBERTa.",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=0,
            help="Seed for randomness in train-dev-test split.",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=0,
            help="Number of workers in dataloader.",
        )
        return parent_parser
