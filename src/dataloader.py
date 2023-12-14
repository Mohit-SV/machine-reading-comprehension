"""
Creates and/or loads data as batches
"""

import torch
from datasets import Dataset
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ImageProcessor,
    LayoutLMv3TokenizerFast,
    RobertaTokenizerFast,
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
from data_preprocessing.squad.preprocess_squad import squad_load
from data_preprocessing.squad.run_preprocess import create_visual_squad
from typing import Dict, Union, List, Tuple
import configargparse


class VisualSquadDataset:
    def __init__(
        self,
        dataset: Dataset,
        model_name: str,
        batch_size: int,
        tokenizer_max_length: int,
        tokenizer_stride: int,
        num_workers: int,
        is_layout_dependent: bool,
        llmv3_correction,
        llmv3_checkpoint: str = LLMv3_BACKBONE,
        roberta_checkpoint: str = RoBERTa_BACKBONE
    ):
        """
        :param dataset: train/val/test Dataset
        :param model_name: LLMv3 or RoBERTa
        :param batch_size: size of batches outputted by dataloader
        :param tokenizer_max_length: max_length used for tokenizing each question-
            answer while offsetting answer positions
        :param tokenizer_stride: stride for while tokenizing each question-answer
            while offsetting answer positions
        :param num_workers: number of workers in dataloader to parallel process
        :param is_layout_dependent: whether RoBERTa should know the layout by \n's
        :param llmv3_checkpoint: LLMv3 checkpoint for tokenizing and processing
        :param roberta_checkpoint: RoBERTa checkpoint for tokenizing
        """

        super().__init__()

        self.batch_size = batch_size
        self.tokenizer_max_length = tokenizer_max_length
        self.tokenizer_stride = tokenizer_stride
        self.num_workers = num_workers
        self.is_layout_dependent = is_layout_dependent
        self.dataset = dataset
        self.llmv3_correction = llmv3_correction

        if model_name == "LLMv3":
            self.llmv3_tokenizer = LayoutLMv3TokenizerFast.from_pretrained(
                llmv3_checkpoint
            )
            self.image_feature_extractor = LayoutLMv3ImageProcessor(
                apply_ocr=False, ocr_lang="eng", size=None, do_resize=True
            )
            self.tokenizer = LayoutLMv3Processor(
                self.image_feature_extractor, self.llmv3_tokenizer
            )
            self._prepare_examples = self.prepare_LLMv3_examples
            self.tokenizer_roberta = RobertaTokenizerFast.from_pretrained(roberta_checkpoint)
        elif model_name == "RoBERTa":
            self.tokenizer = RobertaTokenizerFast.from_pretrained(roberta_checkpoint)
            self._prepare_examples = self.prepare_RoBERTa_examples
        else:
            raise ValueError(
                f"Unexpected type of model specified ({self.model_name}), "
                f"choose one of 'LLMv3' or 'RoBERTa'"
            )

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

    ###################################################
    # Dataloader
    ###################################################

    def to_dataloader(self, shuffle: bool = False) -> DataLoader:
        """
        Returns data as torch DataLoader object for given split (train/dev/test).
        """
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=self.num_workers,
            collate_fn=self._prepare_examples,
        )

    ###################################################
    # Collate functions
    ###################################################

    def prepare_RoBERTa_examples(
        self, examples: List[Dict[str, Union[str, int]]]
    ) -> Dict[str, Union[str, torch.Tensor, List[int]]]:
        """
        Encodes the batch data (context + questions).

        :param examples: batch of samples from dataloader input
        :return: dictionary containing encoded input_ids, attention_mask,
            overflow_to_sample_mapping, start_positions and end_positions.
        """
        contexts_with_linebreaks, questions = list(), list()

        for example in examples:
            questions.append(example["question"].strip())

            text_file_path = os.path.join(
                SRC_DIRECTORY, example["image_path"].replace("image.png", "text.txt")
            )
            with open(text_file_path, "r", encoding="utf-8") as file:
                contexts_with_linebreaks.append(file.read().replace("\n", "\n "))

        if self.is_layout_dependent:
            for i, example in enumerate(examples):
                example["answer_start_char_idx"] = self.map_indices(
                    contexts_with_linebreaks[i], example["answer_start_char_idx"]
                )
                example["answer_end_char_idx"] = self.map_indices(
                    contexts_with_linebreaks[i], example["answer_end_char_idx"]
                )
            contexts = contexts_with_linebreaks
        else:
            contexts = [
                context.replace("\n", "") for context in contexts_with_linebreaks
            ]

        tokenized_input = self.tokenizer(
            questions,
            contexts,
            max_length=self.tokenizer_max_length,
            stride=self.tokenizer_stride,
            truncation="only_second",
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        encoding = {}
        (
            encoding["start_positions"],
            encoding["end_positions"],
        ) = self._to_token_positions(tokenized_input, examples, tokenized_input["offset_mapping"])
        encoding.update(tokenized_input)

        encoding = {key: torch.LongTensor(value) for key, value in encoding.items()}

        # Needed for calculating metrics:
        encoding["answer_text"] = np.array(
            [example["answer_text"].strip() for example in examples]
        )

        encoding["overflow_to_sample_mapping"] = tokenized_input[
            "overflow_to_sample_mapping"
        ]

        return encoding
    
    @staticmethod    
    def normalize_bbox(bbox, width, height):
        return [
             int(1000 * (bbox[0] / 224)),
             int(1000 * (bbox[1] / 224)),
             int(1000 * (bbox[2] / 224)),
             int(1000 * (bbox[3] / 224)),
         ]

    def prepare_LLMv3_examples(
        self, examples: List[Dict[str, Union[str, int]]]
    ) -> Dict[str, Union[str, torch.Tensor, List[int]]]:
        """
        Encodes the combined data from batch information and image data got from paths in batch.

        :param examples: batch of samples from dataloader input
        :return: dictionary containing encoded input_ids, attention_mask, bounding box data,
            encoded pixel values, overflow_to_sample_mapping, start_positions and end_positions.
        """

        words, bboxes, images, questions, contexts = list(), list(), list(), list(), list()

        for example in examples:
            questions.append(example["question"].strip())
            text_file_path = os.path.join(
                SRC_DIRECTORY, example["image_path"].replace("image.png", "text.txt")
            )
            with open(text_file_path, "r", encoding="utf-8") as file:
                contexts.append(file.read().replace("\n", " "))

        tokenized_input_roberta = self.tokenizer_roberta(
            questions,
            contexts,
            max_length=self.tokenizer_max_length,
            stride=self.tokenizer_stride,
            truncation="only_second",
            return_overflowing_tokens=False,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = tokenized_input_roberta["offset_mapping"]

        del tokenized_input_roberta, contexts

        for example in examples:
            images.append(
                Image.open(os.path.join(SRC_DIRECTORY, example["image_path"]))
            )
            word_bboxes_path = os.path.join(
                SRC_DIRECTORY, example["image_path"].replace("image.png", "bboxes.json")
            )
            with open(word_bboxes_path, "r", encoding="utf-8") as f:
                word_bboxes = json.load(f)
            bboxes = [self.normalize_bbox(bbox) for bbox in word_bboxes["bboxes"]]
            words.append(word_bboxes["words"])
            bboxes.append(bboxes)

        tokenized_input = self.tokenizer(
            images,
            questions,
            words,
            boxes=bboxes,
            max_length=self.tokenizer_max_length,
            stride=self.tokenizer_stride,
            truncation="only_second",
            return_overflowing_tokens=True,
            return_offsets_mapping=False,
            padding="max_length",
        )
        
        # if self.llmv3_correction:
        #     tokenized_input["offset_mapping"] = [
        #         self.corrected_llmv3_offsets(
        #             offsets, tokenized_input.word_ids(i), self.tokenizer_max_length
        #         )
        #         for i, offsets in enumerate(tokenized_input["offset_mapping"])
        #     ]

        encoding = {}
        (
            encoding["start_positions"],
            encoding["end_positions"],
        ) = self._to_token_positions(tokenized_input, examples, offset_mapping)
        encoding.update(tokenized_input)

        for key, value in encoding.items():
            if key == "pixel_values":
                encoding[key] = torch.FloatTensor(np.array(value))
            else:
                encoding[key] = torch.LongTensor(np.array(value))

        # Needed for calculating metrics:
        encoding["answer_text"] = [
            example["answer_text"].strip() for example in examples
        ]

        encoding["overflow_to_sample_mapping"] = tokenized_input[
            "overflow_to_sample_mapping"
        ]

        return encoding

    ###################################################
    # Helper functions
    ###################################################

    @staticmethod
    def corrected_llmv3_offsets(
        offsets: List[Tuple[int]], word_ids: List[int], max_length: int
    ):
        """
        LLMv3 offsets look weird. Function converts to similar structure as that
        of RoBERTa offsets.
        e.g., offsets of a span:
            LLMv3: [(0, 0), (0, 4), (5, 8), (9, 14), (14, 16), (17, 22), (23, 30),
            (30, 32), (33, 38), (39, 42), (43, 49), (50, 51), (52, 56), (57, 63),
            (63, 64), (0, 0), (0, 0), (0, 5), (5, 7), (8, 13), (13, 15), (16, 26),
            (27, 29), (30, 36), (37, 43), (44, 47), (48, 57), (58, 63), (64, 68),
            (68, 69), (70, 74), (75, 79), (80, 85), (85, 87), (88, 89), (89, 91),
            (91, 95), (96, 100), (100, 103), (104, 113)]
            RoBERTa: [(0, 0), (0, 4), (5, 8), (9, 12), (13, 17), (18, 20),
            (21, 26), (26, 28), (28, 30), (31, 36), (37, 41), (42, 47), (47, 48),
            (0, 0), (0, 0), (0, 3), (3, 5), (5, 7), (7, 8), (9, 14), (14, 16),
            (17, 27), (28, 30), (31, 37), (38, 44), (45, 48), (48, 49), (50, 59),
            (60, 65), (66, 70), (70, 71), (72, 76), (76, 77), (78, 82), (82, 83),
            (84, 89), (89, 91), (92, 93), (93, 95), (95, 99)]

        :param offsets: LLMv3 offsets
        :param word_ids: A list indicating the word corresponding to each token.
            Special tokens added by the tokenizer are mapped to None and other tokens
            are mapped to the index of their corresponding word (several tokens will
            be mapped to the same word index if they are parts of that word).
        :param max_length: max sequence length of each span
        :return: corrected offsets
        """
        # for each token in input sequence
        for i in range(max_length):
            # if not padding token
            if offsets[i][1] != 0:
                # if
                if offsets[i - 1][1] != 0 and word_ids[i] != word_ids[i - 1]:
                    offsets[i] = (
                        offsets[i - 1][1] + 1,
                        offsets[i - 1][1] + (offsets[i][1] - offsets[i][0]) + 1,
                    )
                else:
                    offsets[i] = (
                        offsets[i - 1][1],
                        offsets[i - 1][1] + (offsets[i][1] - offsets[i][0]),
                    )
        return offsets

    @staticmethod
    def _to_token_positions(
        tokenized_examples: Dict[str, torch.Tensor],
        input_examples: List[Dict[str, Union[str, int]]],
        offset_mapping
    ) -> Tuple[List[int]]:
        """
        Returns answer start and end token indices in model input sequence using
        start and end character indices in page

        :param tokenized_examples: huggingface tokenizer output of a batch
        :param input_examples: batch of a=samples as list of dictionaries,
            each dictionary contains answer_start_char_idx, answer_end_char_idx
        :returns:
            - list of answer start token indices in given batch
            - list of answer end token indices in given batch
        """
        sample_mapping = tokenized_examples["overflow_to_sample_mapping"]
        cls_index = 0
        n_spans = len(sample_mapping)
        start_positions = [cls_index] * n_spans
        end_positions = [cls_index] * n_spans

        for i, offsets in enumerate(offset_mapping):

            input_ids = tokenized_examples["input_ids"][i]
            sequence_ids = tokenized_examples.sequence_ids(i)
            sample_index = sample_mapping[i]

            # If no answers are given, set the cls_index as answer. Else...
            if len(input_examples) != 0:
                # Start/end character index of the answer in the text.
                start_char = input_examples[sample_index]["answer_start_char_idx"]
                end_char = input_examples[sample_index]["answer_end_char_idx"]

                # Start token index of the current span in the text.
                context_start_index = 0
                while sequence_ids[context_start_index] != 1:
                    context_start_index += 1

                # End token index of the current span in the text.
                context_end_index = len(input_ids) - 1
                while sequence_ids[context_end_index] != 1:
                    context_end_index -= 1

                # If the answer is not fully inside the context, label is (0, 0). Else...
                if (
                    offsets[context_start_index][0] <= start_char
                    and offsets[context_end_index][1] >= end_char
                ):
                    idx = context_start_index
                    while idx <= context_end_index and offsets[idx][0] <= start_char:
                        idx += 1
                    start_positions[i] = idx - 1

                    idx = context_end_index
                    while idx >= context_start_index and offsets[idx][1] >= end_char:
                        idx -= 1
                    end_positions[i] = idx + 1

        return start_positions, end_positions

    @staticmethod
    def map_indices(string2, string1_target_index):
        """
        Returns index found in page text with linebreakes corresponding to index 
        of page text without linebreakes
        e.g.,
        string1 = "01  45 7   X"
        string2 = "01 \n 45\n 7\n \n  X"
        each character in string1 should match with that of string2

        :param string2: string with line brakes
        :param string1_target_index: An index from string 1 which is without line brakes
        :return: 
        """
        # string1, string2 are text obtained from pdf over which "\n" are
        # replaced by " " and "\n " respectively. Example text obtained from pdf:
        # "Headline\nsome text\nSection\nsome more text..."

        # Find the corresponding index in string2
        string1_index = -1
        for string2_index in range(len(string2)):
            if string2[string2_index] != "\n":
                string1_index += 1
            if string1_index == string1_target_index:
                break

        return string2_index
    

class VisualSquadDataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name: str,
        batch_size: int,
        dev_ratio: float,
        test_ratio: float,
        split_seed: int,
        tokenizer_max_length: int,
        tokenizer_stride: int,
        num_workers: int,
        is_layout_dependent: bool,
        include_references: bool,
        image_width: Union[int, None],
        image_height: Union[int, None],
        percent,
        n_spans,
        llmv3_correction,
        dataset_dir: str = V_SQUAD_DIR,
        train_data_seed: Union[int, None] = None
    ):
        """
        :param model_name: LLMv3 or RoBERTa
        :param batch_size: size of batches outputted by dataloader
        :param dev_ratio: Validation set to total dataset size ratio
        :param test_ratio: Test set to total dataset size ratio
        :param split_seed: seed of randomness in splitting the dataset
        :param tokenizer_max_length: max_length used for tokenizing each question-
            answer while offsetting answer positions
        :param tokenizer_stride: stride for while tokenizing each question-answer
            while offsetting answer positions
        :param num_workers: number of workers in dataloader to parallel process
        :param is_layout_dependent: whether RoBERTa should know the layout by \n's
        :param include_references: to include citation numbers from webpages or not
        :param image_size: size to which images are to be resized
        :param dataset_dir: directory in which dataset files live
        :param train_data_seed: seed with which train dataloader should shuffle
        """

        super().__init__()

        self.batch_size = batch_size
        self.dev_ratio = dev_ratio
        self.test_ratio = test_ratio
        self.tokenizer_max_length = tokenizer_max_length
        self.tokenizer_stride = tokenizer_stride
        self.num_workers = num_workers
        self.is_layout_dependent = is_layout_dependent
        self.model_name = model_name
        self.train_data_seed = train_data_seed

        self.dataset_dir = dataset_dir
        self.fitting_dataset_json_path = os.path.join(self.dataset_dir, f"final_data_{n_spans}spans", f"modelling_fitting_data_{percent}.json")
        self.test_dataset_json_path = os.path.join(self.dataset_dir, f"final_data_{n_spans}spans", f"modelling_test_data.json")
        
        self.llmv3_correction=llmv3_correction

        # if not os.path.exists(self.dataset_json_path):
        #     self.create_dataset(
        #         doc_limit=2,
        #         include_references=include_references,
        #         zip_path=None,
        #         image_width=image_width,
        #         image_height=image_height
        #     )

        self.fitting_dataset = self.load_dataset(self.fitting_dataset_json_path)
        self.test_dataset = self.load_dataset(self.test_dataset_json_path)
        self.split_seed = split_seed
        self.splitted_dataset = self.split_data()

    ###################################################
    # Dataset creation
    ###################################################

    @staticmethod
    def create_dataset(
        data_paths: List[str] = [TRAINSET_PATH, DEVSET_PATH],
        output_dir: str = V_SQUAD_DIR,
        doc_limit: Union[int, None] = DOC_LIMIT,
        zip_path: Union[str, None] = ZIP_PATH,
        html_version: str = "2017",
        threshold_min: int = 1,
        threshold_max: int = 1,
        image_width: Union[int, None] = None,
        image_height: Union[int, None] = None,
        include_references: bool = False,
    ):
        """
        Calls create_visual_squad from src\data_preprocessing\squad\run_preprocess.py to create
        visual squad dataset.
        """
        data = list()
        for data_path in data_paths:
            data.extend(squad_load(data_path))

        create_visual_squad(
            data=data,
            output_dir=output_dir,
            doc_limit=doc_limit,
            zip_path=zip_path,
            html_version=html_version,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            image_width=image_width,
            image_height=image_height,
            include_references=include_references,
        )

    ###################################################
    # Loading dataset
    ###################################################

    @staticmethod
    def load_dataset(data_path) -> Dataset:
        """
        Loads dataset from input dataset json file.
        """
        return Dataset.from_json(data_path)

    def split_data(self) -> Dict[str, Dataset]:
        """
        Splits input dataset and returns dictionary of Dataset objects for train, dev, test.
        """
        # No shuffle
        if self.split_seed is None:
            train_dev = self.fitting_dataset.train_test_split(
                test_size=self.dev_ratio, shuffle=False
            )
        # With shuffling
        else:
            train_dev = self.fitting_dataset.train_test_split(
                test_size=self.dev_ratio, seed=self.split_seed
            )
        return {
            "train": train_dev["train"],
            "val": train_dev["test"]
        }

    def setup(self, stage=None):
        """
        Initialize object detection datasets distinguished into datasets during fitting proces and testing proces.
        """
        if stage == "fit":
            self.train_dataset = VisualSquadDataset(
                dataset=self.splitted_dataset["train"],
                model_name=self.model_name,
                batch_size=self.batch_size,
                tokenizer_max_length=self.tokenizer_max_length,
                tokenizer_stride=self.tokenizer_stride,
                num_workers=self.num_workers,
                is_layout_dependent=self.is_layout_dependent,
                llmv3_correction=self.llmv3_correction
            )
            self.val_dataset = VisualSquadDataset(
                dataset=self.splitted_dataset["val"],
                model_name=self.model_name,
                batch_size=self.batch_size,
                tokenizer_max_length=self.tokenizer_max_length,
                tokenizer_stride=self.tokenizer_stride,
                num_workers=self.num_workers,
                is_layout_dependent=self.is_layout_dependent,
                llmv3_correction=self.llmv3_correction
            )
        if stage == "test":
            self.test_dataset = VisualSquadDataset(
                dataset=self.test_dataset,
                model_name=self.model_name,
                batch_size=self.batch_size,
                tokenizer_max_length=self.tokenizer_max_length,
                tokenizer_stride=self.tokenizer_stride,
                num_workers=self.num_workers,
                is_layout_dependent=self.is_layout_dependent,
                llmv3_correction=self.llmv3_correction
            )
  
    ###################################################
    # Dataloaders
    ###################################################

    def train_dataloader(self) -> DataLoader:
        """
        Calls to_dataloader function that wraps data into dataloader object for training set.
        """
        shuffle_trainset = False if self.train_data_seed is None else True
        return self.train_dataset.to_dataloader(shuffle=shuffle_trainset)

    def val_dataloader(self) -> DataLoader:
        """
        Calls to_dataloader function that wraps data into dataloader object for validation set.
        """
        return self.val_dataset.to_dataloader()

    def test_dataloader(self) -> DataLoader:
        """
        Calls to_dataloader function that wraps data into dataloader object for test set.
        """
        return self.test_dataset.to_dataloader()

    ###################################################
    # Arguments
    ###################################################

    @staticmethod
    def add_argparse_args(
        parent_parser: configargparse.ArgParser,
    ) -> configargparse.ArgParser:
        """
        Imports args needed for running SquadDataset instance in src\main.py
        """
        parser = parent_parser.add_argument_group("SquadDataset")
        parser.add_argument(
            "--include_references",
            action="store_true",
            default=False,
            help="To include citation numbers from webpages or not.",
        )
        parser.add_argument(
            "--html_version",
            type=str,
            default="2017",
            help="Version of wiki articles to scrape: 2017 or online.",
        )
        parser.add_argument(
            "--percent",
            type=int,
            default=1,
            help="percentage of fitting data.",
        )
        parser.add_argument(
            "--n_spans",
            type=int,
            default=14,
            help="n spans.",
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
            "--dev_ratio",
            type=float,
            default=0.1,
            help="Dev set to whole dataset ratio.",
        )
        parser.add_argument(
            "--test_ratio",
            type=float,
            default=0.1,
            help="Test set to whole dataset ratio.",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=10,
            help="Train/validation/test batch size.",
        )
        parser.add_argument(
            "--image_width",
            type=int,
            default=None,
            help="Image width with which we want to save.",
        )
        parser.add_argument(
            "--image_height",
            type=int,
            default=None,
            help="Image height with which we want to save.",
        )
        parser.add_argument(
            "--tokenizer_max_length",
            type=int,
            default=512,
            help="max_length used for tokenizing each question-answer.",
        )
        parser.add_argument(
            "--tokenizer_stride",
            type=int,
            default=50,
            help="stride for while tokenizing each question-answer while offsetting answer positions.",
        )
        parser.add_argument(
            "--split_seed",
            type=int,
            default=None,
            help="Seed for randomness in train-dev-test split. If None, it won't be shuffled",
        )
        parser.add_argument(
            "--dataloader_seed",
            type=int,
            default=None,
            help="Shuffles train set with this seed. If None, it won't be shuffled",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=0,
            help="Number of workers in dataloader.",
        )
        parser.add_argument(
            "--is_layout_dependent",
            action="store_true",
            default=True,
            help="Should RoBERTa know the layout by \\n's.",
        )
        parser.add_argument(
            "--llmv3_correction",
            action="store_true",
            default=False,
            help="llmv3_correction.",
        )
        return parent_parser
