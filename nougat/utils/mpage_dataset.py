import logging
import os
from math import prod
from pathlib import Path
from functools import partial
import random
from typing import Dict, Tuple, Callable
from PIL import Image, UnidentifiedImageError
from typing import List, Optional

import torch
import pypdf
import orjson
from torch.utils.data import Dataset
from transformers.modeling_utils import PreTrainedModel
from nougat.dataset.rasterize import rasterize_paper


class MultiPageDocumentsDataset(Dataset):
    """
    Args:
        dataset_path: the path to the jsonl file
    """

    def __init__(
        self,
        images_path: str,
        gt_path: str,
        nougat_model: PreTrainedModel,
        max_length: int,
        split: str = "train"
    ):
        super().__init__()
        self.images_path = images_path
        self.gt_path = gt_path
        self.nougat_model = nougat_model
        self.max_length = max_length
        self.perturb = "NOUGAT_PERTURB" in os.environ and os.environ["NOUGAT_PERTURB"]
        self.split = split
        
        self.dataset = os.path.listdir(gt_path)
        
        self.dataset_length = len(self.dataset)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels.
        Convert gt data into input_ids (tokenized string)

        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
        """
        sample = self.dataset[idx]

        img_paths = os.listdir(os.path.join(self.images_path, sample))
        
        with open(os.listdir(os.path.join(self.gt_path, sample)), "r") as f:
            gt = f.read()

        imgs = [Image.open(path) for path in img_paths]

        images_tensor = torch.cat([self.nougat.model.encoder.prepare_input(img, random_padding=self.split == "train") for img in imgs])

        tokenizer_out = self.nougat_model.decoder.tokenizer(
            gt,
            max_length=self.max_length,
            padding="max_length",
            return_token_type_ids=False,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenizer_out["input_ids"].squeeze(0)
        attention_mask = tokenizer_out["attention_mask"].squeeze(0)
        # randomly perturb ground truth tokens
        if self.split == "train" and self.perturb:
            # check if we perturb tokens
            unpadded_length = attention_mask.sum()
            while random.random() < 0.1:
                try:
                    pos = random.randint(1, unpadded_length - 2)
                    token = random.randint(
                        23, len(self.nougat_model.decoder.tokenizer) - 1
                    )
                    input_ids[pos] = token
                except ValueError:
                    break
        return images_tensor, input_ids, attention_mask    
