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


class ThreePageSciPDFDataset(Dataset):
    """
    Custom dataset for scientific PDF data.

    This dataset loads data from JSONL files and provides access to images, ground truth,
    and metadata.

    Args:
        path_to_index (str): Path to the index file.
        split (str, optional): Split of the dataset (e.g., "train", "test"). Default is "train".
        root_name (str, optional): Root directory name. Default is an empty string.
        template (str, optional): Template for split naming. Default is "%s".

    Attributes:
        empty_sample: Placeholder for empty samples.
    """

    empty_sample = torch.zeros(1)

    def __init__(
        self,
        path_to_index: str,
        split: str = "train",
        root_name="",
        template="%s",
        prev_and_next = True
    ) -> None:
        super().__init__()
        self.prev_and_next = prev_and_next
        self.path_to_index = Path(path_to_index)
        self.root_name = root_name
        self.path_to_root = self.path_to_index.parent
        if not split in self.path_to_index.stem:
            pti = self.path_to_root / (template % split + ".jsonl")
            if pti.exists():
                self.path_to_index = pti
            else:
                raise ValueError(f'Dataset file for split "{split}" not found: {pti}')
        self.dataset_file = None  # mulitprocessing
        # load seek map
        seek_path = self.path_to_root / (self.path_to_index.stem + ".seek.map")
        if seek_path.exists():
            self.seek_map = orjson.loads(seek_path.open().read())
        else:
            raise ValueError(
                'No "%s" found in %s' % (seek_path.name, str(self.path_to_root))
            )
        self.dataset_length = len(self.seek_map)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, index: int) -> Dict:
        position = self.seek_map[index]
        if self.dataset_file is None:
            self.dataset_file = self.path_to_index.open()
        self.dataset_file.seek(position)
        line = self.dataset_file.readline()
        try:
            data: Dict = orjson.loads(line)
        except Exception as e:
            logging.info(
                "JSONL for sample %i could not be loaded at position %i: %s\n%s",
                index,
                position,
                str(e),
                line,
            )
            return None
        img_name = data.pop("image")
        img_path: Path = self.path_to_root / self.root_name / img_name
        if not img_path.exists():
            logging.info("Sample %s could not be found.", img_path)
            return None
        try:
            img = Image.open(img_path)
        except UnidentifiedImageError:
            logging.info("Image %s could not be opened.", img_path)
            return None
        
        prev_image, next_image, prev_gt, next_gt = self.find_prev_and_next(img_name)

        return {"image": img, "prev_img": prev_image, "next_img": next_image, "prev_gt": prev_gt, "next_gt": next_gt, "ground_truth": data.pop("markdown"), "meta": data}
    
    def find_prev_and_next(self, img_name):
        n = int(img_name.split('/')[1].split('.')[0])
        prefix = img_name.split('/')[0]
        next_img_name: Path = self.path_to_root / self.root_name / "{}/{:02}.png".format(prefix, n+1)
        next_gt_name: Path = self.path_to_root / self.root_name / "{}/{:02}.mmd".format(prefix, n+1)
        previous_image_name: Path = self.path_to_root / self.root_name / "{}/{:02}.png".format(prefix, n-1)
        previous_gt_name: Path = self.path_to_root / self.root_name / "{}/{:02}.mmd".format(prefix, n-1)

        try:
            next_img = Image.open(next_img_name)
            with open(next_gt_name, "r") as f:
                next_gt = f.read()
        except:
            # logging.info("Image %s (to be used as next) could not be opened.", next_name)
            next_img = self.empty_sample
            next_gt = ""
        
        try:
            prev_img = Image.open(previous_image_name)
            with open(previous_gt_name, "r") as f:
                prev_gt = f.read()
        except:
            # logging.info("Image %s (to be used as prev) could not be opened.", previous_name)
            prev_img = self.empty_sample
            prev_gt = ""

        return prev_img, next_img, prev_gt, next_gt

    def __iter__(self):
        for i in range(self.dataset_length):
            yield self[i]


class ThreePageDataset(Dataset):
    """
    Args:
        dataset_path: the path to the jsonl file
    """
    def __init__(
        self,
        dataset_path: str,
        nougat_model: PreTrainedModel,
        max_length: int,
        split: str = "train",
        root_name: str = "arxiv",
        prev_and_next = False
    ):
        super().__init__()
        self.nougat_model = nougat_model
        self.max_length = max_length
        self.split = split
        self.perturb = "NOUGAT_PERTURB" in os.environ and os.environ["NOUGAT_PERTURB"]
        # TODO improve naming conventions
        template = "%s"
        self.dataset = ThreePageSciPDFDataset(
            dataset_path, split=self.split, template=template, root_name=root_name, prev_and_next=prev_and_next
        )
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
        if sample is None:
            # if sample is broken choose another randomly
            return self[random.randint(0, self.dataset_length - 1)]
        if sample is None or sample["image"] is None or prod(sample["image"].size) == 0:
            input_tensor = None
        else:
            input_tensor = self.nougat_model.encoder.prepare_input(
                sample["image"], random_padding=self.split == "train"
            )

        if sample is None or sample["next_img"] is None or (isinstance(sample["next_img"], torch.Tensor) and sample["next_img"].max() == 0) or prod(sample["image"].size) == 0:
            next_image_tensor = torch.zeros(1)
        else:
            next_image_tensor = self.nougat_model.encoder.prepare_input(
                sample["next_img"], random_padding=self.split == "train"
            )

        if sample is None or sample["prev_img"] is None or (isinstance(sample["prev_img"], torch.Tensor) and sample["prev_img"].max() == 0) or prod(sample["image"].size) == 0:
            prev_image_tensor = torch.zeros(1)
        else:
            prev_image_tensor = self.nougat_model.encoder.prepare_input(
                sample["prev_img"], random_padding=self.split == "train"
            )


        tokenizer_out = self.nougat_model.decoder.tokenizer(
            f'{sample["prev_gt"]}\n{sample["ground_truth"]}\n{sample["next_gt"]}',
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
        return input_tensor, next_image_tensor, prev_image_tensor, input_ids, attention_mask


class ThreePageDocumentsDataset(Dataset):
    """
    Args:
        dataset_path: the path to the jsonl file
    """

    def __init__(
        self,
        dataset_path: str,
        nougat_model: PreTrainedModel,
        max_length: int,
        split: str = "train",
        root_name: str = "arxiv",
        prev_and_next = False
    ):
        super().__init__()
        self.nougat_model = nougat_model
        self.max_length = max_length
        self.split = split
        self.perturb = "NOUGAT_PERTURB" in os.environ and os.environ["NOUGAT_PERTURB"]
        # TODO improve naming conventions
        template = "%s"
        self.dataset = ThreePageSciPDFDataset(
            dataset_path, split=self.split, template=template, root_name=root_name, prev_and_next=prev_and_next
        )
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
        if sample is None:
            # if sample is broken choose another randomly
            return self[random.randint(0, self.dataset_length - 1)]
        if sample is None or sample["image"] is None or prod(sample["image"].size) == 0:
            input_tensor = None
        else:
            input_tensor = self.nougat_model.encoder.prepare_input(
                sample["image"], random_padding=self.split == "train"
            )

        if sample is None or sample["next_img"] is None or (isinstance(sample["next_img"], torch.Tensor) and sample["next_img"].max() == 0) or prod(sample["image"].size) == 0:
            next_image_tensor = torch.zeros(1)
        else:
            next_image_tensor = self.nougat_model.encoder.prepare_input(
                sample["next_img"], random_padding=self.split == "train"
            )

        if sample is None or sample["prev_img"] is None or (isinstance(sample["prev_img"], torch.Tensor) and sample["prev_img"].max() == 0) or prod(sample["image"].size) == 0:
            prev_image_tensor = torch.zeros(1)
        else:
            prev_image_tensor = self.nougat_model.encoder.prepare_input(
                sample["prev_img"], random_padding=self.split == "train"
            )


        tokenizer_out = self.nougat_model.decoder.tokenizer(
            f'{sample["prev_gt"]}\n{sample["ground_truth"]}\n{sample["next_gt"]}',
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
        return input_tensor, next_image_tensor, prev_image_tensor, input_ids, attention_mask    
