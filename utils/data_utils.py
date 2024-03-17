import os
from typing import Tuple 
import torch
import random
import numpy as np
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

def import_data(dataset_name, dataset_config_name, train_data_dir, cache_dir, image_column, caption_column):
    """
    TODO: Refactor to handle various types of dataset.
    For evaluation, planned datasets are:
        - ImageFolder(Taggable)
        - Images with captions
    """
    if dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            dataset_name,
            dataset_config_name,
            cache_dir=cache_dir,
            data_dir=train_data_dir,
        )
    else:
            data_files = {}
            if train_data_dir is not None:
                data_files["train"] = os.path.join(train_data_dir, "**")
            dataset = load_dataset(
                "imagefolder",
                data_files=data_files,
                cache_dir=cache_dir)

    column_names = dataset["train"].column_names

    if image_column is None:
        raise ValueError("Image column is not specified.")
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if caption_column is None:
        raise ValueError("Caption column is not specified.")
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{caption_column}' needs to be one of: {', '.join(column_names)}"
    )
            
    return dataset, image_column, caption_column

def generate_default_transform(resolution, center_crop, random_flip):
    return transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

class DataPreprocessor:
    def __init__(self, 
                resolution,
                center_crop,
                random_flip,
                caption_column, 
                tokenizer, 
                is_train, 
                image_column,
                train_transforms="default"
                ):
                self.resolution = resolution
                self.center_crop = center_crop
                self.random_flip = random_flip
                self.caption_column = caption_column
                self.tokenizer = tokenizer
                self.is_train = is_train
                self.image_column = image_column
                self.caption_column = caption_column
                if train_transforms == "default":
                    print("Using default transforms, be careful with this option as it may not be suitable for your dataset.")
                    self.train_transforms = generate_default_transform(resolution, center_crop, random_flip)
                elif train_transforms.lower() == "none" or not train_transforms:
                    self.train_transforms = transforms.Compose([transforms.ToTensor()]) # image -> tensor
                else:
                    self.train_transforms = train_transforms
    def get_caption_column(self, examples):
        """
        Get the caption column from the dataset.
        If dropout or shuffle is not used, caching can be used to speed up the process.
        """
        captions = []
        for caption in examples[self.caption_column]:
                if isinstance(caption, str):
                    captions.append(caption)
                elif isinstance(caption, (list, np.ndarray)):
                    # take a random caption if there are multiple
                    captions.append(random.choice(caption) if self.is_train else caption[0])
                else:
                    raise ValueError(
                        f"Caption column `{self.caption_column}` should contain either strings or lists of strings."
                    )
        return captions
    
    #tag tokenizing
    def tokenize_captions(self, examples: dict):
        """
        Tokenize the captions using the tokenizer.
        """
        captions = self.get_caption_column(examples)
        inputs = self.tokenizer(
            captions, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    def preprocess_train(self, examples):
        # Preprocessing the datasets.
        images = [image.convert("RGB") for image in examples[self.image_column]]
        examples["pixel_values"] = [self.train_transforms(image) for image in images]
        examples["input_ids"] = self.tokenize_captions(examples)
        return examples

    def collate_fn(self, examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        batch = {"pixel_values": pixel_values, "input_ids": input_ids}
        return batch

    def get_data(self,
                accelerator,
                max_train_samples,
                dataset, 
                seed, 
                preprocess_train_transforms, 
                train_batch_size, 
                dataloader_num_workers,
                collate_fn
                ) -> Tuple[Dataset, DataLoader]:
        """
        Get the data for training and validation.
        """
        with accelerator.main_process_first():
            if max_train_samples is not None:
                dataset["train"] = dataset["train"].shuffle(seed=seed).select(range(max_train_samples))
            # Set the training transforms
            train_dataset = dataset["train"].with_transform(preprocess_train_transforms)
         #dataloader 
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=train_batch_size,
            num_workers=dataloader_num_workers
        )
        return train_dataset, train_dataloader
    
class CatDataPreprocessor(DataPreprocessor):
    def __init__(self, 
                resolution,
                center_crop,
                random_flip,
                caption_column, 
                tokenizer, 
                is_train, 
                image_column,
                trigger_word
                ):
        self.trigger_word = trigger_word
        super().__init__(
            resolution,
            center_crop,
            random_flip,
            caption_column, 
            tokenizer, 
            is_train, 
            image_column,
        )
    
    def cat_tokenize_captions(self, examples):
        """
        TODO: support for multiple trigger words. 
        """
        captions = super().get_caption_column(examples)
        for idx, text in enumerate(captions):
            captions[idx] = text.replace(f"{self.trigger_word}, ", "")
        inputs = self.tokenizer(
            captions, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids   
    
    def preprocess_train(self, examples):
        """
        Preprocessing the datasets. adds input_ids_no_trigger to the examples.
        """
        examples = super().preprocess_train(examples)
        examples["input_ids_no_trigger"] = self.cat_tokenize_captions(examples)
        return examples
    
    def collate_fn(self, examples):
        """
        Collate function for the dataloader. Adds input_ids_no_trigger to the batch.
        """
        batch = super().collate_fn(examples)
        batch["input_ids_no_trigger"] = torch.stack([example["input_ids_no_trigger"] for example in examples])
        return batch

   
class FirstReplaceCatPreprocessor(DataPreprocessor):
    def __init__(self, 
                resolution,
                center_crop,
                random_flip,
                caption_column, 
                tokenizer, 
                is_train, 
                image_column,
                ):
        """
        This is the class that replaces the first word in the caption with the trigger word.
        """
        print("WARN: This class replaces the first word in the caption with the trigger word. This may result regular captions to be meaningless. Use with caution.")
        super().__init__(
            resolution,
            center_crop,
            random_flip,
            caption_column, 
            tokenizer, 
            is_train, 
            image_column,
        )
        
    
    def cat_tokenize_captions(self, examples):
        captions = super().get_caption_column(examples)
        for idx, text in enumerate(captions):
            if "," not in text:
                raise ValueError("The caption should contain a comma.")
            captions[idx] = ', '.join(text.split(',')[1:]) # WARN : This assumes that the first word is the trigger word. There can be case that does not follow this pattern.
        inputs = self.tokenizer(
            captions, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids
    
    def preprocess_train(self, examples):
        examples = super().preprocess_train(examples)
        examples["input_ids_no_trigger"] = self.cat_tokenize_captions(examples)
        return examples
    
    def collate_fn(self, examples):
        batch = super().collate_fn(examples)
        batch["input_ids_no_trigger"] = torch.stack([example["input_ids_no_trigger"] for example in examples])
        return batch
