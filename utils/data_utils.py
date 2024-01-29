import os 
import torch
import random
import numpy as np
from datasets import load_dataset
from torchvision import transforms

DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}

def import_from_hub(dataset_name, dataset_config_name, train_data_dir, cache_dir, image_column, caption_column):
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

    dataset_columns = DATASET_NAME_MAPPING.get(dataset_name, None)

    if image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{caption_column}' needs to be one of: {', '.join(column_names)}"
    )
    return dataset, image_column, caption_column

#tag tokenizing
def tokenize_captions(examples, caption_column, tokenizer, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

def preprocess_train(examples, resolution, center_crop, random_flip, image_column):
        # Preprocessing the datasets.
        train_transforms = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

def collate_fn_hub(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

def get_dataloader(accelerator, max_train_samples, dataset, seed, collate_fn, train_batch_size, dataloader_num_workers):
    with accelerator.main_process_first():
        if max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=seed).select(range(max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)
     #dataloader 
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=train_batch_size,
        num_workers=dataloader_num_workers
    )
    return train_dataloader