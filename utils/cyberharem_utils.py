import os
from typing import List
import random
import zipfile
import jsonlines
from huggingface_hub import HfApi
from huggingface_hub.hf_api import DatasetInfo

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', ".webp"]
NSFW_CAPTIONS = ["nsfw", "sex", "nude"]

def count_images(folder_path: str, filter_nsfw:bool=False) -> int:
    """
    Count the number of files in the folder
        :param folder_path: folder path
        :return: number of files in the folder
    """
    return len(name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name)) and name.lower().endswith(tuple(IMAGE_EXTENSIONS)) and (not filter_nsfw or not filter_nsfw_from_caption(os.path.join(folder_path, name))))

def sample_folders(folder_path: str, sample_count:int = 10, filter_nsfw:bool=False) -> List[str]:
    """
    Count the number of files in the folder
        :param folder_path: folder path
        :return: number of files in the folder
    """
    folders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    # get folders which contain at least 1 image
    folders = [folder for folder in folders if count_images(os.path.join(folder_path, folder), filter_nsfw) > 0]
    if len(folders) < sample_count:
        return folders
    return random.sample(folders, sample_count)

def sample_files_from_folder(folder_path: str, filter_nsfw:bool=False, target_count:int= 10) -> str:
    """
    Sample a file from the folder
        :param folder_path: folder path
        :return: file path (absolute path)
    """
    files = [name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name)) and name.lower().endswith(tuple(IMAGE_EXTENSIONS)) and (not filter_nsfw or not filter_nsfw_from_caption(os.path.join(folder_path, name)))]
    files = [os.path.join(folder_path, file) for file in files]
    files = [os.path.abspath(file) for file in files]
    if len(files) < target_count:
        if len(files) == 0:
            raise FileNotFoundError(f"No image found in the folder: {folder_path}")
        return files
    return random.sample(files, target_count)

def filter_nsfw_from_caption(image_path:str) -> bool:
    """
    Filter NSFW from caption
        :param caption: caption
        :return: True if caption contains NSFW, False otherwise
    """
    # change extension to txt
    caption_path = os.path.splitext(image_path)[0] + ".txt"
    if not os.path.exists(caption_path):
        raise FileNotFoundError(f"Caption file not found: {caption_path}")
    with open(caption_path, "r", encoding="utf-8") as file:
        caption = file.read()
    return any(nsfw_caption in caption.lower() for nsfw_caption in NSFW_CAPTIONS)

def get_dataset_list(author:str) -> List[str]:
    """
    Get dataset list from Hugging Face Hub
        :param user_name: user name
        :return: dataset list
    """
    api = HfApi()
    datasets :List[DatasetInfo]= api.list_datasets(author=author)
    return [dataset.id for dataset in datasets]

def download_datasets(datasets:List[str], output_dir:str, target_file = "dataset-1200.zip") -> List[str]:
    """
    Download datasets from Hugging Face Hub
        :param datasets: dataset list
        :param output_dir: output directory
    """
    for dataset in datasets:
        api = HfApi()
        dataset_info = api.dataset_info(dataset)
        dataset_name = dataset_info.id
        dataset_path = os.path.join(output_dir, dataset_name)
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path, exist_ok=True)
        # download specific file
        try:
            api.hf_hub_download(dataset, target_file, local_dir=dataset_path, local_dir_use_symlinks=False, repo_type="dataset")
            print(f"Downloaded {dataset_name} to {dataset_path}")
        except Exception as e:
            print(f"Error downloading {dataset_name}: {e}")
    # extract zip files
    unzipped_path= []
    for dataset in datasets:
        dataset_info = api.dataset_info(dataset)
        dataset_name = dataset_info.id
        dataset_path = os.path.join(output_dir, dataset_name)
        # extract zip file
        zip_file = os.path.join(dataset_path, target_file)
        if os.path.exists(zip_file):
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(dataset_path)
            os.remove(zip_file)
            print(f"Extracted {zip_file}")
            unzipped_path.append(dataset_path)
        else:
            print(f"Zip file not found: {zip_file}")
    return unzipped_path

def prepare_samples(output_dir:str, author:str = "CyberHarem", sample_count:int = 10, filter_nsfw:bool=False, image_sample_count:int = 10) -> dict:
    """
    Sample and download datasets from Hugging Face Hub
        :param output_dir: output directory
        :param author: author
        :param sample_count: sample count
        :param filter_nsfw: filter NSFW
        :param image_sample_count: image sample count
        :return Dictionary of images as {character_name: [image_paths]}
    """
    datasets = get_dataset_list(author)
    sampled_datasets = random.sample(datasets, sample_count)
    dataset_folders = download_datasets(sampled_datasets, output_dir)
    images = {}
    for dataset_folder in dataset_folders:
        character_name = os.path.basename(dataset_folder)
        character_images = sample_files_from_folder(dataset_folder, filter_nsfw, image_sample_count)
        images[character_name] = character_images
    return images

def prepare_metadata_and_samples(output_dir:str, author:str = "CyberHarem", sample_count:int = 10, filter_nsfw:bool=False, image_sample_count:int = 10) -> dict:
    """
    Sample and download datasets from Hugging Face Hub
        :param output_dir: output directory
        :param author: author
        :param sample_count: sample count
        :param filter_nsfw: filter NSFW
        :param image_sample_count: image sample count
        :return metadata as list of dictionary [{file_name: image_path, text: caption}]
    """
    datasets = get_dataset_list(author)
    sampled_datasets = random.sample(datasets, sample_count)
    dataset_folders = download_datasets(sampled_datasets, output_dir)
    images = {}
    metadata = []
    for dataset_folder in dataset_folders:
        character_name = os.path.basename(dataset_folder)
        character_images = sample_files_from_folder(dataset_folder, filter_nsfw, image_sample_count)
        images[character_name] = character_images
        for image in character_images:
            caption_path = os.path.splitext(image)[0] + ".txt"
            if not os.path.exists(caption_path):
                raise FileNotFoundError(f"Caption file not found: {caption_path}")
            with open(caption_path, "r", encoding="utf-8") as file:
                caption = file.read()
            caption = character_name + ", " + caption
            metadata.append({"file_name": image, "text": caption})
    # save jsonl to output_dir/metadata.jsonl
    metadata_path = os.path.join(output_dir, "metadata.jsonl")
    with jsonlines.open(metadata_path, mode='w', encoding="utf-8") as writer:
        writer.write_all(metadata)
    print(f"Saved metadata to {metadata_path}")
    return metadata

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Prepare samples from Hugging Face Hub')
    parser.add_argument('--output_dir', type=str, default="samples", help='Output directory')
    parser.add_argument('--author', type=str, default="CyberHarem", help='Author')
    parser.add_argument('--sample_count', type=int, default=10, help='Sample count')
    parser.add_argument('--filter_nsfw', type=bool, default=False, help='Filter NSFW')
    parser.add_argument('--image_sample_count', type=int, default=10, help='Image sample count')
    args = parser.parse_args()
    result = prepare_metadata_and_samples(args.output_dir, args.author, args.sample_count, args.filter_nsfw, args.image_sample_count)
    print(result)
