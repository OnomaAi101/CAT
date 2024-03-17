import torch 
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

#define the device
device = "cuda" if torch.cuda.is_available() else "cpu"
#Load the CLIP model and processor
model_name = "openai/clip-vit-base-patch32"
PROCESSOR_MODEL = None
CLIP_MODEL = None

def dynamic_load_model(model_name: str, device: str):
    """
    Load the CLIP model and processor
    """
    global PROCESSOR_MODEL, CLIP_MODEL
    if PROCESSOR_MODEL is not None and CLIP_MODEL is not None:
        return PROCESSOR_MODEL, CLIP_MODEL
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    clip_model = CLIPModel.from_pretrained(model_name).to(device)
    PROCESSOR_MODEL = clip_processor
    CLIP_MODEL = clip_model
    return clip_processor, clip_model

def unload_model():
    """
    Unload the CLIP model and processor. This may be called manually to free up memory.
    """
    global PROCESSOR_MODEL, CLIP_MODEL
    PROCESSOR_MODEL = None
    CLIP_MODEL = None
    return True

#cosine similarity
cos = torch.nn.CosineSimilarity(dim=0)

def image_similarity(image1, image2):
    """
    Using CLIPImageProcessor to get the image features and calculate the similarity between two images.
    """
    dynamic_load_model(model_name, device)
    #getting clip features
    image1_preprocess = PROCESSOR_MODEL(images=Image.open(image1), return_tensors="pt").to(device)
    with torch.no_grad():
        image1_features = CLIP_MODEL.get_image_features(**image1_preprocess).to(device)
    image2_preprocess = PROCESSOR_MODEL(images=Image.open(image2), return_tensors="pt").to(device)
    with torch.no_grad():
        image2_features = CLIP_MODEL.get_image_features(**image2_preprocess).to(device)
    #calculate similarity
    similarity = cos(image1_features[0], image2_features[0]).item()
    similarity = (similarity + 1) / 2

    return similarity

def prompt_similarity(prompt, image_path):
    """
    Using CLIPProcessor to get the text and image features and calculate the similarity between the prompt and the image.
    """
    dynamic_load_model(model_name, device)
    #Tokenize the prompt
    inputs = PROCESSOR_MODEL(prompt, return_tensors="pt", padding=True).to(device)
    # Get the text features from the model
    with torch.no_grad():
        prompt_features = CLIP_MODEL.get_text_features(**inputs).to(device)
    # Load and preprocess the image
    image = Image.open(image_path)
    inputs = PROCESSOR_MODEL(images=image, return_tensors="pt").to(device)
    # Get the image features from the model
    with torch.no_grad():
        image_features = CLIP_MODEL.get_image_features(**inputs).to(device)
    # Calculate similarity
    similarity = cos(prompt_features[0], image_features[0]).item()
    return similarity
