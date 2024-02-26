import torch 
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

#define the device
device = "cuda" if torch.cuda.is_available() else "cpu"
#Load the CLIP model and processor
model_name = "openai/clip-vit-base-patch32"
clip_processor = CLIPProcessor.from_pretrained(model_name)
clip_model = CLIPModel.from_pretrained(model_name).to(device)
#cosine similarity
cos = torch.nn.CosineSimilarity(dim=0)

def image_similarity(image1, image2):
    #getting clip features
    image1_preprocess = clip_processor(images=Image.open(image1), return_tensors="pt").to(device)
    with torch.no_grad():
        image1_features = clip_model.get_image_features(**image1_preprocess).to(device)

    image2_preprocess = clip_processor(images=Image.open(image2), return_tensors="pt").to(device)
    with torch.no_grad():
        image2_features = clip_model.get_image_features(**image2_preprocess).to(device)
    #calculate similarity
    similarity = cos(image1_features[0], image2_features[0]).item()
    similarity = (similarity + 1) / 2

    return similarity

def prompt_similarity(prompt, image_path):
    #Tokenize the prompt
    inputs = clip_processor(prompt, return_tensors="pt", padding=True).to(device)

    # Get the text features from the model
    with torch.no_grad():
        prompt_features = clip_model.get_text_features(**inputs).to(device)

    # Load and preprocess the image
    image = Image.open(image_path)
    inputs = clip_processor(images=image, return_tensors="pt").to(device)

    # Get the image features from the model
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs).to(device)

    # Calculate similarity
    similarity = cos(prompt_features[0], image_features[0]).item()

    return similarity