from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("/home/paneah/Desktop/lora_training/lora_trained/", weight_name="pytorch_lora_weights.safetensors")
image = pipeline("A pokemon with blue eyes").images[0]
image.save("./results/pokemon.png")