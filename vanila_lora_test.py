from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("/data7/OnomaAi101/CAT/results/cat_statue_20240201_new_prompts/checkpoint-500", weight_name="pytorch_lora_weights.safetensors")
image = pipeline("a cat statue on a green field").images[0]
image.save("/data7/OnomaAi101/CAT/results/cat_statue_20240201_new_prompts/checkpoint-500/no lora_a cat statue on a green field_0.png")