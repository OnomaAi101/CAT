from diffusers import DiffusionPipeline

model_path = "/data7/OnomaAi101/CAT/results/dreambooth/cat_statue_20240207_new/"  # Adjust this to your model's path
pipeline = DiffusionPipeline.from_pretrained(model_path, use_auth_token=True).to("cuda")
image = pipeline("cat statue, <shs>, on a green field").images[0]
image.save("/data7/OnomaAi101/CAT/results/dreambooth/cat_statue_20240207_new/dreambooth_<shs>, on a green field_0.png")