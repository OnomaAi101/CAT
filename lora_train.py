import accelerate
import torch
import torch.nn.functional as F
import os
from accelerate.utils import set_seed
from packaging import version
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel, DiffusionPipeline
from diffusers.utils import check_min_version
from diffusers.loaders import AttnProcsLayers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.models.attention_processor import LoRAAttnProcessor
from datasets import load_dataset
from torchvision import transforms
import numpy as np
import random
import math
from tqdm import tqdm
import shutil
import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
#check_min_version("0.26.0.dev0")

#args
DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}
gradient_accumulation_steps = 1
mixed_precision = None
report_to = None
accelerator_project_config = None
seed = 42
output_dir = "./lora_trained"
pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
revision = None
variant = None
rank = 1
non_ema_revision = None
gradient_checkpointing = True
scale_lr = None
learning_rate = 1e-4
train_batch_size = 1
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_weight_decay = 1e-2
adam_epsilon = 1e-08
dataset_name = "lambdalabs/pokemon-blip-captions"
train_data_dir = None
dataset_config_name = None
cache_dir = None
image_column = "image"
caption_column = "text"
resolution = 512
center_crop = False
random_flip = True
max_train_samples = None
dataloader_num_workers = 0
max_train_steps = 10
num_train_epochs = None
lr_warmup_steps = 0
lr_scheduler = "constant"
resume_from_checkpoint = None
noise_offset = None
input_perturbation = 0
prediction_type = None
snr_gamma = None
max_grad_norm = 1.0
checkpointing_steps = max_train_steps
checkpoints_total_limit = None
validation_prompt = "A pokemon with blue eyes"
validation_epochs = 5
weight_dtype = torch.float32
num_validation_images = 3

#creating accelerator instance
accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=report_to,
        project_config=accelerator_project_config
    )
#setting seed
set_seed(seed)

#creating model repository
if accelerator.is_main_process:
    os.makedirs(output_dir, exist_ok=True)

#load scheduler 
noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer", revision=revision)
#load vae, text encoder and unet model
text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder", revision=revision, variant=variant
        )
vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae", revision=revision, variant=variant
        )

unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet", revision=non_ema_revision
)
#freeze vae and text encoder and set to model to trainable mode 

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.requires_grad_(False)

# Set correct lora layers
lora_attn_procs = {}
for name in unet.attn_processors.keys():
    cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
    if name.startswith("mid_block"):
        hidden_size = unet.config.block_out_channels[-1]
    elif name.startswith("up_blocks"):
        block_id = int(name[len("up_blocks.")])
        hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
    elif name.startswith("down_blocks"):
        block_id = int(name[len("down_blocks.")])
        hidden_size = unet.config.block_out_channels[block_id]

    lora_attn_procs[name] = LoRAAttnProcessor(
        hidden_size=hidden_size,
        cross_attention_dim=cross_attention_dim,
        rank=rank
    )

unet.set_attn_processor(lora_attn_procs)

lora_layers = AttnProcsLayers(unet.attn_processors)

if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
)

optimizer_cls = torch.optim.AdamW

optimizer = optimizer_cls(
        lora_layers.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon
)

#getting dataset 

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

# Preprocessing the datasets.
# We need to tokenize input captions and transform the images.
def tokenize_captions(examples, is_train=True):
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

def preprocess_train(examples):
    images = [image.convert("RGB") for image in examples[image_column]]
    examples["pixel_values"] = [train_transforms(image) for image in images]
    examples["input_ids"] = tokenize_captions(examples)
    return examples

with accelerator.main_process_first():
    if max_train_samples is not None:
        dataset["train"] = dataset["train"].shuffle(seed=seed).select(range(max_train_samples))
    # Set the training transforms
    train_dataset = dataset["train"].with_transform(preprocess_train)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}

#dataloader 
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=collate_fn,
    batch_size=train_batch_size,
    num_workers=dataloader_num_workers
)

# Scheduler and math around the number of training steps.
overrode_max_train_steps = False
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
if max_train_steps is None:
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    overrode_max_train_steps = True

lr_scheduler = get_scheduler(
    lr_scheduler,
    optimizer=optimizer,
    num_warmup_steps=lr_warmup_steps * accelerator.num_processes,
    num_training_steps=max_train_steps * accelerator.num_processes,
)

# Prepare everything with our `accelerator`.
unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    unet, optimizer, train_dataloader, lr_scheduler
)

#datatype setting
if accelerator.mixed_precision == "fp16":
    weight_dtype = torch.float16
    mixed_precision = accelerator.mixed_precision
elif accelerator.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16
    mixed_precision = accelerator.mixed_precision

# Move text_encode and vae to gpu and cast to weight_dtype
text_encoder.to(accelerator.device, dtype=weight_dtype)
vae.to(accelerator.device, dtype=weight_dtype)

# We need to recalculate our total training steps as the size of the training dataloader may have changed.
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
if overrode_max_train_steps:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
# Afterwards we recalculate our number of training epochs
num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)


 # Function for unwrapping if model was compiled with `torch.compile`.
def unwrap_model(model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

# Train!
total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

print(f"***** Running training *****")
print(f"  Num examples = {len(train_dataset)}")
print(f"  Num Epochs = {num_train_epochs}")
print(f"  Instantaneous batch size per device = {train_batch_size}")
print(
    f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
)
print(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
print(f"  Total optimization steps = {max_train_steps}")

global_step = 0
first_epoch = 0

# Potentially load in the weights and states from a previous save
if resume_from_checkpoint:
    if resume_from_checkpoint != "latest":
        path = os.path.basename(resume_from_checkpoint)
    else:
        # Get the most recent checkpoint
        dirs = os.listdir(output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None

    if path is None:
        accelerator.print(
            f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run."
        )
        resume_from_checkpoint = None
        initial_global_step = 0
    else:
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(output_dir, path))
        global_step = int(path.split("-")[1])

        initial_global_step = global_step
        first_epoch = global_step // num_update_steps_per_epoch

else:
    initial_global_step = 0

progress_bar = tqdm(
    range(0, max_train_steps),
    initial=initial_global_step,
    desc="Steps",
    # Only show the progress bar once on each machine.
    disable=not accelerator.is_local_main_process,
)

for epoch in range(first_epoch, num_train_epochs):
    train_loss = 0.0
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(unet):
            # Convert images to latent space
            latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            if noise_offset:
                # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                noise += noise_offset * torch.randn(
                    (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                )
            if input_perturbation:
                new_noise = noise + input_perturbation * torch.randn_like(noise)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            if input_perturbation:
                noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
            else:
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

            # Get the target for loss depending on the prediction type
            if prediction_type is not None:
                # set prediction_type of scheduler if defined
                noise_scheduler.register_to_config(prediction_type=prediction_type)

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Predict the noise residual and compute loss
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

            if snr_gamma is None:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            else:
                # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                # This is discussed in Section 4.2 of the same paper.
                snr = compute_snr(noise_scheduler, timesteps)
                if noise_scheduler.config.prediction_type == "v_prediction":
                    # Velocity objective requires that we add one to SNR values before we divide by them.
                    snr = snr + 1
                mse_loss_weights = (
                    torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                )

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
            train_loss += avg_loss.item() / gradient_accumulation_steps

            # Backpropagate
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if checkpoints_total_limit is not None:
                            checkpoints = os.listdir(output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                print(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                print(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        print(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

        if accelerator.is_main_process:
            print(f"Running validation... \n Generating {num_validation_images} images with prompt:")
            print(f" {validation_prompt}.")
            # create pipeline
            pipeline = DiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet),
                revision=revision,
                torch_dtype=weight_dtype,
            )
            pipeline = pipeline.to(accelerator.device)
            pipeline.set_progress_bar_config(disable=True)

            # run inference
            generator = torch.Generator(device=accelerator.device)
            if seed is not None:
                generator = generator.manual_seed(seed)
            images = []
            for _ in range(num_validation_images):
                images.append(
                    pipeline(validation_prompt, num_inference_steps=30, generator=generator).images[0]
                )

            del pipeline
            torch.cuda.empty_cache()

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        unet.save_attn_procs(output_dir)

    # Final inference
    # Load previous pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path, revision=revision, torch_dtype=weight_dtype
    )
    pipeline = pipeline.to(accelerator.device)

    # load attention processors
    pipeline.unet.load_attn_procs(output_dir)

    # run inference
    generator = torch.Generator(device=accelerator.device)
    if seed is not None:
        generator = generator.manual_seed(seed)
    images = []
    for _ in range(num_validation_images):
        images.append(pipeline(validation_prompt, num_inference_steps=30, generator=generator).images[0])


    accelerator.end_training()