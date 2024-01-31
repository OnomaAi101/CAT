import torch 
import math
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from transformers import CLIPTokenizer, CLIPTextModel

def import_model(pretrained_model_name_or_path,
                revision, 
                variant, 
                non_ema_revision, 
                accelerator,
                weight_dtype,
                lora_rank,
                ):
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
    # Move model to gpu and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    #freeze all the parameters except the adapter
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    if lora_rank is not None:
        unet, layers_to_train = get_lora(unet, lora_rank)

    return noise_scheduler, tokenizer, text_encoder, vae, unet, layers_to_train

def get_lora(unet, rank):
    #load lora attn processors
    #this will be changed regrading the type of adapters
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

    layers_to_train = AttnProcsLayers(unet.attn_processors)

    return unet, layers_to_train

def get_optimizer(learning_rate, 
                adam_beta1,
                adam_beta2,
                adam_weight_decay,
                adam_epsilon,
                layers_to_train):
    
    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
            layers_to_train.parameters(),
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            weight_decay=adam_weight_decay,
            eps=adam_epsilon
    )

    return optimizer

def get_scheduler(accelerator, train_dataloader, gradient_accumulation_steps, num_train_epochs, max_train_steps, optimizer, lr_scheduler, lr_warmup_steps):
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

    return lr_scheduler