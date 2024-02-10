import accelerate
import torch
import os
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
import math
from tqdm import tqdm
import argparse
import json
from utils.train_utils import cat_import_model, get_optimizer
from utils.data_utils import import_data, cat_data_preprocessor
from utils.train_loops import setting_steps, get_prediction_type, snr_loss, save_and_validate


def main(args):
    #creating accelerator instance
    accelerator = accelerate.Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_config=args.accelerator_project_config
    ) 
    
    #accelerator seed setting
    set_seed(args.seed)

    #setting weight dtype
    if args.weight_dtype == "torch.float16":
        weight_dtype = torch.float16
    elif args.weight_dtype == "torch.float32":
        weight_dtype = torch.float32

    #getting diffuser model
    noise_scheduler, tokenizer, text_encoder, vae, unet, lora_unet, layers_to_train = cat_import_model(args.pretrained_model_name_or_path,  
                                                                                        args.revision, 
                                                                                        args.variant, 
                                                                                        args.non_ema_revision,
                                                                                        accelerator,
                                                                                        weight_dtype,
                                                                                        args.rank)
    #setting scalering factor
    if args.scale_lr:
            learning_rate = (
                learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
    )

    optimizer = get_optimizer(
            args.learning_rate,
            args.adam_beta1, 
            args.adam_beta2,
            args.adam_weight_decay,
            args.adam_epsilon,
            layers_to_train=layers_to_train,
    )

    #getting dataset 
    dataset, image_column, caption_column = import_data(
            args.dataset_name,
            args.dataset_config_name,
            args.train_data_dir,
            args.cache_dir,
            args.image_column,
            args.caption_column
    )

    #preprocessing dataset and creating dataloader
    preprocessor = cat_data_preprocessor(
            args.resolution,
            args.center_crop,
            args.random_flip,
            caption_column,
            tokenizer,
            True,
            image_column,
            args.trigger_word
    )

    train_dataset, train_dataloader = preprocessor.get_data(
        accelerator,
        args.max_train_samples,
        dataset,
        args.seed,
        preprocessor.preprocess_train,
        args.train_batch_size,
        args.dataloader_num_workers,
        preprocessor.collate_fn
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    
    # Prepare everything with our `accelerator`.
    layers_to_train, optimizer, train_dataloader, lr_scheduler, args.validation_prompt, unet = accelerator.prepare(
        layers_to_train, optimizer, train_dataloader, lr_scheduler, args.validation_prompt, unet
    )

    #creating model repository
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
            args.max_train_steps = num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    
    #applying multiprocessing in batch
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    #train
    if accelerator.is_local_main_process:
        print(f"***** Running training *****")
        print(f"  Num examples = {len(train_dataset)}")
        print(f"  Num Epochs = {num_train_epochs}")
        print(f"  Instantaneous batch size per device = {args.train_batch_size}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        print(f"  Total optimization steps = {args.max_train_steps}")
    
    #setting steps with resuming or without resuming
    #initiates resume_from_checkpoint = ./results/checkpoint-1000
    initial_global_step, first_epoch = setting_steps(
        args.resume_from_checkpoint,
        accelerator,
        args.output_dir,
        num_update_steps_per_epoch
    )
    #total step and checkpointing steps
    total_step = 0 
    if args.checkpointing_steps is None:
        checkpointing_steps = args.max_train_steps
    else: 
        checkpointing_steps = args.checkpointing_steps
    #setting progress bar
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, num_train_epochs):
        lora_unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(lora_unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                encoder_hidden_states_no_trigger = text_encoder(batch["input_ids_no_trigger"])[0]

                # Get the target for loss depending on the prediction type
                #initiate prediction_type = "epsilon" for epsilon prediction and "v_prediction" for velocity prediction
                target = get_prediction_type(args.prediction_type, noise_scheduler, noise, latents, timesteps)

                # Predict the noise residual and compute loss
                model_pred = lora_unet(noisy_latents, timesteps, encoder_hidden_states).sample
                #cat application
                model_pred_no_trigger = lora_unet(noisy_latents, timesteps, encoder_hidden_states_no_trigger).sample
                base_pred = unet(noisy_latents, timesteps, encoder_hidden_states_no_trigger).sample
                

                #apply snr_gamma_loss
                model_loss = snr_loss(args.snr_gamma, model_pred, target, noise_scheduler, timesteps)
                cat_loss = snr_loss(args.snr_gamma, model_pred_no_trigger, base_pred, noise_scheduler, timesteps)
                #cat application
                loss = model_loss + args.cat_factor * cat_loss

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                total_step = save_and_validate(
                    accelerator,
                    progress_bar,
                    total_step,
                    train_loss,
                    checkpointing_steps,
                    args.checkpoints_total_limit,
                    args.output_dir,
                    lora_unet,
                    args.validation_prompt,
                    args.num_validation_images,
                    args.pretrained_model_name_or_path,
                    weight_dtype,
                    args.seed,
                    loss,
                    lr_scheduler
                )

            if total_step >= args.max_train_steps:
                accelerator.end_training()
                break
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tuning_config_path', type=str, default='./configs/tuning_config.json', help = "tuning config path")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help = "")
    parser.add_argument('--mixed_precision', type=str, default=None, help="")
    parser.add_argument('--report_to', type=str, default=None, help="")
    parser.add_argument('--accelerator_project_config', type=str, default=None, help = "tuning config path")
    parser.add_argument('--seed', type=int, default=42, help = "")
    parser.add_argument('--output_dir', type=str, default="./lora_trained", help="")
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='runwayml/stable-diffusion-v1-5', help="")
    parser.add_argument('--revision', type=str, default=None, help = "tuning config path")
    parser.add_argument('--variant', type=str, default=None, help = "")
    parser.add_argument('--rank', type=int, default=1, help="")
    parser.add_argument('--non_ema_revision', type=str, default=None, help="")
    parser.add_argument('--gradient_checkpointing', type=bool, default=True, help = "tuning config path")
    parser.add_argument('--scale_lr', type=int, default=None, help = "")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="")
    parser.add_argument('--train_batch_size', type=int, default=1, help="")
    parser.add_argument('--adam_beta1', type=float, default=0.9, help="")
    parser.add_argument('--adam_beta2', type=float, default=0.999, help = "tuning config path")
    parser.add_argument('--adam_weight_decay', type=float, default=1e-2, help = "")
    parser.add_argument('--adam_epsilon', type=float, default=1e-08, help="")
    parser.add_argument('--dataset_name', type=str, default="lambdalabs/pokemon-blip-captions", help="")
    parser.add_argument('--train_data_dir', type=str, default=None, help = "tuning config path")
    parser.add_argument('--dataset_config_name', type=str, default=None, help = "")
    parser.add_argument('--cache_dir', type=str, default=None, help="")
    parser.add_argument('--image_column', type=str, default="image", help="")
    parser.add_argument('--caption_column', type=str, default="text", help="")
    parser.add_argument('--resolution', type=int, default=512, help = "tuning config path")
    parser.add_argument('--center_crop', type=bool, default=False, help = "")
    parser.add_argument('--random_flip', type=bool, default=True, help="")
    parser.add_argument('--max_train_samples', type=str, default=None, help="")
    parser.add_argument('--dataloader_num_workers', type=int, default=0, help = "tuning config path")
    parser.add_argument('--max_train_steps', type=int, default=10, help = "")
    parser.add_argument('--num_train_epochs', type=int, default=None, help="")
    parser.add_argument('--lr_warmup_steps', type=int, default=0, help="")
    parser.add_argument('--lr_scheduler', type=str, default="constant", help="")
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help = "tuning config path")
    parser.add_argument('--noise_offset', type=str, default=None, help = "")
    parser.add_argument('--prediction_type', type=str, default=None, help="")
    parser.add_argument('--snr_gamma', type=int, default=None, help = "tuning config path")
    parser.add_argument('--cat_factor', type=int, default=1, help = "")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help = "")
    parser.add_argument('--checkpointing_steps', type=int, default=1, help="")
    parser.add_argument('--checkpoints_total_limit', type=str, default=1, help="")
    parser.add_argument('--trigger_word', type=str, default="pokemon", help = "tuning config path")
    parser.add_argument('--validation_prompt', type=str, default=["A pokemon with blue eyes"], help = "tuning config path")
    parser.add_argument('--weight_dtype', type=str, default=torch.float32, help="")
    parser.add_argument('--num_validation_images', type=int, default=1, help="")
    args = parser.parse_args()
    with open(args.tuning_config_path) as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)
    main(args)