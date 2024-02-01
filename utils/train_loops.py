import os 
from tqdm import tqdm
import torch
import torch.nn.functional as F
import shutil
from diffusers import AutoPipelineForText2Image
from diffusers.training_utils import compute_snr

def setting_steps(resume_from_checkpoint,
                accelerator,
                output_dir,
                num_update_steps_per_epoch):
    global_step = 0
    first_epoch = 0

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
            total_step = int(path.split("-")[1])

            initial_global_step = total_step
            first_epoch = total_step // num_update_steps_per_epoch
        return initial_global_step, first_epoch
    else:
        initial_global_step = 0
        return initial_global_step, first_epoch

def get_prediction_type(prediction_type,
                        noise_scheduler,
                        noise,
                        latents,
                        timesteps):
    if prediction_type is not None:
        # set prediction_type of scheduler if defined
        noise_scheduler.register_to_config(prediction_type = prediction_type)
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
    return target

def snr_loss(snr_gamma,
    model_pred,
    target,
    noise_scheduler,
    timesteps):
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
    return loss

def save_and_validate(accelerator, 
                    progress_bar,
                    total_step,
                    train_loss,
                    checkpointing_steps,
                    checkpoints_total_limit,
                    output_dir,
                    unet,
                    validation_prompt,
                    num_validation_images,
                    pretrained_model_name_or_path,
                    weight_dtype,
                    seed,
                    loss,
                    lr_scheduler,
                    ):
    # Checks if the accelerator has performed an optimization step behind the scenes
    if accelerator.sync_gradients:
        progress_bar.update(1)
        total_step += 1
        accelerator.log({"train_loss": train_loss}, step=total_step)
        train_loss = 0.0

        if total_step % checkpointing_steps == 0:
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

                    save_path = os.path.join(output_dir, f"checkpoint-{total_step}")
                    unet = unet.to(torch.float32)
                    unet.save_attn_procs(save_path)
                    print(f"Saved state to {save_path}")

                for prompt in tqdm(validation_prompt, desc="Validation"):
                    print(f"Running validation... \n Generating {num_validation_images} images with prompt:")
                    print(f" {prompt}.")
                    # create pipeline
                    pipeline = AutoPipelineForText2Image.from_pretrained(pretrained_model_name_or_path, 
                                                                            torch_dtype=weight_dtype)
                    pipeline.load_lora_weights(save_path, weight_name="pytorch_lora_weights.safetensors")
                    pipeline = pipeline.to(accelerator.device)

                    # run inference
                    generator = torch.Generator(device=accelerator.device)
                    if seed is not None:
                        generator = generator.manual_seed(seed)
                    images = []
                    for _ in range(num_validation_images):
                        images.append(
                            pipeline(prompt, num_inference_steps=50, generator=generator).images[0]
                        )

                    # save images
                    for i, image in enumerate(images):
                        image.save(os.path.join(save_path, f"{prompt}_{i}.png"))

                    del pipeline
                    torch.cuda.empty_cache()

                    logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                    progress_bar.set_postfix(**logs)

    return total_step