import os 
import torch
import torch.nn.functional as F
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