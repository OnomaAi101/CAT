import os 
import json
import importlib
import torch
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_processes", default=1, type=int, help="Number of processes to use for conversion.")
    parser.add_argument("--kohya_path", default="/data3/lora_dev/kohya_ss", type=str, help="Path to the checkpoint to convert.")
    parser.add_argument("--enable_bucket", action="store_true", help="Enable bucket.")
    parser.add_argument("--min_bucket_reso", default=256, type=int, help="Minimum bucket resolution.")
    parser.add_argument("--max_bucket_reso", default=2048, type=int, help="Maximum bucket resolution.")
    parser.add_argument("--pretrained_model_name_or_path", default="runwayml/stable-diffusion-v1-5", type=str, help="The pretrained model name or path.")
    parser.add_argument("--train_data_dir", default="/data7/OnomaAi101/CAT/data/textual_inversion/cat_statue", type=str, help="The training data directory.")
    parser.add_argument("--reg_data_dir", default="/data7/OnomaAi101/CAT/data/textual_inversion/cat_statue", type=str, help="The regularizing data directory.")
    parser.add_argument("--train_repeat", default=1, type=int, help="The train repeat.")
    parser.add_argument("--reg_repeat", default=1, type=int, help="The regulation repeat.")
    parser.add_argument("--resolution", default="512,512", type=str, help="The resolution.")
    parser.add_argument("--output_dir", default="/data7/OnomaAi101/CAT/results/dreembooth/cat_statue_20240205", type=str, help="The output directory.")
    parser.add_argument("--logging_dir", default="/data7/OnomaAi101/CAT/results/dreembooth/cat_statue_20240205", type=str, help="The logging directory.")
    parser.add_argument("--save_model_as", default="safetensors", type=str, help="The output name.")
    parser.add_argument("--output_name", default="model", type=str, help="The output name.")
    parser.add_argument("--lr_scheduler_num_cycles", default=1, type=int, help="The number of cycles for the learning rate scheduler.")
    parser.add_argument("--max_data_loader_n_workers", default=0, type=int, help="The maximum number of data loader workers.")
    parser.add_argument("--learning_rate_te", default=1e-05, type=int, help="The learning rate for the text encoder.")
    parser.add_argument("--learning_rate", default=1e-05, type=int, help="The learning rate.")
    parser.add_argument("--lr_scheduler", default="cosine", type=str, help="The learning rate scheduler.")
    parser.add_argument("--lr_warmup_steps", default=0, type=int, help="The learning rate warmup steps.")
    parser.add_argument("--train_batch_size", default=1, type=int, help="The training batch size.")
    parser.add_argument("--max_train_steps", default=36, type=int, help="The maximum number of training steps.")
    parser.add_argument("--save_every_n_epochs", default=1, type=int, help="The number of epochs to save the model.")
    parser.add_argument("--mixed_precision", default="fp16", type=str, help="The mixed precision.")
    parser.add_argument("--save_precision", default="fp16", type=str, help="The save precision.")
    parser.add_argument("--cache_latents", action="store_true", help="Cache latents.")
    parser.add_argument("--optimizer_type", default="AdamW8bit", type=str, help="The optimizer type.")
    parser.add_argument("--xformers", action="store_true", help="Use xformers.")
    parser.add_argument("--bucekt_reso_steps", default=64, type=int, help="The bucket resolution steps.")
    parser.add_argument("--bucket_no_upscale", action="store_true", help="Bucket no upscale.")
    parser.add_argument("--noise_offset", default="0.0", type=str, help="The noise offset.")
    parser.add_argument("--caption_extension", default=".txt", type=str, help="The caption extension.")
    parser.add_argument("--tuning_config_path", default="/data7/OnomaAi101/CAT/configs/dreambooth_tuning_config.json", type=str, help="The tuning config path.")
    # !wget https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml
    parser.add_argument(
        "--original_config_file",
        default=None,
        type=str,
        help="The YAML config file corresponding to the original architecture.",
    )
    parser.add_argument(
        "--config_files",
        default=None,
        type=str,
        help="The YAML config file corresponding to the architecture.",
    )
    parser.add_argument(
        "--num_in_channels",
        default=None,
        type=int,
        help="The number of input channels. If `None` number of input channels will be automatically inferred.",
    )
    parser.add_argument(
        "--scheduler_type",
        default="pndm",
        type=str,
        help="Type of scheduler to use. Should be one of ['pndm', 'lms', 'ddim', 'euler', 'euler-ancestral', 'dpm']",
    )
    parser.add_argument(
        "--pipeline_type",
        default=None,
        type=str,
        help=(
            "The pipeline type. One of 'FrozenOpenCLIPEmbedder', 'FrozenCLIPEmbedder', 'PaintByExample'"
            ". If `None` pipeline will be automatically inferred."
        ),
    )
    parser.add_argument(
        "--image_size",
        default=None,
        type=int,
        help=(
            "The image size that the model was trained on. Use 512 for Stable Diffusion v1.X and Stable Siffusion v2"
            " Base. Use 768 for Stable Diffusion v2."
        ),
    )
    parser.add_argument(
        "--prediction_type",
        default=None,
        type=str,
        help=(
            "The prediction type that the model was trained on. Use 'epsilon' for Stable Diffusion v1.X and Stable"
            " Diffusion v2 Base. Use 'v_prediction' for Stable Diffusion v2."
        ),
    )
    parser.add_argument(
        "--extract_ema",
        action="store_true",
        help=(
            "Only relevant for checkpoints that have both EMA and non-EMA weights. Whether to extract the EMA weights"
            " or not. Defaults to `False`. Add `--extract_ema` to extract the EMA weights. EMA weights usually yield"
            " higher quality images for inference. Non-EMA weights are usually better to continue fine-tuning."
        ),
    )
    parser.add_argument(
        "--upcast_attention",
        action="store_true",
        help=(
            "Whether the attention computation should always be upcasted. This is necessary when running stable"
            " diffusion 2.1."
        ),
    )
    parser.add_argument(
        "--from_safetensors",
        action="store_true",
        help="If `--checkpoint_path` is in `safetensors` format, load checkpoint with safetensors instead of PyTorch.",
    )
    parser.add_argument(
        "--to_safetensors",
        action="store_true",
        help="Whether to store pipeline in safetensors format or not.",
    )
    parser.add_argument("--device", type=str, help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)")
    parser.add_argument(
        "--stable_unclip",
        type=str,
        default=None,
        required=False,
        help="Set if this is a stable unCLIP model. One of 'txt2img' or 'img2img'.",
    )
    parser.add_argument(
        "--stable_unclip_prior",
        type=str,
        default=None,
        required=False,
        help="Set if this is a stable unCLIP txt2img model. Selects which prior to use. If `--stable_unclip` is set to `txt2img`, the karlo prior (https://huggingface.co/kakaobrain/karlo-v1-alpha/tree/main/prior) is selected by default.",
    )
    parser.add_argument(
        "--clip_stats_path",
        type=str,
        help="Path to the clip stats file. Only required if the stable unclip model's config specifies `model.params.noise_aug_config.params.clip_stats_path`.",
        required=False,
    )
    parser.add_argument(
        "--controlnet", action="store_true", default=None, help="Set flag if this is a controlnet checkpoint."
    )
    parser.add_argument("--half", action="store_true", help="Save weights in half precision.")
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        required=False,
        help="Set to a path, hub id to an already converted vae to not convert it again.",
    )
    parser.add_argument(
        "--pipeline_class_name",
        type=str,
        default=None,
        required=False,
        help="Specify the pipeline class name",
    )
    args = parser.parse_args()
    with open(args.tuning_config_path) as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)
    if args.enable_bucket == True:
        bucket = f" --enable_bucket --min_bucket_reso={args.min_bucket_reso} --max_bucket_reso={args.max_bucket_reso} "
    else: 
        bucket = ""
    if args.cache_latents == True:
        cache_latents = " --cache_latents"
    else:
        cache_latents = ""
    if args.xformers == True:
        xformers = " --xformers"
    else:
        xformers = ""
    if args.bucket_no_upscale == True:
        bucket_no_upscale = " --bucket_no_upscale"
    else:
        bucket_no_upscale = ""
    data_class = args.train_data_dir.split("/")[-1]
    os.mkdir("data_temp")
    os.mkdir("data_temp/train")
    os.mkdir("data_temp/reg")
    os.system(f"ln -s {args.train_data_dir} ./data_temp/train" + f"{args.train_repeat}_{data_class}")
    os.system(f"ln -s {args.reg_data_dir} ./data_temp/reg" + f"{args.reg_repeat}_{data_class}")
    args.train_data_dir = "./data_temp/train"
    args.reg_data_dir = "./data_temp/reg"
    command = (f"""
            {args.kohya_path}/venv/bin/accelerate launch \
            --num_processes={args.num_processes} \
            "{args.kohya_path}/train_db.py" \
            """ + bucket + f""" --pretrained_model_name_or_path={args.pretrained_model_name_or_path} \
            --train_data_dir={args.train_data_dir}  \
            --reg_data_dir={args.reg_data_dir} \
            --resolution={args.resolution} \
            --output_dir={args.output_dir} \
            --logging_dir={args.logging_dir} \
            --save_model_as={args.save_model_as} \
            --output_name={args.output_name} \
            --lr_scheduler_num_cycles={args.lr_scheduler_num_cycles} \
            --max_data_loader_n_workers={args.max_data_loader_n_workers}  \
            --learning_rate_te={args.learning_rate_te} \
            --learning_rate={args.learning_rate} \
            --lr_scheduler={args.lr_scheduler} \
            --lr_warmup_steps={args.lr_warmup_steps}  \
            --train_batch_size={args.train_batch_size} \
            --max_train_steps={args.max_train_steps}  \
            --save_every_n_epochs={args.save_every_n_epochs} \
            --mixed_precision={args.mixed_precision} \
            --save_precision={args.save_precision}  \
            """ + cache_latents + f""" --optimizer_type={args.optimizer_type} \
            --bucket_reso_steps={args.bucekt_reso_steps} \
            """ + xformers + bucket_no_upscale + f""" --noise_offset={args.noise_offset} \
            --caption_extension {args.caption_extension}
            """)
    os.system(command)
    os.system(f"rm -rf ./data_temp")
    args.dump_path = args.output_dir
    args.checkpoint_path = args.output_dir + "/" + args.output_name + ".safetensors"
    if args.pipeline_class_name is not None:
        library = importlib.import_module("diffusers")
        class_obj = getattr(library, args.pipeline_class_name)
        pipeline_class = class_obj
    else:
        pipeline_class = None
    pipe = download_from_original_stable_diffusion_ckpt(
        checkpoint_path_or_dict=args.checkpoint_path,
        original_config_file=args.original_config_file,
        config_files=args.config_files,
        image_size=args.image_size,
        prediction_type=args.prediction_type,
        model_type=args.pipeline_type,
        extract_ema=args.extract_ema,
        scheduler_type=args.scheduler_type,
        num_in_channels=args.num_in_channels,
        upcast_attention=args.upcast_attention,
        from_safetensors=True,
        device=args.device,
        stable_unclip=args.stable_unclip,
        stable_unclip_prior=args.stable_unclip_prior,
        clip_stats_path=args.clip_stats_path,
        controlnet=args.controlnet,
        vae_path=args.vae_path,
        pipeline_class=pipeline_class,
    )
    if args.half:
        pipe.to(torch_dtype=torch.float16)
    if args.controlnet:
        # only save the controlnet model
        pipe.controlnet.save_pretrained(args.dump_path, safe_serialization=args.to_safetensors)
    else:
        pipe.save_pretrained(args.dump_path, safe_serialization=args.to_safetensors)
    os.system(f"rm {args.output_dir}/{args.output_name}.safetensors")