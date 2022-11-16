import os
import gc
import mimetypes
import shutil
import tempfile
from zipfile import ZipFile
from subprocess import call
from argparse import Namespace
import torch

from cog import BasePredictor, Input, Path

from dreambooth import main


def run_cmd(command):
    try:
        call(command, shell=True)
    except KeyboardInterrupt:
        print("Process interrupted")
        sys.exit(1)


class Predictor(BasePredictor):
    def setup(self):
        pass

    def predict(
        self,
        # pretrained_model: str = Input(
        #     description="Model identifier from huggingface.co/models",
        #     default="runwayml/stable-diffusion-v1-5",
        # ),
        # huggingface_token: str = Input(
        #     description="Provide your huggingface token to download the models.",
        #     default=None,
        # ),
        # pretrained_vae: str = Input(
        #     description="Pretrained vae or vae identifier from huggingface.co/models",
        #     default="stabilityai/sd-vae-ft-mse",
        # ),
        # revision: str = Input(
        #     description="Revision of pretrained model identifier from huggingface.co/models",
        #     choices=["fp16", "None"],
        #     default="fp16",
        # ),
        # mixed_precision: str = Input(
        #     description="Whether to use mixed precision. Choose"
        #     "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        #     "and an Nvidia Ampere GPU.",
        #     choices=["fp16", "bf16", "no"],
        #     default="fp16",
        # ),
        # tokenizer_name: str = Input(
        #     description="Pretrained tokenizer name or path if not the same as model_name",
        #     default=None,
        # ),
        instance_prompt: str = Input(
            description="The prompt with identifier specifying the instance",
        ),
        class_prompt: str = Input(
            description="The prompt to specify images in the same class as provided instance images.",
        ),
        instance_data: Path = Input(
            description="A ZIP file containing the training data of instance images",
        ),
        class_data: Path = Input(
            description="A ZIP file containing the training data of class images. Images will be generated if you do not provide.",
            default=None,
        ),
        save_sample_prompt: str = Input(
            description="The prompt used to generate sample outputs to save.",
            default=None,
        ),
        save_sample_negative_prompt: str = Input(
            description="The negative prompt used to generate sample outputs to save.",
            default=None,
        ),
        n_save_sample: int = Input(
            description="The number of samples to save.",
            default=4,
        ),
        save_guidance_scale: float = Input(
            description="CFG for save sample.",
            default=7.5,
        ),
        save_infer_steps: int = Input(
            description="The number of inference steps for save sample.",
            default=50,
        ),
        pad_tokens: bool = Input(
            description="Flag to pad tokens to length 77.",
            default=False,
        ),
        with_prior_preservation: bool = Input(
            description="Flag to add prior preservation loss.",
            default=True,
        ),
        prior_loss_weight: float = Input(
            description="Weight of prior preservation loss.",
            default=1.0,
        ),
        num_class_images: int = Input(
            description="Minimal class images for prior preservation loss. If not have enough images, additional images will be"
            " sampled with class_prompt.",
            default=50,
        ),
        seed: int = Input(description="A seed for reproducible training", default=1337),
        resolution: int = Input(
            description="The resolution for input images. All the images in the train/validation dataset will be resized to this"
            " resolution.",
            default=512,
        ),
        center_crop: bool = Input(
            description="Whether to center crop images before resizing to resolution",
            default=False,
        ),
        train_text_encoder: bool = Input(
            description="Whether to train the text encoder",
            default=True,
        ),
        train_batch_size: int = Input(
            description="Batch size (per device) for the training dataloader.",
            default=1,
        ),
        sample_batch_size: int = Input(
            description="Batch size (per device) for sampling images.",
            default=4,
        ),
        num_train_epochs: int = Input(default=1),
        max_train_steps: int = Input(
            description="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
            default=500,
        ),
        gradient_accumulation_steps: int = Input(
            description="Number of updates steps to accumulate before performing a backward/update pass.",
            default=1,
        ),
        gradient_checkpointing: bool = Input(
            description="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
            default=False,
        ),
        learning_rate: float = Input(
            description="Initial learning rate (after the potential warmup period) to use.",
            default=1e-6,
        ),
        scale_lr: bool = Input(
            description="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
            default=False,
        ),
        lr_scheduler: str = Input(
            description="The scheduler type to use",
            choices=[
                "linear",
                "cosine",
                "cosine_with_restarts",
                "polynomial",
                "constant",
                "constant_with_warmup",
            ],
            default="constant",
        ),
        lr_warmup_steps: int = Input(
            description="Number of steps for the warmup in the lr scheduler.",
            default=0,
        ),
        use_8bit_adam: bool = Input(
            description="Whether or not to use 8-bit Adam from bitsandbytes.",
            default=False,
        ),
        adam_beta1: float = Input(
            default=0.9,
            description="The beta1 parameter for the Adam optimizer.",
        ),
        adam_beta2: float = Input(
            default=0.999,
            description="The beta2 parameter for the Adam optimizer.",
        ),
        adam_weight_decay: float = Input(
            default=1e-2,
            description="Weight decay to use",
        ),
        adam_epsilon: float = Input(
            default=1e-8,
            description="Epsilon value for the Adam optimizer",
        ),
        max_grad_norm: float = Input(
            default=1.0,
            description="Max gradient norm.",
        ),
        save_interval: int = Input(
            default=10000,
            description="Save weights every N steps.",
        ),
    ) -> Path:

        cog_instance_data = "cog_instance_data"
        cog_class_data = "cog_class_data"
        cog_output_dir = "cog_out"
        for path in [cog_instance_data, cog_output_dir, cog_class_data]:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)

        # extract zip contents, flattening any paths present within it
        with ZipFile(str(instance_data), "r") as zip_ref:
            for zip_info in zip_ref.infolist():
                if zip_info.filename[-1] == '/':
                    continue
                mt = mimetypes.guess_type(zip_info.filename)
                if mt and mt[0] and mt[0].startswith('image/'):
                    zip_info.filename = os.path.basename(zip_info.filename)
                    zip_ref.extract(zip_info, cog_instance_data)

        if class_data is not None:
            with ZipFile(str(class_data), "r") as zip_ref:
                zip_ref.extractall(cog_class_data)

        # some settings are fixed for the replicate model
        args = {
            "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
            "pretrained_vae_name_or_path": "stabilityai/sd-vae-ft-mse",
            "revision": "fp16",
            "tokenizer_name": None,
            "instance_data_dir": f"{cog_instance_data}",
            "class_data_dir": f"{cog_class_data}/class_data",
            "instance_prompt": instance_prompt,
            "class_prompt": class_prompt,
            "save_sample_prompt": save_sample_prompt,
            "save_sample_negative_prompt": save_sample_negative_prompt,
            "n_save_sample": n_save_sample,
            "save_guidance_scale": save_guidance_scale,
            "save_infer_steps": save_infer_steps,
            "pad_tokens": pad_tokens,
            "with_prior_preservation": with_prior_preservation,
            "prior_loss_weight": prior_loss_weight,
            "num_class_images": num_class_images,
            "seed": seed,
            "resolution": resolution,
            "center_crop": center_crop,
            "train_text_encoder": train_text_encoder,
            "train_batch_size": train_batch_size,
            "sample_batch_size": sample_batch_size,
            "num_train_epochs": num_train_epochs,
            "max_train_steps": max_train_steps,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "gradient_checkpointing": gradient_checkpointing,
            "learning_rate": learning_rate,
            "scale_lr": scale_lr,
            "lr_scheduler": lr_scheduler,
            "lr_warmup_steps": lr_warmup_steps,
            "use_8bit_adam": use_8bit_adam,
            "adam_beta1": adam_beta1,
            "adam_beta2": adam_beta2,
            "adam_weight_decay": adam_weight_decay,
            "adam_epsilon": adam_epsilon,
            "max_grad_norm": max_grad_norm,
            "push_to_hub": False,
            "hub_token": None,
            "hub_model_id": None,
            "save_interval": save_interval,
            "save_min_steps": 0,
            "mixed_precision": "fp16",
            "not_cache_latents": False,
            "local_rank": -1,
            "output_dir": cog_output_dir,
            "concepts_list": None,
            "logging_dir": "logs",
            "log_interval": 10,
            "hflip": False,
        }

        args = Namespace(**args)

        main(args)

        gc.collect()
        torch.cuda.empty_cache()
        call("nvidia-smi")

        out_path = "output.zip"

        directory = Path(cog_output_dir)
        with ZipFile(out_path, "w") as zip:
            for file_path in directory.rglob("*"):
                print(file_path)
                zip.write(file_path, arcname=file_path.relative_to(directory))

        return Path(out_path)
