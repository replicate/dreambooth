from cog import BasePredictor, Input, Path

from .dreambooth import main


class Predictor(BasePredictor):
    def setup(self):
        pass

    def predict(
        self,
        pretrained_model: str = Input(
            description="Model identifier from huggingface.co/models",
            default="runwayml/stable-diffusion-v1-5",
        ),
        revision: str = Input(
            description="Revision of pretrained model identifier from huggingface.co/models",
            default=None,
        ),
        tokenizer_name: str = Input(
            description="Pretrained tokenizer name or path if not the same as model_name",
            default=None,
        ),
        instance_data: Path = Input(
            description="A ZIP file containing the training data of instance images",
        ),
        class_data_dir: Path = Input(
            description="A ZIP file containing the training data of class images",
            default=None,
        ),
        with_prior_preservation: bool = Input(
            description="Flag to add prior preservation loss.",
            default=False,
        ),
        prior_loss_weight: float = Input(
            description="Weight of prior preservation loss.",
            default=1.0,
        ),
        num_class_images: int = Input(
            description="Minimal class images for prior preservation loss. If not have enough images, additional images will be"
            " sampled with class_prompt.",
            default=100,
        ),
        seed: int = Input(description="A seed for reproducible training", default=None),
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
            default=False,
        ),
        train_batch_size: int = Input(
            description="Batch size (per device) for the training dataloader.",
            default=4,
        ),
        sample_batch_size: int = Input(
            description="Batch size (per device) for sampling images.",
            default=4,
        ),
        num_train_epochs: int = Input(default=1),
        max_train_steps: int = Input(
            description="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
            default=None,
        ),
        gradient_accumulation_steps: int = Input(
            description="Number of updates steps to accumulate before performing a backward/update pass.",
            default=1,
        ),
        gradient_checkpointing: bool = Input(
            help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
            default=False,
        ),
        learning_rate: float = Input(
            description="Initial learning rate (after the potential warmup period) to use.",
            default=5e-6,
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
            default=500,
        ),
        use_8bit_adam: bool = Input(
            description="Whether or not to use 8-bit Adam from bitsandbytes.",
            default=False,
        ),
        attn_only: bool = Input(
            description="Only train U-Net cross-attention layers",
            default=False,
        ),
        use_ema: bool = Input(
            description="Whether to use EMA model.",
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
        mixed_precision: str = Input(
            description="Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU.",
            choices=["fp16", "bf16", "no"],
            default="no",
        ),
    ) -> Path:
        # TODO: unzip instance_data and class_data_dir

        main(
            pretrained_model=pretrained_model,
            revision=revision,
            tokenizer_name=tokenizer_name,
            instance_data=instance_data,
            class_data_dir=class_data_dir,
            with_prior_preservation=with_prior_preservation,
            prior_loss_weight=prior_loss_weight,
            num_class_images=num_class_images,
            seed=seed,
            resolution=resolution,
            center_crop=center_crop,
            train_text_encoder=train_text_encoder,
            train_batch_size=train_batch_size,
            sample_batch_size=sample_batch_size,
            num_train_epochs=num_train_epochs,
            max_train_steps=max_train_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            learning_rate=learning_rate,
            scale_lr=scale_lr,
            lr_scheduler=lr_scheduler,
            lr_warmup_steps=lr_warmup_steps,
            use_8bit_adam=use_8bit_adam,
            attn_only=attn_only,
            use_ema=use_ema,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_weight_decay=adam_weight_decay,
            adam_epsilon=adam_epsilon,
            max_grad_norm=max_grad_norm,
            mixed_precision=mixed_precision,
        )

        # TODO: return result (is it just a single file? or multiple files? could be an output object with multiple files, or a zip, or something)
