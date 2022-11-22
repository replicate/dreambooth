# DreamBooth

We built an API that simplifies the process of using this model. **To get started training and publishing your own custom model, see [replicate.com/blog/dreambooth-api](https://replicate.com/blog/dreambooth-api).**

From Wikipedia:

> DreamBooth is a deep learning generation model used to fine-tune existing text-to-image models, developed by researchers from Google Research and Boston University in 2022. Originally developed using Google's own Imagen text-to-image model, DreamBooth implementations can be applied to other text-to-image models, where it can allow the model to generate more fine-tuned and personalised outputs after training on three to five images of a subject.

This repository is a copy of the [canonical DreamBooth code](https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth), modified to work with [Cog](https://replicate.com/replicate/cog).

 [![Replicate](https://replicate.com/replicate/dreambooth/badge)](https://replicate.com/replicate/dreambooth)

## Usage

This model takes your training images as input, and outputs trained weights that can be used to publish your own custom variant of Stable Diffusion.

The default stable diffusion model used is `runwayml/stable-diffusion-v1-5` (fp16), and `stabilityai/sd-vae-ft-mse` as `pretrained_vae`. 

## Run locally with Cog

First, download the pre-trained weights [with your Hugging Face auth token](https://huggingface.co/settings/tokens):

    cog run script/download-weights <your-hugging-face-auth-token>

Then, you can run train your dreambooth:

    cog predict -i instance_prompt="" -i class_prompt="" -i instance_data=@data.zip


There are a few inputs you should know about when training with this model:

- `instance_data` (required) - A ZIP file containing your training images (JPG, PNG, etc. size not restricted). These images contain your "subject" that you want the trained model to embed in the output domain for later generating customised scenes beyond the training images. For best results, use images without noise or unrelated object in the background.

- `instance_prompt` (required) - This is the prompt you use to describe your training images, in the format: `a [identifier] [class noun]`, where the `[identifier]` should be a rare-token - it is found that relatively short sequences with 1-3 letters work the best (e.g. `sks`, `xjy`). `[class noun]` is a coarse class descriptor of the subject (e.g. cat, dog, watch, etc.). For example, your `instance_prompt` can be: `a sks dog`, or with some extra description `a photo of a xjy dog`. The trained model will learn to bind a unique identifier with your specific subject in the `instance_data`.

- `class_prompt` (required) - This is the prompt or description of the coarse class of your training images, in the format of `a [class noun]` (or with some extra description).  `class_prompt` is used to alleviate overfitting to your customised images (the trained model should still keep the learnt prior so that it can still generate different dogs when the `[identifier]` is not in the prompt). Corresponding to the examples of the `instant_prompt` above, the `class_prompt` can be `a dog` or `a photo of a dog`. 

- `class_data` (optional). This corresponds to `class_prompt` above, also with the purpose to keep the generalisability of the model. By default, the pretrained stable-diffusion model will generate N (determined by the `num_class_images` you set) images based on the `class_prompt` provided above. But to save time or to to have your preferred specific set of `class_data`, you can also provide them in a ZIP file.

You may also want to change `num_class_images` and the `max_train_steps` settings, to trade-off speed and quality.

`seed` is randomly initialised to 1337, feel free to change it!
