# DreamBooth

✋ Notice: This is an experimental model in early development. ✋

From Wikipedia:

> DreamBooth is a deep learning generation model used to fine-tune existing text-to-image models, developed by researchers from Google Research and Boston University in 2022. Originally developed using Google's own Imagen text-to-image model, DreamBooth implementations can be applied to other text-to-image models, where it can allow the model to generate more fine-tuned and personalised outputs after training on three to five images of a subject.

This repository is a copy of the [canonical DreamBooth code](https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth), modified to work with [Cog](https://replicate.com/replicate/cog).

## Usage

This model takes your training images as input, and outputs trained weights that can be used to publish your own custom variant of Stable Diffusion.

To get started training and publishing your own custom model, see [github.com/replicate/dreambooth-template](https://github.com/replicate/dreambooth-template)
