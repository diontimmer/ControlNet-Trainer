# ControlNet-Trainer
<a target="_blank" href="https://colab.research.google.com/github/diontimmer/ControlNet-Trainer/blob/main/ControlNet_Trainer.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a><br>
Train your own ControlNet (https://github.com/lllyasviel/ControlNet) models for SD 1.5 and 2.1! Using a dataset of conditioning images, target images and captions you are able to create brand new models to control stable diffusion with.

## Dataset
The preferred dataset exists of three folders; one for the conditioning images, one for the target images and one for the prompt .txt files. These three folders can be placed in google drive so it can be accessed by the notebook.

The colab notebook comes with an integrated dataset converter. The dataset converter will format your conditioning + target + prompt pairs into a json needed for your sanity/convenience + the controlnet trainer. You can skip this if you already have a .json file. Every conditioning image needs at least one target image and prompt .txt with the same name. Multiple target images can be loaded for a single conditioning image by differentiating using an underscore _. *ie: 1_0.png 1_1.png 1_2.png etc..*.

This is a personal training script that I have decided to make public, so there might be some odd settings and/or bugs!


