# ControlNet-Trainer
<a target="_blank" href="https://colab.research.google.com/github/diontimmer/ControlNet-Trainer/blob/main/ControlNet_Trainer.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a><br>
Train your own ControlNet (https://github.com/lllyasviel/ControlNet) models for SD 1.5 and 2.1! Using a dataset of conditioning images, target images and captions you are able to create brand new models to control stable diffusion with.
This is a personal training script that I have decided to make public, so there might be some odd settings and/or bugs!

## Dataset
The preferred dataset exists of three folders; one for the conditioning images, one for the target images and one for the prompt .txt files. These three folders can be placed in google drive so it can be accessed by the notebook.

The colab notebook comes with an integrated dataset converter. The dataset converter will format your conditioning + target + prompt pairs into a json needed for your sanity/convenience + the controlnet trainer. You can skip this if you already have a .json file. Every conditioning image needs at least one target image and prompt .txt with the same name. Multiple target images can be loaded for a single conditioning image by differentiating using an underscore _. *ie: 1_0.png 1_1.png 1_2.png etc..*. In essence, this converts a format of dataset used by various other trainers into the one used in the original repo training page: https://github.com/lllyasviel/ControlNet/blob/main/docs/train.md 

## Local Setup Instructions

To utilize the ControlNet Trainer on your local machine, follow the steps outlined below.
- Create a new Conda Python environment. This ensures that any dependencies installed don't interfere with your global Python environment. `conda create --name controlnet_trainer python=3.10`

- Activate the environment using `conda activate controlnet_trainer`.

- Install PyTorch. PyTorch is a required dependency. Visit https://pytorch.org/get-started/locally/ for specific installation instructions tailored to your system.

- Clone the repository. This will create a local copy of the ControlNet Trainer on your system. `git clone https://github.com/diontimmer/ControlNet-Trainer.git`

- Navigate to the repository directory using `cd ControlNet-Trainer`.

- Install additional requirements. The necessary Python packages are listed in the requirements.txt file. `pip install -r requirements.txt`

### With the setup complete, you're now ready to commence training.

To initiate training, run the train.py script with the path of a configuration JSON file as an argument. This configuration file replaces the need for multiple command line arguments. Default configurations can be located within the config.py file, and an example config file is provided for reference. I recommend storing your config files in the configs directory, but they can be placed wherever you find convenient. NOTE: Make sure your dataset is in the format needed by the original ControlNet repo: https://github.com/lllyasviel/ControlNet/blob/main/docs/train.md

## Additional Information

The ControlNet Trainer automatically saves a checkpoint (ckpt) file containing the training weights. This file allows you to resume training at a later point. Additionally, a SafeTensor file is created for sharing and inference purposes. These files are saved in the designated output folder.
To manage storage, you may wish to use the wipe_older_ckpts parameter. This option allows you to retain only the most recent checkpoint file on disk, which can be useful given the substantial size of these files.
Please note, the ControlNet Trainer performs image resizing automatically and in real-time. It ensures that your image will be correctly sized without any cropping
