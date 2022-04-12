# Discord Bot Template for StyleGANv1,2,3
Create a discord bot that generates images with StyleGAN! All credit for this code goes to NVIDIA, this is just an example use case I recommend you check out if you're looking to extend your discord bot's abilities.

## Requirements
Essential requirements are Windows/Linux, CUDA Toolkit v11.0 or higher, and Python 3.7.

After setting these up, simply run `pip3 install -r requirements` in the project folder to get everything else running.

## Setup
1. You'll need to save your networks PKL file into this projects directory and update the `network_path` variable of `bot.py`. If you don't have a PKL, or don't know what a PKL file is, it is a pickle file that tells pytorch how to get your network loaded. This file should be named `network-final.pkl` in the results folder after you train your GAN. I recommend checking out the[StyleGANV2 project](https://github.com/NVlabs/stylegan2-ada) for more details on this. 
2. You'll need to add your bot's token to the `token` variable of `bot.py`.

Other than that, this project runs like a charm and is a simple and easy addition to add GAN and DeepFake Techniques to your discord bot. 
