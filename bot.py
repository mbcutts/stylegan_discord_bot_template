import os
import re
from typing import List, Optional
import dnnlib
import numpy as np
import PIL.Image
import torch
import legacy
from discord.ext import commands
import discord
import random
import time

'''
Helper functions modified from the STYLEGANv2-ADA github repository. All of this code belongs to NVIDIA. Disclaimer:

Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

'''

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

def generate_images(network_pkl: str, seeds: Optional[List[int]], outdir: str):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    if seeds is None:
        print('Error: seeds argument is required')

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        img = G(z, label)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')

'''
Discord Functionality Lies Below.
'''

token = ''
bot = commands.Bot(command_prefix='!')

@bot.command()
async def generate(ctx, arg):
    
    await ctx.send("Got your message. Going to generate your image with STYLEGANv2, just give me a few moments.")
    
    #setting up variables for network
    seed = [random.randint(1, 1000)]
    outdir = os.getcwd() + "/out"
    network_path = os.getcwd() + "./network-final.pkl" 
    
    #generating the image
    generate_images(network_pkl = "./network-final.pkl", seeds=seed, outdir=outdir)
    
    #get the path of your image
    image_path = f'{outdir}/seed{seed[0]:04d}.png'
    with open(image_path, 'rb') as f:
        picture = discord.File(f)
        await ctx.send(file=picture)


bot.run(token)
