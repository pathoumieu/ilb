from PIL import Image
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
max_h = []
max_w = []
aspect_ratio = []

i = 0
for dir in tqdm(os.listdir('./data/reduced_images_ILB/reduced_images/train/')):
    for img in os.listdir(f'./data/reduced_images_ILB/reduced_images/train/{dir}'):
        i += 1
        # if max_h <= Image.open(f'./data/reduced_images_ILB/reduced_images/train/{dir}/{img}').size[0]:
        width, height = Image.open(f'./data/reduced_images_ILB/reduced_images/train/{dir}/{img}').size
        max_h.append(height)
        max_w.append(width)
        aspect_ratio.append(width / height)
        if Image.open(f'./data/reduced_images_ILB/reduced_images/train/{dir}/{img}').size[0] == 743:
            print(f'./data/reduced_images_ILB/reduced_images/train/{dir}/{img}')
print(pd.Series(max_w).describe())
print(pd.Series(max_h).describe())
print(pd.Series(aspect_ratio).describe())