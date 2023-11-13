# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import glob
import imageio
import shutil

import numpy as np
import torch

res = [1024, 1024]

# datasets = ['ethiopianHead', 'moldGoldCape']
datasets = ['briar'] #['moldGoldCape']
folders  = ['masks', 'images']
input_format = 'png'

for dataset in datasets:
    dataset_rescaled = dataset + "_rescaled"
    os.makedirs(dataset_rescaled, exist_ok=True)
    shutil.copyfile(os.path.join(dataset, "poses_bounds.npy"), os.path.join(dataset_rescaled,
                                                                                  "poses_bounds.npy"))
    for folder in folders:
        os.makedirs(os.path.join(dataset_rescaled, folder), exist_ok=True)
        files = glob.glob(os.path.join(dataset, folder, f'*.{input_format}')) + glob.glob(os.path.join(dataset, folder, f'*.{input_format.upper()}'))
        for file in files:
            img = torch.tensor(imageio.v2.imread(file).astype(np.float32) / 255.0, )
            try:
                img = img[None, ...].permute(0, 3, 1, 2)
            except RuntimeError as re:
                # handle greyscale images
                np_image = np.array(img)
                np_image = np_image.astype(float)
                np_image = np_image - 128.0
                np_image = np_image / 128.0
                new_image = np.zeros((np_image.shape[0], np_image.shape[1], 3))
                new_image[:,:,0] = np_image
                new_image[:,:,1] = np_image
                new_image[:,:,2] = np_image
                img = torch.tensor(new_image)[None, ...].permute(0, 3, 1, 2)
            rescaled_img = torch.nn.functional.interpolate(img, res, mode='area')
            rescaled_img = rescaled_img.permute(0, 2, 3, 1)[0, ...]
            out_file = os.path.join(dataset_rescaled, folder, os.path.basename(file))
            imageio.imwrite(out_file, np.clip(np.rint(rescaled_img.numpy() * 255.0), 0, 255).astype(np.uint8))
