import os
import numpy as np
from glob import glob
import shutil
import cv2


input_path = "data/clean_images/"

shutil.rmtree(input_path.replace("clean_images", "noisy_images"), ignore_errors=True)
shutil.rmtree(input_path.replace("clean_images", "masked_images"), ignore_errors=True)
shutil.rmtree(input_path.replace("clean_images", "outlier_images"), ignore_errors=True)

os.makedirs(input_path.replace("clean_images", "noisy_images"), exist_ok=True)
os.makedirs(input_path.replace("clean_images", "masked_images"), exist_ok=True)
os.makedirs(input_path.replace("clean_images", "outlier_images"), exist_ok=True)

in_images = list(glob(input_path+"*.jpg"))

for x in in_images:
    img = cv2.imread(x)

    scale = np.random.randint(1, 128)
    nb_masks = np.random.randint(1, 20)

    img_uniform_noise = np.random.randint(0, 255, size=img.shape)

    img_g_noise = img.copy()+np.random.normal(size=img.shape, scale=scale)

    img_masked = img.copy()

    for _ in range(nb_masks):
        i = np.random.randint(0, img.shape[0])
        j = np.random.randint(0, img.shape[1])
        mask_size_i = np.random.randint(1, img.shape[0] // 2)
        mask_size_j = np.random.randint(1, img.shape[1] // 2)

        img_masked[i:(i+mask_size_i), j:(j+mask_size_j), :] = np.random.choice([0, 255])


    cv2.imwrite(x.replace("clean_images", "noisy_images"), img_g_noise)
    cv2.imwrite(x.replace("clean_images", "masked_images"), img_masked)
    cv2.imwrite(x.replace("clean_images", "outlier_images"), img_uniform_noise)
