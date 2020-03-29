"""me manually coding out a kernel filter effect"""
from pathlib import Path

PATH_MOLE: Path = Path(__file__).parent / 'mole_mug_shot.jpg'

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image

img = Image.open(PATH_MOLE).convert('LA')
img.save('grey_mole.png')

img = mpimg.imread('grey_mole.png', format='png')
# img = mpimg.imread(PATH_MOLE, format='jpg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


img = mpimg.imread(PATH_MOLE)
gray = rgb2gray(img)

# gray_norm = gray / gray.max()
# print(gray_norm)

ones = np.ones((50, 25))
zeros = np.zeros((50, 25))
kernel = np.concatenate([ones, zeros], axis=1)


def apply_kernel(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    nrow_img, ncol_img = img.shape
    nrow_k, ncol_k = kernel.shape
    row_max = nrow_img - nrow_k + 1
    col_max = ncol_img - ncol_k + 1
    img_new: np.ndarray = np.zeros((row_max, col_max))

    row_i: int = 0
    while row_i < row_max:

        col_i: int = 0
        while col_i < col_max:
            window = img[row_i: row_i + nrow_k, col_i: col_i + ncol_k]
            transformed_pixel = np.sum(window * kernel) / kernel.size
            img_new[row_i, col_i] = transformed_pixel
            col_i += 1
        row_i += 1
        print(f'row: {row_i} of {col_max}')

        print(img_new)
    return img_new


img_new = apply_kernel(gray, kernel=kernel)

print(img_new)
print(img_new.shape)

fig, ax = plt.subplots(ncols=2, figsize=(12, 8))
ax[0].imshow(gray, cmap='gray')
ax[1].imshow(img_new, cmap='gray')
plt.show()
