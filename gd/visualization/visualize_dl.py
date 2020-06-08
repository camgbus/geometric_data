# ------------------------------------------------------------------------------
# Visualizes x and y of a dataloader by switching each image by the 
# concatenation of two images.
# ------------------------------------------------------------------------------

import numpy as np
from PIL import Image
from ptt.visualization.visualize_imgs import get_img_grid

def get_img_pairs_from_dl(dataloader, nr_pairs):
    imgs = []
    for x, y in dataloader:
        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        assert x.shape == y.shape
        for ix in range(len(x)):
            if len(imgs) < nr_pairs:
                imgs.append((x[ix], y[ix]))
        if len(imgs) == nr_pairs:
            break  
    return imgs

import sys
def create_img_pairs_grid(img_grid = [[]], img_size = (512, 512), 
    margin = (10, 10), within_pair_margin=5, background_color = (255, 255, 255, 255), 
    save_path=None):
    """
    Creates an image grip of x, y pairs.
    """
    bg_height = len(img_grid)*img_size[1] + (len(img_grid)+1)*margin[1]
    bg_width = len(img_grid[0])*((img_size[0]*2)+within_pair_margin) + (len(img_grid[0])+1)*margin[0]
    new_img = Image.new('RGBA', (bg_width, bg_height), background_color)
    # Initial margins
    left = margin[0]
    top = margin[1]
    for row in img_grid:
        for img_pair in row:
            for img in img_pair:
                # If grayscale image
                if img.shape[0]==1:
                    img = img[0]
                # If channels first
                elif np.argpartition(img.shape, 1)[0] == 0:
                    img = np.moveaxis(img, 0, 2) 
                img = Image.fromarray((img * 255).astype(np.uint8)).resize(img_size).convert('RGB')
                new_img.paste(img, (left, top))
                left += img_size[0] + within_pair_margin
            left += margin[0] - within_pair_margin
        top += img_size[1] + margin[1]
        left = margin[0]
    if save_path is None:
        new_img.show()
    else:
        new_img.save(save_path)

def visualize_data_pairs(dataloader, grid_size=(5, 5), save_path=None, img_size=(512, 512)):
    img_pairs = get_img_pairs_from_dl(dataloader, grid_size[0]*grid_size[1])
    img_grid = get_img_grid(img_pairs, grid_size[0], grid_size[1])
    create_img_pairs_grid(img_grid=img_grid, save_path=save_path, img_size=img_size)