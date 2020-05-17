# ------------------------------------------------------------------------------
# Example of generating a geometric dataset (2)
# ------------------------------------------------------------------------------

import os
from gd.drawing.shape import Shape
from gd.drawing.draw_shapes import create_image_with_shapes, create_mask

def mask_not_empty(img):
    return img.convert("L").getextrema()[1]>0

class Ellipsis():
    """This data consists of two ellipsis, a smaller one with texture and a larger one without"""
    def __init__(self, small_ellipsis_data, big_ellipsis_data):
        nr_dummy_per_img = len(big_ellipsis_data) // len(small_ellipsis_data)
        nr_empty = 0
        for ix in range(len(small_ellipsis_data)):
            small_length_long, small_length_short, small_x_start, small_y_start, small_angle = small_ellipsis_data[ix]
            for ix_dummy in range(nr_dummy_per_img):
                ix_big = ix*nr_dummy_per_img+ix_dummy
                big_length_long, big_length_short, big_x_start, big_y_start, big_angle = big_ellipsis_data[ix_big]
                name = str(ix)+'-'+str(ix_dummy)
                not_empty = self.create_observation(name, big_length_long, big_length_short, big_x_start, big_y_start, big_angle,
                    small_length_long, small_length_short, small_x_start, small_y_start, small_angle, dummy=ix_dummy==0)
                nr_empty += int(not not_empty)
        print('Nr. observations: {}, ratio empty: {}'.format(len(small_ellipsis_data), nr_empty/len(big_ellipsis_data)))

    def create_observation(self, name, big_length_long, big_length_short, big_x_start, big_y_start, big_angle,
        small_length_long, small_length_short, small_x_start, small_y_start, small_angle, img_shape=(64, 64),
        save_path=r'storage/ellipsis/', dummy=False):

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        big_ellipsis = Shape(
            length_long = big_length_long,
            length_short = big_length_short,
            x_start = big_x_start,
            y_start = big_y_start,
            angle = big_angle,
            color_tuple = 255,
            shape='ellipse',
            texture_tuple=(None, 0)
        )
        small_ellipsis = Shape(
            length_long = small_length_long,
            length_short = small_length_short,
            x_start = small_x_start,
            y_start = small_y_start,
            angle = small_angle,
            color_tuple = 255,
            shape='ellipse',
            texture_tuple=('Gaussian Noise', 20)
        )
        shapes = [big_ellipsis, small_ellipsis]

        img = create_image_with_shapes(img_shape=img_shape, shapes=shapes)
        img.save(os.path.join(save_path, name+'_x.png'))

        dummy_mask = create_mask(img_shape=img_shape, shapes=[big_ellipsis], intersection=True)
        dummy_mask.save(os.path.join(save_path, name+'_z.png'))

        mask = create_mask(img_shape=img_shape, shapes=shapes, intersection=True)
        mask.save(os.path.join(save_path, name+'_y.png'))

        return mask_not_empty(mask)
