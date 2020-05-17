# ------------------------------------------------------------------------------
# Code to draw shapes from PIL images.
# ------------------------------------------------------------------------------

from PIL import Image, ImageDraw
import numpy as np
from gd.drawing.texturize import corrupt_img

def create_image_with_shapes(img_shape, shapes):
    img = Image.new(mode="L", size=(img_shape[0], img_shape[1]))
    img = add_shapes(img, shapes)
    return img

def create_mask(img_shape, shapes, intersection=True):
    img = Image.new(mode="L", size=(img_shape[0], img_shape[1]))
    if intersection:
        img = shapes_intersection(img, shapes)
    else:
        img = shapes_union(img, shapes)
    return img

def shapes_intersection(img, shapes):
    intersection = None
    for shape in shapes:
        canvas_img = Image.new(mode="L", size=img.size)
        shape = add_shape(canvas_img, shape, color=1)
        if intersection is None:
            intersection = np.array(shape)
        else:
            intersection *= np.array(shape)
    return Image.fromarray(np.uint8(intersection*255))

def shapes_union(img, shapes):
    for shape in shapes:
        canvas_img = Image.new(mode="L", size=img.size)
        shape = add_shape(canvas_img, shape, color=255)
        texture = Image.new(mode="L", size=img.size, color=255)
        img = Image.composite(texture, img, shape)
    return img

def add_shapes(img, shapes):
    for shape in shapes:
        canvas_img = Image.new(mode="L", size=img.size)
        if shape.texture_tuple[0] is None:
            rotated = add_shape(canvas_img, shape, color=255)
            img = Image.composite(Image.new(mode="L", size=img.size, color=shape.color_tuple), img, rotated)
            #img.paste(rotated, (0, 0), rotated)
        else:
            rotated_mask = add_shape(canvas_img, shape, color=255)
            texture = Image.new(mode="L", size=img.size, color=shape.color_tuple)
            texture = corrupt_img(texture, texture_name=shape.texture_tuple[0], texture_level=shape.texture_tuple[1])
            img = Image.composite(texture, img, rotated_mask)
    return img

def add_shape(img, shape, color=None):
    if color is None:
        color=shape.color_tuple
    img_x, img_y = img.size
    y = min(shape.length_long, shape.length_short)
    x = max(shape.length_long, shape.length_short)
    x1, y1, x2, y2 = int(img_x*shape.x_start), int(img_y*shape.y_start), int(img_x*shape.x_start+img_x*x), int(img_y*shape.y_start+img_y*y)
    draw = ImageDraw.Draw(img)
    bbox =  (x1, y1, x2, y2)
    if shape.shape == 'ellipse':
        draw.ellipse(bbox, fill=color)
    elif shape.shape == 'rectangle':
        draw.rectangle(bbox, fill=color)
    elif shape.shape == 'line':
        draw.line(bbox, fill=color)
    img = img.rotate(shape.angle*360,  expand=0)
    return img