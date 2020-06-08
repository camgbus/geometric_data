# ------------------------------------------------------------------------------
# Classes for generating datasets consisting of parametric representations.
# Each geometric instance has:
# - A name -> shape dictionary of gd.drawing.shape.Shape objects
# - functions to produce x, y and intermediate features
# ------------------------------------------------------------------------------

import os
from ptt.data.dataset import Dataset, Instance
from ptt.utils.load_restore import pkl_dump, pkl_load
from gd.paths import data_path
from gd.drawing.shape import Shape
import gd.drawing.draw_shapes as draw
from gd.distributions.sampling import sample

def sample_shape(shape_params):
    return Shape(
        length_long = sample(shape_params['length_long']),
        length_short = sample(shape_params['length_short']),
        x_start = sample(shape_params['x_start']),
        y_start = sample(shape_params['y_start']),
        angle = sample(shape_params['angle']),
        color_tuple = shape_params['color_tuple'],
        shape = shape_params['shape'],
        texture_tuple = shape_params['texture_tuple']
    )

class LatentInstance(Instance):
    """A LatentInstance has X, Z and Y values, i.e. a defined latent space.
    Z is a dictionary with multiple named features.
    """
    def __init__(self, x, y, z, class_ix=0, name=None):
        self.z = z
        super().__init__(x=x, y=y, class_ix=class_ix, name=name)

class GeometricInstance(LatentInstance):
    """ Each geometric instance has a name -> shape dictionary of gd.drawing.shape.Shape objects.
    """
    def __init__(self, x, y, z, shapes, class_ix=0, name=None):
        self.shapes = shapes
        super().__init__(x=x, y=y, z=z, class_ix=class_ix, name=name)

class GeometricDataset(Dataset):
    """A geometric dataset.
    """
    def __init__(self, subclass_name, shape_definition, fn_x, fn_y, fn_z, 
        nr_instances, input_shape=(1, 64, 64)):
        self.name = subclass_name + '_' + str(nr_instances) + '_' + str(input_shape).replace(' ','')
        self.shape_definition = shape_definition
        self.fn_x, self.fn_y, self.fn_z = fn_x, fn_y, fn_z
        # Check whether the instances already exists. If so, load, else create.
        instances = pkl_load(self.name+'_instances', data_path)
        if instances is None:
            instances = []
            for ix in range(nr_instances):
                shapes = dict()
                for shape_name, shape_params in shape_definition.items():
                    shapes[shape_name] = sample_shape(shape_params)
                instances.append(GeometricInstance(x=fn_x(shapes), y=fn_y(shapes), 
                    z={z_name: fn(shapes) for z_name, fn in fn_z.items()}, 
                    shapes=shapes, name=str(ix)))
            # Save instances
            pkl_dump(instances, self.name+'_instances', data_path)
        super().__init__(name=self.name, instances=instances, 
            input_shape=input_shape, output_shape=input_shape)

