from gd.data.dataset_geometric import GeometricDataset
import gd.drawing.draw_shapes as draw

class rectangleEllipseDataset(GeometricDataset):
    """Each x of this dataset contains two shapes: a rectangle and an 
    ellipse. The image generation process renders the rectangle gray and the 
    ellipse white. 
    The class value y is the segmnetation mask of their union.
    The latent space consists of:
    - 'z': segmentation mask for the ellipse
    """
    def __init__(self, nr_instances, input_shape=(1, 64, 64)):
        # Definition of shapes
        shape_definition = {'ellipsis': dict(), 'rectangle': dict()}
        ## Ellipsis
        shape_definition['ellipsis']['shape'] = 'ellipse'
        shape_definition['ellipsis']['color_tuple'] = 255
        shape_definition['ellipsis']['texture_tuple'] = (None, 0)
        shape_definition['ellipsis']['length_long'] = ('normal', 0.5, 0.1, 0.2, 1)
        shape_definition['ellipsis']['length_short'] =  ('normal', 0.4, 0.1, 0.2, 1)
        shape_definition['ellipsis']['x_start'] = ('normal', 0.25, 0.1, 0, 1)
        shape_definition['ellipsis']['y_start'] = ('normal', 0.25, 0.1, 0, 1)
        shape_definition['ellipsis']['angle'] = ('uniform', 0, 1)
        ## Rectangle
        shape_definition['rectangle']['shape'] = 'rectangle'
        shape_definition['rectangle']['color_tuple'] = 100
        shape_definition['rectangle']['texture_tuple'] = (None, 0)
        shape_definition['rectangle']['length_long'] = ('normal', 0.3, 0.1, 0.2, 1)
        shape_definition['rectangle']['length_short'] = ('normal', 0.3, 0.1, 0.2, 1)
        shape_definition['rectangle']['x_start'] = ('normal', 0.25, 0.1, 0, 1)
        shape_definition['rectangle']['y_start'] = ('normal', 0.25, 0.1, 0, 1)
        shape_definition['rectangle']['angle'] = ('uniform', 0, 1)
        # Functions defining how to create features
        def x(shapes):
            return draw.create_image_with_shapes(img_shape=input_shape[1:], shapes=[shapes['ellipsis'], shapes['rectangle']])
        def y(shapes):
            return draw.create_mask(img_shape=input_shape[1:],  shapes=[shapes['ellipsis'], shapes['rectangle']], intersection=False)
        def z(shapes): 
            return draw.create_mask(img_shape=input_shape[1:],  shapes=[shapes['ellipsis']], intersection=False)
        super().__init__(subclass_name='rectangleEllipseDataset', 
            shape_definition=shape_definition, fn_x=x, fn_y=y, fn_z={'z': z},
            nr_instances=nr_instances, input_shape=input_shape)
