# ------------------------------------------------------------------------------
# Representation of a 2D shape.
# ------------------------------------------------------------------------------

class Shape:
    def __init__(self, 
    length_long=0.5, 
    length_short=0.5, 
    x_start=0.5, 
    y_start=0.5, 
    angle=45, 
    shape='rectangle', 
    color_tuple=255,
    texture_tuple=(None, 0)):
        self.length_long = length_long
        self.length_short = length_short
        self.x_start = x_start
        self.y_start = y_start
        self.angle = angle
        self.shape = shape
        self.color_tuple = color_tuple
        self.texture_tuple = texture_tuple
