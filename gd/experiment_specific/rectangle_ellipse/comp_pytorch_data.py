from ptt.data.pytorch_dataset import PytorchDataset
from gd.data.dataset_geometric import sample_shape
import gd.drawing.draw_shapes as draw
from torchvision import transforms

class LatentSpaceDataset(PytorchDataset):
    """Outputs x and z.
    """
    def __init__(self, dataset, ix_lst=None):
        super().__init__(dataset=dataset, ix_lst=ix_lst)

    def __getitem__(self, idx):
        # Get images
        x = self.instances[idx].x
        z = self.instances[idx].z['z']
        # Apply transforms
        x = self.transform(x)
        z = transforms.ToTensor()(z)
        return x, z

class RectangleDataset(PytorchDataset):
    """Outputs x and the rectangule mask.
    """
    def __init__(self, dataset, ix_lst=None):
        self.orig_dataset = dataset
        super().__init__(dataset=dataset, ix_lst=ix_lst)

    def __getitem__(self, idx):
        # Get images
        x = self.instances[idx].x
        rectangle = draw.create_mask(img_shape=self.orig_dataset.input_shape[1:],  shapes=[self.instances[idx].shapes['rectangle']], intersection=False)
        # Apply transforms
        x = self.transform(x)
        rectangle = transforms.ToTensor()(rectangle)
        return x, rectangle

def _perturbe_x(geom_instance, geom_dataset):
    """
    This method outputs an alternative x with a different ellipse.
    The goal is to make sure that for certain features, changes in the ellipse 
    do not affect the features, as they should only capture rectangule info.
    """
    alt_ellipse = sample_shape(geom_dataset.shape_definition['ellipsis'])
    return draw.create_image_with_shapes(img_shape=geom_dataset.input_shape[1:], shapes=[alt_ellipse, geom_instance.shapes['rectangle']])

class AlternativeEllipse(PytorchDataset):
    """Outputs the original x and an alternative x with a perturbed ellipse.
    """
    def __init__(self, dataset, ix_lst=None):
        self.orig_dataset = dataset
        super().__init__(dataset=dataset, ix_lst=ix_lst)

    def __getitem__(self, idx):
        # Get images
        x = self.instances[idx].x
        alt_x = _perturbe_x(self.instances[idx], self.orig_dataset)
        # Apply transforms
        x = self.transform(x)
        alt_x = self.transform(alt_x)
        return x, alt_x