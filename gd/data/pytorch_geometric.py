from ptt.data.pytorch_dataset import PytorchDataset
from torchvision import transforms

class GeometricPytorchDataset(PytorchDataset):
    """Geometric dataset that creates geometric images.
    """
    def __init__(self, dataset, ix_lst=None):
        super().__init__(dataset=dataset, ix_lst=ix_lst)

    def __getitem__(self, idx):
        # Get images
        x = self.instances[idx].x
        y = self.instances[idx].y
        # Apply transforms
        x = self.transform(x)
        y = transforms.ToTensor()(y)
        return x, y



