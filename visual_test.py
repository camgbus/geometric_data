#%%
from IPython import get_ipython
get_ipython().magic('load_ext autoreload') 
get_ipython().magic('autoreload 2')

import torch
from gd.data.pytorch_geometric import GeometricPytorchDataset
from gd.experiment_specific.rectangle_ellipse.dataset import rectangleEllipseDataset
from ptt.visualization.visualize_imgs import visualize_dataloader

from gd.experiment_specific.rectangle_ellipse.comp_pytorch_data import LatentSpaceDataset, AlternativeEllipse, RectangleDataset


#%%
ds = rectangleEllipseDataset(nr_instances=100, input_shape=(1, 64, 64))

py_ds = GeometricPytorchDataset(ds)
dl = torch.utils.data.DataLoader(py_ds, batch_size=1, shuffle=False) 

py_z = LatentSpaceDataset(ds)
dl_z = torch.utils.data.DataLoader(py_z, batch_size=5, shuffle=False) 

py_alt = AlternativeEllipse(ds)
dl_alt = torch.utils.data.DataLoader(py_alt, batch_size=1, shuffle=False) 

py_rect_ds = RectangleDataset(ds)
dl_rect_ds = torch.utils.data.DataLoader(py_rect_ds, batch_size=1, shuffle=False) 

# %%
visualize_dataloader(dl, img_size=(64, 64))
#for x, y in dl:
#    break


# %%
from gd.visualization.visualize_dl import visualize_data_pairs
visualize_data_pairs(dl_z, save_path=None, img_size=(64, 64))
# %%

# %%
s = (1, 64, 64)

# %%
s = list(s)
s[0]*=2
s = tuple(s)
print(s)

# %%
