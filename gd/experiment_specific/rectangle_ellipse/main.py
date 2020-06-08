
import sys
import torch
import torch.optim as optim

from ptt.argument_parsing import parse_args_as_dict
from ptt.experiment.experiment import Experiment
from ptt.eval.result import Result
from gd.experiment_specific.rectangle_ellipse.dataset import rectangleEllipseDataset
from gd.data.pytorch_geometric import GeometricPytorchDataset
from gd.models.segmentation.unet import UNet
from gd.models.multievel.mm_unet import MMUNet
from gd.agents.segmentation_agent import SegmentationAgent
from gd.agents.multilevel_agent import MultilevelAgent
from gd.experiment_specific.rectangle_ellipse.comp_pytorch_data import LatentSpaceDataset, AlternativeEllipse, RectangleDataset

def run(exp, exp_run, data):
    config = exp.config

    # Transform data to PyTorch format and build dataloaders
    datasets = dict()
    ellipse_datasets = dict()
    rectangle_datasets = dict()
    for data_name, data_ixs in exp.splits[exp_run.run_ix].items():
        if len(data_ixs) > 0:
            datasets[data_name] = GeometricPytorchDataset(data, ix_lst=data_ixs)
            ellipse_datasets[data_name] = LatentSpaceDataset(data, ix_lst=data_ixs)
            rectangle_datasets[data_name] = RectangleDataset(data, ix_lst=data_ixs)

    dataloaders = dict()
    ellipse_dataloaders = dict()
    rectangle_dataloaders = dict()
    for split in datasets.keys():
        shuffle = not(split == 'test')
        dataloaders[split] = torch.utils.data.DataLoader(datasets[split], batch_size=config['batch_size'], shuffle=shuffle)
        ellipse_dataloaders[split] = torch.utils.data.DataLoader(ellipse_datasets[split], batch_size=config['batch_size'], shuffle=shuffle)
        rectangle_dataloaders[split] = torch.utils.data.DataLoader(rectangle_datasets[split], batch_size=config['batch_size'], shuffle=shuffle)

    # Get model
    model_G = UNet(input_shape=data.input_shape)
    model_H = UNet(input_shape=data.input_shape)
    model_F = MMUNet(model_g=model_G, model_h=model_H, input_shape=data.input_shape)
    #model = UNet(input_shape=data.input_shape)
    #model.to(config['device'])
    model_G.to(config['device'])
    model_H.to(config['device'])
    model_F.to(config['device'])

    # Define optimizer
    #ptimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
    optimizer_G = optim.SGD(model_G.parameters(), lr=config['lr'], momentum=config['momentum'])
    optimizer_H = optim.SGD(model_H.parameters(), lr=config['lr'], momentum=config['momentum'])
    optimizer_F = optim.SGD(model_F.parameters(), lr=config['lr'], momentum=config['momentum'])

    # Train model
    results = Result(name='training_trajectory')
    agent = SegmentationAgent(config=config, verbose=True)
    #agent = SegmentationAgent(config=config, verbose=True)
    #agent.train(results, model, optimizer, trainloader=dataloaders['train'], dataloaders=dataloaders)
    agent.train(results, model_F, optimizer_F, trainloader=dataloaders['train'], dataloaders=dataloaders)


    return results
    

if __name__ == '__main__':
    # Use console arguments
    config = parse_args_as_dict(sys.argv[1:])
    exp = Experiment(config=config, name=config['experiment_name'], 
        notes='Segmentation of the ellipse-rectangle problem', reload_exp=True)
    # Get data
    data = rectangleEllipseDataset(nr_instances=5000, input_shape=(1, 64, 64))
    # Divide indexes into splits/folds
    exp.set_data_splits(data)
    # Iterate over repetitions and run
    for ix in [0]: #range(config['nr_runs']):
        print('Running repetition {} of {}'.format(ix+1, config['nr_runs']))
        exp_run = exp.get_run(run_ix=ix)
        #try:
        results = run(exp=exp, exp_run=exp_run, data=data)
        exp_run.finish(results=results)
        #except Exception as e: 
        #    exp_run.finish(exception=e)
