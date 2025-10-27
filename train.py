import os
import numpy as np
import torch
from torch.utils.data import TensorDataset

import setup
import losses
import models
import datasets
import utils

class Trainer():

    def __init__(self, model, train_loader, params):

        self.params = params

        # define dataset
        self.params['dataset'] = train_loader.dataset
        self.params['temperature_map'] = np.ones(self.params['num_classes']) * 0.2 

        # define loaders:
        self.train_loader = train_loader

        # define model:
        self.model = model

        # define important objects:
        self.compute_loss = losses.get_loss_function(params)
        self.encode_location = self.train_loader.dataset.enc.encode

        # define optimization objects:
        self.optimizer = torch.optim.Adam(self.model.parameters(), params['lr'])
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=params['lr_decay'])

        # initialize loss tracking
        self.train_epoch_losses = []


    def train_one_epoch(self):

        self.model.train()
        # initialise run stats
        running_train_loss = 0.0
        samples_processed = 0
        for _, batch in enumerate(self.train_loader):
            # reset gradients:
            self.optimizer.zero_grad()
            # compute loss and log predictions if tracking (second to last batch in epoch):
            batch_loss = self.compute_loss(batch, self.model, self.params, self.encode_location)
            # backwards pass:
            batch_loss.backward()
            # update parameters:
            self.optimizer.step()
            # track and report:
            running_train_loss += float(batch_loss.item()) * batch[0].shape[0]
            samples_processed += batch[0].shape[0]
        # update learning rate according to schedule:
        self.lr_scheduler.step()
        # Compute and store mean loss for the epoch
        train_epoch_loss = running_train_loss / samples_processed
        self.train_epoch_losses.append(train_epoch_loss)


    def save_model(self):
        # Save model
        save_path = os.path.join(self.params['save_path'], 'model.pt')
        op_state = {'state_dict': self.model.state_dict(), 'params' : self.params}
        torch.save(op_state, save_path)
        # Save train loss trajectory
        train_loss_path = os.path.join(self.params['save_path'], 'train_loss_trajectory.npy')
        np.save(train_loss_path, np.array(self.train_epoch_losses))
        # Save per-species loss trajectory

def launch_training_run(ovr):
    print(f"Process working directory: {os.getcwd()}")
    # setup:
    params = setup.get_default_params_train(ovr)
    params['save_path'] = os.path.join(params['save_base'], params['experiment_name'])
    if params['timestamp']:
        params['save_path'] = params['save_path'] + '_' + utils.get_time_stamp()
    os.makedirs(params['save_path'], exist_ok=False)

    # training data:
    dataset = datasets.get_train_data(params)
    params['input_dim'] = dataset.input_dim
    params['num_classes'] = dataset.num_classes
    params['class_to_taxa'] = dataset.class_to_taxa

    # train loader
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=4)

    # model:
    model = models.get_model(params)
    model = model.to(params['device'])

    # train:
    trainer = Trainer(model, train_loader, params)
    for epoch in range(0, params['num_epochs']):
        print(f'epoch {epoch+1}')
        params['epoch'] = epoch
        trainer.train_one_epoch()
    trainer.save_model()
