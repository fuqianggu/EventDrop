import argparse
from os.path import dirname
import torch
import torchvision
import os
import numpy as np
import tqdm
import time
from datetime import date
from utils.model import Classifier
from torch.utils.tensorboard import SummaryWriter
from utils.loader import Loader
from utils.loss import cross_entropy_loss_and_accuracy
from utils.dataset import NCaltech101, NCars, NMNIST
import random

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(100)
torch.cuda.manual_seed_all(100)
np.random.seed(100)
random.seed(100)

names = '_NCars_EventDrop_'
num_run = 5

# representation using ESt
representations = ['EventFrame', 'EventCount', 'VoxGrid', 'EST'] 
in_channels = [1, 2, 1, 2]
voxel_dims = [(120, 100), (120, 100), (9, 120, 100), (9, 120, 100)]  # for NCars
#voxel_dims = [(180, 240), (180, 240), (9, 180, 240), (9, 180, 240)] #for NCaltech101
res_model = 'resnet34' # 'resnet18' 
num_classes = 2
options = len(representations)

def FLAGS():
    parser = argparse.ArgumentParser("""Train classifier using a learnt quantization layer.""")

    # training / validation dataset
    parser.add_argument("--validation_dataset", default="N-Cars/validation/") #, required=True)
    parser.add_argument("--training_dataset", default="N-Cars/training/") #, required=True)
    parser.add_argument("--testing_dataset", default="N-Cars/testing/")

    # logging options
    parser.add_argument("--log_dir", default="log/temp_NCars") #, required=True)

    # loader and device options
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--save_every_n_epochs", type=int, default=5)

    flags = parser.parse_args()

#    assert os.path.isdir(dirname(flags.log_dir)), f"Log directory root {dirname(flags.log_dir)} not found."
    assert os.path.isdir(flags.validation_dataset), f"Validation dataset directory {flags.validation_dataset} not found."
    assert os.path.isdir(flags.training_dataset), f"Training dataset directory {flags.training_dataset} not found."
    assert os.path.isdir(flags.testing_dataset), f"Testing dataset directory {flags.testing_dataset} not found."

    print(f"----------------------------\n"
          f"Model: {names}\n"
          f"Starting training with \n"
          f"num_epochs: {flags.num_epochs}\n"
          f"batch_size: {flags.batch_size}\n"
          f"device: {flags.device}\n"
          f"log_dir: {flags.log_dir}\n"
          f"training_dataset: {flags.training_dataset}\n"
          f"validation_dataset: {flags.validation_dataset}\n"
          f"testing_dataset: {flags.testing_dataset}\n"
          f"----------------------------")

    return flags

if __name__ == '__main__':
    flags = FLAGS()

    # datasets, add augmentation to training set
    training_dataset = NCars(flags.training_dataset, augmentation=True)
    validation_dataset = NCars(flags.validation_dataset)
    testing_dataset = NCars(flags.testing_dataset)

    # construct loader, handles data streaming to gpu
    training_loader = Loader(training_dataset, flags, device=flags.device)
    validation_loader = Loader(validation_dataset, flags, device=flags.device)
    testing_loader = Loader(testing_dataset, flags, device=flags.device)

    # for multiple runs
    best_testing_acc = torch.zeros(num_run)
    best_testing_epoch = torch.zeros(num_run)
    best_validation_acc = torch.zeros(num_run)
    training_acc_list = list([])
    testing_acc_list = list([])
    validation_acc_list = list([])
    training_loss_list = list([])
    validation_loss_list = list([])
    testing_loss_list = list([])
    net_list = list([])
    for run in range(num_run * options):
        fold = run % num_run
        option = int(run/num_run)
        representation = representations[option]
        in_channel = in_channels[option]
        voxel_dim = voxel_dims[option]

        if (run % num_run) == 0:
            best_testing_acc = torch.zeros(num_run)
            best_testing_epoch = torch.zeros(num_run)
            best_validation_acc = torch.zeros(num_run)
            training_acc_list = list([])
            testing_acc_list = list([])
            validation_acc_list = list([])
            training_loss_list = list([])
            validation_loss_list = list([])
            testing_loss_list = list([])
            net_list = list([])

        # model, and put to device
        model = Classifier(voxel_dimension=voxel_dim, in_channel=in_channel, num_classes= num_classes, representation=representation, res_model=res_model)
        model = model.to(flags.device)

        # optimizer and lr scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
        iteration = 0
        min_validation_loss = 1000
        max_validation_accuracy = 0   

        tr_loss = torch.zeros(flags.num_epochs)
        tr_acc = torch.zeros(flags.num_epochs)
        te_loss = torch.zeros(flags.num_epochs)
        te_acc = torch.zeros(flags.num_epochs)
        val_loss = torch.zeros(flags.num_epochs)
        val_acc = torch.zeros(flags.num_epochs)
        best_model_dict = model.state_dict()
        
        best_test_acc = 0
        best_test_epoch = 0
        for i in range(flags.num_epochs):
            print(f"For representation: {representation:s}, at run [{run:3d}/{num_run:3d}], epoch [{i:3d}/{flags.num_epochs:3d}]\n")

            # Training
            sum_accuracy = 0   
            sum_loss = 0     
            model = model.train()   
            start_time = time.time()      
            for events, labels in training_loader: #tqdm.tqdm(training_loader):       
                optimizer.zero_grad()
                pred_labels = model(events)
                loss, accuracy = cross_entropy_loss_and_accuracy(pred_labels, labels)
                loss.backward()
                optimizer.step()
                sum_accuracy += accuracy
                sum_loss += loss
                iteration += 1
            if (i > 100) & (i % 10 == 9):
                lr_scheduler.step()

            training_loss = sum_loss.item() / len(training_loader)
            training_accuracy = sum_accuracy.item() / len(training_loader)
            print(names)
            print(f"Iteration {iteration:5d}  Loss {training_loss:.4f}  Accuracy {training_accuracy:.4f} Time elasped {time.time() - start_time:.4f}")
            tr_loss[i] = training_loss
            tr_acc[i] = training_accuracy

            # validation
            sum_accuracy = 0
            sum_loss = 0
            model = model.eval()

            for events, labels in validation_loader: #tqdm.tqdm(validation_loader):
                with torch.no_grad():
                    start_time = time.time()
                    pred_labels = model(events)
                    loss, accuracy = cross_entropy_loss_and_accuracy(pred_labels, labels)

                sum_accuracy += accuracy
                sum_loss += loss

            validation_loss = sum_loss.item() / len(validation_loader)
            validation_accuracy = sum_accuracy.item() / len(validation_loader)
            print(f"Validation Loss {validation_loss:.4f}  Accuracy {validation_accuracy:.4f}")

            val_loss[i] = validation_loss
            val_acc[i] = validation_accuracy

            # Testing
            sum_accuracy = 0
            sum_loss = 0
            model = model.eval()
            for events, labels in testing_loader: #tqdm.tqdm(testing_loader):
                with torch.no_grad():
                    pred_labels = model(events)
                    loss, accuracy = cross_entropy_loss_and_accuracy(pred_labels, labels)
                sum_accuracy += accuracy
                sum_loss += loss
            testing_loss = sum_loss.item() / len(testing_loader)
            testing_accuracy = sum_accuracy.item() / len(testing_loader)
            te_loss[i] = testing_loss
            te_acc[i] = testing_accuracy

            if validation_accuracy > max_validation_accuracy: 
                best_testing_acc[fold] = testing_accuracy
                best_test_acc = testing_accuracy
                best_test_epoch = i
                best_testing_epoch[fold] = i
                max_validation_accuracy = validation_accuracy
                best_validation_acc[fold] = validation_accuracy
                best_model_dict = model.state_dict()
            
            print(f"Testing Loss {testing_loss:.4f} Best Accuracy {best_test_acc:.4f} at Epoch {best_test_epoch:3d}")

            print(f"----------------------------\n")

        # save statistics
        training_loss_list.append(tr_loss)
        training_acc_list.append(tr_acc)
        testing_loss_list.append(te_loss)
        testing_acc_list.append(te_acc)
        validation_loss_list.append(val_loss)
        validation_acc_list.append(val_acc)
        net_list.append(best_model_dict)
        if ((run + 1) % num_run) == 0:
            state = {
                    'net_list': net_list,
                    'best_testing_acc': best_testing_acc,
                    'best_testing_epoch': best_testing_epoch,
                    'best_validation_acc': best_validation_acc,
                    'num_epochs': flags.num_epochs,
                    'training_acc_list': training_acc_list,
                    'training_loss_list': training_loss_list,
                    'testing_loss_list': testing_loss_list,
                    'testing_acc_list': testing_acc_list,
                    'validation_loss_list': validation_loss_list,
                    'validation_acc_list': validation_acc_list,
                    }
            dateStr = date.today().strftime("%Y%m%d")

            if not os.path.isdir('log_data'):
                os.mkdir('log_data')
            torch.save(state, './log_data/' + dateStr + names + '_' + res_model + '_' + representation + '_runs_' +  str(num_run) + '.t7')
            print('Avg testing acc: %f, std: %f: ' % (torch.mean(state['best_testing_acc']), torch.std(state['best_testing_acc'])))

