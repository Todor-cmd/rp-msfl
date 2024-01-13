from __future__ import print_function

import csv
from federated_learning import FederatedLearning

import torch.nn as nn

from cifar10.cifar10_normal_train import *

from utils.misc import get_time_string
from arguments import Arguments
from data import tensor_loader

from client import Client

def create_clients(args, loaders, criterion):
    clients = []
    for i in range(args.clients):
        if i >= args.num_attackers:
            clients.append(Client(i, False, args, loaders[i], criterion))
        else:
            clients.append(Client(i, True, args, loaders[i], criterion))
    return clients

def run_experiment(args):
    # Experiment setup
    loaders, validation, test_set = tensor_loader(args)
    criterion = nn.CrossEntropyLoss()

    # Create a file to write to
    chkpt = './' + args.topology + '-' + args.aggregation

    results_file = './results/' + get_time_string()       + '-'\
                                + args.dataset            + '-'\
                                + args.topology           + '-'\
                                + args.aggregation        + '-'\
                                + str(args.epochs)        +'e-'\
                                + str(args.num_attackers) + 'att-'\
                                + args.attack             + '-'\
                                + args.arch               + '.csv'
    results = []
    if args.batch_write:
        print('Results will be saved in: ' + results_file)
        with open(results_file, 'w') as csvfile:
            csv.writer(csvfile).writerow(['Accuracy', 'Loss'])

    # Variables to keep track of performance
    epoch_num = 0
    best_global_acc = 0
    best_global_te_acc = 0
    best_global_loss = 100
    best_global_te_loss = 100
    last_pos_step_epoch = 0

    # Create clients
    clients = create_clients(args, loaders, criterion)
    
    # Instantiate Federated Learning
    federated_learning = FederatedLearning(args, clients)
    
    # Empty cuda cache if neccessary
    if args.cuda:
        torch.cuda.empty_cache()

    # Start training model
    while epoch_num < args.epochs:
        # Get Global Model after epoch of training
        global_model = federated_learning.run_trainning_epoch(epoch_num)
    
        # Perform Validation test
        val_loss, val_acc = test(validation[0], validation[1], global_model.fed_model, criterion, args.cuda)
        te_loss, te_acc = test(test_set[0], test_set[1], global_model.fed_model, criterion, args.cuda)

        # Check if best accuracy has changed
        is_best_acc = best_global_acc < val_acc

        best_global_acc = max(best_global_acc, val_acc)

        if is_best_acc:
            best_global_te_acc = te_acc

        # Check if best loss changed
        is_best_loss = best_global_loss > val_loss

        best_global_loss = min(best_global_loss, val_loss)

        if is_best_loss:
            best_global_te_loss = te_loss

        # mark epoch in which global model improved in loss or accuracy
        if is_best_loss or is_best_acc:
            last_pos_step_epoch = epoch_num
        
        print("Acc: " + str(val_acc) + " Loss: " + str(val_loss))
        results.append([val_acc, val_loss])
        if epoch_num % 10 == 0 or epoch_num == args.epochs - 1:
            print('%s, %s: at %s n_at %d e %d fed_model val loss %.4f val acc %.4f best val_acc %f' % (
                args.topology, args.aggregation, args.attack, args.num_attackers, epoch_num, val_loss, val_acc, best_global_acc))

        if args.batch_write and epoch_num % args.batch_write == 0:
            print('Writing next batch of results at e ' + str(epoch_num))
            with open(results_file, 'a') as csvfile:
                csv.writer(csvfile).writerows(results)
                results.clear()

        # End experiment if loss gets too large
        if val_loss > 10:
            print('val loss %f too high' % val_loss)
            break
        
        # End experiment if model converges
        if (epoch_num - last_pos_step_epoch) > 200:
            print('model convergence, last positive step in epoch %f' % last_pos_step_epoch)
            break

        epoch_num += 1
    
    # Save to results file at the end of experiment 
    print('Saving to ' + results_file)
    if args.batch_write:
        with open(results_file, 'a') as csvfile:
            csv.writer(csvfile).writerows(results)
    else:
        with open(results_file, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Accuracy', 'Loss'])
            csvwriter.writerows(results)
    if args.save_final_model:
        weights_file = './pretrained/' + get_time_string() + '-'\
                                       + args.arch         + '-'\
                                       + args.dataset      + '-'\
                                       + str(args.epochs)  + '.zip'
        print('Saving weights to ' + weights_file)
        torch.save(global_model.fed_model.state_dict(), weights_file)
    print('Done')