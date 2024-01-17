from __future__ import print_function

import os
import pickle
import csv

import numpy as np

from topology import Topology

import aggregations
from cifar10.cifar10_normal_train import *

from cifar10.cifar10_models import *

from cifar10.sgd import SGD

from arguments import Arguments

from attack import our_attack_dist
from client import Client

args = Arguments()

# X, Y = load_data(args)
data = pickle.load(open('./cifar10_data_ind.pkl', 'rb'))
X = data[0]
Y = data[1]
# data loading

nusers = args.clients
user_tr_len = 2500

total_tr_len = user_tr_len * nusers
val_len = 5000
te_len = 5000

print('total data len: ', len(X))

if not os.path.isfile('./cifar10_shuffle.pkl'):
    all_indices = np.arange(len(X))
    np.random.shuffle(all_indices)
    pickle.dump(all_indices, open('./cifar10_shuffle.pkl', 'wb'))
else:
    all_indices = pickle.load(open('./cifar10_shuffle.pkl', 'rb'))

total_tr_data = X[:total_tr_len]
total_tr_label = Y[:total_tr_len]

val_data = X[total_tr_len:(total_tr_len + val_len)]
val_label = Y[total_tr_len:(total_tr_len + val_len)]

te_data = X[(total_tr_len + val_len):(total_tr_len + val_len + te_len)]
te_label = Y[(total_tr_len + val_len):(total_tr_len + val_len + te_len)]

total_tr_data_tensor = torch.from_numpy(total_tr_data).type(torch.FloatTensor)
total_tr_label_tensor = torch.from_numpy(total_tr_label).type(torch.LongTensor)

val_data_tensor = torch.from_numpy(val_data).type(torch.FloatTensor)
val_label_tensor = torch.from_numpy(val_label).type(torch.LongTensor)

te_data_tensor = torch.from_numpy(te_data).type(torch.FloatTensor)
te_label_tensor = torch.from_numpy(te_label).type(torch.LongTensor)

print('total tr len %d | val len %d | test len %d' % (
    len(total_tr_data_tensor), len(val_data_tensor), len(te_data_tensor)))

# ==============================================================================================================

user_tr_data_tensors = []
user_tr_label_tensors = []

for i in range(nusers):
    user_tr_data_tensor = torch.from_numpy(total_tr_data[user_tr_len * i:user_tr_len * (i + 1)]).type(torch.FloatTensor)
    user_tr_label_tensor = torch.from_numpy(total_tr_label[user_tr_len * i:user_tr_len * (i + 1)]).type(
        torch.LongTensor)

    user_tr_data_tensors.append(user_tr_data_tensor)
    user_tr_label_tensors.append(user_tr_label_tensor)
    print('user %d tr len %d' % (i, len(user_tr_data_tensor)))

batch_size = 165
resume = 0
schedule = [700, 1000, 1200, 1400]
nbatches = user_tr_len // batch_size

gamma = .5
opt = 'vgg'
fed_lr = 0.8
criterion = nn.CrossEntropyLoss()
use_cuda = torch.cuda.is_available()

aggregation = 'average' # 'median', 'average', 'trimmed-mean', 'krum', 'multi-krum', 'bulyan', 'dnc'
candidates = []

dev_type = 'std'
z_values = {3: 0.69847, 5: 0.7054, 8: 0.71904, 10: 0.72575, 12: 0.73891}
n_attackers = [0]

arch = 'alexnet'
chkpt = './' + aggregation

results = []

topology = Topology('multi-cross')  # 'multi-cross', 'multi-line'

# Keep track of the clients each server reaches
server_control_dict = topology.get_server_control()
# Keep track of weights where keys refer to server and lists contain weights associated with overlap of clients with
# same index in server_control_dict.
overlap_weight_index = topology.get_overlap_index()

for n_attacker in n_attackers:
    epoch_num = 0
    best_global_acc = 0
    best_global_te_acc = 0
    best_global_loss = 100
    best_global_te_loss = 100
    last_pos_step_epoch = 0

    clients = []

    # Create clients
    print("creating %d clients" % (nusers))
    for i in range(nusers):
        if i >= n_attacker:
            clients.append(Client(i, False, arch, fed_lr, criterion))
        else:
            clients.append(Client(i, True, arch, fed_lr, criterion))

    torch.cuda.empty_cache()
    r = np.arange(user_tr_len)
    while epoch_num <= args.epochs:
        user_grads = []

        # Shuffle data for each epoch except the first one
        if not epoch_num and epoch_num % nbatches == 0:
            np.random.shuffle(r)
            for i in range(nusers):
                user_tr_data_tensors[i] = user_tr_data_tensors[i][r]
                user_tr_label_tensors[i] = user_tr_label_tensors[i][r]
                
        # Iterate over users, excluding attackers
        for i in range(n_attacker, nusers):
            # Get a batch of inputs and targets for the current user
            inputs = user_tr_data_tensors[i][
                     (epoch_num % nbatches) * batch_size:((epoch_num % nbatches) + 1) * batch_size]
            targets = user_tr_label_tensors[i][
                      (epoch_num % nbatches) * batch_size:((epoch_num % nbatches) + 1) * batch_size]

            param_grad = clients[i].train(inputs, targets)

            # Concatenate user gradients to the list
            user_grads = param_grad[None, :] if len(user_grads) == 0 else torch.cat((user_grads, param_grad[None, :]),
                                                                                    0)

        # Store the collected user gradients as malicious gradients
        malicious_grads = user_grads.cuda()
        # Update learning rate of clients
        for client in clients:
            client.update_learning_rate(epoch_num, schedule, gamma)

        # Add the parameters of the malicious clients depending on attack type
        if n_attacker > 0:  
            agg_grads = torch.mean(malicious_grads, 0)
            malicious_grads = our_attack_dist(malicious_grads, agg_grads, n_attacker, dev_type=dev_type)

        if not epoch_num:
            print(malicious_grads.shape)

        # Store of aggregate gradients for servers
        server_aggregates = []
        
        # For each server find the aggregate gradients of clients it reaches
        for server in server_control_dict.keys():
            clients_in_reach = []
            for clientId in server_control_dict[server]:
                clients_in_reach.append(malicious_grads[clientId])

            agg_grads = []
            stacked_clients_in_reach = torch.stack(clients_in_reach, dim=0)

            if aggregation == 'median':
                agg_grads = aggregations.fedmes_median(stacked_clients_in_reach, overlap_weight_index[server])

            elif aggregation == 'average':
                agg_grads = aggregations.fedmes_mean(stacked_clients_in_reach, overlap_weight_index[server])

            elif aggregation == 'trimmed-mean':
                agg_grads = aggregations.fedmes_tr_mean_v2(stacked_clients_in_reach, 2, overlap_weight_index[server])

            elif aggregation == 'krum':
                agg_grads = aggregations.fedmes_multi_krum(stacked_clients_in_reach, 2, overlap_weight_index[server])

            elif aggregation == 'multi-krum':
                agg_grads = aggregations.fedmes_multi_krum(
                    stacked_clients_in_reach,
                    n_attacker,
                    overlap_weight_index[server],
                    True)

            elif aggregation == 'bulyan':
                agg_grads = aggregations.fedmes_bulyan(stacked_clients_in_reach, 2, overlap_weight_index[server])
                
            elif aggregation == 'dnc':
                agg_grads = aggregations.fmes_dnc(stacked_clients_in_reach, 2, overlap_weight_index[server], 1, 1, 10000)

            server_aggregates.append(agg_grads)

        del user_grads

        del malicious_grads

        server_aggregates = torch.stack(server_aggregates, dim=0)

        # Update models of clients taking into account the servers that reach it
        for client in clients:
            if client.client_idx in topology.get_set_0():
                client.update_model(server_aggregates[0])
            elif client.client_idx in topology.get_set_1():
                client.update_model(server_aggregates[1])
            elif client.client_idx in topology.get_set_2():
                client.update_model(server_aggregates[2])
            elif client.client_idx in topology.get_set_0_1():
                comb_0_1 = torch.mean(server_aggregates[:2], dim=0)
                client.update_model(comb_0_1)
            elif client.client_idx in topology.get_set_1_2():
                comb_1_2 = torch.mean(server_aggregates[1:], dim=0)
                client.update_model(comb_1_2)
            elif client.client_idx in topology.get_set_0_2():
                comb_0_2 = torch.mean(server_aggregates[[0, 2]], dim=0)
                client.update_model(comb_0_2)
            elif client.client_idx in topology.get_set_0_1_2():
                comb_all = torch.mean(server_aggregates, dim=0)
                client.update_model(comb_all)

        val_loss, val_acc = test(val_data_tensor, val_label_tensor, clients[0].fed_model, criterion, use_cuda)
        te_loss, te_acc = test(te_data_tensor, te_label_tensor, clients[0].fed_model, criterion, use_cuda)

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

        # Print result of training in epoch
        print("Acc: " + str(val_acc) + " Loss: " + str(val_loss))
        results.append([val_acc, val_loss])
        if epoch_num % 10 == 0 or epoch_num == args.epochs - 1:
            print('%s: at %s n_at %d e %d fed_model val loss %.4f val acc %.4f best val_acc %f te_acc %f' % (
                aggregation, args.attack, n_attacker, epoch_num, val_loss, val_acc, best_global_acc,
                best_global_te_acc))

        if val_loss > 10:
            print('val loss %f too high' % val_loss)
            break

        if (epoch_num - last_pos_step_epoch) > 300:
            print('model convergence, last positive step in epoch %f' % last_pos_step_epoch)
            break

        epoch_num += 1

print(results)
print("Saving to results.csv")

with open("results/results.csv", 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)

    # writing the data rows
    csvwriter.writerows(results)
