from __future__ import print_function

import aggregations
import attack

from topology import Topology

from cifar10.cifar10_normal_train import *

from arguments import Arguments

class FederatedLearning:
    def __init__(self, args, clients):
        self.args = args
        self.clients = clients
        
        # Create the network topology         
        self.topology = Topology(args.topology) 
        # Keep track of the clients each server reaches
        self.server_control_dict = self.topology.get_server_control()
        # Keep track of weights where keys refer to server and lists contain weights associated with overlap of clients with
        # same index in server_control_dict.
        self.overlap_weight_index = self.topology.get_overlap_index()


    def run_trainning_epoch(self, epoch_num):
        # Store benign client gradients
        user_grads = []

        # Iterate over users, excluding attackers and train their models
        for i in range(self.args.num_attackers, len(self.clients)):
            param_grad = self.clients[i].train()
            user_grads = param_grad[None, :] if len(user_grads) == 0 else torch.cat((user_grads, param_grad[None, :]),
                                                                                    0)

        # Store the collected user gradients as malicious gradients
        malicious_grads = user_grads

        
        # Update learning rate of clients
        for client in self.clients:
            client.update_learning_rate(epoch_num, self.args.schedule, self.args.gamma)

        # Add the parameters of the malicious clients depending on attack type
        if self.args.num_attackers > 0:
            agg_grads = torch.mean(malicious_grads, 0)
            malicious_grads = attack.min_max(malicious_grads, agg_grads, self.args.num_attackers, dev_type=self.args.dev_type)
        
        # Store of aggregate gradients for servers
        server_aggregates = []

        # For each server find the aggregate gradients of clients it reaches
        for server in self.server_control_dict.keys():
            clients_in_reach = []
            for clientId in self.server_control_dict[server]:
                clients_in_reach.append(malicious_grads[clientId])

            agg_grads = []
            stacked_clients_in_reach = torch.stack(clients_in_reach, dim=0)

            if self.args.aggregation == 'Fmes-median':
                agg_grads = aggregations.fedmes_median(stacked_clients_in_reach, self.overlap_weight_index[server])

            elif self.args.aggregation == 'Fedmes':
                agg_grads = aggregations.fedmes_mean(stacked_clients_in_reach, self.overlap_weight_index[server])

            elif self.args.aggregation == 'FMes-trimmed-mean':
                agg_grads = aggregations.fedmes_tr_mean_v2(stacked_clients_in_reach, 1, self.overlap_weight_index[server])

            elif self.args.aggregation == 'FMes-krum':
                agg_grads = aggregations.fedmes_multi_krum(stacked_clients_in_reach, 1, self.overlap_weight_index[server])

            elif self.args.aggregation == 'FMes-multi-krum':
                agg_grads = aggregations.fedmes_multi_krum(
                    stacked_clients_in_reach,
                    self.args.num_attackers,
                    self.overlap_weight_index[server],
                    True)

            elif self.args.aggregation == 'FMes-bulyan':
                agg_grads = aggregations.fedmes_bulyan(stacked_clients_in_reach, 1, self.overlap_weight_index[server])

            server_aggregates.append(agg_grads)

        del user_grads

        del malicious_grads

        server_aggregates = torch.stack(server_aggregates, dim=0)

        # Update models of clients taking into account the servers that reach it
        for client in self.clients:
            if client.client_idx in self.topology.get_set_0():
                client.update_model(server_aggregates[0])
            elif client.client_idx in self.topology.get_set_1():
                client.update_model(server_aggregates[1])
            elif client.client_idx in self.topology.get_set_2():
                client.update_model(server_aggregates[2])
            elif client.client_idx in self.topology.get_set_0_1():
                comb_0_1 = torch.mean(server_aggregates[:2], dim=0)
                client.update_model(comb_0_1)
            elif client.client_idx in self.topology.get_set_1_2():
                comb_1_2 = torch.mean(server_aggregates[1:], dim=0)
                client.update_model(comb_1_2)
            elif client.client_idx in self.topology.get_set_0_2():
                comb_0_2 = torch.mean(server_aggregates[[0, 2]], dim=0)
                client.update_model(comb_0_2)
            elif client.client_idx in self.topology.get_set_0_1_2():
                comb_all = torch.mean(server_aggregates, dim=0)
                client.update_model(comb_all)
        
        return self.clients[0]