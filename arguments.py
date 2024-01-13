import torch

SEED = 1
torch.manual_seed(SEED)

class Arguments:
    def __init__(self):
        
        # Data-set
        self.available_datasets = ["cifar10", "fashionmnist"]
        self.dataset = self.available_datasets[0]
        self.val_size = 5000
        self.te_size = 5000 # Requirement: val_size + te_size = 10k
        
        # Model
        self.arch = "alexnet" # "alexnet"
        self.dev_type = "std" # "std"
        
        self.batch_size = 250
        self.epochs = 1500
        self.fed_lr = 0.5
        
        self.schedule = [1000, 1200]
        self.gamma = 0.5

        self.clients = 20
        
        # Attack
        self.num_attackers = 0
        self.attack = "min-max"
    

        # Aggregation/Defense
        self.aggregation = "Fedmes" # "Fedmes", "FMes-trimmed-mean", "FMes-krum", "FMes-multi-krum", "FMes-bulyan"

        # CUDA
        self.cuda = True
        self.parallel = False
        
        self.save_final_model = False
        
        # How many epochs before the results are saved, disable with 0
        self.batch_write = 0
        
        self.topology = "multi-cross" # "multi-cross", "multi-line


