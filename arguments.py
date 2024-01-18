import torch

SEED = 1
torch.manual_seed(SEED)

class Arguments:
    def __init__(self):
        
        # Data-set
        self.available_datasets = ["cifar10", "fashionmnist"]
        self.dataset = self.available_datasets[1]
        self.val_size = 5000
        self.te_size = 5000 # Requirement: val_size + te_size = 10k
        
        # Model
        self.arch = "vgg11" # "alexnet", "vgg11"
        self.dev_type = "std" # "std"
        
        self.batch_size = 150 #cifar10-vgg:165, fashionmnist-vgg: 150
        self.epochs = 1500
        self.fed_lr = 0.5 # cifar10: 0.8, fashionmnist: 0.5
        
        self.schedule = [600, 800] # cifar10-vgg: [800, 900, 980, 1000], fashionmnist-vgg: [600, 800]
        self.gamma = 0.5

        self.clients = 20
        
        # Attack
        self.num_attackers = 2
        self.attack = "min-max"
    

        # Aggregation/Defense
        self.aggregation = "Fedmes" # "Fedmes", "FMes-trimmed-mean", "FMes-krum", "FMes-multi-krum", "FMes-bulyan", "FMes-dnc", "FMes-median"

        # CUDA
        self.cuda = True
        self.parallel = True
        
        # Logging
        self.save_final_model = False
        
        # How many epochs before the results are saved, disable with 0
        self.batch_write = 0
        
        self.topology = "attack case 2" # "attack case 1", "attack case 2", "multi-line


