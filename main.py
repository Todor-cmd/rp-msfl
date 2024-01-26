from arguments import Arguments
from experiment import run_experiment


def main():

    
    args1 = Arguments()
    args2 = Arguments()
    args3 = Arguments()
    args4 = Arguments()
    args5 = Arguments()
    args6 = Arguments()
    args7 = Arguments()
    args9 = Arguments()
    args10 = Arguments()
    
    
    args1.num_attackers = 0
    args2.topology = "attack case 1"
    args3.aggregation = "FMes-trimmed-mean"
    args4.aggregation = "FMes-multi-krum"
    args5.aggregation = "FMes-bulyan"
    args6.aggregation = "FMes-dnc"
    args7.aggregation = "FMes-median"
    
    args9.aggregation = "FMes-dnc"
    args9.dataset = "cifar10"
    args9.batch_size = 250
    args9.fed_lr = 0.8
    args9.schedule = [1000, 1200]
    
    
  
    args = [args6, args6, args6, args9, args9, args9]
    
    for a in args:
        run_experiment(a)

if __name__ == '__main__':
    main()
