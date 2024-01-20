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
    args8 = Arguments()
    args9 = Arguments()
    args10 = Arguments()
    
    args1.topology = "attack case 1"
    args2.aggregation = "FMes-trimmed-mean"
    args3.aggregation = "FMes-krum"
    args4.aggregation = "FMes-multi-krum"
    args5.aggregation = "FMes-bulyan"
    args6.aggregation = "FMes-dnc"
    args7.aggregation = "FMes-median"
    args8.num_attackers = 0
    
    
    args = [args1, args2, args3, args4, args5, args6, args7, args8, args9, args10]
    for a in args:
        run_experiment(a)

if __name__ == '__main__':
    main()
