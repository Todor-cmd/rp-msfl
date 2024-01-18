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
    
    args1.topology = "attack case 1"
    args2.topology = "attack case 2"
    
    args3.aggregation = "FMes-trimmed-mean"
    args4.aggregation = "FMes-krum"
    args5.aggregation = "FMes-multi-krum"
    args6.aggregation = "FMes-median"
    args7.aggregation = "FMes-bulyan"
    args8.aggregation = "FMes-dnc"
    
    args = [args1, args2, args3, args4, args5, args6, args7, args8]
    for a in args:
        run_experiment(a)

if __name__ == '__main__':
    main()
