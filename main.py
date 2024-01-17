from arguments import Arguments
from experiment import run_experiment


def main():

    args1 = Arguments()
    args2 = Arguments()
    
    args1.topology = 'attack case 1'
    args2.topology = 'attack case 2'
    args = [args1, args2]
    for a in args:
        run_experiment(a)

if __name__ == '__main__':
    main()
