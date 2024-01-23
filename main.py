from arguments import Arguments
from experiment import run_experiment


def main():

    
    args1 = Arguments()
  
    args = [args1]
    
    for a in args:
        run_experiment(a)

if __name__ == '__main__':
    main()
