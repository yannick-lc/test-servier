import argparse

def main():
    """
    Entry point of the program.
    Parse arguments and call relevant parts based on arguments.
    """
    parser = argparse.ArgumentParser(prog='servier', description='Predict properties of molecules.')
    parser.add_argument('action',
        choices=['train', 'evaluate', 'predict'], metavar='action',
        help='Whether to train, evaluate, or use the model to make predictions.')
    # parser.print_help()

    args = parser.parse_args()

    if args.action == 'train':
        print('Train')
    elif args.action == 'evaluate':
        print('Evaluate')
    else: # predict
        print('Predict')

if __name__ == "__main__":
    main()