import argparse

def main():
    """
    Entry point of the program.
    Parse arguments and call relevant parts based on arguments.
    """
    parser = argparse.ArgumentParser(prog='servier', description='Predict properties of molecules.')
    parser.add_argument('action',
        choices=['train', 'evaluate', 'predict'], metavar='action',
        help='[train|evaluate|predict] whether to train, evaluate, or use the model to make predictions.')
    parser.add_argument('--dataset',
        default='models/dataset_single_train.csv',
        help='path to csv file used to train the model or make predictions')
    parser.add_argument('--features',
        choices=['morgan', 'smile'], default='morgan',
        help='choose model using Morgan fingerprints (default) or SMILE bag-of-words as features.')
    parser.add_argument('--model',
        default='models/morgan_pretrained.zip',
        help='path to state dict to load pretrained model if action is evaluate or predict.')
    # parser.print_help()

    args = parser.parse_args()
    print(args)

    if args.action == 'train':
        print('Train')
    elif args.action == 'evaluate':
        print('Evaluate')
    else: # predict
        print('Predict')

if __name__ == "__main__":
    main()