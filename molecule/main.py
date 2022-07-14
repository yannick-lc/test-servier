import argparse
from pathlib import Path

from molecule.model import FeatureType, Model
from molecule.dataset import Dataset


def get_project_root_directory() -> Path:
    """Return project root directory"""
    return Path(__file__).parent.parent

def train_model(feature_type: FeatureType, path_dataset: str, path_model: str=None) -> Model:
    """
    Train and return model
    Also sometimes saves the model
    """
    model = Model(feature_type)
    dataset = Dataset(path_dataset)
    X, y = dataset.get_features(feature_type), dataset.labels
    model.fit(X, y, validation_set_ratio=0.)
    if path_model is not None:
        pass
    return model

def main() -> None:
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