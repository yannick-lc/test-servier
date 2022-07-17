"""
Main module.
Contains entry point of the program.
"""

import argparse

from molecule.config import configuration, PATH_TO_DEFAULT_CONFIG_FILE
from molecule.prediction.model import FeatureType
from molecule.prediction.train_and_predict import train, predict, evaluate


def main() -> None:
    """
    Entry point of the program.
    Parse arguments and train, evaluate etc based on arguments.
    """
    parser = argparse.ArgumentParser(prog="servier",
                description="Train or apply a previously trained machine learning model " \
                            "to predict properties of molecules",
                epilog=f"Default paths can be edited in {PATH_TO_DEFAULT_CONFIG_FILE}")
    parser.add_argument("action",
                choices=["train", "evaluate", "predict"], metavar="action",
                help="{train,evaluate,predict} whether to train, evaluate, " \
                     "or use the model to make predictions")
    parser.add_argument("-f", "--features",
                choices=["morgan", "smile"], default="morgan",
                help="choose model based on Morgan fingerprints (default) " \
                     "or SMILE text as features")
    parser.add_argument("-d", "--dataset",
                help="path to input csv file used to train the model or make predictions "\
                     "(if not specified, default path to train dataset is " \
                     f"{configuration['default_path_to_train_dataset']})")
    parser.add_argument("-m", "--model",
                help="path to pretrained model if action is evaluate or predict, " \
                     "or path to save the model if action is train " \
                     f"(default path is {configuration['default_path_to_model_save']})")
    parser.add_argument("-o", "--output",
                help=f"path to save predictions if action is predict " \
                     f"(default path is {configuration['default_path_to_save_predictions']})")

    args = parser.parse_args()

    feature_type = FeatureType[args.features.upper()] # convert string to Enum type

    # Train
    if args.action == "train":
        train(feature_type, args.dataset, args.model)
    # Predict
    elif args.action == "predict":
        output_path = args.output
        if output_path is None: # use default path to save predictions
            output_path = configuration["default_path_to_save_predictions"]
        predict(feature_type, args.dataset, args.model, output_path)
    # Evaluate
    else:
        evaluate(feature_type, args.dataset, args.model)


if __name__ == "__main__":
    main()
