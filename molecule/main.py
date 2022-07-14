import argparse

from molecule.config import configuration
from molecule.prediction.model import FeatureType
from molecule.prediction.train_and_predict import train, predict, evaluate


def main() -> None:
    """
    Entry point of the program.
    Parse arguments and train, evaluate etc based on arguments.
    """
    parser = argparse.ArgumentParser(prog="servier", description="Predict properties of molecules.")
    parser.add_argument("action",
        choices=["train", "evaluate", "predict"], metavar="action",
        help="[train|evaluate|predict] whether to train, evaluate, or use the model to make predictions")
    parser.add_argument("--features", "-f",
        choices=["morgan", "smile"], default="morgan",
        help="choose model based on Morgan fingerprints (default) or SMILE text as features")
    parser.add_argument("--dataset", "-d",
        help="path to input csv file used to train the model or make predictions")
    parser.add_argument("--model", "-m",
        help="path to pretrained model if action is evaluate or predict, " \
                "or path to save model if action is train "\
                f"(if not specified, default path is {configuration['default_path_to_model_save']})")
    parser.add_argument("--output", "-o",
        help=f"path to save predictions if action is predict " \
                f"(default path is {configuration['default_path_to_save_predictions']})")
    # parser.print_help()

    args = parser.parse_args()
    print(args)

    feature_type = FeatureType[args.features.upper()]

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