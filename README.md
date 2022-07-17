# Test Servier

![Python version](https://img.shields.io/badge/python-v3.7-blue)
![Platform version](https://img.shields.io/badge/platform-linux-lightgrey)

![Molecules illustration](https://github.com/yannick-lc/test-servier/blob/main/data/images/banner.png)

**Author**: Yannick Le Cacheux

## How to install

Requires Python 3.7
*(Python 3.6 has reached end of security support in December 2021, Rdkit, PyTorch etc do not seem well supported with Python 3.6).*

Recommended: create virtual env with Python 3.7 using conda, and activate environment:
```bash
conda create -n servier python=3.7
conda activate servier
```

Install dependencies using pip and install local packages:

```bash
pip install -r requirements.txt
```

Tested on Ubuntu 22.04 LTS.

## How to run

Launch training of the model and display performance:
```bash
servier train
servier evaluate
```

You can similarly use keyword 'predict' instead of 'train'.

Use pre-trained model to make predictions on a dataset using the model based on SMILE text representations:
```bash
servier predict --dataset data/datasets/dataset_single_test.csv \
--output data/output/predictions_single_test.csv \
--model models/model_smile.pth \
--features smile
```

More information regarding the accepted input parameters can be obtained with:
```bash
servier --help
```

## Tests

Unit tests are located in *tests/*, and are based on pytest.

To run tests:
```bash
pytest
```

Some of the tests (those in files postfixed with "*from_data.py*") rely on sample data located in *tests/test_data/*.

## About

### Models

Two deep learning models are available:
- a 3-layer fully-connected network based on Morgan fingerprints of the molecules.
- a 2-layer RNN (GRU) based on the text representations (SMILE) of the molecules.

Models are relatively light-weight (up to a dozen of MB) so weights of pre-trained model have been included in *models/*.

Below is a brief overview of the architectures of the 2 models:
![Models illustration](https://github.com/yannick-lc/test-servier/blob/main/data/images/DNNs.png)

*Left: Fully connected model used on Morgan fingerprint features. Right: RNN used on features based on text representation.*

More information are provided in the notebooks (*notebooks/* folder) and the file defining the architecture of the models (*molecule/prediction/deep_architectures.py*).

### Performance

As detailed in "*notebooks/1 - Exploration*", since the dataset is imbalanced, accuracy is not a good metrics. The "default" metrics used is thus the Area Under Curve (AUC) of the Receiver Operation Charactistic (ROC) curve.

Below are the ROC curves and AUCs of the 2 models above:
![ROC curves](https://github.com/yannick-lc/test-servier/blob/main/data/images/ROCs.png)

More metrics, figures and baselines regarding the exploration, training and evaluation process are provided in the 3 notebooks in *notebooks/*.

## To do

Things to do if I find some time:

- More documentation
- Improve model performance, see the end of "*notebooks/3 - Model 2 - SMILE text representation.ipynb*" for more details
- Create separate requirements for production (train or run model from command line) and developement (explore baseline in Jupyter notebooks etc)
- Better unit test coverage
- Dockerization
