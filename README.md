# Test Servier

![Python version](https://img.shields.io/badge/python-v3.7-blue)
![Platform version](https://img.shields.io/badge/platform-linux-lightgrey)

![Molecules illustration](https://github.com/yannick-lc/test-servier/blob/main/data/images/banner.png)

**Author**: Yannick Le Cacheux

## Install locally

The easiest way to run the model is through a Docker container — see the *Install using Docker* section.
This section explains how to install Python dependencies to run and edit the code locally.

This project uses Python 3.7, and has been tested on Ubuntu 22.04 LTS.
*(Python 3.6 has reached end of security support in December 2021, Rdkit, PyTorch etc do not seem well supported with Python 3.6).*

**Recommended**: create virtual env with Python 3.7 using conda, and activate environment:
```bash
conda create -n servier python=3.7
conda activate servier
```

Install dependencies using pip and install local packages in edit mode:

```bash
pip install -r requirements/prod.txt
pip install -e .
```

If you also want to run the notebooks and modify the code:
```bash
pip install -r requirements/dev.txt
```

## Install using Docker

You can directly download and run a Docker container with the model(s) with:
```bash
docker run -it -v /path/to/data:/app/data yannicklc/servier:0.0.1 /bin/bash
```

Simply replace */path/to/data* with the path to the dataset on your local drive.
(A dataset split into training and testing sets is available in *data/*).

Alternatively, build the Docker image locally:
```bash
docker build -t servier:0.0.1 .
```

Once connected to the shell, steps to run the model are detailed in the following section.

## Run the model

Launch training of the model based on Morgan fingerprints* and display performance:
```bash
servier train
servier evaluate
```

*The model based on Morgan fingerprints is the fastest but not best-performing, see section *About the models*.*

You can similarly use keyword 'predict' instead of 'train'.

Further options are available. For instance, use a pre-trained model to make predictions on a dataset using the model based on SMILE text representations:
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

## Code and tests

The application specific code is contained within the package *molecule*, located in *src/molecule/*.

Unit tests are located in *tests/*, and are based on pytest.

To run tests:
```bash
pytest
```

Some of the tests (those in *\*_from_data.py* files) rely on sample data located in *tests/test_data/*.

**Warning**: due to possible differences in floating point precision, predictions done on a CPU with a model trained on GPU may differ slightly, which may result in some failed unit test(s).

*(#TODO: allow for small rounding errors in unit tests).*

## About the models

### Architecture

Two deep learning models are available:
- a 3-layer fully-connected network based on Morgan fingerprints of the molecules.
- a 2-layer RNN (GRU) based on the text representations (SMILE) of the molecules.

Models are relatively light-weight (up to a dozen of MB) so weights of pre-trained model have been included in *models/*.

Below is a brief overview of the architectures of the 2 models:

![Models illustration](https://github.com/yannick-lc/test-servier/blob/main/data/images/DNNs.png)

*Left: Fully connected model using Morgan fingerprint as features. Right: RNN model using features based on text representation.*

More information are provided in the notebooks (*notebooks/*) and the file defining the architecture of the models (*molecule/train/deep_architectures.py*).

### Performance

As explained in "*notebooks/1 - Exploration.ipynb*", since the dataset is imbalanced, accuracy is not a very good metrics. The default metrics used in this project is thus the Area Under Curve (AUC) of the Receiver Operation Charactistic (ROC) curve.

Below are the ROC curves and AUCs of the 2 models above:

![ROC curves](https://github.com/yannick-lc/test-servier/blob/main/data/images/ROCs.png)

Here are a few more details on the performance of the better-performing model (the GRU-based model), for instance the Precision-Recall curve:

![PR curve](https://github.com/yannick-lc/test-servier/blob/main/data/images/PR2.png)

As evidenced by the top-left of the PR curve (as well as the bottom left of the ROC curve), for a certain decision threshold, the molecules predicted as having property P1 do have this property with near certainty.

A threshold of 0.95 is enough to obtain 100% precision on the test set used. Here is the confusion matrix obtained with such a threshold:

![Confusion matrix](https://github.com/yannick-lc/test-servier/blob/main/data/images/CM2.png)

More metrics, figures and baselines regarding the exploration, training and evaluation process are provided in the 3 notebooks in *notebooks/*.

A few ideas which may lead to better performance given more time are suggested at the end of the third notebook.