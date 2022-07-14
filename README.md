# test-servier

![Molecules illustration](https://github.com/yannick-lc/test-servier/blob/main/data/images/banner.png)


Technical test for Servier

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

## How to run

Launch training of the model and display performance:
```bash
servier train
servier evaluate
```

You can similarly use keyword 'predict' instead of 'train'.

Use pre-trained model to make predictions on a dataset using the model based on SMILE text representations:
```bash
servier predict --dataset data/datasets/dataset_single_test.csv --predictions data/output/predictions_single_test.csv --model models/model_smile.pth --features smile
```

More information regarding the accepted input parameters can be obtained with
```bash
servier --help
```

Dockerization is coming soon.

## About the models

Two deep learning models are available:
- a 3-layer fully-connected network based on Morgan fingerprints of the molecules.
- a 2-layer LSTM based on the text representations (SMILE) of the molecules.

## Performance

Coming soon

## Further information

More information, figures and baselines regarding the exploration and training process are provided in the 3 notebooks in the 'notebooks' folder.