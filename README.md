# test-servier
Technical test for Servier

**Author**: Yannick Le Cacheux

## How to run

Requires Python 3.7
(Python 3.6 has reached end of security support in December 2021, rdkit, PyTorch etc do not seem well supported with Python 3.6).

Recommended: create virtual env with Python 3.7 using conda, and activate environment:
```bash
conda create -n servier python=3.7
conda activate servier
```

Install dependencies using pip and install local packages:

```bash
pip install -r requirements.txt
pip install -e .
```

Launch training of the model:
```bash
servier train
```

You can similarly use keywords 'evaluate' and 'predict' instead of 'train'.

**Note**: nothing is implemented so far, so this won't do anything.

Dockerization is coming soon.