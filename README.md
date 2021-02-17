# CS539-HW-2

### Learning Goal
Learn how to write the first part of an interpreter for the FOPPL. In particular
learn how to manipulate Daphne compiler abstract syntax tree and graphical model
outputs to sample from the prior, i.e. no conditioning (yet).


## Setup
**Note:** This code base was developed on Python3.7

Clone Daphne directly into this repo:
```bash
git clone git@github.com:plai-group/daphne.git
```
(To use Daphne you will need to have both a JVM installed and Leiningen installed)

```bash
pip3 install -r requirements.txt
```

## Usage
1. Change the daphne path in `evaluation_based_sampling.py` and run:
```bash
python3 evaluation_based_sampling.py
```

2. Change the daphne path in `graph_based_sampling.py` and run:
```bash
python3 evaluation_based_sampling.py
```
