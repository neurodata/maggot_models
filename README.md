maggot_models
==============================

Modeling the Drosophila larva connectome

Installation and setup
--------

If you are new to Python development, my recommended practices are [here](https://github.com/bdpedigo/giskard/blob/master/python_start.md).

Currently, the recommended setup is to use conda or [miniconda](https://docs.conda.io/en/latest/miniconda.html) to create a virtual
environment.

The following will assume some familiarity with the command line, `git`, `Github`, and `conda`.

## Recommended setup for development 

### Fork the repo

- Hit "Fork" in the upper left corner on Github

### Create a folder for everything
- We'll be downloading a few repos, so I'd recommend having a dedicated folder to store them all. I'll refer to this folder as the "top-level directory" throughout.

### Clone the repo 

- Click "Clone or download" button in the top right
- Copy the <link> provided
- From the top-level directory, 
    - (Recommended) Clone just the most recent version of master:
    
        ``git clone --depth 1 -b master <link>``
    
    - OR to clone the whole repo and all history (large) do
    
        ``git clone <link>``

### Add conda-forge as a channel if you haven't already

``conda config --append channels conda-forge``

``conda config --set channel_priority strict``

### Create a new conda environment

``conda create -n {insert name} python==3.7``

### To verify that the environment was created run 
``conda info --envs``

### Activate the environment

``conda activate {insert name}``

### Install (most of) the dependencies

``conda install --file requirements.txt``

### Install conda-build

``conda install conda-build``

### Install GraSPy and Hyppo
Both of these packages, while available on `PyPI`, are still undergoing development. For now, I recommend installing these two packages via cloning and installing locally
rather than doing so via `pip` or similar. 

From the directory where you would like to store GraSPy and Hyppo, do

``git clone https://github.com/neurodata/graspy.git``

``cd graspy``

``conda develop .``

Rather than installing a "static" version of `GraSPy`, the command above will install the
package located in the current folder (in this case, `GraSPy`) while tracking any changes
made to those files. 

Similarly for Hyppo, navigate to the top level directory for this project, and do

``git clone https://github.com/neurodata/hyppo.git``

``cd hyppo``

``conda develop .``

Now you should have `GraSPy` and `Hyppo` installed, and if you need to get the latest version, 
this is as simple as

``cd graspy``

``git pull``

### Install `src`, a.k.a. the maggot_models package itself
From the top level directory for this project, do

``cd maggot_models``

``conda develop .``

### Get the data
The data is not yet public, talk to @bdpedigo about how to find and store the data.

### Store the data 
Place the `.graphml` files in `maggot_models/data/processed/<data>/`

### See if it all worked
From the top-level directory, do

``python`` <-- Start python in the terminal

``from src.data import load_metagraph`` <-- Import a function to load data

``mg = load_metagraph("G")`` <-- Load the "sum" graph

``print(mg.adj)`` <-- Access and print the adjacency matrix

``print(mg.meta.head())`` <-- Access and print the first few rows of metadata
<!-- Using this package is also possible with ``pip`` and a virtual environment manager. 
If you would like to use ``pip`` please contact @bdpedigo and I can make sure the pip
``requirements.txt`` is up to date (it isn't right now) -->


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │   |                      the creator's initials, and a short `-` delimited description, e.g.
    │   |                      `1.0-jqp-initial-data-exploration`.
    |   |
    |   └── outs           <- figures and intermediate results labeled by notebook that generated them.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── simulations        <- Synthetic data experiments and outputs
    │   └── runs           <- Sacred output for individual experiment runs
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
