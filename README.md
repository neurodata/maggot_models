maggot_models
==============================

Modeling the Drosophila larva connectome

Installation and setup
--------

If you are new to Python development, my recommended practives are [here](https://github.com/bdpedigo/giskard/blob/master/python_start.md) 

Currently, the recommended setup is to use conda or miniconda to create a virtual
environment: https://docs.conda.io/en/latest/miniconda.html

conda environments are recommended. To create a new conda environment for this project, 
navigate to this directory in a terminal and run

## Recommended setup for development 

Fork the repo

- Hit "Fork" in the upper left corner on Github


Clone the repo

- Click clone or download
- Copy the `link` provided
- From the folder where you would like this repo to live, 
    - (Recommended) Clone just the most recent version of master:
    
        ``git clone --depth 1 -b master https://github.com/neurodata/maggot_models.git``
    
    - To clone the whole repo and all history (large) do
    
        ``git clone {link}``

Add conda-forge as a channel if you haven't already

``conda config --append channels conda-forge``

``conda config --set channel_priority strict``

Create a new conda environment

``conda create -n {insert name} python==3.7``


To verify that the environment was created run 
``conda info --envs``

Activate the environment

``conda activate {insert name}``



Using this package is also possible with ``pip`` and a virtual environment manager. 
If you would like to use ``pip`` please contact @bdpedigo and I can make sure the pip
``requirements.txt`` is up to date (it isn't right now)


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
