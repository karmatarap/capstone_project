# capstone project

## Directory Structure
The folder structure follows the best practices suggested by [Data Science Cookiecutter](https://drivendata.github.io/cookiecutter-data-science/).

```
├── LICENSE
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.py           <- Make this project pip installable with `pip install -e`
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
│
└── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
```

## Experiment Tracking

Experiment tracking is performed using [Mlflow](https://mlflow.org/). This metadata is shared online via [Neptune](https://neptune.ai).

### Setting up mlflow

After installing the dependencies from requirements.txt, an environment variable must be defined to specify where to store the metadata. The full path has to be provided. 
Example of .bashrc

`export MLFLOW_URI="/home/karma/Projects/zambezi/capstone_project/mlruns"`

For more information on what can be logged, see the quickstart [here](https://www.mlflow.org/docs/latest/quickstart.html)
                   
### Setting up neptune

For neptune integration, the best practice is to store API keys as environmental variables in .bashrc also. Our neptune project name is `capstone-project/capstone`.

`export NEPTUNE_API_TOKEN='your_long_api_token'`

To sync your mlflow runs to neptune, run the following on the command line

`neptune mlflow --project capstone-project/capstone`

More information can be found [here](https://github.com/neptune-ai/neptune-mlflow)

## Metrics

The metrics class with a standard set of metrics has been defined in the utils folder. This is to allow both notebooks and scripts to access from a central location. 

To be able to import this module, we have to append the path to the sys path.

`
sys.path.append('../utils') # <-- relative path of utils library

from metrics import Metrics
`

This will be updated based on teams feedback and experimentation with mlflow


## Project Tracking

Weekly tasks are planned using critical path analysis that can be found [here](https://docs.google.com/spreadsheets/d/1rqE4yLXR02qDLuFRmBkCYsmpdwcgjuj8/edit?usp=sharing&ouid=112406226383179847866&rtpof=true&sd=true)
