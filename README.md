# A Weighted Matching Model for Georeferenced Microsimulations
This repository contains all relevant code for my MSc Thesis.

## Installation
The project is created with Python 3.
Use the package manager [pip](https://pypi.org/project/pip/) to install all required modules

```bash
pip install -r requirements.txt
```
The optimization problems are built programmed using the solver [Gurobi](https://www.gurobi.com/) and a valid Gurobi license is required to run the models.

## Structure and Files
### Code
A short overview of the provided code files.

#### ilp_formulation.py
Runs the optimization in ILP formulation using the Gurobi Solver. See Equation 6.9 in Section 6.2 Integer Linear Programming.

#### matching_formulation.py
Runs the optimization in form of a Maximum Matching using the Gurobi Solver. See Equation 6.18 in Section 6.3 Matching Formulation.

#### simulated_annealing.py
Runs the Simulated Annealing algorithm on a given dataset. See Section 7.2 Simulated Annealing.

#### GMM_data.py
Generate data from a Distribution-Based Urban Model, i.e., using GMMs. It follows the method outlined in Section 5.3 Urban Model using GMMs.

#### get_GMM_input.py
Guides the user through generating all necessary input paramters for GMM data creation for any number of clusters.

#### binary_search_variation.py
Calculates the best penalty weight parameters for all given penalty weight functions for the matching formulation, specific to one dataset and its associated model.

#### get_input.py
Contains functions to calculate and read variables from input data to pass onto the solvers.

#### get_files.py
Contains two functions to return specific subdirectories or files in the directory.

### Data
The two setup files used to generate the sampled datasets of three and five clusters for the Distribution-Based Urban Model are provided. These can be read using the `GMM_data.py` script.

## Abstract

## License
