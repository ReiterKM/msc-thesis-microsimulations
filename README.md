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

* ```ilp_formulation.py``` runs the optimization in ILP formulation using the Gurobi Solver. See Equation 6.9 in Section 6.2 Integer Linear Programming.
* ```matching_formulation.py``` runs the optimization in form of a Maximum Matching using the Gurobi Solver. See Equation 6.18 in Section 6.3 Matching Formulation.
* ```simulated_annealing.py``` runs the Simulated Annealing algorithm on a given dataset. See Section 7.2 Simulated Annealing.
* ```GMM_data.py``` generate data from a Distribution-Based Urban Model, i.e., using GMMs. It follows the method outlined in Section 5.3 Urban Model using GMMs.
* ```get_GMM_input.py``` guides the user through generating all necessary input paramters for GMM data creation for any number of clusters.
* ```binary_search_variation.py``` calculates the best penalty weight parameters for all given penalty weight functions for the matching formulation, specific to one dataset and its associated model.
* ```get_input.py``` contains functions to calculate and read variables from input data to pass onto the solvers.
* ```get_files.py``` contains two functions to return specific subdirectories or files in the directory.

### Data
The two setup files used to generate sampled datasets of three and five clusters for the Distribution-Based Urban Model are provided. These can be read using the `GMM_data.py` script to then create new data files.
All other data used in this thesis is either publically available at the cited sources or was only shared for these research purposes and will not be published here.

## Abstract

Microsimulation models form a central role in examining the socio-economic impacts of new policies on a population. As a base for the microsimulations, extensive and detailed datasets are required to accurately model individual units. In this thesis, a novel application of optimization methods to assign households to dwellings with precise geo-coordinates, while preserving the statistical properties of the area, is presented. This aims to provide an assignment between a set of households and a set of dwellings in a specific geographic area. Three methods are developed: an Integer Linear Program, an optimization problem formulation based on Maximum Weight Bipartite Matchings, and a Simulated Annealing heuristic approach. Furthermore, input datasets based on the German 2011 Census are evaluated and examined, and a novel method for creating correlated datasets using Gaussian Mixture Models is presented. The results of a comparison in accuracy and runtime on both datasets indicate that the optimization formulations have a runtime cubic in the number of dwellings while the Simulated Annealing approach is quadratic in the input. The ILP and matching formulations are both capable of finding feasible solutions quickly and reach a small MIP Gap (less than one percent) within the first ten minutes of the optimization. All three approaches are suitable for solving the presented problem, although future work on large datasets, possibly based on the German Census 2022, is required to evaluate the robustness of these findings.

## License

This work is licensed under the GNU General Public License v3.0.
