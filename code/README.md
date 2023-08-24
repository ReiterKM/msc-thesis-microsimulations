# Input Format
All data should be cleaned and pre-processed before passing it as input to the provided formulations. The following provides an overview of the required inputs for the different models:


### Variables
This dictionary provides all relevant parameters and variables to the problem.

| Key | Name | Description | Size | Type |
|--------------|-----------|------------|------------|------------|
| D | Dwellings | Number of dwellings | | int |
| H | Household  | Number of households | | int |
| K | Gridcells | Number of grid cell | | int |
| p_h | Persons | Number of persons in household $h \in H$| $\lvert H \rvert$| Numpy Array |
| c_d" | Capacity | Capacity of dwelling $d \in D$ | $\lvert D \rvert$| Numpy Array |
| s | | Indicates if dwelling $d$ is in grid cell $k$| $\lvert D \rvert \times \lvert K \rvert$| Numpy Array |
|B_hh | | Max. number of households per grid cell $k \in K$| $\lvert K \rvert$| List|
| B_per || Max. number of persons per grid cell $k \in K$| $\lvert K \rvert$| List |

### Lambdas
This dictionary provides the penalty weights $\lambda$ for the objective function.

| Key | Name | Description |
|--------------|-----------|------------|
| hhd | Households | Exceeding number of households per grid cell $k \in K$ |
| per | Persons  | Exceeding number of persons per grid cell $k \in K$ |
| cap | Capacity | Exceeding capacity for dwelling $d \in D$ | 