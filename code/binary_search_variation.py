#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Kendra M. Reiter


This script calculates the best penalty weight parameters lambda
for all given penalty weight indices. This is specific to one dataset and its
associated model.
The script requires a .mps model where the objective function is only the
'base' objective function
(\sum_{h \in H} \sum_{d \in D} w_{h,d} \times x_{h,d})
without any penalty parameters.
For an in-dept description of the algorithm, see Algorithm 4 in Section 7.1.5
on Penalty Weights.
The method 'run' will start the search.

Parameters
----------
data_dict : dict
    dictionary of data to use for the model.
path_to_model : str
    path to the model (.mps file) to use for optimization.
indices : list, optional
    Indices of the lambda parameters. The default is ["hhd", "per", "cap"].
precision : int, optional
    Precision k. The default is 2.
starting_interval : list, optional
    Search interval for the lambdas. The default is [0,1].
optimization_parameters : dict, optional
    Optional parameters to pass to the gurobi model for optimization.
    The default is None.
verbose_optimization : bool, optional
    If the optimization should be verbose, i.e., print values and information
    while its running. The default is False.
"""

# imports
import gurobipy as gp


class lambda_midpoints:
    def __init__(self,
                 variables: dict,
                 path_to_model: str,
                 indices: list = ["hhd", "per", "cap"],
                 precision: int = 2,
                 starting_interval: list = [0, 1],
                 optimization_parameters: dict = None,
                 verbose_optimization: bool = False):

        # initialize the class

        self.variables = variables
        self.path_to_model = path_to_model

        # check if starting interval is valid
        if starting_interval[1] < starting_interval[0]:
            raise Exception("Starting interval is not a valid interval.")
        # build search list for each lambda
        self.searchlists = {}
        search_list = [starting_interval[0]]
        while search_list[-1] < starting_interval[1]:
            search_list.append(round(search_list[-1] + 10**(-1 * precision),
                                     precision))
        # assign search lists to each lambda
        for index in indices:
            self.searchlists[index] = search_list

        self.lambdas = indices
        self.midpoints = {}
        self.costs = {}

        self.optimal_values = {}

        for ll in self.lambdas:
            self.costs[ll] = None
            self.optimal_values[ll] = starting_interval[1]
            self.midpoints[ll] = None

        # settings for the optimization
        self.opt_params = optimization_parameters
        self.verbose = verbose_optimization

        # get midpoints of first intervals
        self.find_all_midpoints()

    def optimize_and_calculate_costs(self):
        """
        This function loads the given half-model,
        adds the penalty functions with their corresponding
        lambda values to the objective function, and optimizes the model.
        The resulting penalty function values (i.e., violations) are calculated
        and self.costs is updated.

        Raises
        ------
        Exception
            If the number of variables in the loaded model does not match
            the expected number of variables.

        Returns
        -------
        None.

        """

        # get variables from variable dict
        len_D = self.variables["D"]
        len_H = self.variables["H"]
        len_K = self.variables["K"]

        D = range(len_D)
        H = range(len_H)
        K = range(len_K)

        # read model from file
        model = gp.read(self.path_to_model)

        # clear previous solution
        model.reset(0)

        # get model variables
        mod_vars = model.getVars()

        # calculate amount of x variables = H * D
        len_of_x = len(H) * len(D)

        # total length needs to equal
        total_length = len_of_x + 2 * len(K) + len(D)
        if len(mod_vars) != total_length:
            raise Exception("Number of variables in the model is " +
                            str(len(mod_vars)) + " does not match expected " +
                            str(total_length))

        # initialize penalty variables
        ys = {}
        for index in self.lambdas:
            ys[index] = []

        # get variables from model and assign to variable list
        for i in range(len_of_x, len(mod_vars)):
            var_name = mod_vars[i].varName
            if any(index in var_name for index in self.lambdas):
                idx = [s for s in self.lambdas if s in var_name][0]
                ys[idx].append(mod_vars[i])

        model.update()

        # get objective function from model
        obj_func = model.getObjective()

        # add to objective function what is dependent on lambda
        for y, y_vars in ys.items():
            obj_func += self.midpoints[y] * gp.quicksum(y_vars)

        # set objective function
        model.setObjective(obj_func)

        model.update()

        # set verbose setting
        if self.verbose:
            model.setParam('OutputFlag', 1)
            model.printStats()
        else:
            model.setParam('OutputFlag', 0)
            model.setParam('LogToConsole', 0)

        # set optional parameter settings
        if self.opt_params:
            for key, val in self.opt_params.items():
                print("Setting", key, "to", val)
                model.setParam(key, val)

        model.optimize()

        if self.verbose:
            model.printQuality()

        # get costs for each penalty function
        costs = {}
        for y, y_vars in ys.items():
            costs[y] = sum([var.x for var in y_vars])

        # update costs
        self.costs = costs

    def return_lambdas(self):
        """ Returns the midpoints. """
        return self.midpoints

    def find_all_midpoints(self):
        """ finds the midpoint of the current search list for each lambda """
        for ll, interval in self.searchlists.items():
            mid = len(interval) // 2
            self.midpoints[ll] = interval[mid]

    def reduce_searchlist(self, ll: str, left_right: str):
        """
        Bisects the searchlist for the given lambda l. If
        left_right = left, then searchlist[0, mid] is returned, else
        searchlist[mid + 1, 0] is returned, where mid is the current midpoint.
        Updates the searchlist attribute.

        Parameters
        ----------
        l : str
            lambda.
        left_right : str
            either "left" (or "l) or "right" (or "r"). Defines which sublist
                           to return.

        Returns
        -------
        None.

        """
        mid = self.midpoints[ll]
        searchlist = self.searchlists[ll]
        mid_idx = searchlist.index(mid)
        if (left_right == "left") or (left_right == "l"):
            searchlist = searchlist[:mid_idx]
        elif (left_right == "right") or (left_right == "r"):
            searchlist = searchlist[mid_idx + 1:]
        self.searchlists[ll] = searchlist

    def output_values(self, costs: bool = False, midpoints: bool = False):
        """
        Prints a small table to console with the current values.
        Either prints the costs or the midpoints for each lambda

        Parameters
        ----------
        costs : bool, optional
            If costs should be printed. The default is False.
        midpoints : bool, optional
            If midpoints should be printed. The default is False.

        Returns
        -------
        None.

        """
        max_len_l = max([len(ll) for ll in self.lambdas])
        max_width_l = max_len_l + 2

        if costs:
            max_len_col = max([len(str(c)) for c in self.costs.values()])
            max_width_col = max_len_col + 1
            neg_vals = any(c < 0 for c in self.costs.values())

        elif midpoints:
            max_len_col = max([len(str(m)) for m in self.midpoints])
            max_width_col = max_len_col + 2
            neg_vals = False

        l_str = {}
        val_str = {}

        total_width = 6 + max_len_l + max_width_col

        for ll in self.lambdas:
            right_width_l = max_width_l - len(ll) - 1
            l_str[ll] = " " + ll + " " * right_width_l

            left_width_vals = 1
            if costs:
                right_width_vals = max_width_col - len(str(self.costs[ll]))
                if self.costs[ll] > 0:
                    right_width_vals -= 1
                    if neg_vals:
                        left_width_vals = 2
                val_str[ll] = " " * left_width_vals + str(self.costs[ll]) +\
                    " " * right_width_vals
            elif midpoints:
                right_width_vals = max_width_col - len(str(self.midpoints[ll]))
                val_str[ll] = " " * left_width_vals +\
                    str(self.midpoints[ll]) + " " * right_width_vals
        if costs:
            print("Costs:")
        elif midpoints:
            print("Midpoints")
        print("-" * total_width)
        for ll in self.lambdas:
            print("|" + l_str[ll] + "|" + val_str[ll] + "|")
        print("-" * total_width)

    def write_lowest_optimal_values(self):
        """ updates the lowest value of lambda with zero costs found so far """
        for ll, c in self.costs.items():
            if c == 0:
                if self.midpoints[ll] < self.optimal_values[ll]:
                    self.optimal_values[ll] = self.midpoints[ll]

    def adjust_midpoints(self):
        """ decides how to subdivide the search list based on costs """
        for ll in self.lambdas:
            # if we have a violation : choose midpoint in right interval
            if self.costs[ll] < 0:
                left_right = "right"
            elif self.costs[ll] == 0:
                left_right = "left"

            # get new interval
            self.reduce_searchlist(ll, left_right)
        # update midpoints
        self.find_all_midpoints()

    def run(self):
        stop_while = False
        self.last_midpoints = []

        while (not stop_while):
            # add last midpoints to list
            self.last_midpoints.append(self.midpoints.copy())
            if self.verbose:
                self.output_values(midpoints=True)
            # optimize and calculate costs
            self.optimize_and_calculate_costs()
            if self.verbose:
                self.output_values(costs=True)
            # save the value if we are optimal
            self.write_lowest_optimal_values()
            # get new midpoints / subdivide the searchlist
            self.adjust_midpoints()
            # stopping condition
            if any(len(sl) == 1 for sl in self.searchlists.values()):
                stop_while = True


if __name__ == "__main__":
    # example function call
    variables = {
        "H": 3,
        "D": 4,
        "K": 1,
    }

    # Initialize class
    lambda_mid = lambda_midpoints(variables=variables,
                                  path_to_model="example_model.mps",
                                  indices=["hh", "per", "c"])
    # run search
    lambda_mid.run()
    # output parameters
    opt_lambdas = lambda_mid.return_lambdas()
    print("\nBest values:")
    lambda_mid.output_values(midpoints=True)
