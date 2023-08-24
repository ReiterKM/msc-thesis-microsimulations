#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Kendra model. Reiter

This Script is for running the optimization in ILP formulation
using the Gurobi Solver.

"""

import numpy as np
import gurobipy as gp
import pandas as pd
from tqdm import tqdm  # show progress


def optimize(variables: dict,
             save_path: str = None,
             return_opt: bool = False,
             model_path: str = None,
             verbose: bool = False,
             log_file_path: str = None,
             params: dict = None):
    """
    This function builds a Gurobi model and optimizes it. This is the
    ILP formulation optimization problem as defined in
    Equation 6.9 in Section 6.2 Integer Linear Programming.

    Parameters
    ----------
    variables : dict
        provides all relevant data for the optimization. See README for more details.
    save_path : str, optional
        path to save the optimal solution x. If included, this is where we save
        the optimal x. The default is None.
    model_path : str, optional
        path to save the model with the 'simple' objective function.
        If included, this is where the model file will be saved.
        The default is None.
    verbose : bool, optional
        True if we want to show the Gurobi output. False otherwise.
        The default is True.
    return_opt : bool, optional
        True if we want to return the optimal solution as a numpy matrix.
        The default is False.
    log_file_path : str, optional
        if true, a log file of the run is saved to the same location
        as the code, named as the string. The default is None.
    params : dict, optional
        additional paramters to be passed to the Gurobi model, as
        {name : value}, example: {"MIPGap" : 0.0005}. The default is None.


    Returns
    -------
    solution_df : pandas DataFrame
        dataframe of size H x D with x values, if return_opt = True

    """
    # define a function to only print if verbose = True
    verboseprint = print if verbose else lambda *a, **k: None

    # ------- Load Inputs -------
    verboseprint("\nStep 1: load inputs")

    # get variables from dict
    len_D = variables["D"]
    len_H = variables["H"]
    len_K = variables["K"]
    p_h = variables["p_h"]
    c_d = variables["c_d"]
    s = variables["s"]
    B_hh = variables["B_hh"]
    B_per = variables["B_per"]

    H = range(len_H)
    D = range(len_D)
    K = range(len_K)

    verboseprint("Number of Dwellings:", len_D)
    verboseprint("Number of Households:", len_H)
    verboseprint("Number of Grid Cells", len(K))
    verboseprint("\nStep 2: create model")

    # create model
    model = gp.Model("ILP Model")

    # ------- Defining the optimization weights -------
    verboseprint("\nStep 3: calculate optimization weights w")
    w_arr = np.full((len(H), len(D)), 0.0)

    for d_idx, d in enumerate(tqdm(D)):
        for h_idx, h in enumerate(H):
            if c_d[d] >= p_h[h]:
                w_arr[h_idx, d_idx] = 1.0/(1 + c_d[d] - p_h[h])

    # ------- Defining the Variable x -------
    verboseprint("\nStep 4: add optimization weights x")

    x = model.addMVar(shape=(len_H, len_D), vtype=gp.GRB.BINARY, name="x")

    # ------- Defining the Constraints -------
    verboseprint('\nStep 5: add constraints and objective function')

    H_ones_vec = np.ones(len_H)
    D_ones_vec = np.ones(len_D)
    p_h_list = np.array([p_h[h] for h in H])

    # initialize objective function
    obj_func = 0

    verboseprint("\n> add dwelling and capacity constraints")
    for d in tqdm(D):
        model.addConstr(H_ones_vec @ x[:, d] <= 1)
        model.addConstr(p_h_list @ x[:, d] <= c_d[d])
        obj_func += w_arr[:, d] @ x[:, d]

    verboseprint('\n> add household constraints')
    for h in tqdm(H):
        model.addConstr(D_ones_vec @ x[h, :] <= 1)

    verboseprint("\n> add B_hh and B_per constraints")
    for k in tqdm(K):
        expr_B_hh = 0
        expr_B_per = 0

        temp_D = s[k]
        for d in temp_D:
            expr_B_hh += H_ones_vec @ x[:, d]
            expr_B_per += p_h_list @ x[:, d]

        model.addConstr(expr_B_hh <= B_hh[k])
        model.addConstr(expr_B_per <= B_per[k])

    model.update()

    verboseprint("\n> set model objective function")
    model.setObjective(obj_func)
    model.modelSense = gp.GRB.MAXIMIZE

    # set optional model parameters
    if log_file_path:
        model.setParam("LogFile", log_file_path)
    if params:
        for key, value in params.items():
            model.setParam(key, value)

    # optionally save model
    if model_path:
        model.write(model_path + ".lp")
        model.write(model_path + ".mps")
        verboseprint("\n> model saved")

    verboseprint('\nStep 6: Optimize')
    model.optimize()

    # save the solution, if desired
    values_of_x = x.X
    solution_matrix = [[int(round(values_of_x[h, d], 0))
                        for h in H] for d in D]
    solution_df = pd.DataFrame(solution_matrix, columns=H, index=D)

    if save_path:
        solution_df.to_csv(save_path)
        verboseprint("\nSaved the solution at:", save_path)

    # print some info about the solution quality

    verboseprint('\nTotal Weight:', round(model.objVal, 0))
    # count how many households have been assigned
    tester = solution_df.sum(axis=0)
    amount_of_matched_households = sum(tester)

    verboseprint("Amount of matched households:", amount_of_matched_households)

    if amount_of_matched_households != len(H):
        verboseprint('\nThere are unassigned households!')
        verboseprint('Number:', len(H) - amount_of_matched_households)

    if return_opt:
        solution_df.to_numpy()


if __name__ == "__main__":
 # example function call

    variables = {
        "H": 3,
        "D": 4,
        "K": 1,
        "p_h": np.array([4, 1, 2]),
        "c_d": np.array([6, 6, 4, 2]),
        "s": np.array([[1, 1, 1, 1]]),
        "B_hh": [4],
        "B_per": [18],
    }

    # OPTIMIZATION
    # these inputs define additional parameters and options of the optimization
    log_file = False  # produce a log file
    verbose = True  # verbose optimization
    params = {"MIPGap": 0.0005}  # dictionary of parameters to pass to Gurobi
    # in the form: {parameter : value}, ex: {"MIPGap" : 0.0005}

    # run optimization
    optimize(variables,
             save_path="example_solution.csv",
             model_path="example_model.mps",
             verbose=verbose,
             return_opt=False,
             params=params)
