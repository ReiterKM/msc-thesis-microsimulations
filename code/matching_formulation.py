#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Kendra M. Reiter

This Script is for running the optimization in form of a Maximum Matching
using the Gurobi Solver.
"""

import numpy as np
import gurobipy as gp
import pandas as pd
from tqdm import tqdm  # display progress


def optimize(variables: dict,
             save_path: str = None,
             model_path: str = None,
             verbose: bool = True,
             return_opt: bool = False,
             lambdas: dict = {"hhd": 0.5, "per": 0.5, "cap": 0.5},
             log_file_path: str = None,
             params: dict = None):
    """
    This function builds a Gurobi model and optimizes it. This is the
    matching formulation optimization problem as defined in
    Equation 6.18 in Section 6.3 Matching Formulation.

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
    lambdas : dict, optional
        A dictionary of values to be used as the lambdas, i.e.,
        the penalty weight parameters. The default is None.
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

    D = range(len_D)
    H = range(len_H)
    K = range(len_K)
    # len_H = len(H)
    # len_D = len(D)
    # len_K = len(K)

    verboseprint("Number of Dwellings:", len_D)
    verboseprint("Number of Households:", len_H)
    verboseprint("Number of Grid Cells", len_K)
    verboseprint("\nStep 2: create model")

    # create model
    model = gp.Model("Matching Model")

    # ------- Defining penalty weights -------
    verboseprint("\nStep 3: define penalty weights lambda")

    # get penalty weights from input
    lambda_hhd = lambdas["hhd"]
    lambda_per = lambdas["per"]
    lambda_cap = lambdas["cap"]

    # ------- Defining the optimization variable -------
    verboseprint("\nStep 4: define optimization variable x")
    x = model.addMVar(shape=(len_H, len_D), vtype=gp.GRB.BINARY, name="x")

    model.update()

    # ------- Model Constraints -------
    verboseprint("\nStep 5: add model constraints")

    # help variables / vectors of ones
    H_ones_vec = np.ones(len_H)
    D_ones_vec = np.ones(len_D)
    K_ones_vec = np.ones(len_K)

    verboseprint("\n> add dwelling constraints")

    for d in tqdm(D):
        model.addConstr(H_ones_vec @ x[:, d] <= 1)

    verboseprint("\n> add household constraints")

    for h in tqdm(H):
        model.addConstr(D_ones_vec @ x[h, :] <= 1)

    # ------- Objective Function -------
    verboseprint("\n> define help variables")
    # define help variables y
    y_hhd = model.addMVar(len_K, lb=-gp.GRB.INFINITY, ub=0.0, name="y_hhd")
    y_per = model.addMVar(len_K, lb=-gp.GRB.INFINITY, ub=0.0, name="y_per")
    y_cap = model.addMVar(len_D, lb=-gp.GRB.INFINITY, ub=0.0, name="y_cap")

    model.update()
    verboseprint("\n> add B_hh and B_per constraints")
    for k in tqdm(K):
        expr_B_hh = 0
        expr_B_per = 0

        # consider only the dwellings in grid cell k but all households
        for d in s[k]:
            expr_B_hh += H_ones_vec @ x[:, d]  # sum of x_[:, d]
            expr_B_per += p_h @ x[:, d]  # sum of p[:] * x[:,d]

        expr_B_hh += -1 * B_hh[k]
        expr_B_per += -1 * B_per[k]

        model.addConstr(y_hhd[k] <= -1 * expr_B_hh)
        model.addConstr(y_per[k] <= -1 * expr_B_per)

    verboseprint("\n> add c_d constraints")

    for d in tqdm(D):
        model.addConstr(y_cap[d] <= (-1 * p_h) @ x[:, d] + c_d[d])

    model.update()

    verboseprint("\n> add weights w to base objective function")

    w_arr = np.zeros(len_H, dtype=np.float32)
    idx = np.zeros(len_H, dtype=np.uint16)

    verboseprint("\n> build objective function (w * x)")
    obj_func = 0
    for d in tqdm(D):
        idx_count = 0
        for h in H:
            if c_d[d] >= p_h[h]:
                w_arr[idx_count] = 1.0/(1 + c_d[d] - p_h[h])
                idx[idx_count] = h
                idx_count += 1
            else:
                x[h, d].ub = 0
                x[h, d].lb = 0
        # consider only the relevant indices
        obj_func += w_arr[:idx_count] @ x[idx[:idx_count], d]

    # set base objective function
    model.setObjective(obj_func)
    model.modelsense = gp.GRB.MAXIMIZE

    verboseprint("\n> set base objective function of model")
    if model_path:
        # save "half" model as MPS
        model.write(model_path + "_half.mps")
        verboseprint("\n> limited model saved")

    verboseprint("\n> add penalties to objective function")
    # compute actual objective and set
    obj_func += lambda_hhd * K_ones_vec @ y_hhd
    obj_func += lambda_per * K_ones_vec @ y_per
    obj_func += lambda_cap * D_ones_vec @ y_cap

    verboseprint("\n> set full objective function")
    model.setObjective(obj_func)

    verboseprint("\n> set optional model parameters")
    # set optional model parameters
    # log file
    if log_file_path:
        model.setParam("LogFile", log_file_path)
    # verbose output
    if verbose:
        model.setParam('OutputFlag', 1)
    else:
        model.setParam('OutputFlag', 0)
    # other parameters
    if params:
        for key, value in params.items():
            model.setParam(key, value)

    # Save model
    if model_path:
        model.write(model_path + "_full.mps")
        verboseprint("\n> full model saved to", model_path + "_full")

    verboseprint('\nModel set-up completed')
    verboseprint('\nStep 6: Optimize')

    model.optimize()

    verboseprint('\nStep 7: calculate solution')

    # save the solution, if desired
    solution_matrix = [[int(np.round(x[h, d].x, 0)) for h in H] for d in D]
    solution_df = pd.DataFrame(solution_matrix, columns=H, index=D)
    solution_df = solution_df.transpose()

    if save_path:
        solution_df.to_csv(save_path)
        verboseprint("\nSaved the solution at:", save_path)

    # print some info about the solution quality
    print('\nTotal Weight:', round(model.objVal, 0))
    # count how many households have been assigned
    tester = solution_df.sum(axis=0)
    amount_of_matched_households = sum(tester)

    print("Amount of matched households:", amount_of_matched_households)

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
    lambdas = {"hhd": 0.5, "per": 0.5, "cap": 0.5}

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
             lambdas=lambdas,
             params=params)
