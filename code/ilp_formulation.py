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

# import code for reading in data
from get_input import get_input, input_from_csv_ILP


def optimize(data_dict: dict,
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
    data_dict : dict
        of the format {"d" : os.path (to .csv file of data of dwellings),
            "h": os.path (to .csv file of data of households)}
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

    """
    This function builds and optimizes a "standard" MIP model based on our
    problem, using the given input in data_dict.

    Parameters
    ----------
    data_dict: dictionary
        of the format {"d" : os.path (to .csv file of data of dwellings),
        "h": os.path (to .csv file of data of households)}
    save_path : string, optional
        path to save the optimal solution x. If included, this is where we save
        the optimal x.
        The default is None.
    return_opt : bool, optional
        True if we want to return the optimal solution as a numpy matrix
        The default is False
    model_path: string, optional
        path to save the complete model. If included,
        this is where the model file will be saved.
        The default is None.
    vebose : bool, optional
        True if we want to show the Gurobi output. False otherwise.
        The default is True.
    max_time: float, optional
        if included, a time limit is set on the gurobi model ("TimeLimit")
        Should be given in seconds?.
        The default is None
    mip_gap: float, optional
        if included, the MIPGap is set to this value on the gurobi sovler
        The default value is None
    log_file : string, optional
        if true, a log file of the run is saved to the same location
        as the code, named as the string.
        The default is None.
    log_runtime : bool, optional
        if True, the runtime is logged and returned. The default is False.
    extended : bool, optional
        if True, a different function for loading the inputs is used.
        Denotes if we are using the extended dataset or not.
        The default is False.
    timer : str, optional
        option for timer (from the 'time' module). The default is None.

    Returns
    -------
    None.

    """

    # define a function to only print if verbose = True
    verboseprint = print if verbose else lambda *a, **k: None

    # ------- Load Inputs -------
    verboseprint("\nStep 1: load inputs")
    variable_dict = input_from_csv_ILP(data_dict)

    # get variables from dict
    D = variable_dict["D"]
    H = variable_dict["H"]
    K = variable_dict["K"]
    p_h = variable_dict["p_h"]
    c_d = variable_dict["c_d"]
    s = variable_dict["s"]
    B_hh = variable_dict["B_hhd"]
    B_per = variable_dict["B_per"]

    len_H = len(H)
    len_D = len(D)

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
    solution_matrix = [[int(round(x[h, d].x[0], 0)) for h in H] for d in D]
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
    """ Inputs """
    # all inputs are defined here

    # DATASET
    # these inputs define the dataset to use for the optimization
    number_of_dwellings = 10
    number_of_households = 10
    number_of_grid_cells = None

    sub_dir = None  # specific subdirector where the data is located
    sub_str = ""  # specific substring / name of the data

    # OPTIMIZATION
    # these inputs define additional parameters and options of the optimization
    log_file = True  # produce a log file
    verbose = True  # verbose optimization
    params = {"MIPGap": 0.0005}  # dictionary of parameters to pass to Gurobi
    # in the form: {parameter : value}, ex: {"MIPGap" : 0.0005}
    # get inputs
    data_dict, model_path, save_str, solution_path, log_file_path = get_input(
        number_of_cells=number_of_grid_cells,
        num_households=number_of_households,
        num_dwellings=number_of_dwellings,
        substr=sub_str,
        sub_dir=sub_dir,
        log_file=log_file)

    # run optimization
    optimize(data_dict,
             save_path=solution_path,
             model_path=model_path,
             verbose=verbose,
             return_opt=False,
             log_file_path=log_file_path,
             params=params)
