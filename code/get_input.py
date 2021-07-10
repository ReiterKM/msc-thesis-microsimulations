#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Kendra M. Reiter

This Script contains functions to calculate and read variables
from input data to pass onto the solvers.

The function 'get_input' takes user-input parameters to define
the correct paths where the input files are located.

The functions 'input_from_csv_ILP' and 'input_from_csv_matching'
read the input files to calculate various parameters needed to build
the optimization models.
"""

# import code for folder-path-generation
from get_files import get_path_to_folder, get_file_path
import os
import json
import pandas as pd
import numpy as np


def input_from_csv_ILP(data_dict: dict):
    """
    This function takes the dictionary of paths to various input files
    and generates a dictionary of parameters to pass to the ILP model.

    Parameters
    ----------
    data_dict : dict
        of the format {"d" : os.path (to .csv file of data of dwellings),
        "h": os.path (to .csv file of data of households)}.

    Returns
    -------
    return_dict : dict
        dictionary with lists of all inputs needed:
            D (Dwellings), H (Households), K (grid cells),
            p_h (persons per household), c_d (capacity per dwelling),
            B_hhd (number of households per grid cell),
            B_per (number of persons per grid cell)
    """

    # ------- Load B-variables from JSON files -------
    with open(data_dict["B_per"], "r") as B_per_file:
        B_per_orig = json.load(B_per_file)
    with open(data_dict["B_hhd"], "r") as B_hhd_file:
        B_hhd_orig = json.load(B_hhd_file)

    # create a match between index (idx) and ID of each dwelling, household,
    # and grid cell.
    # both a forward (idx : id) and a backward (id : idx) match is saved
    idx_to_ids = {}

    # ------- Get dwelling data -------
    dwelling_df = pd.read_csv(data_dict["d"])
    dwelling_df = dwelling_df.reset_index(drop=True)

    # list of all grid cells K
    K_list = list(dwelling_df["Gitter_ID_100m"].unique())
    K = list(range(len(K_list)))
    idx_to_ids["K"] = dict(zip(K, K_list))
    idx_to_ids["K_inv"] = {v: k for k, v in idx_to_ids["K"].items()}

    # list of all dwellings, as per their ID
    D_list = list(dwelling_df["unique_ID"])
    D = list(range(len(D_list)))
    idx_to_ids["D"] = dict(zip(D, D_list))
    idx_to_ids["D_inv"] = {v: k for k, v in idx_to_ids["D"].items()}

    # create s[d,k] which indicates if dwelling d is in grid cell k
    # as a dictionary where key = k (Gitter_ID) and the value
    # is a list of dwellings in that grid cell
    s_df = dwelling_df[["Gitter_ID_100m", "unique_ID"]].groupby(
        "Gitter_ID_100m").agg(
        lambda x: x.unique().tolist())
    s_df = s_df.to_dict()
    s_list = s_df["unique_ID"]

    s = {}
    for key, temp in s_list.items():
        new_temp = [idx_to_ids["D_inv"][d] for d in temp]
        s[idx_to_ids["K_inv"][key]] = new_temp

    # get max capacity of dwellings, size D
    c_d_list = dwelling_df[["unique_ID", "capacity"]].set_index(
        "unique_ID").to_dict("dict")
    c_d_list = c_d_list["capacity"]
    c_d = {}
    for key, val in c_d_list.items():
        c_d[idx_to_ids["D_inv"][key]] = val

    del dwelling_df

    # ------- Get Household data -------
    household_df = pd.read_csv(data_dict["h"])

    household_df = household_df.astype({"household_size": int})
    # list of all households (taken from the index)
    H_list = list(household_df["unique_ID"].astype(int))
    H = list(range(len(H_list)))
    idx_to_ids["H"] = dict(zip(H, H_list))
    idx_to_ids["H_inv"] = {v: k for k, v in idx_to_ids["H"].items()}

    # create the variable p_h (number of persons per household h)
    p_h_list = household_df[["unique_ID", "household_size"]].set_index(
        "unique_ID").to_dict("dict")
    p_h_list = p_h_list["household_size"]
    p_h = {}
    for key, val in p_h_list.items():
        p_h[idx_to_ids["H_inv"][key]] = val

    # adjust B_per and B_hhd variables with the new K index
    B_per = dict((key, B_per_orig[value])
                 for (key, value) in idx_to_ids["K"].items())
    B_hhd = dict((key, B_hhd_orig[value])
                 for (key, value) in idx_to_ids["K"].items())

    # save the idx to ids matching to a JSON file
    with open(data_dict["idx_to_ids"], "w") as f:
        json.dump(idx_to_ids, f)

    return_dict = {"D": D, "H": H, "K": K, "s": s, "p_h": p_h,
                   "c_d": c_d, "B_hhd": B_hhd, "B_per": B_per}

    return return_dict


def input_from_csv_matching(data_dict):
    """
    This function takes the dictionary of paths to various input files
    and generates a dictionary of parameters to pass to the Matching model.
    Change from the ILP version: uses np.uint16 variables

    Parameters
    ----------
    data_dict : dict
        of the format {"d" : os.path (to .csv file of data of dwellings),
        "h": os.path (to .csv file of data of households)}.

    Returns
    -------
    return_dict : dict
        dictionary with lists of all inputs needed:
            D (Dwellings), H (Households), K (grid cells),
            p_h (persons per household), c_d (capacity per dwelling),
            B_hhd (number of households per grid cell),
            B_per (number of persons per grid cell)
    """

    # ------- Load B-variables from JSON files -------
    with open(data_dict["B_per"], "r") as B_per_file:
        B_per_orig = json.load(B_per_file)
    with open(data_dict["B_hhd"], "r") as B_hhd_file:
        B_hhd_orig = json.load(B_hhd_file)

    # create a match between index (idx) and ID of each dwelling, household,
    # and grid cell.
    # both a forward (idx : id) and a backward (id : idx) match is saved
    idx_to_ids = {}

    # ------- Get dwelling data -------
    dwelling_df = pd.read_csv(data_dict["d"])
    dwelling_df = dwelling_df.reset_index(drop=True)
    # list of all grid cells K
    K_list = list(dwelling_df["Gitter_ID_100m"].unique())
    K = np.array(range(len(K_list)), dtype=np.uint16)
    K_keys = list(range(len(K_list)))

    idx_to_ids["K"] = dict(zip(K_keys, K_list))
    idx_to_ids["K_inv"] = {v: k for k, v in idx_to_ids["K"].items()}

    # list of all dwellings, as per their ID
    D_list = list(dwelling_df["unique_ID"])
    D = np.array(range(len(D_list)), dtype=np.uint16)
    D_keys = list(range(len(D_list)))
    idx_to_ids["D"] = dict(zip(D_keys, D_list))
    idx_to_ids["D_inv"] = {v: k for k, v in idx_to_ids["D"].items()}

    # create s[d,k] which indicates if dwelling d is in grid cell k
    # as a dictionary where key = k (Gitter_ID) and the value
    # is a list of dwellings in that grid cell
    s_df = dwelling_df[["Gitter_ID_100m", "unique_ID"]].groupby(
        "Gitter_ID_100m").agg(
        lambda x: x.unique().tolist())
    s_df = s_df.to_dict()
    s_list = s_df["unique_ID"]
    s = {}

    for key, temp in s_list.items():
        new_temp = [idx_to_ids["D_inv"][d] for d in temp]
        s[idx_to_ids["K_inv"][key]] = new_temp

    # get max capacity of dwellings, size D
    c_d_list = dwelling_df[["unique_ID", "capacity"]
                           ].set_index("unique_ID").to_dict("dict")
    c_d_list = c_d_list["capacity"]
    c_d_dict = {}
    for key, val in c_d_list.items():
        c_d_dict[idx_to_ids["D_inv"][key]] = val

    c_d = np.array([np.uint16(c_d_dict[key]) for key in sorted(c_d_dict)])

    del dwelling_df

    # ------- Get Household data -------
    household_df = pd.read_csv(data_dict["h"])

    household_df = household_df.astype({"household_size": int})
    # list of all households (taken from the index)
    H_list = list(household_df["unique_ID"].astype(int))
    H_keys = list(range(len(H_list)))
    H = np.array(range(len(H_list)), dtype=np.uint16)
    idx_to_ids["H"] = dict(zip(H_keys, H_list))
    idx_to_ids["H_inv"] = {v: k for k, v in idx_to_ids["H"].items()}

    # create the variable p_h
    p_h_list = household_df[["unique_ID", "household_size"]].set_index(
        "unique_ID").to_dict("dict")
    p_h_list = p_h_list["household_size"]

    p_h_dict = {}
    for key, val in p_h_list.items():
        p_h_dict[idx_to_ids["H_inv"][key]] = val

    p_h = np.array([np.uint16(p_h_dict[key]) for key in sorted(p_h_dict)])

    # adjust B_per and B_hhd variables with the new K index
    B_per_dict = dict((key, B_per_orig[value]) for (key, value)
                      in idx_to_ids["K"].items())
    B_per = np.array([np.uint16(B_per_dict[key]) for key
                      in sorted(B_per_dict)])

    B_hhd_dict = dict((key, B_hhd_orig[value]) for (key, value)
                      in idx_to_ids["K"].items())
    B_hhd = np.array([np.uint16(B_hhd_dict[key]) for key
                      in sorted(B_hhd_dict)])

    # save to JSON file
    with open(data_dict["idx_to_ids"], "w") as f:
        json.dump(idx_to_ids, f)

    return_dict = {"D": D, "H": H, "K": K, "s": s, "p_h": p_h,
                   "c_d": c_d, "B_hh": B_hhd, "B_per": B_per}

    return return_dict


def get_input(number_of_cells: int = None,
              num_households: int = None,
              num_dwellings: int = None,
              substr: str = None,
              sub_dir: str = None,
              log_file: bool = None,
              get_model: bool = False,
              model_comments: str = None):
    """
    From the input parameters defining the desired input data, paths to the
    required files are generated here. All paths are gathered in one
    dictionary, the "data_dict".
    Additionally, the path to the model, path where the solution should be
    saved, and the string of the saved files is returned.
    Usually the data_dict is passed onwards to the other input functions
    to read the data located in the files.

    Parameters
    ----------
    number_of_cells : int, optional
        Number of cells in the dataset. The default is None.
    num_households : int, optional
        Number of households in the dataset. The default is None.
    num_dwellings : int, optional
        Number of dwellings in the dataset. The default is None.
    substr : str, optional
        comment for file name to load. Should be used when a custom data file
        has been created. The default is None.
    sub_dir : str, optional
        optional sub_directory where data is located. The default is None.
    log_file : bool, optional
         write a log file?. If yes, the path to the log file is defined and
         returned. The default is False.
    get_model : bool, optional
        True if the model already exists and the path to the model should
        be returned. The default is False.
    model_comments : str
        optional file comments to find the model to be returned. Is only
        used if get_model = True.

    Returns
    -------
    data_dict : dict
        a dictionary where key : parameter, value : value for possible returns
        from the optimizer (depends on input parameters).
    model_path : str
        path where the model is / should be saved.
    save_str : str
        name of the saved files.
    save_path : str
        path where the solution should be saved.
    log_file_path : str
        path where the log file should be saved.

    """

    data_dict = {}

    # ------- build substring to find files -------
    comment = ""
    if substr:
        comment += "_" + substr
    if num_dwellings:
        comment += "_" + str(num_dwellings) + "d"
    if num_households:
        comment += "_" + str(num_households) + "h"
    if number_of_cells:
        comment += "_" + str(number_of_cells) + "cells"

    # ------- get paths to all .csv files -------
    if sub_dir:
        path_to_folder = get_path_to_folder(sub_dir)
    else:
        path_to_folder = get_path_to_folder(sub_dir="data")

    # get all .csv files with the comment
    files = [f for f in os.listdir(path_to_folder) if (f[-4:] == ".csv") and
             (comment in f)]
    # get paths
    path_to_files = [os.path.join(path_to_folder, f) for f in files]

    # ------- get household and dwelling files -------
    for path in path_to_files:

        name_of_file = path.split("/")[-1].split(".")[0].lower()
        type_of_file = ["h" if "hold" in name_of_file
                        else "d" if "house" in name_of_file else None][0]
        if type_of_file:
            data_dict[type_of_file] = path

    # ------- get B-variable files -------
    B_files = [f for f in os.listdir(path_to_folder) if f[-4:] == "json"]
    B_files = [f for f in B_files if comment in f]
    path_to_B_files = [os.path.join(path_to_folder, f) for f in B_files]

    for path in path_to_B_files:
        type_of_file = ["B_per" if "B_per" in path
                        else "B_hhd" if ("B_hhd" in path) else None][0]
        if type_of_file:
            data_dict[type_of_file] = path

    # ------- define save paths  -------

    # name of file to be saved
    save_str = comment

    # model save path
    # if the model already exists and its path should be returned
    if get_model:
        if model_comments:
            model_comments = [model_comments, ".mps", "half"]
            if num_dwellings:
                model_comments.append(str(num_dwellings) + "d")
            if num_households:
                model_comments.append(str(num_households) + "h")
            if number_of_cells:
                model_comments.append("_" + str(number_of_cells) + "cells")

            model_path = get_file_path(sub_dir="data/models",
                                       file_comments=model_comments,
                                       ignore_file_comments=None,
                                       num_files=1)[0]
        else:
            model_path = get_file_path(sub_dir="data/models",
                                       file_comments=[".mps", "half"],
                                       ignore_file_comments=None,
                                       num_files=None)[0]
    # else define the path where it should be saved
    else:
        model_path = get_path_to_folder(sub_dir="data/models")
        model_path = os.path.join(model_path, save_str)

    # solution save path
    save_path = get_path_to_folder(sub_dir="data/matching_solutions")
    save_path = os.path.join(save_path, save_str + ".csv")

    # log file path
    if log_file:
        log_file_path = get_path_to_folder("data/log_files")
        log_file_path = os.path.join(log_file_path, save_str + ".txt")
    else:
        log_file_path = None

    # idx to ids mapping
    idx_to_ids_file_name = "ids_to_idx_" + save_str + ".json"
    data_dict["idx_to_ids"] = os.path.join(
        path_to_folder, idx_to_ids_file_name)

    return data_dict, model_path, save_str, save_path, log_file_path
