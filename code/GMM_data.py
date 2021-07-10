#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Kendra M. Reiter

This script is used to generate data from a Distribution-Based Urban Model,
i.e., using GMMs. It follows the method outlined in Section 5.3
Urban Model using GMMs

Required is a .JSON file with the input, which may be generated using the
script 'get_GMM_input.py', which guides the user through all required parts.
The input is a list with
    1. list of parameter names as strings
    2. dictionary of means, with keys for each cluster
    3. dictionary of standard deviations, with keys for each cluster
    4. dictionary of correlation matrices, with keys for each cluster

"""
import numpy as np
import math
from get_files import get_path_to_folder, get_file_path
import os
import pandas as pd
import json


def generate_cluster_points(num_points: int,
                            cluster_stds: list,
                            cluster_means: list,
                            cm: np.array,
                            types: list,
                            cluster_probabilities: list = None,
                            min_vals: list = None,
                            vacancy_rate: float = .01):
    """
    generates num_points points across the given clusters. Clusters for each
    point are chosen randomly and the point's attributes are determined
    from the cluster's covariance matrix. These are correlated random
    variables.
    Cluster_stds and cluster_means need to be in the same order!

    Parameters
    ----------
    num_points : int
        number of points to generate (in total).
    cluster_stds : list
        list of standard deviations of the clusters.
    cluster_means : list
        list of means of the clusters.
    cm : np.array
        correlation matricies of the clusters.
    types: list
        converts values to the given types.
    repeat_idx : int, optional
        index of the point to repeat the point (e.g., the number of dwellings -
        so the point is appended as often as there are number of dwellings).
        This way each point represents a house with individual units at the
        same (x,y) coordinate.
    cluster_probabilities : list, optional
        probability weighting for the choice of clusters. If None is given,
        the probabilities are set to 1. The default is None.
    min_vals : list, optional
        the minimum of any accepted value. If None is passed then no
        minimum value is set. The default is None.
    vacancy_rate : float, optional
        the default "emptyness" level, i.e. percent of dwellings which are
        allowed to be empty. 1% is a guideline for large cities.
        The default is .01.


    Raises
    ------
    Exception
        if the number of cluster stds does not match the number of means
        or if a cluster does not have a positive definite covariance matrix.

    Returns
    -------
    points : list
        list of points, each point is a list of attributes, in the same order
        as the cluster_stds / cluster_means.

    """

    num_clusters = len(cluster_stds)

    # set equal probabilities if none are given
    if not cluster_probabilities:
        cluster_probabilities = np.ones(num_clusters) / num_clusters

    if len(cluster_stds) != len(cluster_means):
        raise Exception("Input of unequal lengths. The lists 'cluster_stds' \
                        and 'cluster_means' need to have equal lengths.")

    cluster_choices = [i for i in range(num_clusters)]

    covs = {}
    dim = {}
    # generate covariance matrices for each cluster
    for i in range(num_clusters):
        covs[i] = np.asarray(cm[str(i)])
        dim[i] = covs[i].shape[0]

    points = []

    # ------- Generate points -------
    num_generated_points = 0

    while num_generated_points < num_points:
        append_point = True
        # randomly choose cluster
        cluster = np.random.choice(cluster_choices,
                                   size=1,
                                   replace=False,
                                   p=cluster_probabilities)[0]

        num_vars = dim[cluster]

        # generate normal variable in appropriate size
        n = np.random.normal(0, 1, size=num_vars)
        # calculate correlatated normal
        corr_normals = cm[str(cluster)] @ n
        n_sigma = cluster_means[str(cluster)] + \
            np.sqrt(np.array(cluster_stds[str(cluster)])) * corr_normals

        # check minimum value
        if min_vals:
            non_zero_idx = [i for i, val in enumerate(min_vals) if
                            (val is not None)]

            # if all values are greater than or equal to, we accept the value
            for i in non_zero_idx:
                if (n_sigma[i] < min_vals[i]):
                    append_point = False

        if append_point:
            num_generated_points += 1
            # convert to given types
            point = [types[i](round(val, 0)) if types[i] == int else
                     types[i](val) for i, val in enumerate(n_sigma)]
            points.append(np.append(point, int(cluster)))

    return points


def add_gitter_id(data: dict,
                  cell_width: int,
                  cell_reference_point: tuple,
                  x_col: str = "X",
                  y_col: str = "Y"):
    """
    adds data to the column "Gitter_ID_100m" of the data. Creates a grid of
    width cell_width, starting at the cell_reference_point. Each cell is
    defined by the south-west corner of the square.

    Parameters
    ----------
    data : dict
        dictionary of house data.
    cell_width : int
        width of each cell.
    cell_reference_point : tuple
        (x,y) coordinate of the grid reference point.
    x_col : str, optional
        column name which contains all x-coordinates. The default is "X".
    y_col : str, optional
        column name which contains all y-coordinates. The default is "Y".

    Returns
    -------
    house_data : dict
        dictionary of house data.

    """

    gitter_ids = []
    for x, y in zip(data[x_col], data[y_col]):
        # find x-grid cell
        grid_x = math.floor(x / cell_width)
        # grid_x = math.floor(grid_x) if (grid_x < 0) else math.ceil(grid_x)
        # find y-grid cell
        grid_y = math.floor(y / cell_width)
        # grid_y = math.floor(grid_y) if (grid_y < 0) else math.ceil(grid_y)
        grid = str(cell_width) + "N" + str(grid_y) + "E" + str(grid_x)
        gitter_ids.append(grid)

    # set data
    data["Gitter_ID_100m"] = gitter_ids
    return data


def add_capacity(data: dict, factor: float = 2, col: str = "rooms"):
    """
    This function adds values to the "capacity" column of the houses data,
    based on the given column and the factor. Then capacity = factor * col.
    For example, one could say that a max. of 2 persons per room are allowed.
    Then, factor = 2 and col = "rooms", meaning that capacity = 2 * "rooms".

    Parameters
    ----------
    data : dict
        dictionary of all housing data in the format "col" : [list of data].
    factor : float, optional
        mulitiplicative factor. The default is 2.
    col : str, optional
        column to multiply by factor. The default is "sqm".

    Returns
    -------
    house_data : dict
        dictionary of all housing data, where "capacity" has the new data.

    """
    capacity = []
    for room in data[col]:
        capacity.append(int(round(room * factor, 0)))

    # make sure it matches the given data type - int

    data["capacity"] = capacity
    return data


def calculate_B_variables(selected_df,
                          save_dir="data",
                          save_comment=""):
    """
    This function calculates the values of the variables B_per and B_hh
    for the given .csv file of census data input.
    The data is saved at the given save_dir.

    Parameters
    ----------
    selected_df : pd.Dataframe
        households data in a dataframe
    save_dir : str, optional
        path to save the two variables as individual JSON files.
        The default is "data".
    save_comment : str, optional
        optional comment for the file name when saving. The default is "".

    Returns
    -------
    None
    """

    # the variables we create are dictionaries
    B_per = {}
    B_hhd = {}

    # calculate number of persons and households per grid cell
    grouped_df = selected_df[["Gitter_ID_100m",
                              "household_size"]].groupby(
        "Gitter_ID_100m").agg(["sum", "count"])
    grouped_df.columns = ["sum", "count"]
    data = grouped_df.to_dict()

    B_hhd = data["count"]
    B_per = data["sum"]

    def convert(o):
        """ convert to Int64 """
        if isinstance(o, np.int64):
            return int(o)
        raise TypeError

    save_path = get_path_to_folder(save_dir)

    # if we include a save comment, we want to add it with a leading underscore
    if save_comment:
        save_comment = "_" + save_comment.strip("_")
    B_per_path = os.path.join(save_path, "B_per" + save_comment + ".json")
    # write B_per values
    with open(B_per_path, "w") as B_per_file:
        json.dump(B_per, B_per_file, default=convert)
    print("B_per dictionary saved to ", B_per_path)

    # write B_hh values
    B_hhd_path = os.path.join(save_path, "B_hhd" + save_comment + ".json")
    with open(B_hhd_path, "w") as B_hhd_file:
        json.dump(B_hhd, B_hhd_file, default=convert)
    print("B_hh dictionary saved to ", B_hhd_path)


def save_points(points, data_comment: str, cell_width: int,
                cell_reference_point: tuple = (0, 0),
                point_idx: list = ["x", "y", "sqm", "persons",
                                   "cluster"],
                file_dir: str = "data",
                household_columns: list = ["unique_ID", "household_size"],
                household_idx: list = ["id", 3],
                household_types: list = [int, int],
                house_columns: list = ["Source", "unique_ID", "X", "Y",
                                       "Gitter_ID_100m", "rooms", "capacity"],
                house_idx: list = ["synthetic", "id", 0, 1, None, 2, None],
                house_types: list = [str, int, float, float, str, int, int],
                factor: float = .10, capacity_reference_col: str = "sqm"):
    """

    This function saves the points in a format fitting the input of the
    optimization code. Points will be saved once as a total csv and then
    separated into Households and Dwellings and saved separately, with the
    fitting columns.The input needs to specify which column should be filled
    with which value from the points.
    Data is saved in the subdirectory file_dir.
    In the _idx lists, the values determine the fill of the columns:
        int: indicates which index of the point tuples is relevant
        str:
            if "id": indicates a unique ID value should be assigned
            else: entire column will be filled with the given string
        None: indicates this column has a special, calculated fill


    Parameters
    ----------
    points : list (of lists)
        list of points.
    data_comment : str
        description of the data will be used as the save string of
        the .csv file.
    cell_width : int
        width of grid cell.
    cell_reference_point : tuple, optional
        reference point of the grid. The default is (0,0).
    point_idx : list, optional
        gives the description of each point, i.e. which value is at which idx.
        The default is ["x", "y", "rooms", "persons", "cluster"].
    file_dir : str, optional
        subdirectory where the data should be saved.
        The default is "data".
    household_columns : list, optional
        list of columns in the household .csv file.
        The default is ["unique_ID", "household_size"].
    household_idx : list, optional
        what to fill the household dataset with. The default is ["id", 3].
    household_types : list, optional
        data type of the household columns. The default is [int, int].
    house_columns : list, optional
        columns of the dwelling/house dataset.
        The default is ["Source", "unique_ID", "X", "Y",
                        "Gitter_ID_100m", "sqm", "capacity"].
    house_idx : list, optional
        what to fill the house dataset with.
        The default is ["synthetic", "id", 0, 1, None, 2, None].
    house_types : list, optional
        data type of the house columns.
        The default is [str, int, float, float, str, int, int].
    factor : float
        factor for determining capacity values, if not given. The default is 2.
    capacity_reference_col : str
        column to calculate capacity values, if not given.
        The default is "rooms".

    Raises
    ------
    Exception
        If unknown _idx is given.

    Returns
    -------
    None.

    """

    folder_path = get_path_to_folder(file_dir)

    household_data = {col: [] for col in household_columns}
    house_data = {col: [] for col in house_columns}
    total_cols = list(set(household_columns + house_columns))
    total_cols.remove("unique_ID")
    total_data = {col: [] for col in total_cols}

    for i, pt in enumerate(points):
        for col_list, idx_list, data_type, data_dict in zip(
                [household_columns, house_columns],
                [household_idx, house_idx],
                [household_types, house_types],
                [household_data, house_data]):
            for idx, col in enumerate(col_list):
                if idx_list[idx] == "id":
                    data_dict[col].append(data_type[idx](i))
                elif isinstance(idx_list[idx], str):
                    data_dict[col].append(data_type[idx](idx_list[idx]))
                    total_data[col].append(data_type[idx](idx_list[idx]))
                elif isinstance(idx_list[idx], int):
                    data_dict[col].append(data_type[idx](pt[idx_list[idx]]))
                    total_data[col].append(data_type[idx](pt[idx_list[idx]]))
                elif idx_list[idx] is None:
                    pass
                else:
                    raise Exception("Unknown type.")

    household_df = pd.DataFrame(household_data)
    household_save_path = os.path.join(folder_path,
                                       "Households_" + data_comment + ".csv")
    household_df.to_csv(household_save_path, index=False)

    # check if a value is given to fill the "Gitter_ID_100m" column
    gitter_idx = [i for i, val in enumerate(house_columns) if
                  (val == "Gitter_ID_100m")]
    try:
        gitter_idx = gitter_idx[0]
        gitter_fill = house_idx[gitter_idx]
    except:
        raise Exception("The data needs to contain a column 'Gitter_ID_100m'.")

    # if None is given, add Gitter_ID_100m
    # based on cell width and reference point
    if gitter_fill is None:
        house_data = add_gitter_id(
            house_data, cell_width, cell_reference_point)
        total_data = add_gitter_id(
            total_data, cell_width, cell_reference_point)

    # check if a value is given to fill the "capacity" column
    capacity_idx = [i for i, val in enumerate(house_columns) if
                    (val == "capacity")]
    try:
        capacity_idx = capacity_idx[0]
        capacity_fill = house_idx[capacity_idx]
    except:
        raise Exception("The data needs to contain a column 'capacity'.")

    # if not, add capacity according to the number of persons per room
    if capacity_fill is None:
        house_data = add_capacity(house_data, factor=factor,
                                  col=capacity_reference_col)
        total_data = add_capacity(total_data, factor=factor,
                                  col=capacity_reference_col)

    house_df = pd.DataFrame(house_data)
    # shuffle rows
    house_df = house_df.sample(frac=1)
    # save to csv
    house_save_path = os.path.join(folder_path,
                                   "Houses_" + data_comment + ".csv")
    house_df.to_csv(house_save_path, index=False)

    total_data_df = pd.DataFrame(total_data)
    total_save_path = os.path.join(
        folder_path, "total_" + data_comment + ".csv")
    total_data_df.to_csv(total_save_path, index=False)

    # calculate and save B variables for the data
    calculate_B_variables(selected_df=total_data_df, save_comment=data_comment)


def read_input(sub_dir: str, comment: str):
    """ read input from .JSON file """
    read_path = get_file_path(sub_dir, comment)[0]
    with open(read_path, "r") as f:
        data = json.load(f)

    # order: parameter names, dict of means, dict of stds, dict of cms
    print("Parameters in data:", data[0])
    clusters = list(data[1].keys())
    print("Clusters:", clusters)

    return data[1], data[2], data[3]


if __name__ == "__main__":
    """
    Parameters
    ----------
    number_of_points : int
        number of points to generate
    subdir_of_input : str
        subdirectory where the input .JSON file is located
    input_file_comments : list
        list of strings of file comments, i.e., name of the input file
    save_str : str
        name of file where the data is saved
    """

    number_of_points = 10
    subdir_of_input = "data"
    input_file_comments = ["3", ".json"]
    save_str = "GMM_of_3clusters"

    # ------- Load Inputs -------
    means, stds, cms = read_input(subdir_of_input, input_file_comments)

    # ------- Generate cluster points -------
    mixed_points = generate_cluster_points(number_of_points,
                                           cluster_stds=stds,
                                           cluster_means=means,
                                           cm=cms,
                                           types=[float, float, int, int],
                                           cluster_probabilities=[
                                               1/3, 1/3, 1/3],
                                           min_vals=[None, None, 10, 0])

    save_str = save_str + "_" + \
        str(number_of_points) + "d_" + \
        str(number_of_points) + "h"

    save_points(mixed_points, save_str,
                cell_width=1,
                factor=.1,
                capacity_reference_col="sqm",
                house_columns=["Source", "unique_ID", "X", "Y",
                               "Gitter_ID_100m", "sqm", "capacity"])
