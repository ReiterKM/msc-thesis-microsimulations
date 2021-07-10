#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Kendra M. Reiter

This script guides the user through the inputs for a GMM. The finished
file is saved under the input name.

"""

import json
import numpy as np
from get_files import get_path_to_folder
import os


def symmetrize(A):
    """
    Takes a square numpy matrix and returns a symmetrized version of it.
    A should be upper or lower diagonal, i.e. A_ij = 0 or A_ji = 0 for i != j

    Parameters
    ----------
    A : np.array
        square matrix.

    Returns
    -------
    np.array
        symmetrized version of A.

    """
    return A + A.T - np.diag(A.diagonal())


print("This process guides you through inputting parameters" +
      "\nto be used to create synthetic data \nsaved as a .JSON file")

get_input = True
while get_input:
    num_input_params = input("Enter the number of parameters for each point: ")

    if not num_input_params.isnumeric():
        print("Only integers are allowed!")
        continue

    num_input_params = int(num_input_params)

    if num_input_params <= 0:
        print("Only positive integers are allowed!")
        continue
    get_input = False

labels = []
for i in range(num_input_params):
    x = input("Label parameter " + str(i) + ": ")
    labels.append(x)

get_input = True
while get_input:
    num_clusters = input("Enter the number of clusters: ")
    if not num_clusters.isnumeric():
        print("Only integers are allowed!")
        continue
    num_clusters = int(num_clusters)

    if num_clusters <= 0:
        print("Only positive integers are allowed!")
        continue
    get_input = False

inputs_needed = ["mean", "standard deviation", "correlation matrix"]

means = {}
stds = {}
cms = {}

for cluster in range(num_clusters):
    print("\n######## Cluster", cluster, "########")
    cluster_means = []
    cluster_stds = []
    for input_type, save_list in zip(["mean", "standard deviation"],
                                     [cluster_means, cluster_stds]):
        print("\n## Input:", input_type, "##")
        get_input = True
        while get_input:
            params = input("Please input " + input_type + " for " +
                           " ".join(labels) +
                           " separated by a space: ")
            params = params.split()
            # validate input
            if len(params) != num_input_params:
                print(
                    "Length of input does not match required length" +
                    " of input parameters")
                continue
            for val in params:
                try:
                    float(val)
                except:
                    print("Only floats are allowed!")
                    continue
            get_input = False

        save_list.append([float(i) for i in params])

    # get correlation matrix
    print("\n## Input: Correlation Matrix ##")
    corr_mat = []
    for i, val in enumerate(labels[:-1]):
        get_input = True
        while get_input:
            corrs = input("Enter correlation between parameter " + val +
                          " and parameters " + " ".join(labels[i+1:]) + ": ")
            corrs = corrs.split()

            # validate input
            if len(corrs) != (num_input_params - (i + 1)):
                print(
                    "Length of input does not match required length" +
                    " of input parameters")
                continue
            for val in corrs:
                try:
                    float(val)
                except:
                    print("Only integers are allowed!")
                    continue
            get_input = False

        corrs = [0] * i + [1] + [float(i) for i in corrs]
        print(labels[i], ":", corrs)
        corr_mat.append(corrs)
    corr_mat.append([0] * (len(labels) - 1) + [1])

    means[cluster] = cluster_means[0]
    stds[cluster] = cluster_stds[0]
    corr_mat = symmetrize(np.array(corr_mat))
    cms[cluster] = corr_mat.tolist()

save_comment = input("Enter a specific save comment: ")

save_path = get_path_to_folder("data1/synthetic_data")
save_str = "setup_" + str(num_clusters) + "clusters_" + \
    "_".join(save_comment.split()) + ".json"

save_path = os.path.join(save_path, save_str)

with open(save_path, 'w') as f:
    json.dump([labels, means, stds, cms], f)

print("Saved data to:", save_path)
