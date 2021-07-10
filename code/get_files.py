#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Kendra M. Reiter

This script containts two functions to get either the path to a subdirectory
or to a file (or list of files) in the same general directory.
"""

import os


def get_file_path(sub_dir, file_comments, ignore_file_comments=None,
                  num_files=1):
    """
    get a file which contains all substrings in 'file_comments' and is located
    in 'sub_dir'. Returns a list of file paths.

    Parameters
    ----------
    sub_dir : string
        subdirectory. Should be up one directory level of the current
        file location..
    file_comments : list of strings
        list of substrings.
    ignore_file_comments : list of strings, optional
        list of substrings to ignore. The default is None.
    num_files : int, optional
        Number of files to find. The default is 1.

    Raises
    ------
    Exception
        If too many / too little / no files are found.

    Returns
    -------
    file_path : list of strings
        list of paths to files found. Length corresponds to num_files

    """
    # get path to subdirectory
    path = get_path_to_folder(sub_dir)

    # find all files with all comments in their name
    files = [f for f in os.listdir(path) if all(
        comm in f for comm in file_comments)]
    file_paths = [os.path.join(path, f) for f in files]

    # if any ignore_file_comments are given, remove files containing these
    if ignore_file_comments:
        file_paths = [f for f in file_paths if
                      all(comm not in f for comm in ignore_file_comments)]

    if num_files is None:
        return file_paths

    if len(file_paths) > num_files:
        print(files)
        raise Exception("Too many files found!")
    elif len(file_paths) < num_files:
        raise Exception("Not enough files found!")
    elif len(file_paths) == 0:
        raise Exception("No files found!")

    return file_paths


def get_path_to_folder(sub_dir="data"):
    """
    This function takes a string of a folder name and returns a path to the
    given subdirectory. It needs to be located in the same directory as this
    script.

    Parameters
    ----------
    sub_dir : string, optional
        subdirectory. Should be up one directory level of the
        current file location.
        The default is "data1/original_datasets".

    Returns
    -------
    path_to_folder : os.path
        path to folder i.e. to the desired subdirectory.

    """
    # clean input
    sub_dir = sub_dir.strip("/")
    # path to current file
    file_path = os.path.dirname(os.path.realpath(__file__))
    # go up one directory
    dir_path = os.path.dirname(file_path)
    # get correct sub-directory
    path_to_folder = os.path.join(dir_path, sub_dir)

    if not os.path.exists(path_to_folder):
        raise Exception("The path: " + path_to_folder + " does not exist.")

    return path_to_folder
