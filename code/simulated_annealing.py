#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Kendra M. Reiter

This script is for running the Simulated Annealing algorithm
on a given dataset. Note all possible stopping criteria, which are passed
as keyword arguments to Python.
"""

import random
import numpy as np
import json
import time

# import code for reading in data
from get_input import input_from_csv_ILP, get_input


class Simulated_Annealing():
    def __init__(self, data_dict: dict,
                 lambdas_dict: dict = {"hh": 0.3, "per": 0.3, "c": 0.3},
                 niters: int = 100,
                 starting_temp: int = 100,
                 cooling_schedule: str = "log",
                 cooling_param: float = 0.9,
                 **kwargs):
        """
        This function handles all input and settings for the
        Simulated Annealing algorithm.

        Parameters
        ----------
        data_dict : dict
            input dictionary of the format
            {"d" : os.path (to .csv file of data of dwellings),
             "h": os.path (to .csv file of data of households)}
        lambdas_dict : dict, optional
            penalty weights. The default is {"hh":0.3, "per":0.3, "c":0.3}.
        niters : int, optional
            number of iterations at each temperature before reannealing.
            The default is 100.
        starting_temp : int, optional
            initial temperature. The default is 100.
        cooling_schedule : str, optional
            choice of cooling schedule. The default is "log". Choices are:
                "exp" or "exponential": T_n = T_0 * alpha ** (n + 1)
                "log" or "logarithmic": T_n = T_0 / log(n + 1)
                "mult" or "multiplicative": T_n = T_0 / (1 + alpha * (n + 1))
                "add" or "additive": T_n = T_0 - n * alpha
        cooling_param : float, optional
            alpha; cooling parameter for cooling schedule. The default is 0.9.
        **kwargs :
            Optional parameters for the stopping criterion.
            max_iters: int
                maximum number of total iterations. The default is inf.
            min_temp: float
                minimum temperature to reach. The default is 0.
            max_func_evals: int
                maximum number of objective function evaluations. The default
                is 3000 * (number of variables)
            max_time: int
                maximum runtime (in seconds). The default is inf.
            acceptance_tol: float
                ratio of accepted vs. proposed edges over
                500 * (number of variables) iterations.
                The default is 10**(-6).

        Returns
        -------
        None.

        """

        self.niters = niters  # number of iterations per temperature T

        self.lambdas_dict = lambdas_dict  # weights for objective function

        self.T0 = starting_temp  # starting temperature
        self.T = starting_temp  # running temperature

        # optional fraction of temperature decrease
        self.alpha = cooling_param
        self.cooling_schedule = cooling_schedule.lower()
        self.c = self.T0 * np.log(2)

        # get inputs
        variable_dict = input_from_csv_ILP(data_dict)

        # get variables from dict
        self.D = variable_dict["D"]
        self.H = variable_dict["H"]
        self.K = variable_dict["K"]
        self.p_h = variable_dict["p_h"]
        self.c_d = variable_dict["c_d"]
        self.s = variable_dict["s"]
        self.B_hhd = variable_dict["B_hhd"]
        self.B_per = variable_dict["B_per"]

        # add nullspace (represented by -1) to H and D
        self.full_D = self.D + [-1]
        self.full_H = self.H + [-1]

        self.len_H = len(self.H)
        self.len_D = len(self.D)
        self.num_vars = self.len_H * self.len_D

        self.p_h_list = np.array([self.p_h[h] for h in self.H])

        self.edges = []
        self.func_evals = 0  # number of obj. function evaluations
        self.start_time = time.time()  # save start time to determine runtime
        # number of iters to consider for the acceptance tolerance
        self.acceptance_iters = 500 * self.num_vars
        self.func_vals = []

        # check values for stopping criterions
        # limit on max iterations
        if "max_iters" in kwargs:
            self.max_iters = kwargs["max_iters"]
        else:
            self.max_iters = np.inf
        # minimum temperature
        if "min_temp" in kwargs:
            self.min_temp = kwargs["min_temp"]
        else:
            self.min_temp = 5**(-324)
        # max function evaluations
        if "max_func_evals" in kwargs:
            self.max_func_evals = kwargs["max_func_evals"]
        else:
            self.max_func_evals = 3000 * self.num_vars
        # max time
        if "max_time" in kwargs:
            self.max_time = kwargs["max_time"]
        else:
            self.max_time = np.inf
        # acceptance tolerance
        if "acceptance_tol" in kwargs:
            self.acceptance_tol = kwargs["acceptance_tol"]
        else:
            self.acceptance_tol = 10**(-6)
        if "max_obj_val" in kwargs:
            self.max_obj_val = kwargs["max_obj_val"]
        else:
            self.max_obj_val = np.inf

    def calculate_weights(self):
        """
        calculates weights w_{h,d} for each household h and dwelling d.
        w_arr is a numpy array of size |H| x |D|.

        Returns
        -------
        None.

        """
        w_arr = np.full((self.len_H, self.len_D), 0.0)
        for d_idx, d in enumerate(self.D):
            for h_idx, h in enumerate(self.H):
                if self.c_d[d] >= self.p_h[h]:
                    w_arr[h_idx, d_idx] = 1.0/(1 + self.c_d[d] - self.p_h[h])

        self.w_arr = w_arr

    def get_inital_solution(self):
        """
        Given two sets of vertices (H and D), generate a set of edges
        (from H U {-1} to D U {-1}) such that each vertex is assigned
        (possibly to the "unassigned" vertex -1) and the solution is feasible.

        Returns
        -------
        None.

        """
        temp_H = self.H.copy()
        temp_D = self.D.copy()
        edges = []

        for h in temp_H:
            possible_d = [d for d in temp_D if self.p_h[h] <= self.c_d[d]]
            if len(possible_d) == 0:
                # print("exhausted d?")
                # add edge between h and nullspace
                edges.append((h, -1))
            else:
                # choose 'best' value for d
                max_c = max(self.w_arr[h][possible_d])
                max_index = np.where(self.w_arr[h][possible_d] == max_c)[0][0]
                d = possible_d[max_index]

                edges.append((h, d))
                temp_D.remove(d)

        # add edges for any remaining dwellings in D
        for d in temp_D:
            edges.append((-1, d))

        self.edges = edges

    def create_x_matrix(self, edges: list):
        """
        creates and returns a matrix of edges in the matching.

        Parameters
        ----------
        edges : list
            list of tuples of the form (h,d) for a household h, dwelling d.
            Each tuple represents an edge between h and d.

        Returns
        -------
        x : numpy array
            x[h,d] == 1 corresponds to a matching between household h
            and dwelling d. x[h,d] == 0 otherwise. For all h in H, d in D.

        """
        x = np.zeros((self.len_H, self.len_D))

        for edge in edges:
            v1, v2 = edge
            # if either one is matched to the nullspace: there is no edge
            if (v1 == -1) or (v2 == -1):
                continue
            x[v1][v2] = 1
        return x

    def get_candidate(self):
        """
        generates a candidate (h', d') for a possible new edge.
        Ensures that the candidate is feasible
        (i.e., capacity of d' >= size of h') and the edge (h', d')
        does not already exist.

        Returns
        -------
        h : int
            household h' in H U {-1}.
        d : int
            dwelling d' in D U {-1}.

        """
        loop = True

        while loop:
            h = random.choice(self.full_H)
            d = random.choice(self.full_D)

            # stopping conditions
            # if either one is the null space
            if (h == -1) or (d == -1):
                loop = False
            # if neither is the null space and the edge is permitted
            if (h != -1) and (d != -1) and (self.p_h[h] <= self.c_d[d]):
                loop = False
            # if the edge already exists: do not pick this / try again
            if (h, d) in self.edges:
                loop = True

        return h, d

    def permute_edges(self):
        """
        given the current state of the edges, this function removes an
        existing edge and adds a new, feasible edge,
        to propose a new candidate solution to the problem.

        Returns
        -------
        new_edges : list
            list of tuples of proposed edges.

        """
        h, d = self.get_candidate()  # get candidates (h', d') for new edge

        # proposed set of edges
        new_edges = self.edges.copy()

        # get edge (h', *)
        h_edge = [e for e in self.edges if e[0] == h]
        # if we have found an edge, we use that edge
        if len(h_edge) > 0:
            h_edge = h_edge[0]
            # delete old edges
            new_edges.remove(h_edge)
        # else it will be mapped to the nullspace
        else:
            h_edge = (h, -1)

        # get edge (*, d')
        d_edge = [e for e in self.edges if e[1] == d]
        # if we have found an edge, we use that edge
        if len(d_edge) > 0:
            d_edge = d_edge[0]
            # delete old edges
            new_edges.remove(d_edge)
        # else it will be mapped to the nullspace
        else:
            d_edge = (-1, d)

        # add new edges:
            # result from (h',d')
        new_edges.append((h, d))
        # resulting from the previous edges of h' and d'
        new_edges.append((d_edge[0], h_edge[1]))

        return new_edges

    def evaluate_matching(self, edges):
        """
        evaluates the objective function of the matching on the given
        set of edges

        Parameters
        ----------
        edges : list of tuples
            list of edges in the matching. Edges are represented as tuples
            (h,d) with h = household, d = dwelling.

        Returns
        -------
        obj_val : float
            objective value of the given edges.

        """
        self.func_evals += 1

        x = self.create_x_matrix(edges)

        obj_val = (self.w_arr * x).sum()

        # save the objective function without penalties
        self.base_obj_val = obj_val

        for k in self.K:
            s_k = self.s[k]
            temp_hh = x[:, s_k].sum() - self.B_hhd[k]
            temp_per = (self.p_h_list @ x[:, s_k]).sum() - self.B_per[k]

            obj_val -= self.lambdas_dict["hh"] * max(0, temp_hh)
            obj_val -= self.lambdas_dict["per"] * max(0, temp_per)

        for d in self.D:
            temp_c = self.p_h_list @ x[:, d] - self.c_d[d]
            obj_val -= self.lambdas_dict["c"] * max(0, temp_c)

        return obj_val

    def acceptance_probability(self, new_value: float):
        """
        calculate the acceptance probability of a new solution.
        Note that this is only called when new_value < best_value
        (in a max. problem).

        Parameters
        ----------
        new_value : float
            new proposed value.

        Returns
        -------
        prob: float
            returns the probability of accepting the new solution.

        """
        prob = np.exp((new_value - self.best_value) / self.T)
        return min(prob, 1)

    def calculate_temperature(self):
        """
        Calculates the next temperature in the temperature cooling schedule,
        using the given cooling schedule.

        Raises
        ------
        Exception
            if an invalid cooling schedule was passed to the SA code.

        Returns
        -------
        None.

        """
        # exponential
        if ((self.cooling_schedule == "exp") or
                (self.cooling_schedule == "exponential")):
            self.T = self.T0 * self.alpha ** (self.t_n + 1)
        # logarithmic
        elif ((self.cooling_schedule == "log") or
                  (self.cooling_schedule == "logarithmic")):
            self.T = self.c / np.log((self.t_n + 1))
        # multiplicative
        elif ((self.cooling_schedule == "mult") or
                  (self.cooling_schedule == "multiplicative")):
            self.T = self.T0 / (1 + self.alpha * (self.t_n + 1))
        # additive
        elif ((self.cooling_schedule == "add") or
                  (self.cooling_schedule == "additive")):
            self.T -= self.alpha
        else:
            raise Exception("Invalid cooling schedule passed.")

    def calc_temp_diff(self, last_5_temps):
        diff = [i-j for i, j in zip(last_5_temps[:-1], last_5_temps[1:])]
        avg_diff = sum(diff) / len(last_5_temps)
        if avg_diff < 0:
            raise Exception("Weird temperatures encountered -- seem to go up?")
        return avg_diff

    def stopping_criterion(self, *kwargs):
        """
        Returns TRUE if any stopping criterion is reached. Used the standard
        stopping criteria if no other inputs are given.

        Parameters
        ----------
        *kwargs : float
            possible values for the stopping criteria.

        Returns
        -------
        bool
            True if the stopping criterion is reached, false otherwise.

        """
        # if number of iterations (in T) > max iters
        if self.t_n > self.max_iters:
            print("Stopping Criterion: Max. iterations reached")
            return True
        # if temperature T < min temp
        if self.T < self.min_temp:
            print("Stopping Criterion: Min. temperature reached")
            return True
        # if obj. func. has been evaluated more than max_func_evals
        if self.func_evals > self.max_func_evals:
            print("Stopping Criterion: Max. function evaluations reached")
            return True
        # if max runtime is exceeded
        if (time.time() - self.start_time) > self.max_time:
            print("Stopping Criterion: Max. runtime is exceeded")
            return True

        # calculate average relative change of the obj. func.
        if len(self.func_vals) >= self.acceptance_iters:
            func_change = (abs(self.func_vals[-1] - self.func_vals[0]) /
                           (self.acceptance_iters * max(1,
                                            abs(self.func_vals[-1]))))
            # if average relative change < acceptance tolerance
            if func_change < self.acceptance_tol:
                print("Stopping Criterion: Acceptance tolerance reached")
                return True
        # check best objective value
        if self.best_value >= self.max_obj_val:
            print("Stopping Criterion: Maximum Objective Value reached")
            return True
        # else return False
        return False

    def iter_at_temperature(self):
        """
        at each temperature, a certain number of iterations are carried out
        before the algorithm re-anneals. This is controlled by the parameter
        niters.
        This function tracks some statistics as well.

        Returns
        -------
        iter_stats : dict
            contains three key values: number of better solutions that were
            accepted ("better_accepted"), number of worse solutions that were
            accepted ("worse_accepted"), and number of worse solutions that
            were rejected ("worse_rejected").

        """
        iter_stats = {"better_accepted": 0,
                      "worse_accepted": 0,
                      "worse_rejected": 0}
        # we iterate over niters
        for it in range(self.niters):
            # generate candidate edges and calculate its objective value
            candidate_edges = self.permute_edges()
            candidate_value = self.evaluate_matching(candidate_edges)

            # if the current matching improves the previous matching
            if candidate_value > self.best_value:
                self.edges = candidate_edges
                # save as best we have found so far
                self.best_edges = candidate_edges
                self.best_value = candidate_value
                # track statistics
                iter_stats["better_accepted"] += 1

            # if it doesn't, we still accept with a certain probability
            else:
                # calculate the acceptance criterion
                prob = self.acceptance_probability(candidate_value)
                rand_val = random.random()

                if rand_val < prob:
                    self.edges = candidate_edges
                    self.best_value = candidate_value
                    self.best_edges = candidate_edges

                    # track statistics
                    iter_stats["worse_accepted"] += 1

                else:
                    # track statistics
                    iter_stats["worse_rejected"] += 1

            # save best values
            if len(self.func_vals) < self.acceptance_iters:
                self.func_vals.append(self.best_value)
            else:
                temp = self.func_vals[1:]
                temp.append(self.best_value)
                self.func_vals = temp

        self.stats["better_accepted"] += iter_stats["better_accepted"]
        self.stats["worse_accepted"] += iter_stats["worse_accepted"]
        self.stats["worse_rejected"] += iter_stats["worse_rejected"]

        return iter_stats

    def calculate_acceptance_rate(self, stats):
        """ calculate the acceptance rate """
        total_proposed = stats["worse_accepted"] + stats["worse_rejected"]
        if total_proposed != 0:
            acceptance_rate = stats["worse_accepted"] / total_proposed
        else:
            acceptance_rate = 0
        return acceptance_rate

    def optimize(self):
        """
        The central function of the class, running the optimization part
        of the algorithm. Continues until at least one stopping criterion is
        reached and tracks some statistics through each iteration.

        Returns
        -------
        list
            returns a list containing at index
                0: list of best edges (as list [h,d])
                1: best objective value found
                2: statistics (better / worse accepted and worse rejected)
                3: dictionary with keys = Temperature and values = best
                    objective function value over all temperatures visited
                4: dictionary with keys = Temperature and values = acceptance
                    ratio over all temperatures visited

        """
        # calculate weights w
        self.calculate_weights()
        # initialize edges
        self.get_inital_solution()

        # keep track of some statistics:
        # how many better solutions accepted
        # how many worse solutions accepted
        self.stats = {"better_accepted": 0,
                      "worse_accepted": 0,
                      "worse_rejected": 0}

        self.best_edges = self.edges
        self.best_value = self.evaluate_matching(self.edges)

        continue_while = True
        self.t_n = 1  # counter for number of temperature steps

        # save best value at each temperature for plots
        value_dict = {}
        acceptance_rate_dict = {}

        while continue_while:
            # at temperature: search for new candiates niter times
            iter_stats = self.iter_at_temperature()
            acceptance_rate = self.calculate_acceptance_rate(iter_stats)
            acceptance_rate_dict[self.T] = acceptance_rate
            # print("T = " + str(self.T) + ", Rate = " + str(acceptance_rate))
            value_dict[self.T] = self.best_value

            # update temperature
            self.calculate_temperature()
            self.t_n += 1  # increase temperature step

            if self.stopping_criterion():
                continue_while = False

        self.print_optimum()
        print("Ending temperature:", self.T)

        return [self.best_edges, self.best_value, self.stats,
                value_dict, acceptance_rate_dict]


if __name__ == "__main__":
    """ Inputs """
    # all inputs are defined here

    # DATASET
    # these inputs define the dataset to use for the optimization
    number_of_dwellings = 10
    number_of_households = 10
    number_of_grid_cells = None

    sub_dir = "data"
    save_best_edges = True  # save the optimal edges

    # OPTIMIZATION
    # these inputs define additional parameters and options of the optimization
    cooling_parameter = 0.8
    repetitions = 5
    cooling_schedule = "exp"
    starting_temperature = 1.6
    number_of_iterations = 10

    # INPUTS
    # set a different random seed for each repetition
    seeds = [random.randint(0, 100) for _ in range(repetitions)]

    # get input
    data_dict, model_path, save_str, _, _ = get_input(
        number_of_cells=number_of_grid_cells,
        num_households=number_of_households,
        num_dwellings=number_of_dwellings,
        sub_dir=sub_dir)

    best_val_by_temp = {}
    for rep in range(repetitions):
        random.seed(seeds[rep])
        anneal = Simulated_Annealing(data_dict,
                                     starting_temp=starting_temperature,
                                     niters=number_of_iterations,
                                     cooling_schedule=cooling_schedule,
                                     cooling_param=cooling_parameter)
        data = anneal.optimize()

        # save best edges found in this repetition
        if save_best_edges:
            save_path_temp = save_str + "_" +\
                str(cooling_schedule).replace(".", "") + "_" + \
                str(cooling_parameter).replace(".", ",") + "_" +\
                str(starting_temperature) + "_" + \
                str(rep) + ".json"
            with open(save_path_temp, "w+") as f:
                json.dump(data, f)
