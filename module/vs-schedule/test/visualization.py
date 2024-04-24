#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
import re

def plot_subfigure (ax, title, var_name, var_vec, latency_vec):
    ax.set_title(title)
    ax.set_xlabel(var_name)
    ax.set_ylabel("Latency [s]")
    ax.scatter(var_vec, latency_vec, color='#666666', label='raw data', s=50)
    # plot the line of best fit
    m, b = np.polyfit(var_vec, latency_vec, 1)
    mse_linear = np.mean((m*var_vec + b - latency_vec)**2)
    ax.plot(var_vec, m*var_vec + b, color='green', label='linear fit: MSE='+'{:.3f}'.format(mse_linear))
    # plot the best fit exponential curve
    p = np.polyfit(var_vec, np.log(latency_vec), 1)
    mse_exponential = np.mean((np.exp(p[1])*np.exp(p[0]*var_vec) - latency_vec)**2)
    ax.plot(var_vec, np.exp(p[1])*np.exp(p[0]*var_vec), color='red', label='exp fit: MSE='+'{:.3f}'.format(mse_exponential))
    # plot the best fit logarithmic curve
    p = np.polyfit(np.log(var_vec), latency_vec, 1)
    mse_logarithmic = np.mean((np.log(var_vec)*p[0]+p[1] - latency_vec)**2)
    ax.plot(var_vec, np.log(var_vec)*p[0]+p[1], color='blue', label='log fit: MSE='+'{:.3f}'.format(mse_logarithmic))
    ax.legend(loc='upper left')

def collect_data(directory: str) -> dict:
    testcase_dir = os.path.join(directory, "testcases")
    if not os.path.exists(testcase_dir):
        print("Directory does not exist")
        sys.exit(1)
    
    # get all directories in the testcases directory
    dirs = [d for d in os.listdir(testcase_dir) if os.path.isdir(os.path.join(testcase_dir, d))]

    # Figure out the categories, each category should start with a word, then a underscore, then a number
    # store the category name in a list
    categories = set()
    for d in dirs:
        # check if the name matches the pattern
        if re.match(r"^(\w)+_\d+$", d):
            categories.add(d.split("_")[0])
        else:
            logging.warning("Encounter invalid directory name: "+d+", ignored!")
    
    # make a list for each category
    category_dirs = {}
    for c in categories:
        category_dirs[c] = []
    for d in dirs:
        for c in categories:
            if d.startswith(c):
                category_dirs[c].append(d)
    
    # sort each list based on the numbers
    for c in categories:
        category_dirs[c].sort(key=lambda x: int(x.split("_")[1]))
    
    # create a var_vec for each category, the value is the number in the directory name
    var_vec = {}
    for c in categories:
        var_vec[c] = np.array([int(d.split("_")[1]) for d in category_dirs[c]])
    
    # collect the time record for each category, the time record is number stored in time_record.txt in each directory
    latency_vec = {}
    for c in categories:
        latency_vec[c] = []
        for d in category_dirs[c]:
            with open(os.path.join(testcase_dir, d, "time_record.txt"), "r") as file:
                latency_vec[c].append(float(file.readline()))
    
    # create final return structure
    ret_dict = {}
    for c in categories:
        ret_dict[c] = (var_vec[c], latency_vec[c])
    
    return ret_dict



def create_figure(directory: str, vars: list):
    testcase_dir = os.path.join(directory, "testcases")
    if not os.path.exists(testcase_dir):
        print("Directory does not exist")
        sys.exit(1)
    
    data = collect_data(directory)


    fig, axs = plt.subplots(len(vars), 1, layout='constrained', figsize=(6, 8))
    for i in range(len(vars)):
        key = vars[i]
        var_vec, latency_vec = data[key]
        plot_subfigure(axs[i], "Latency increases with "+key, key, var_vec, latency_vec)

    # save the figure to pdf
    plt.savefig(os.path.join(directory, "plot.pdf"))



directory = "testcases"
create_figure(directory, ['N', 'M', 'C'])