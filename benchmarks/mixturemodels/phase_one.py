import ast
import decimal
import gc
import getopt
import json
import math
import multiprocessing as mp
import os
import pdb
# import pdb
import sys
import time
import warnings
from os.path import abspath, dirname
from pathlib import Path
import pickle

import numpy as np
import matplotlib.pyplot as plt

import polars as pl
from loguru import logger
from pyspark.sql import SparkSession
from operator import itemgetter
from pr3d.de import ConditionalGaussianMM, ConditionalGaussianMixtureEVM

warnings.filterwarnings("ignore")

import faulthandler

faulthandler.enable()

# 设置图形边框的粗细
def adjust_plot_borders(ax):
    # 设置四个边框的粗细
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)  # 1是边框的粗细，可以根据需要调整

def get_max_pair(list1, list2):
    return np.array([max(a, b) for a, b in zip(list1, list2)])


def get_min_pair(list1, list2):
    return np.array([min(a, b) for a, b in zip(list1, list2)])


def parse_validate_pred_args_one(argv: list[str]):
    # parse arguments to a dict
    args_dict = {}
    try:
        opts, args = getopt.getopt(
            argv,
            "hd:t:x:n:m:l:r:c:y:u:f:i:p:g:o:s:",
            [
                "dataset=",
                "target=",
                "condition-nums=",
                "condition-markers=",
                "models=",
                "label=",
                "rows=",
                "columns=",
                "y-points=",
                "prob-lims=",
                "plot-pdf",
                "plot-cdf",
                "plot-tail",
                "log",
                "loglog",
                "split",
            ],
        )
    except getopt.GetoptError:
        print('Wrong args, type "python -m models_benchmark validate -h" for help')
        sys.exit(2)
    # import pdb
    # pdb.set_trace()
    # opts = [('-d', 'wifi/prepped_length'), ('-t', 'receive'), ('-x', '0,1,2,3'), ('-m', 'wifi/trained_length_dl_l.gmevm.0,wifi/trained_length_dl_l.gmm.0'), ('-l', 'wifi/validate_length_l'), ('--plot-tail', ''), ('--log', ''), ('-r', '2'), ('-c', '2'), ('-y', '0,400,200')]
    # default values
    args_dict["prob_lims"] = None
    args_dict["condition_markers"] = None
    args_dict["y_points"] = [0, 100, 400]
    args_dict["plotcdf"] = False
    args_dict["plotpdf"] = False
    args_dict["plottail"] = False
    args_dict["logplot"] = False
    args_dict["loglogplot"] = False

    # parse the args
    for opt, arg in opts:
        if opt == "-h":
            print(
                "python -m models_benchmark validate "
                + "-q <qlens> -d <dataset> -m <trained models> -l <label> -e <ensemble num> ",
            )
            sys.exit()
        elif opt in ("-d", "--dataset"):
            args_dict["dataset"] = arg
        elif opt in ("-t", "--target"):
            args_dict["target"] = arg
        elif opt in ("-x", "--condition-nums"):
            args_dict["condition_nums"] = [int(s.strip()) for s in arg.split(",")]
        elif opt in ("-n", "--condition-markers"):
            args_dict["condition_markers"] = [s.strip() for s in arg.split(",")]
        elif opt in ("-m", "--models"):  # 使用"."分割
            args_dict["models"] = [s.strip().split(".") for s in arg.split(",")]
        elif opt in ("-l", "--label"):
            args_dict["label"] = arg
        elif opt in ("-r", "--rows"):
            args_dict["rows"] = int(arg)
        elif opt in ("-s", "--split"):
            args_dict["split"] = int(arg)
        elif opt in ("-c", "--columns"):
            args_dict["columns"] = int(arg)
        elif opt in ("-y", "--y-points"):
            args_dict["y_points"] = [int(s.strip()) for s in arg.split(",")]
        elif opt in ("-u", "--log-lims"):
            args_dict["prob_lims"] = [np.float64(s.strip()) for s in arg.split(",")]
        elif opt in ("-f", "--plot-cdf"):
            args_dict["plotcdf"] = True
        elif opt in ("-i", "--plot-tail"):  #
            args_dict["plottail"] = True
        elif opt in ("-p", "--plot-pdf"):
            args_dict["plotpdf"] = True
        elif opt in ("-g", "--log"):  #
            args_dict["logplot"] = True
        elif opt in ("-o", "--loglog"):
            args_dict["loglogplot"] = True

    if not args_dict["condition_markers"]:
        args_dict["condition_markers"] = ["." for cond in args_dict["condition_nums"]]

    return args_dict


def lookup_df_split(folder_path, cond_num, k, spark):
    # import pdb
    # pdb.set_trace()
    # json_path = os.path.join(folder_path, f"{k}_conditions.json")
    json_path = os.path.join(folder_path, f"{cond_num}_conditions.json")
    with open(json_path, "r") as file:
        info_json = json.load(file)

    # parquet_path = os.path.join(folder_path, f"{cond_num}{k}_records_n.parquet")
    parquet_path = os.path.join(folder_path, f"{cond_num}{k}_records.parquet")
    cond_df = spark.read.parquet(parquet_path)
    total_count = cond_df.count()
    logger.info(f"Parquet file {parquet_path} is loaded.")
    logger.info(f"Total number of samples in this empirical dataset: {total_count}")

    return info_json, cond_df


def lookup_df(folder_path, cond_num, spark):
    # import pdb
    # pdb.set_trace()
    json_path = os.path.join(folder_path, f"{cond_num}_conditions.json")
    with open(json_path, "r") as file:
        info_json = json.load(file)

    parquet_path = os.path.join(folder_path, f"{cond_num}_records.parquet")
    cond_df = spark.read.parquet(parquet_path)
    total_count = cond_df.count()
    logger.info(f"Parquet file {parquet_path} is loaded.")
    logger.info(f"Total number of samples in this empirical dataset: {total_count}")

    return info_json, cond_df


# Detecte the convex part
def Convex_detection(tail_prob):
    # import pdb
    # pdb.set_trace()
    k = len(tail_prob)
    range_value = 2
    convex_point_array = []

    for n in range(0, k - range_value):
        # import pdb
        # pdb.set_trace()
        # mean_value = (tail_prob[n] + tail_prob[n + range_value]) / 2
        value_on_line = ((tail_prob[n] - tail_prob[n + range_value]) / (-2)) * (-1) + tail_prob[n + range_value]
        delta_ = tail_prob[int(n + range_value / 2)] - value_on_line
        if delta_ > 0:
            convex_point_array.append((int(n + range_value / 2), delta_))

    return convex_point_array


def Cluster_convex_points(convex_point_array):
    k = len(convex_point_array)

    cluster_temp = []
    cluster_resluts = []
    for n in range(0, k):
        # import pdb
        # pdb.set_trace()
        cluster_temp.append(convex_point_array[n])
        if n != k - 1:
            if convex_point_array[n][0] + 1 != convex_point_array[n + 1][0]:
                cluster_resluts.append(cluster_temp)
                cluster_temp = []
            else:
                continue

    cluster_resluts.append(cluster_temp)

    return cluster_resluts


def Cluster_convex_start_point(cluster_resluts):
    k = len(cluster_resluts)

    Cluster_convex_start_list = []
    for n in range(0, k):
        Cluster_convex_start_list.append(cluster_resluts[n][0])

    return Cluster_convex_start_list


def Cluster_convex_range(Cluster_convex_start_list, tail_prob):
    k = len(Cluster_convex_start_list)
    len_tail_prob = len(tail_prob)

    Cluster_convex_range_list = []

    for n in range(0, k):
        # import pdb
        # pdb.set_trace()
        convex_start_index = Cluster_convex_start_list[n][0]
        # convex_start_index = Cluster_convex_start_list[n][0] - 1
        convex_start_value = tail_prob[convex_start_index]
        n1 = convex_start_index + 2
        # convex_tag = False
        for n1 in range(convex_start_index + 2, len_tail_prob):

            value_on_line = ((convex_start_value - tail_prob[n1]) / (convex_start_index - n1)) * (-1) + tail_prob[n1]
            if value_on_line < tail_prob[n1 - 1]:
                convex_tag = True
                continue
            else:
                convex_tag = False
                break
        if n1 == len_tail_prob - 1:
            if not convex_tag:
                Cluster_convex_range_list.append((convex_start_index, n1-1, convex_start_value - tail_prob[n1-1]))
            else:
                Cluster_convex_range_list.append((convex_start_index, n1, convex_start_value - tail_prob[n1]))
        else:
            Cluster_convex_range_list.append((convex_start_index, n1-1, convex_start_value - tail_prob[n1-1]))
    # import pdb
    # pdb.set_trace()

    return Cluster_convex_range_list


def Cluster_convex_range2(Cluster_convex_start_list, tail_prob):
    k = len(Cluster_convex_start_list)
    len_tail_prob = len(tail_prob)

    Cluster_convex_range_list = []

    for n in range(0, k):
        # import pdb
        # pdb.set_trace()
        convex_start_index = Cluster_convex_start_list[n][0] - 1
        convex_start_value = tail_prob[convex_start_index]
        n1 = convex_start_index + 2
        for n1 in range(convex_start_index + 2, len_tail_prob):

            value_on_line = ((convex_start_value - tail_prob[n1]) / (convex_start_index - n1)) * (-1) + tail_prob[n1]
            if value_on_line < tail_prob[n1 - 1]:
                continue
            else:
                break
        if n1 == len_tail_prob - 1:
            Cluster_convex_range_list.append((convex_start_index, n1, convex_start_value - tail_prob[n1]))
        else:
            # print(n1)
            Cluster_convex_range_list.append((convex_start_index, n1-1, convex_start_value - tail_prob[n1-1]))

    return Cluster_convex_range_list


def Cluster_convex_range3(Cluster_convex_list, tail_prob):

    k = len(Cluster_convex_list)
    Cluster_convex_range_list = []

    for n in range(0, k):
        # Cluster_convex_range_list.append((Cluster_convex_list[n][0][0]-1, Cluster_convex_list[n][-1][0]+1, tail_prob[Cluster_convex_list[n][0][0]-1]-tail_prob[Cluster_convex_list[n][-1][0]+1]))
        Cluster_convex_range_list.append((Cluster_convex_list[n][0][0], Cluster_convex_list[n][-1][0],
                                      tail_prob[Cluster_convex_list[n][0][0]] - tail_prob[
                                          Cluster_convex_list[n][-1][0]]))

    return Cluster_convex_range_list

def Threshold_calculate(cluster_resluts):
    k = len(cluster_resluts)

    treshold_values = []
    treshold_convex = []

    treshold_convex.append(cluster_resluts[0])
    treshold_convex.append(cluster_resluts[1])
    len_vonvex_pre = len(cluster_resluts[1])
    for n in range(2, k):
        # import pdb
        # pdb.set_trace()
        len_vonvex = len(cluster_resluts[n])
        if len_vonvex > len_vonvex_pre:
            treshold_convex[1] = cluster_resluts[n]
            len_vonvex_pre = len_vonvex
        else:
            continue
    # import pdb
    # pdb.set_trace()
    if len(treshold_convex[0]) > 1:
        treshold_values.append(treshold_convex[0][1])
    else:
        treshold_values.append(treshold_convex[0][0])

    if len(treshold_convex[1]) > 1:
        treshold_values.append(treshold_convex[1][1])
    else:
        treshold_values.append(treshold_convex[1][0])
    return treshold_values


def Max_Diff_range_list(diff_list):

    k = len(diff_list)
    DiffRange_list = []
    for n in range(0, k):
        DiffRange_list.append((n+1, n+1, abs(diff_list[n])))

    Max_Diff_range_list_ = sorted(DiffRange_list, key=itemgetter(2), reverse=True)
    Max_Diff_range_list = []
    Max_Diff_range_list.append(Max_Diff_range_list_[0])
    return Max_Diff_range_list


def Threshold_calculate2(Cluster_convex_range_list):

    k = len(Cluster_convex_range_list)

    treshold_values = []

    treshold_values.append(Cluster_convex_range_list[0])
    treshold_values.append(Cluster_convex_range_list[1])
    convex_range_pre = Cluster_convex_range_list[1][2]
    for n in range(2, k):
        # import pdb
        # pdb.set_trace()
        convex_range = Cluster_convex_range_list[n][2]
        if convex_range > convex_range_pre:
            treshold_values[1] = Cluster_convex_range_list[n]
            convex_range_pre = convex_range
        else:
            continue
    # import pdb
    # pdb.set_trace()

    return treshold_values


def Threshold_section(convex_point_array):
    # pdb.set_trace()
    k = len(convex_point_array)

    treshold_values = []
    treshold_values.append(convex_point_array[0])
    for n in range(1, k - 1):
        # import pdb
        # pdb.set_trace()
        if convex_point_array[n][0] + 1 == convex_point_array[n + 1][0]:
            treshold_values.append(convex_point_array[n])
        else:
            continue
    return treshold_values


def run_validate_pred_processes_one(exp_args: list):

    logger.info(
        "Prepare models benchmark validate args "
        + f"with command line args: {exp_args}"
    )

    spark = (
        SparkSession.builder.master("local")
        .appName("LoadParquets")
        .config("spark.executor.memory", "50g")  # 原来6g
        .config("spark.driver.memory", "70g")
        .config("spark.driver.maxResultSize", "50g")  # 原来 0
        .getOrCreate()
    )

    # this project folder setting
    p = Path(__file__).parents[0]  #p: /project/wireless-pr3d/benchmarks/mixturemodels
    main_path = str(p) + "/"
    project_path = main_path + exp_args["label"] + "_results/"
    # project_path: /project/wireless-pr3d/benchmarks/mixturemodels/wifi/validate_length_l_results/
    os.makedirs(project_path, exist_ok=True)

    # dataset project folder setting
    dataset_project_path = main_path + exp_args["dataset"] + "_results/"
    # dataset_project_path: /project/wireless-pr3d/benchmarks/mixturemodels/wifi/prepped_length_results/

    # find dataframe with the desired condition
    # inputs: exp_args["condition_nums"]
    conditions = []
    cond_dataframes = []
    u_values = []
    # k = exp_args["split"]
    for cond_num in exp_args["condition_nums"]:
        # cond_dict, cond_df = lookup_df_split(dataset_project_path, cond_num, k, spark)
        cond_dict, cond_df = lookup_df(dataset_project_path, cond_num, spark)
        cond_dataframes.append(cond_df)
        conditions.append(cond_dict)
        u_value = cond_df.approxQuantile("send_scaled", [0.9], 0.0)[0]
        # 将 u_value 存储到 u_values 数组中
        u_values.append(u_value)
        # 打印当前条件的 u_value
        print(f"The 1% quantile (u value) of 'send_scaled' for condition {cond_num} is: {u_value}")
    # json_path = os.path.join(dataset_project_path, f"{k}_info.json")
    # json_path = os.path.join(dataset_project_path, "info.json")
    # with open(json_path, "r") as file:
    #     info_json = json.load(file)

    key_label = exp_args["target"]
    # key_mean = info_json[key_label]["mean"]
    # key_scale = info_json[key_label]["scale"]

    # bulk plot axis
    y_points = np.linspace(
        start=exp_args["y_points"][0],
        stop=exp_args["y_points"][2],
        num=exp_args["y_points"][1],
    )
    # y_points_standard = np.linspace(
    #     start=exp_args["y_points"][0] - (key_mean * key_scale),
    #     stop=exp_args["y_points"][2] - (key_mean * key_scale),
    #     num=exp_args["y_points"][1],
    # )

    single_plot = False
    if exp_args["rows"] == 0 and exp_args["columns"] == 0:
        single_plot = True

    # figure
    # CDF figure
    if exp_args["plotcdf"]:
        if not single_plot:
            nrows = exp_args["rows"]
            ncols = exp_args["columns"]
            cdf_fig, cdf_axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 5 * nrows))
            cdf_axes = cdf_axes.flat
        else:
            cdf_fig, cdf_ax = plt.subplots(nrows=1, ncols=1)

    # Tail figure
    if exp_args["plottail"]:
        if not single_plot:
            nrows = exp_args["rows"]
            ncols = exp_args["columns"]
            tail_fig, tail_axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 5 * nrows))
            tail_axes = tail_axes.flat
        else:
            tail_fig, tail_ax = plt.subplots(nrows=1, ncols=1)

    # PDF figure
    if exp_args["plotpdf"]:
        if not single_plot:
            nrows = exp_args["rows"]
            ncols = exp_args["columns"]
            pdf_fig, pdf_axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 5 * nrows))
            pdf_axes = pdf_axes.flat
        else:
            pdf_fig, pdf_ax = plt.subplots(nrows=1, ncols=1)

    for idx, cond_dict in enumerate(conditions):

        logger.info(f"Plotting dataframe {idx} with conditions {cond_dict}")
        cond_df = cond_dataframes[idx]

        cond_df_pandas = cond_df.toPandas()

        # Calculate emp CDF

        max_point = max(np.array(cond_df_pandas['receive_scaled']))  # 设置间隔边界和间隔大小
        max_point = min(exp_args["y_points"][2], max_point)
        # del cond_df_pandas
        total_count = cond_df.count()
        emp_cdf = list()
        for y in y_points:
            delay_budget = y
            new_cond_df = cond_df.where(cond_df[key_label + '_scaled'] <= delay_budget)
            success_count = new_cond_df.count()
            emp_success_prob = success_count / total_count
            emp_cdf.append(emp_success_prob)

        # Calculate emp PDF
        emp_pdf = np.diff(np.array(emp_cdf))
        scale_ = exp_args["y_points"][1] / (exp_args["y_points"][2] - exp_args["y_points"][0])
        emp_pdf = np.append(emp_pdf, [0]) * scale_
        # emp_pdf = np.append(emp_pdf, [0]) * exp_args["y_points"][1] / (
        #         exp_args["y_points"][2] - exp_args["y_points"][0])

        if exp_args["plotcdf"]:
            if not single_plot:
                ax = cdf_axes[idx]
            else:
                ax = cdf_ax
            ax.plot(y_points,
                    emp_cdf,
                    marker=exp_args["condition_markers"][idx],
                    label=f"meas. {cond_dict}",
                    )

        if exp_args["plottail"]:
            if not single_plot:
                ax = tail_axes[idx]
            else:
                ax = tail_ax
            # import pdb
            # pdb.set_trace()
            tail_prob = np.float64(1.00) - np.array(emp_cdf, dtype=np.float64)
            ax.plot(
                y_points,
                tail_prob,
                marker=exp_args["condition_markers"][idx],
                label=f"meas. {cond_dict}",
            )

            if exp_args["logplot"]:
                ax.set_yscale('log')
            elif exp_args["loglogplot"]:
                ax.set_yscale('log')
                ax.set_xscale('log')

            ax.set_xlim(0, max_point)

            convex_point_array = Convex_detection(tail_prob)
            Cluster_convex_list = Cluster_convex_points(convex_point_array)

            # threshold_values = Threshold_calculate(Cluster_convex_list)

            Cluster_convex_start_list = Cluster_convex_start_point(Cluster_convex_list)

            # Cluster_convex_range_list = Cluster_convex_range3(Cluster_convex_list, tail_prob)
            Cluster_convex_range_list = Cluster_convex_range(Cluster_convex_start_list, tail_prob)
            # Cluster_convex_range_list = Cluster_convex_range(Cluster_convex_start_list, np.log10(tail_prob))

            diff_list = np.diff(tail_prob)
            tail_prob_diff_range_list = Max_Diff_range_list(diff_list)

            # min_index = np.argmin(diff_list)
            # Cluster_convex_diff_range_list = list(set(tail_prob_diff_range_list + Cluster_convex_range_list))

            threshold_values = sorted(Cluster_convex_range_list, key=itemgetter(2), reverse=True)
            # threshold_values = Threshold_calculate2(Cluster_convex_range_list)

            len_threshold_values = len(threshold_values)
            x_index1 = x_index2 = 0
            if len_threshold_values > 0:
                for k in range(0, len_threshold_values):
                    if tail_prob_diff_range_list[0][0] < threshold_values[k][0]:
                        x_index1 = tail_prob_diff_range_list[0][0] / scale_
                        x_index2 = threshold_values[k][0] / scale_
                        break
                    elif tail_prob_diff_range_list[0][0] >= threshold_values[k][0] and tail_prob_diff_range_list[0][
                        0] <= \
                            threshold_values[k][1]:
                        x_index1 = tail_prob_diff_range_list[0][0] / scale_
                        if k + 1 < len(threshold_values):
                            x_index2 = threshold_values[k + 1][0] / scale_
                        else:
                            x_index2 = threshold_values[k][1] / scale_
                        if x_index1 >= x_index2: continue
                        break

            ax.plot([x_index1, x_index1], [0, 1], color='red',linestyle='--')
            ax.plot([x_index2, x_index2], [0, 1], color='blue',linestyle='--')
            if not single_plot:  # 标题
                ax.set_title(f" ", fontsize=10)
            if idx == 0:
                ax.set_xlim(0, 60)
                ax.set_ylim(1e-5, 1)
                ax.text(0.95, 0.95, "Length: 172 bytes", fontsize=23, ha='right', va='top', transform=ax.transAxes)
            elif idx == 1:
                ax.set_xlim(0, 60)
                ax.set_ylim(1e-6, 1)
                ax.text(0.95, 0.95, "Length: 3440 bytes", fontsize=23, ha='right', va='top', transform=ax.transAxes)
            elif idx == 2:
                ax.set_xlim(0, 80)
                ax.set_ylim(1e-6, 1)
                ax.text(0.95, 0.95, "Length: 6880 bytes", fontsize=23, ha='right', va='top', transform=ax.transAxes)
            elif idx == 3:
                ax.set_xlim(0, 100)
                ax.set_ylim(1e-6, 1)
                ax.text(0.95, 0.95, "Length: 10,320 bytes", fontsize=23, ha='right', va='top', transform=ax.transAxes)
            # print("u =", x_index)
            print("u1 =", x_index1)
            print("u2 =", x_index2)
            # ax.set_xlim(0, 100)  # 设置X轴范围
            # ax.set_ylim(1e-6, 1)
            ax.set_xlabel("End-to-End delay [ms]", fontsize=26)  # 字体大小
            ax.set_ylabel("Tail probability", fontsize=26)  # 字体大小
            # 设置XY轴的刻度字体大小和刻度线粗细
            ax.tick_params(axis='both', which='major', labelsize=22, width=1.2)  # 主刻度线
            ax.tick_params(axis='both', which='minor', labelsize=20, width=1.0)  # 次刻度线
            # 控制刻度线的长度
            ax.tick_params(axis='both', which='major', length=6)  # 这里控制刻度线长度
            ax.tick_params(axis='both', which='minor', length=4)
            # 设置图形边框的粗细
            adjust_plot_borders(ax)

        if exp_args["plotpdf"]:
            if not single_plot:
                ax = pdf_axes[idx]
            else:
                ax = pdf_ax
            ax.plot(
                y_points,
                emp_pdf,
                marker=exp_args["condition_markers"][idx],
                label=f"meas. {cond_dict}",
            )

    if exp_args["plotcdf"]:
        # cdf figure
        cdf_fig.tight_layout()
        if exp_args["logplot"]:
            cdf_fig.savefig(project_path + f"{key_label}_log_cdf.png")
        elif exp_args["loglogplot"]:
            cdf_fig.savefig(project_path + f"{key_label}_loglog_cdf.png")
        else:
            cdf_fig.savefig(project_path + f"{key_label}_cdf.png")

    if exp_args["plottail"]:
        # cdf figure
        tail_fig.tight_layout()
        if exp_args["logplot"]:
            # tail_fig.savefig(project_path + f"{key_label}_log_tail_" + f"{ensemble_num}.png")
            tail_fig.savefig(project_path + f"{key_label}_log_tail.png")
        elif exp_args["loglogplot"]:
            tail_fig.savefig(project_path + f"{key_label}_loglog_tail.png")
        else:
            tail_fig.savefig(project_path + f"{key_label}_tail.png")

    if exp_args["plotpdf"]:
        # pdf figure
        pdf_fig.tight_layout()
        if exp_args["logplot"]:
            pdf_fig.savefig(project_path + f"{key_label}_log_pdf.png")
        elif exp_args["loglogplot"]:
            pdf_fig.savefig(project_path + f"{key_label}_loglog_pdf.png")
        else:
            pdf_fig.savefig(project_path + f"{key_label}_pdf.png")
    spark.stop()
