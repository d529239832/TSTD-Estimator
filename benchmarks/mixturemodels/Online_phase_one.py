import getopt
import json
import os
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
from operator import itemgetter
import faulthandler

warnings.filterwarnings("ignore")
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


def parse_online_phase_one_args(argv: list[str]):
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


def lookup_df(folder_path, cond_num):
    json_path = os.path.join(folder_path, f"{cond_num}_conditions.json")
    with open(json_path, "r") as file:
        info_json = json.load(file)

    parquet_path = os.path.join(folder_path, f"{cond_num}_records.parquet")
    cond_df = pd.read_parquet(parquet_path)
    total_count = len(cond_df)
    logger.info(f"Parquet file {parquet_path} is loaded.")
    logger.info(f"Total number of samples in this empirical dataset: {total_count}")

    return info_json, cond_df


def Convex_detection(tail_prob):
    k = len(tail_prob)
    range_value = 2
    convex_point_array = []
    for n in range(0, k - range_value):
        value_on_line = ((tail_prob[n] - tail_prob[n + range_value]) / (-2)) * (-1) + tail_prob[n + range_value]
        delta_ = tail_prob[int(n + range_value / 2)] - value_on_line
        if delta_ > 0:
            convex_point_array.append((int(n + range_value / 2), delta_))
    return convex_point_array


def Cluster_convex_points(convex_point_array):
    k = len(convex_point_array)
    if k == 0: return []
    cluster_temp = []
    cluster_resluts = []
    for n in range(0, k):
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
        convex_start_index = Cluster_convex_start_list[n][0]
        convex_start_value = tail_prob[convex_start_index]
        n1 = convex_start_index + 2
        convex_tag = False
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
                Cluster_convex_range_list.append((convex_start_index, n1 - 1, convex_start_value - tail_prob[n1 - 1]))
            else:
                Cluster_convex_range_list.append((convex_start_index, n1, convex_start_value - tail_prob[n1]))
        else:
            Cluster_convex_range_list.append((convex_start_index, n1 - 1, convex_start_value - tail_prob[n1 - 1]))
    return Cluster_convex_range_list


def Max_Diff_range_list(diff_list):
    k = len(diff_list)
    DiffRange_list = []
    for n in range(0, k):
        DiffRange_list.append((n + 1, n + 1, abs(diff_list[n])))
    if not DiffRange_list: return [(0, 0, 0)]
    Max_Diff_range_list_ = sorted(DiffRange_list, key=itemgetter(2), reverse=True)
    return [Max_Diff_range_list_[0]]


def run_online_phase_one_processes(exp_args: list):
    logger.info("Prepare models benchmark validate args (No-Spark) with command line args: " + f"{exp_args}")

    p = Path(__file__).parents[0]
    main_path = str(p) + "/"
    project_path = main_path + exp_args["label"] + "_results/"
    os.makedirs(project_path, exist_ok=True)

    dataset_project_path = main_path + exp_args["dataset"] + "_results/"

    # 初始化一个空的字典，确保每次运行都是全新的结果，不再读取旧文件
    all_thresholds = {}

    conditions = []
    cond_dataframes = []
    for cond_num in exp_args["condition_nums"]:
        cond_dict, cond_df = lookup_df(dataset_project_path, cond_num)
        cond_dataframes.append(cond_df)
        conditions.append(cond_dict)

    key_label = exp_args["target"]
    y_points = np.linspace(
        start=exp_args["y_points"][0],
        stop=exp_args["y_points"][2],
        num=exp_args["y_points"][1],
    )
    scale_ = exp_args["y_points"][1] / (exp_args["y_points"][2] - exp_args["y_points"][0])

    for idx, cond_dict in enumerate(conditions):
        logger.info(f"Processing dataframe {idx} with conditions {cond_dict}")
        cond_df = cond_dataframes[idx]
        total_count = len(cond_df)

        # 极速计算：使用 np.searchsorted
        sorted_data = np.sort(cond_df[key_label + '_scaled'].values)
        counts = np.searchsorted(sorted_data, y_points, side='right')
        emp_cdf = counts / total_count
        tail_prob = 1.0 - np.array(emp_cdf, dtype=np.float64)

        convex_point_array = Convex_detection(tail_prob)
        Cluster_convex_list = Cluster_convex_points(convex_point_array)
        Cluster_convex_start_list = Cluster_convex_start_point(Cluster_convex_list)
        Cluster_convex_range_list = Cluster_convex_range(Cluster_convex_start_list, tail_prob)

        diff_list = np.diff(tail_prob)
        tail_prob_diff_range_list = Max_Diff_range_list(diff_list)

        threshold_values = sorted(Cluster_convex_range_list, key=itemgetter(2), reverse=True)

        # 最大斜率点（一定存在）
        diff_idx = tail_prob_diff_range_list[0][0]
        x_index1 = diff_idx / scale_

        # 只选“在 diff_idx 右侧”的 convex 起点
        valid_candidates = [
            v for v in threshold_values
            if v[0] > diff_idx
        ]

        if valid_candidates:
            # threshold_values 已经按 diff 排序过，取最近的即可
            x_index2 = valid_candidates[0][0] / scale_
        else:
            # 保底策略：不允许 u2 < u1
            x_index2 = x_index1

        print(f"u1 = {x_index1}, u2 = {x_index2}")

        # 将结果存入字典
        all_thresholds[f"length_{cond_dict}"] = {
            "u1": x_index1,
            "u2": x_index2,
        }


    # 循环结束后，一次性写入文件，从而覆盖掉以前的所有旧阈值
    threshold_file_path = os.path.join(project_path, "thresholds.json")
    with open(threshold_file_path, "w") as f:
        json.dump(all_thresholds, f, indent=4)

    logger.info(f"Thresholds saved to {threshold_file_path}. Old data cleared.")