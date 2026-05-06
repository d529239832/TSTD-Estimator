import ast
import decimal
import gc
import getopt
import json
import multiprocessing as mp
import os
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
from pr3d.de import (ConditionalGaussianMM, ConditionalGaussianMixtureEVM,ConditionalGaussianMixtureEVMW2WU)

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


def parse_validate_pred_args_mean(argv: list[str]):
    # parse arguments to a dict
    args_dict = {}
    try:
        opts, args = getopt.getopt(
            argv,
            "hd:t:x:n:m:l:r:c:y:z:f:i:p:g:o:s:q:u:v:k:e:",
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
                "datatype",
                "u1=",
                "u2=",
                "nol=",
                "plotsample",
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
    args_dict["plotsample"] = False

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
        elif opt in ("-z", "--log-lims"):
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
        elif opt in ("-q", "--datatype"):
            args_dict["datatype"] = arg
        elif opt in ("-u", "--u1"):
            args_dict["u1"] = [float(x) for x in arg.split(",")]
        elif opt in ("-v", "--u2"):
            args_dict["u2"] = [float(x) for x in arg.split(",")]
        elif opt in ("-k", "--nol"):
            args_dict["normalize"] = [float(x) for x in arg.split(",")]
        elif opt in ("-e", "--plot-sample"):
            args_dict["plotsample"] = True
    if not args_dict["condition_markers"]:
        args_dict["condition_markers"] = ["." for cond in args_dict["condition_nums"]]


    return args_dict


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


def run_validate_pred_processes_mean(exp_args: list):
    # import pdb
    # pdb.set_trace()
    logger.info(
        "Prepare models benchmark validate args "
        + f"with command line args: {exp_args}"
    )
    # import pdb
    # pdb.set_trace()
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
    # import pdb
    # pdb.set_trace()
    # dataset project folder setting
    dataset_project_path = main_path + exp_args["dataset"] + "_results/"
    # dataset_project_path: /project/wireless-pr3d/benchmarks/mixturemodels/wifi/prepped_length_results/

    # find dataframe with the desired condition
    # inputs: exp_args["condition_nums"]
    conditions = []
    cond_dataframes = []
    # import pdb
    # pdb.set_trace()
    # k = exp_args["split"]
    for cond_num in exp_args["condition_nums"]:
        cond_dict, cond_df = lookup_df(dataset_project_path, cond_num, spark)
        cond_dataframes.append(cond_df)
        conditions.append(cond_dict)

    # json_path = os.path.join(dataset_project_path, f"{k}_info.json")
    json_path = os.path.join(dataset_project_path, "info.json")
    with open(json_path, "r") as file:
        info_json = json.load(file)

    key_label = exp_args["target"]
    key_mean = info_json[key_label]["mean"]
    key_scale = info_json[key_label]["scale"]

    # bulk plot axis
    y_points = np.linspace(
        start=exp_args["y_points"][0],
        stop=exp_args["y_points"][2],
        num=exp_args["y_points"][1],
    )
    y_points_standard = np.linspace(
        start=exp_args["y_points"][0] - (key_mean * key_scale),
        stop=exp_args["y_points"][2] - (key_mean * key_scale),
        num=exp_args["y_points"][1],
    )

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

        # del cond_df_pandas
        total_count = cond_df.count()
        emp_cdf = list()
        emp_noisycdf = list()
        emp_verynoisycdf = list()
        for y in y_points:
            delay_budget = y
            new_cond_df = cond_df.where(cond_df[key_label + '_scaled'] <= delay_budget)
            success_count = new_cond_df.count()
            emp_success_prob = success_count / total_count
            emp_cdf.append(emp_success_prob)

        emp_pdf = np.diff(np.array(emp_cdf))
        emp_pdf = np.append(emp_pdf, [0]) * exp_args["y_points"][1] / (
                exp_args["y_points"][2] - exp_args["y_points"][0])

        # 画图抽样21%的测量数据集，用作数学统计分析对比
        # 默认：不采样，直接用全量
        cond_df_sample = cond_df
        sample_count = cond_df.count()

        if exp_args.get("plotsample", False):
            cond_df_sample = cond_df.sample(fraction=0.21, seed=42)
            sample_count = cond_df_sample.count()

        emp_cdf_sample = []
        for y in y_points:
            delay_budget = y
            new_df = cond_df_sample.where(
                cond_df_sample[key_label + '_scaled'] <= delay_budget
            )
            success_count = new_df.count()
            emp_cdf_sample.append(success_count / sample_count)

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
            ax.plot(
                y_points,
                np.float64(1.00) - np.array(emp_cdf, dtype=np.float64),
                marker=exp_args["condition_markers"][idx],
                #color='black',
                label=f"MEAS",
                #label=f"meas. {cond_dict}",
                #markersize=30
            )
            # 画图抽样21%的测量数据集，用作数学统计分析对比
            if exp_args["plotsample"]:
                ax.plot(
                    y_points,
                    np.float64(1.00) - np.array(emp_cdf_sample, dtype=np.float64),
                    linestyle="--",
                    linewidth=2,
                    color="purple",
                    alpha=0.8,
                    label="MEAS (21%)"
                )

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

        # import pdb
        # pdb.set_trace()

        # plot predictions
        for model_list in exp_args["models"]:
            # import pdb
            # pdb.set_trace()
            model_project_name = model_list[0]
            model_conf_key = model_list[1]
            ensemble_num = model_list[2]
            model_path = (
                    main_path + model_project_name + "_results/" + model_conf_key + "/"
            )
            # model_path: /project/wireless-pr3d/benchmarks/mixturemodels/wifi/trained_length_dl_l_results/gmevm/
            with open(
                    model_path + f"model_{ensemble_num}.json"
            ) as json_file:
                model_dict = json.load(json_file)

            cond_columns = []
            for cond_label in model_dict["condition_labels"]:
                cond_columns.append(cond_label)

            # select columns and sample the rows
            rows = cond_df.select(cond_columns).sample(False, (len(y_points_standard) * 2) / cond_df.count(),
                                                       seed=0).limit(len(y_points_standard))
            x_rows = rows.collect()

            # define x numpy list
            x_list = []
            for row in x_rows:
                x_list.append(
                    [row[colm] for colm in cond_columns]
                )
            x = np.array(x_list)

            # define y points and run the inference
            y = np.array(y_points_standard, dtype=np.float64)

            if model_dict["type"] == "gmm":
                pr_model = ConditionalGaussianMM(
                    h5_addr=model_path + f"model_{ensemble_num}.h5",
                )
                # import pdb
                # pdb.set_trace()
                prob, logprob, pred_cdf = pr_model.prob_batch(x, y)
                pred_cdf_tail = np.float64(1.00) - np.array(pred_cdf, dtype=np.float64)
                max_cdf = pred_cdf_tail
                min_cdf = pred_cdf_tail

                for k in range(1, 9):
                    # import pdb
                    # pdb.set_trace()
                    # if k != 3 and k!=5:
                    pr_model = ConditionalGaussianMM(
                        h5_addr=model_path + f"model_{k}.h5",
                    )
                    # import pdb
                    # pdb.set_trace()
                    prob_, logprob_, pred_cdf_ = pr_model.prob_batch(x, y)

                    # import pdb
                    # pdb.set_trace()
                    pred_cdf_tail_ = np.float64(1.00) - np.array(pred_cdf_, dtype=np.float64)
                    max_cdf = get_max_pair(max_cdf, pred_cdf_tail_)
                    min_cdf = get_min_pair(min_cdf, pred_cdf_tail_)

                    prob = prob + prob_
                    logprob = logprob + logprob_
                    pred_cdf = pred_cdf + pred_cdf_

                    del pr_model
                    del prob_, logprob_, pred_cdf_, pred_cdf_tail_
                    gc.collect()
                    # else:
                    #     continue

                prob = prob / 9
                logprob = logprob / 9
                pred_cdf = pred_cdf / 9
            elif model_dict["type"] == "gmevm":
                # import pdb
                # pdb.set_trace()
                pr_model = ConditionalGaussianMixtureEVM(
                    h5_addr=model_path + f"model_{ensemble_num}.h5",
                )
                prob, logprob, pred_cdf = pr_model.prob_batch(x, y)
                pred_cdf_tail = np.float64(1.00) - np.array(pred_cdf, dtype=np.float64)
                max_cdf = pred_cdf_tail
                min_cdf = pred_cdf_tail

                for k in range(1, 9):
                    # import pdb
                    # pdb.set_trace()
                    # if k != 3 and k!=5:
                    pr_model = ConditionalGaussianMixtureEVM(
                        h5_addr=model_path + f"model_{k}.h5",
                    )
                    # import pdb
                    # pdb.set_trace()
                    prob_, logprob_, pred_cdf_ = pr_model.prob_batch(x, y)

                    # import pdb
                    # pdb.set_trace()
                    pred_cdf_tail_ = np.float64(1.00) - np.array(pred_cdf_, dtype=np.float64)
                    max_cdf = get_max_pair(max_cdf, pred_cdf_tail_)
                    min_cdf = get_min_pair(min_cdf, pred_cdf_tail_)

                    prob = prob + prob_
                    logprob = logprob + logprob_
                    pred_cdf = pred_cdf + pred_cdf_

                    del pr_model
                    del prob_, logprob_, pred_cdf_, pred_cdf_tail_
                    gc.collect()
                    # else:
                    #     continue

                prob = prob / 9
                logprob = logprob / 9
                pred_cdf = pred_cdf / 9

            elif model_dict["type"] == "gmevmw2wu":
                # import pdb
                # pdb.set_trace()
                pr_model = ConditionalGaussianMixtureEVMW2WU(
                     h5_addr=model_path + f"model_{ensemble_num}.h5",
                    datatype=exp_args["datatype"],
                    u1=exp_args["u1"],
                    u2=exp_args["u2"],
                    normalize=exp_args["normalize"],
                )
                prob, logprob, pred_cdf = pr_model.prob_batch(x, y)
                pred_cdf_tail = np.float64(1.00) - np.array(pred_cdf, dtype=np.float64)
                max_cdf = pred_cdf_tail
                min_cdf = pred_cdf_tail
                del pr_model
                gc.collect()
                for k in range(1, 9):
                    # import pdb
                    # pdb.set_trace()
                    # if k != 3 and k !=5:
                    pr_model = ConditionalGaussianMixtureEVMW2WU(
                        h5_addr=model_path + f"model_{k}.h5",
                        datatype=exp_args["datatype"],
                        u1=exp_args["u1"],
                        u2=exp_args["u2"],
                        normalize=exp_args["normalize"],
                    )
                    prob_, logprob_, pred_cdf_ = pr_model.prob_batch(x, y)

                    pred_cdf_tail_ = np.float64(1.00) - np.array(pred_cdf_, dtype=np.float64)
                    max_cdf = get_max_pair(max_cdf, pred_cdf_tail_)
                    min_cdf = get_min_pair(min_cdf, pred_cdf_tail_)

                    prob = prob + prob_
                    logprob = logprob + logprob_
                    pred_cdf = pred_cdf + pred_cdf_

                    del pr_model
                    del prob_, logprob_, pred_cdf_, pred_cdf_tail_
                    gc.collect()
                    # else:
                    #     continue

                prob = prob / 9
                logprob = logprob / 9
                pred_cdf = pred_cdf / 9

            if exp_args["plotcdf"]:
                if not single_plot:
                    ax = cdf_axes[idx]
                else:
                    ax = cdf_ax
                ax.plot(
                    y_points,
                    pred_cdf,
                    marker="",
                    label="pred. " + model_project_name + "." + model_conf_key + "." + ensemble_num,
                )

                if exp_args["logplot"]:
                    ax.set_yscale('log')
                elif exp_args["loglogplot"]:
                    ax.set_yscale('log')
                    ax.set_xscale('log')

                if exp_args["prob_lims"]:
                    ax.set_ylim(exp_args["prob_lims"][0], exp_args["prob_lims"][1])
                if not single_plot:
                    ax.set_title(f"{cond_dict}")
                ax.set_xlabel(key_label)
                ax.set_ylabel("Success probability")
                ax.grid()
                ax.legend()

            if exp_args["plottail"]:
                if not single_plot:
                    ax = tail_axes[idx]
                else:
                    ax = tail_ax
                # import pdb
                # pdb.set_trace()
                lighter = 1
                if model_dict["type"] == "gmevm":
                    ax.plot(
                        y_points,
                        np.float64(1.00) - np.array(pred_cdf, dtype=np.float64),
                        marker="",
                        color=[0.9*lighter, 0.4*lighter, 0.0*lighter, 1],
                        linestyle='-',
                        label="GMEVM",
                        #linewidth=10
                    )
                    ax.fill_between(y_points, max_cdf, min_cdf, where=(max_cdf > min_cdf), color=[1.0*lighter, 0.3*lighter, 0.0*lighter, 1], linewidth=0.0,alpha=0.2)
                elif model_dict["type"] == "gmm":
                    ax.plot(
                        y_points,
                        np.float64(1.00) - np.array(pred_cdf, dtype=np.float64),
                        marker="",
                        color="green",
                        linestyle='-',
                        label="GMM",
                        #linewidth=10
                    )
                    ax.fill_between(y_points, max_cdf, min_cdf, where=(max_cdf > min_cdf), color="green", linewidth=0.0,alpha=0.2)
                elif model_dict["type"] == "gmevmw2wu":
                    ax.plot(
                        y_points,
                        np.float64(1.00) - np.array(pred_cdf, dtype=np.float64),
                        marker="",
                        color="blue",
                        linestyle='-',
                        label="TSTD",
                        #linewidth=10
                    )
                    ax.fill_between(y_points, max_cdf, min_cdf, where=(max_cdf > min_cdf), color="blue",linewidth=0.0, alpha=0.2)

                if exp_args["logplot"]:
                    ax.set_yscale('log')
                elif exp_args["loglogplot"]:
                    ax.set_yscale('log')
                    ax.set_xscale('log')
                if exp_args["prob_lims"]:
                    ax.set_ylim(exp_args["prob_lims"][0], exp_args["prob_lims"][1])

                # if not single_plot:   # 设置空格标题避免0看不见
                #     ax.set_title(f"{cond_dict}"+" bytes", fontsize=25)

                max_delay_cond = cond_df_pandas[key_label + '_scaled'].max()  # 当前条件的最大延时
                x_max = max_delay_cond + 10  # 每个条件的 X 轴上限
                # 设置 Tail 图 X 轴
                ax.set_xlim(0, x_max)
                ax.set_ylim(1e-6, 1)

                # ax.set_xlim(0, exp_args["y_points"][2])  # 设置X轴范围
                # if idx == 0:
                #     ax.set_ylim(1e-6, 1)
                # elif idx == 1:
                #     ax.set_ylim(1e-6, 1)
                # elif idx == 2:
                #     ax.set_ylim(1e-6, 1)
                # elif idx == 3:
                #     ax.set_ylim(1e-6, 1)
                ax.set_title(f"{cond_dict}" + " bytes", fontsize=25)
                ax.set_xlabel("End-to-End delay [ms]", fontsize=26)  # 字体大小
                ax.set_ylabel("Tail probability", fontsize=26)  # 字体大小
                ax.grid(linestyle='-', linewidth=1, alpha=1) # 网格

                # 放大图例字体
                legend = ax.legend(prop={'size': 22})
                #ax.legend(prop={'size': 21})
                for line in legend.get_lines():
                    line.set_linewidth(2)  # 图例（legend）中的标签线条的线宽
                # 设置XY轴的刻度字体大小和刻度线粗细
                ax.tick_params(axis='both', which='major', labelsize=22, width=1.2)  # 主刻度线
                ax.tick_params(axis='both', which='minor', labelsize=20, width=1.0)  # 次刻度线
                # 控制刻度线的长度
                ax.tick_params(axis='both', which='major', length=6)  # 这里控制刻度线长度
                ax.tick_params(axis='both', which='minor', length=4)

                #ax.legend()
                # 设置图形边框的粗细
                adjust_plot_borders(ax)

            if exp_args["plotpdf"]:
                if not single_plot:
                    ax = pdf_axes[idx]
                else:
                    ax = pdf_ax
                ax.plot(
                    y_points,
                    prob,
                    marker="",
                    label="pred. " + model_project_name + "." + model_conf_key + "." + ensemble_num,
                )
                if exp_args["logplot"]:
                    ax.set_yscale('log')
                elif exp_args["loglogplot"]:
                    ax.set_yscale('log')
                    ax.set_xscale('log')

                if exp_args["prob_lims"]:
                    ax.set_ylim(exp_args["prob_lims"][0], exp_args["prob_lims"][1])

                if not single_plot:
                    ax.set_title(f"{cond_dict}")
                ax.set_xlabel(key_label)
                ax.set_ylabel("probability")
                ax.grid()
                ax.legend()

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
            tail_fig.savefig(project_path + f"{key_label}_log_tail_average.png")
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
