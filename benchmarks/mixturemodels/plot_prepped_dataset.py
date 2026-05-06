import ast
import getopt
import json
import multiprocessing as mp
import os
import sys
import time
import itertools
import polars
import warnings
from os.path import abspath, dirname
from pathlib import Path
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger
from pyspark.sql import SparkSession

import scienceplots
plt.style.use(['science','ieee'])   # science样式绘制

warnings.filterwarnings("ignore")

condition_markers = [         # 设置了用于绘制数据点的标记样式，这里选择了点标记 '.'。
    '.'
]
markersize=6
darker = 0.9
MeasColorPalette = [
    [
        0.1*darker,  # 用蓝色色值绘制实际测量值的图例
        0.2*darker,
        0.8*darker,
        1
    ]
]
lighter = 1

xaxis_label = 'Link delay [ms]'
meas_legend_labels = [         # 设置了测量图例的标签为一个空字符串，可能表示无图例
    ''
]
markerevery=(0.04,0.04)       # 设置了标记间隔，以元组形式指定 x 轴和 y 轴方向上的间隔

def parse_plot_prepped_dataset_args(argv: list[str]):

    # parse arguments to a dict
    args_dict = {}
    try:
        opts, args = getopt.getopt(
            argv,
            "hd:t:x:m:r:c:y:l:f:i:p:g:o:w",
            ["dataset=", "target=", "condition-nums=", "condition-markers=", "rows=", "columns=", "y-points=", "prob-lims=", "plot-pdf", "plot-cdf", "plot-tail", "log", "loglog", "preview"],
        )
    except getopt.GetoptError:
        print('Wrong args, type "python -m models_benchmark validate -h" for help')
        sys.exit(2)

    # default values
    args_dict["condition_nums"] = None
    args_dict["prob_lims"] = None
    args_dict["condition_markers"] = None
    args_dict["y_points"] = [0, 100, 400]
    args_dict["plotcdf"] = False
    args_dict["plotpdf"] = False
    args_dict["plottail"] = False
    args_dict["logplot"] = False
    args_dict["loglogplot"] = False
    args_dict["preview"] = False

    # parse the args
    for opt, arg in opts:
        if opt == "-h":
            print(
                "python -m models_benchmark validate "
                + "-q <qlens> -d <dataset> -m <trained models> -l <label> -e <ensemble num>",
            )
            sys.exit()
        elif opt in ("-d", "--dataset"):
            args_dict["dataset"] = arg
        elif opt in ("-t", "--target"):
            args_dict["target"] = arg
        elif opt in ("-x", "--condition-nums"):
            args_dict["condition_nums"] = [int(s.strip()) for s in arg.split(",")]
        elif opt in ("-m", "--condition-markers"):
            args_dict["condition_markers"] = [s.strip() for s in arg.split(",")]
        elif opt in ("-r", "--rows"):
            args_dict["rows"] = int(arg)
        elif opt in ("-c", "--columns"):
            args_dict["columns"] = int(arg)
        elif opt in ("-y", "--y-points"):
            args_dict["y_points"] = [int(s.strip()) for s in arg.split(",")]
        elif opt in ("-l", "--log-lims"):
            args_dict["prob_lims"] = [np.float64(s.strip()) for s in arg.split(",")]
        elif opt in ("-f", "--plot-cdf"):
            args_dict["plotcdf"] = True
        elif opt in ("-i", "--plot-tail"):
            args_dict["plottail"] = True
        elif opt in ("-p", "--plot-pdf"):
            args_dict["plotpdf"] = True
        elif opt in ("-g", "--log"):
            args_dict["logplot"] = True
        elif opt in ("-o", "--loglog"):
            args_dict["loglogplot"] = True
        elif opt in ("-w", "--preview"):
            args_dict["preview"] = True

    if not args_dict["condition_markers"]:
        args_dict["condition_markers"] = ["." for cond in args_dict["condition_nums"]]
    # condition_markers': ['.', '.', '.', '.'] 因为命令行 -x 0,1,2,3 --->condition_num长度 = 4
    return args_dict

def lookup_df(folder_path, cond_num, spark):
    #返回num_conditions.json 文件内容 和 num_records.parquet数据集内容
    json_path = os.path.join(folder_path, f"{cond_num}_conditions.json")  # os.path.join拼接路径 cond_num = 0,1,2,3
    with open(json_path, "r") as file:
        info_json = json.load(file) # 加载 0_conditions.json 文件内容，将其存储在 info_json

    parquet_path = os.path.join(folder_path, f"{cond_num}_records.parquet")
    cond_df = spark.read.parquet(parquet_path)  # 使用 Spark 读取 Parquet 文件，并将其存储在 cond_df 变量中。
    total_count = cond_df.count()
    logger.info(f"Parquet file {parquet_path} is loaded.")
    logger.info(f"Total number of samples in this empirical dataset: {total_count}")
    return info_json, cond_df

def run_plot_prepped_dataset_processes(exp_args: list):
    logger.info(
        "Prepare models benchmark validate args "
        + f"with command line args: {exp_args}"
    )
    #创建了一个 SparkSession 对象，用于处理大规模数据的框架，指定配置参数
    spark = (
        SparkSession.builder.master("local")
        .appName("LoadParquets")
        .config("spark.executor.memory", "6g")
        .config("spark.driver.memory", "70g")
        .config("spark.driver.maxResultSize", 0)
        .getOrCreate()
    )

    # bulk plot axis  np.linspace函数生成 400 个从 0 到 100 的等间隔数值。
    y_points = np.linspace(
        start=exp_args["y_points"][0],
        stop=exp_args["y_points"][2],
        num=exp_args["y_points"][1],
    )

    # folder setting
    p = Path(__file__).parents[0]
    main_path = str(p) + "/"

    # dataset project folder setting
    dataset_project_path = main_path + exp_args["dataset"] + "_results/"   #prepped_length_results/

    # find dataframe with the desired condition
    # inputs: exp_args["condition_nums"]
    conditions = []                       # 存储每个num_conditions.json 字典文件内容
    cond_dataframes = []                  # 存储每个num_records.parquet数据集内容
    for cond_num in exp_args["condition_nums"]:
        cond_dict, cond_df = lookup_df(dataset_project_path,cond_num,spark)
        cond_dataframes.append(cond_df)
        conditions.append(cond_dict)

    key_label = exp_args["target"]   # t == receive_scaled

    single_plot = False
    if exp_args["rows"] == 0 and exp_args["columns"] == 0:
        single_plot = True  # 单一图表?

    # CDF figure
    if exp_args["plotcdf"]:
        if not single_plot:
            nrows = exp_args["rows"]
            ncols = exp_args["columns"]
            cdf_fig, cdf_axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 5 * nrows)) # 英寸单位
     # plt.subplots()创建一个包含多个子图的图表，返回一个包含整个图（Figure 对象）和所有子图（Axes 轴对象）的元组 (cdf_fig, cdf_axes)。
            cdf_axes = cdf_axes.flat     # 将二维数组展平成一维数组
        else:
            cdf_fig, cdf_ax = plt.subplots(nrows=1, ncols=1)

    # Tail figure
    if exp_args["plottail"] :
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


    if exp_args["plotpdf"] or exp_args["plotcdf"] or exp_args["plottail"]:
        key_label = exp_args["target"]
        for idx, cond_dict in enumerate(conditions):  # conditions存储每个num_conditions.json 字典文件内容
            logger.info(f"Plotting dataframe {idx} with conditions {cond_dict}")
            # Plotting dataframe 0 with conditions {'length': 172}
            cond_df = cond_dataframes[idx]    # cond_dataframes存储每个num_records.parquet数据集内容
            total_count = cond_df.count()

            emp_cdf = list()    #  emp_cdf == [0.0, 0.0, 0.5322921765986492, 0.8965911668835743.......]
            # 判断records.parquet数据集中receive_scaled满足每个 <= y(400 个从 0 到 100 的等间隔数值)成功样本概率
            for y in y_points:  # y_points 为400 个从 0 到 100 的等间隔数值
                delay_budget = y
                new_cond_df = cond_df.where(cond_df[key_label] <= delay_budget)
                success_count = new_cond_df.count()
                emp_success_prob = success_count / total_count
                emp_cdf.append(emp_success_prob)

            if exp_args["plotcdf"]:
                if not single_plot:
                    ax = cdf_axes[idx]          # 如果不是单个图，选择当前子图对象 cdf_axes[idx]，其中 idx 是当前条件的索引。
                else:
                    ax = cdf_ax
                ax.plot(
                    y_points,                      # 是x轴数据，表示延迟预算?
                    emp_cdf,                       # y轴数据，表示对应延迟预算下的成功概率
                    marker=condition_markers[idx], # 指定曲线的标记样式
                    markersize=markersize,         # 设置标记的大小。
                    markevery=markerevery,         # 控制标记密度
                    #marker=exp_args["condition_markers"][idx],
                    color=MeasColorPalette[idx],   # 曲线颜色
                    label=meas_legend_labels[idx]  # 曲线标签
                    #label=f"{cond_dict}",
                )
                if exp_args["logplot"]:       # 如果需要对纵轴进行对数变换，设置纵轴为对数刻度。
                    ax.set_yscale('log')
                elif exp_args["loglogplot"]:  # 如果需要同时对横轴和纵轴进行对数变换，设置横轴和纵轴为对数刻度
                    ax.set_yscale('log')
                    ax.set_xscale('log')

                if exp_args["prob_lims"]:     # 如果设置了纵轴的限制,根据参数 prob_lims 设置纵轴的范围
                    ax.set_ylim(exp_args["prob_lims"][0],exp_args["prob_lims"][1])

                if not single_plot:
                    ax.set_title(f"{cond_dict}")  # 如果不是单个图，设置当前子图的标题为当前条件的字符串表示

                ax.set_xlabel("Link delay [ms]")
                ax.set_ylabel("Success probability")
                ax.grid()                       # 添加网格线
                ax.legend()                     # 添加图例

            if exp_args["plottail"]:
                if not single_plot:
                    ax = tail_axes[idx]
                else:
                    ax = tail_ax
                ax.plot(
                    y_points,
                    np.float64(1.00)-np.array(emp_cdf,dtype=np.float64),
                    # 将 emp_cdf 列表转换为 NumPy 数组，并指定数据类型为 np.float64
                    # 计算了每个延迟点的尾部概率，即 1 减去对应的经验累积分布函数?
                    marker=condition_markers[idx],
                    markersize=markersize,
                    markevery=markerevery,
                    #marker=exp_args["condition_markers"][idx],
                    color=MeasColorPalette[idx],
                    label=meas_legend_labels[idx]
                    #label=f"{cond_dict}",
                )
                if exp_args["logplot"]:
                    ax.set_yscale('log')
                elif exp_args["loglogplot"]:
                    ax.set_yscale('log')
                    ax.set_xscale('log')

                if exp_args["prob_lims"]:
                    ax.set_ylim(exp_args["prob_lims"][0],exp_args["prob_lims"][1])
                    
                if not single_plot:
                    ax.set_title(f"{cond_dict}")
                
                ax.set_xlabel("Link delay [ms]")
                ax.set_ylabel("Tail probability")
                ax.grid()
                ax.legend()

            if exp_args["plotpdf"]:
                if not single_plot:
                    ax = pdf_axes[idx]
                else:
                    ax = pdf_ax
                emp_pdf = np.diff(np.array(emp_cdf))  # 将经验累积分布函数（CDF）emp_cdf转换为NumPy数组
                # np.diff()对NumPy数组中相邻元素进行差分运算，得到相邻元素之间的差值。得到一个数组，其元素表示概率密度函数在相应点的变化量
                emp_pdf = np.append(emp_pdf,[0])  # 将最后一个元素设置为0，以匹配 y_points 的长度
                ax.plot(
                    y_points,
                    emp_pdf,
                    marker=condition_markers[idx],  # ★这句代码有错， debug时print(condition_markers) == ['.']
                                                    # 而print(exp_args) 里 'condition_markers': ['.', '.', '.', '.']
                    markersize=markersize,
                    markevery=markerevery,
                    #marker=exp_args["condition_markers"][idx],
                    color=MeasColorPalette[idx],
                    label=meas_legend_labels[idx]
                    #label=f"{cond_dict}",
                )
                if exp_args["logplot"]:
                    ax.set_yscale('log')
                elif exp_args["loglogplot"]:
                    ax.set_yscale('log')
                    ax.set_xscale('log')
                
                if exp_args["prob_lims"]:
                    ax.set_ylim(exp_args["prob_lims"][0],exp_args["prob_lims"][1])

                if not single_plot:
                    ax.set_title(f"{cond_dict}")

                ax.set_xlabel("Link delay [ms]")
                ax.set_ylabel("Probability")
                ax.grid()
                ax.legend()

        if exp_args["plotcdf"]:
            # cdf figure
            cdf_fig.tight_layout()        # 调整图形布局以确保子图之间的间距合适
            size = cdf_fig.get_size_inches()   # 获取当前图形的尺寸
            size = [size[0],size[1]/2]         # 图形高度减半
            #cdf_fig.set_size_inches(size)
            if exp_args["logplot"]:
                cdf_fig.savefig(dataset_project_path + f"{key_label}_log_cdf.png")
                # cdf_fig.savefig()保存图形为PNG格式的图像文件，并使用相应的文件名后缀
                pickle.dump(cdf_fig,open(dataset_project_path + f"{key_label}_log_cdf.pickle",'wb'))
                # 使用Pickle模块将图形对象保存为.pickle文件，后续便于加载和处理？'w'表示文本模式写入，而'b'表示二进制模式写入
            elif exp_args["loglogplot"]:
                cdf_fig.savefig(dataset_project_path + f"{key_label}_loglog_cdf.png")
                pickle.dump(cdf_fig,open(dataset_project_path + f"{key_label}_loglog_cdf.pickle",'wb'))
            else:
                cdf_fig.savefig(dataset_project_path + f"{key_label}_cdf.png")
                pickle.dump(cdf_fig,open(dataset_project_path + f"{key_label}_cdf.pickle",'wb'))

        if exp_args["plottail"]:
            # tail figure
            tail_fig.tight_layout()
            size = tail_fig.get_size_inches()
            size = [size[0],size[1]/2]
            #tail_fig.set_size_inches(size)
            if exp_args["logplot"]:
                tail_fig.savefig(dataset_project_path + f"{key_label}_log_tail.png")
                pickle.dump(tail_fig,open(dataset_project_path + f"{key_label}_log_tail.pickle",'wb'))
            elif exp_args["loglogplot"]:
                tail_fig.savefig(dataset_project_path + f"{key_label}_loglog_tail.png")
                pickle.dump(tail_fig,open(dataset_project_path + f"{key_label}_loglog_tail.pickle",'wb'))
            else:
                tail_fig.savefig(dataset_project_path + f"{key_label}_tail.png")
                pickle.dump(tail_fig,open(dataset_project_path + f"{key_label}_tail.pickle",'wb'))

        if exp_args["plotpdf"]:
            # pdf figure
            pdf_fig.tight_layout()
            size = pdf_fig.get_size_inches()
            size = [size[0],size[1]/2]
            #pdf_fig.set_size_inches(size)
            if exp_args["logplot"]:
                pdf_fig.savefig(dataset_project_path + f"{key_label}_log_pdf.png")
                pickle.dump(pdf_fig,open(dataset_project_path + f"{key_label}_log_pdf.pickle",'wb'))
            elif exp_args["loglogplot"]:
                pdf_fig.savefig(dataset_project_path + f"{key_label}_loglog_pdf.png")
                pickle.dump(pdf_fig,open(dataset_project_path + f"{key_label}_loglog_pdf.pickle",'wb'))
            else:
                pdf_fig.savefig(dataset_project_path + f"{key_label}_pdf.png")
                pickle.dump(pdf_fig,open(dataset_project_path + f"{key_label}_pdf.pickle",'wb'))
