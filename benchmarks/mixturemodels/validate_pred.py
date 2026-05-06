import ast
import getopt
import json
import multiprocessing as mp
import os
import pdb
import sys
import time
import warnings
from os.path import abspath, dirname
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger
from pr3d.de import (ConditionalGaussianMixtureEVM, ConditionalGaussianMM,ConditionalGaussianMixtureEVMW2WU,GMEVMU2_params_output)
from pyspark.sql import SparkSession
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from pr3d.common.evm import (
    gpd_prob,
    gpd_quantile,
    gpd_tail_prob,
    mixture_log_prob,
    mixture_prob,
    mixture_tail_prob,
    split_bulk_gpd,
)

warnings.filterwarnings("ignore")

def parse_validate_pred_args(argv: list[str]):

    # parse arguments to a dict
    args_dict = {}
    try:
        opts, args = getopt.getopt(
            argv,
            "hd:t:x:n:m:l:r:c:y:u:f:i:p:g:o:",
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
            ],
        )
    except getopt.GetoptError:
        print('Wrong args, type "python -m models_benchmark validate -h" for help')
        sys.exit(2)

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
        elif opt in ("-m", "--models"):
            args_dict["models"] = [s.strip().split(".") for s in arg.split(",")]
        elif opt in ("-l", "--label"):
            args_dict["label"] = arg
        elif opt in ("-r", "--rows"):
            args_dict["rows"] = int(arg)
        elif opt in ("-c", "--columns"):
            args_dict["columns"] = int(arg)
        elif opt in ("-y", "--y-points"):
            args_dict["y_points"] = [int(s.strip()) for s in arg.split(",")]
        elif opt in ("-u", "--log-lims"):
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


    if not args_dict["condition_markers"]:
        args_dict["condition_markers"] = ["." for cond in args_dict["condition_nums"]]

    return args_dict

def lookup_df(folder_path, cond_num, spark):
    # 循环返回num_conditions.json 文件内容 和 num_records.parquet数据集内容
    json_path = os.path.join(folder_path, f"{cond_num}_conditions.json")    # os.path.join拼接路径 cond_num = 0,1,2,3
    with open(json_path, "r") as file:                     # folder_path == .../mixturemodels/wifi/prepped_length_results/
        info_json = json.load(file)
    parquet_path = os.path.join(folder_path, f"{cond_num}_records.parquet")
    cond_df = spark.read.parquet(parquet_path)
    total_count = cond_df.count()
    logger.info(f"Parquet file {parquet_path} is loaded.")
    logger.info(f"Total number of samples in this empirical dataset: {total_count}")

    return info_json, cond_df

def prob_distribution_plot_view(y_input, prediction_res_1, prediction_res_2, tag):

    weights = prediction_res_1[0]
    locs = prediction_res_1[1]
    scales = prediction_res_1[2]
    tail_param = prediction_res_1[3]
    tail_threshold = prediction_res_2[0]
    tail_scale = prediction_res_1[5]

    strdtype = "float64"
    # import pdb
    # pdb.set_trace()
    cat = tfd.Categorical(probs=weights, dtype=strdtype)
    components = [
        tfd.Normal(loc=loc, scale=scale)
        for loc, scale in zip(
            tf.unstack(locs, axis=1), tf.unstack(scales, axis=1)
        )
    ]
    mixture = tfd.Mixture(cat=cat, components=components)

    gmm_pdf = tf.transpose(mixture.prob(tf.transpose(y_input)))
    gmm_log_pdf = tf.transpose(mixture.log_prob(tf.transpose(y_input)))
    gmm_ecdf = tf.transpose(mixture.cdf(tf.transpose(y_input)))
    # gmm_bulk_mean = mixture.mean()

    norm_factor = tf.constant(1.00, dtype=strdtype) - mixture.cdf(
        tf.squeeze(tail_threshold)
    )

    y_batchsize = tf.cast(tf.size(y_input), dtype=strdtype)

    bool_split_tensor, tail_samples_count, bulk_samples_count = split_bulk_gpd(
        tail_threshold=tail_threshold,
        y_input=y_input,  # self.y_input 应该是经过处理的，这样直接和tail_threshold对比，是否合理？
        y_batch_size=y_batchsize,
        dtype=strdtype,
    )

    bulk_prob_t = mixture.prob(tf.squeeze(y_input))
    bulk_cdf_t = mixture.cdf(tf.squeeze(y_input))
    bulk_tail_prob_t = tf.constant(1.00, dtype=strdtype) - bulk_cdf_t
    # import pdb
    # pdb.set_trace()
    gpd_prob_t = gpd_prob(  # 广义帕莱托分布
        tail_threshold=tf.squeeze(tail_threshold),
        tail_param=tf.squeeze(tail_param),
        tail_scale=tf.squeeze(tail_scale),
        norm_factor=norm_factor,
        y_input=tf.squeeze(y_input),
        dtype=strdtype,
    )

    gpd_tail_prob_t = gpd_tail_prob(
        tail_threshold=tail_threshold,
        tail_param=tail_param,
        tail_scale=tail_scale,
        norm_factor=norm_factor,
        y_input=tf.squeeze(y_input),
        dtype=strdtype,
    )

    # # define weighted mixture probabilities
    # weight_alpha = tf.squeeze(weight_alpha, axis=-1)
    # if len(weight_alpha.shape) == 1:
    #     weight_alpha = tf.squeeze(weight_alpha)
    # weighted_prob_t = (tf.constant(1.00, dtype=strdtype) - weight_alpha) * bulk_prob_t + weight_alpha * gpd_prob_t

    mevm_pdf = mixture_prob(
        bool_split_tensor=bool_split_tensor,
        gpd_prob_t=gpd_prob_t,
        bulk_prob_t=bulk_prob_t,
        dtype=strdtype,
    )

    mevm_log_pdf = mixture_log_prob(
        bool_split_tensor=bool_split_tensor,
        gpd_prob_t=gpd_prob_t,
        bulk_prob_t=bulk_prob_t,
        dtype=strdtype,
    )
    mevm_expanded_log_pdf = tf.expand_dims(mevm_log_pdf, axis=1)

    mevm_ecdf = tf.constant(1.00, dtype=strdtype) - mixture_tail_prob(
        bool_split_tensor=bool_split_tensor,
        gpd_tail_prob_t=gpd_tail_prob_t,
        bulk_tail_prob_t=bulk_tail_prob_t,
        dtype=strdtype,
    )

    return gmm_pdf, gmm_log_pdf, gmm_ecdf, mevm_pdf, mevm_log_pdf, mevm_ecdf

def run_validate_pred_processes(exp_args: list):
    logger.info(
        "Prepare models benchmark validate args "
        + f"with command line args: {exp_args}"
    )

    spark = (
        SparkSession.builder.master("local")
        .appName("LoadParquets")
        .config("spark.executor.memory", "6g")
        .config("spark.driver.memory", "70g")
        .config("spark.driver.maxResultSize", 0)
        .getOrCreate()
    )

    # this project folder setting
    p = Path(__file__).parents[0]
    main_path = str(p) + "/"     # .....benchmarks/mixturemodels/
    project_path = main_path + exp_args["label"] + "_results/" # /wifi/validate_length_l_results/
    os.makedirs(project_path, exist_ok=True)

    # dataset project folder setting
    dataset_project_path = main_path + exp_args["dataset"] + "_results/"  # /prepped_length_results/

    # find dataframe with the desired condition
    # inputs: exp_args["condition_nums"]
    conditions = []                # [{'length': 172}, {'length': 3440}, {'length': 6880}, {'length': 10320}]
    cond_dataframes = []           # 存储每个num_records.parquet数据集内容
    for cond_num in exp_args["condition_nums"]:
        cond_dict, cond_df = lookup_df(dataset_project_path,cond_num,spark)  # 赋值num_conditions.json 文件内容 和 num_records.parquet数据集内容
        cond_dataframes.append(cond_df)
        conditions.append(cond_dict)


    json_path = os.path.join(dataset_project_path, "info.json") # os.path.join()接受两个为参数，并将它们连接成一个有效的文件路径
    with open(json_path, "r") as file:
        info_json = json.load(file)

    key_label = exp_args["target"]               # t = receive
    key_mean = info_json[key_label]["mean"]      # "receive"的 mean  == 3410079.898217934
    key_scale = info_json[key_label]["scale"]    # scale: 1e-06


    # bulk plot axis   linspace函数生成 400 个从 0 到 200 的等间隔数值。(-y 0,400,200)
    y_points = np.linspace(
        start=exp_args["y_points"][0],
        stop=exp_args["y_points"][2],
        num=exp_args["y_points"][1],
    )
    y_points_standard = np.linspace(
        start=exp_args["y_points"][0]-(key_mean*key_scale), # 但每个点都减去 3.410079898217934 ？？
        stop=exp_args["y_points"][2]-(key_mean*key_scale),
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
            tail_fig, tail_axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 5 * nrows))  # 英寸单位
            # plt.subplots()画布中包含多少行列子图，这里两行两列，一共四个子图
            # fig 是一个 Figure 对象，代表整个图形。
            # axes 是一个包含所有子图的 Axes 对象的数组。如果 nrows 和 ncols 都为 1，那么 axes 就是一个单个的 Axes 对象
            # 否则，它是一个包含 nrows * ncols 个 Axes 对象的二维数组，可以通过索引来访问每个子图。
            tail_axes = tail_axes.flat   # 将二维数组展平成一维数组
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

    for idx, cond_dict in enumerate(conditions): # 列表[{'length': 172}, {'length': 3440}, {'length': 6880}, {'length': 10320}]
        logger.info(f"Plotting dataframe {idx} with conditions {cond_dict}")
        cond_df = cond_dataframes[idx]      # 存储每个num_records.parquet数据集内容(加噪正规化后四个文件之一)
        # Calculate emp CDF
        total_count = cond_df.count()
        emp_cdf = list()    # 存了400个概率值的列表
        # 判断400次： records.parquet 数据集中所有的receive_scaled  <= 每个y(400 个从 0 到 200 的等间隔数值)  的概率
        for y in y_points:    # y_points == 400 个从 0 到 200 的等间隔数值的数组
            delay_budget = y
            new_cond_df = cond_df.where(cond_df[key_label+'_scaled'] <= delay_budget)   # key_label = receive；where()在这里，它将应用条件到 DataFrame 的每一行，只保留符合条件的行，并将不符合条件的行填充为 NaN
            success_count = new_cond_df.count()     # 统计满足 receive_scaled <= y 有几行
            emp_success_prob = success_count / total_count
            emp_cdf.append(emp_success_prob)

        # Calculate emp PDF
        emp_pdf = np.diff(np.array(emp_cdf))
        # diff() 函数计算数组中相邻元素之间的差值， 返回差值数组， 这相当于对经验累积分布函数（CDF）进行微分？
        emp_pdf = np.append(emp_pdf,[0])*exp_args["y_points"][1]/(exp_args["y_points"][2]-exp_args["y_points"][0])
        # y 0,400,200                  每个差值         * 400 / (200 - 0)  --->    乘2？

        if exp_args["plotcdf"]:
            if not single_plot:
                ax = cdf_axes[idx]  # plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 5 * nrows))
            else:
                ax = cdf_ax
            ax.plot(
                y_points,
                emp_cdf,
                marker=exp_args["condition_markers"][idx], # 'condition_markers': ['.', '.', '.', '.'],
                label=f"meas. {cond_dict}",  # {'length': 172}
            )

        if exp_args["plottail"]:
            if not single_plot:
                ax = tail_axes[idx]
            else:
                ax = tail_ax
            ax.plot(
                y_points,       # X轴
                np.float64(1.00)-np.array(emp_cdf,dtype=np.float64),   # Y轴
                marker=exp_args["condition_markers"][idx],
                #marker="",
                label=f"meas. {cond_dict}",
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

        # plot predictions
        for model_list in exp_args["models"]:  # models: [['wifi/trained_length_dl_l', 'gmevm', '0'], ['wifi/trained_length_dl_l', 'gmm', '0']]
            model_project_name = model_list[0]
            model_conf_key = model_list[1]
            ensemble_num = model_list[2]
            model_path = (
                main_path + model_project_name + "_results/" + model_conf_key + "/"
            )

            with open(
                model_path + f"model_{ensemble_num}.json"
            ) as json_file:
                model_dict = json.load(json_file)

            # define x from the conditional dataset
            # select the columns
            cond_columns = []  # ==  ["length_normed"]
            for cond_label in model_dict["condition_labels"]:  #  "length_normed"
                cond_columns.append(cond_label)

            # select columns and sample the rows
            rows = cond_df.select(cond_columns).sample(False, (len(y_points_standard)*2)/cond_df.count(), seed=0).limit(len(y_points_standard))
            # cond_df: 存储每个num_records.parquet数据集内容(加噪正规化后四个文件之一),select(cond_columns)：从 cond_df 中选择指定的列 "length_normed"
            # False 表示不进行重复抽样  y_points_standard = 400 个从 0 到 200 的等间隔数值 都减去"receive"的 mean 3.410079898217934
            # num_records.parquet数据集内容中 以400 * 2 / 12万 的概率抽取 400个
            x_rows = rows.collect() # 将抽样结果转换为 Python 中的列表。

            # define x numpy list
            x_list = []  # == [[0.0], [0.0], [0.0], [0.0], [0.0]....]
            for row in x_rows:
                x_list.append(
                    [ row[colm] for colm in cond_columns ] # 采样结果之一[length_normed值]
                )

            x = np.array(x_list)

            # define y points and run the inference
            y = np.array(y_points_standard, dtype=np.float64) # 400 个从 0 到 200 的等间隔数值都减去"receive"的 mean 3.410079898217934
            #y = y.clip(min=0.00)

            if model_dict["type"] == "gmm":
                pr_model = ConditionalGaussianMM(
                    h5_addr=model_path + f"model_{ensemble_num}.h5",
                )
            elif model_dict["type"] == "gmevm":
                pr_model = ConditionalGaussianMixtureEVM(
                    h5_addr=model_path + f"model_{ensemble_num}.h5",
                )
            # dyj's change
            elif model_dict["type"] == "gmevmw":
                 pr_model = ConditionalGaussianMixtureEVMW(
                     h5_addr=model_path + f"model_{ensemble_num}.h5",
                 )
            elif model_dict["type"] == "gmevmu":
                 pr_model = ConditionalGaussianMixtureEVMU(
                     h5_addr=model_path + f"model_{ensemble_num}.h5",
                 )
            elif model_dict["type"] == "gmevmw2w":
                 pr_model = ConditionalGaussianMixtureEVMW2W(
                     h5_addr=model_path + f"model_{ensemble_num}.h5",
                 )
            elif model_dict["type"] == "gmevmw2wu":
                 pr_model = ConditionalGaussianMixtureEVMW2WU(
                     h5_addr=model_path + f"model_{ensemble_num}.h5",
                 )
            elif model_dict["type"] == "gmevmu2":
                # 抽取GMM模型的6个参数
                pr_model_1 = ConditionalGaussianMixtureEVM(
                    h5_addr=main_path + "wifi/trained_length_dl_l_results/gmevm/model_0.h5",
                )
                prediction_res_1 = pr_model_1._params_model.predict(x, )

                # 抽取GMEVMW2模型的 weight_alpha 参数
                pr_model_2 = GMEVMU2_params_output(
                    h5_addr=main_path + "wifi/trained_length_dl_l_results/gmevmu2/model_0.h5",
                )
                prediction_res_2 = pr_model_2._params_model.predict(x, )

                # import pdb
                # pdb.set_trace()
                gmm_prob, gmm_logprob, gmm_cdf, mevm_pdf, mevm_logpdf, mevm_cdf = prob_distribution_plot_view(
                    y_points_standard, prediction_res_1,prediction_res_2, model_dict["type"])
                #mevmw_prob, mevmw_logprob, mevmw_cdf = pr_model.prob_batch(x, y)
            if model_dict["type"] == "gmevmu2":
                prob, logprob, pred_cdf = mevm_pdf, mevm_logpdf, mevm_cdf
            else:
                prob, logprob, pred_cdf = pr_model.prob_batch(x, y) #  pr_model 是一个已经训练好的 TensorFlow 模型对象本291行代码  x相当于特征向量


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
                    ax.set_ylim(exp_args["prob_lims"][0],exp_args["prob_lims"][1])

                if not single_plot:
                    ax.set_title(f"{cond_dict}")
                ax.set_xlabel(key_label)
                ax.set_ylabel("Success probability")
                ax.grid()
                ax.legend()

            if exp_args["plottail"]:
                if not single_plot:
                    ax = tail_axes[idx] # 四个子图之一
                else:
                    ax = tail_ax
                if model_dict["type"] == "gmm":
                    ax.plot(        # 画图
                        y_points,
                        np.float64(1.00)-np.array(pred_cdf,dtype=np.float64),
                        marker="",
                        color="green",
                        linestyle=':',
                        label="pred. " + model_project_name + "." + model_conf_key + "." + ensemble_num,
                    )
                elif model_dict["type"] == "gmevm":
                    ax.plot(        # 画图
                        y_points,
                        np.float64(1.00)-np.array(pred_cdf,dtype=np.float64),
                        marker="",
                        linestyle='--',
                        color="red",
                        label="pred. " + model_project_name + "." + model_conf_key + "." + ensemble_num,
                    )
                elif model_dict["type"] == "gmevmw2w":
                    ax.plot(        # 画图
                        y_points,
                        np.float64(1.00)-np.array(pred_cdf,dtype=np.float64),
                        marker="",
                        linestyle='-',
                        color="blue",
                        label="pred. " + model_project_name + "." + model_conf_key + "." + ensemble_num,
                    )
                elif model_dict["type"] == "gmevmw2wu":
                    ax.plot(        # 画图
                        y_points,
                        np.float64(1.00)-np.array(pred_cdf,dtype=np.float64),
                        marker="",
                        linestyle='-',
                        color="blue",
                        label="pred. " + model_project_name + "." + model_conf_key + "." + ensemble_num,
                    )
                else:
                    ax.plot(        # 画图
                        y_points,
                        np.float64(1.00)-np.array(pred_cdf,dtype=np.float64),
                        marker="",
                        label="pred. " + model_project_name + "." + model_conf_key + "." + ensemble_num,
                    )
                if exp_args["logplot"]:    # true
                    ax.set_yscale('log')   # # 设置 y 轴为对数比例
                elif exp_args["loglogplot"]:
                    ax.set_yscale('log')
                    ax.set_xscale('log')

                if exp_args["prob_lims"]:
                    ax.set_ylim(exp_args["prob_lims"][0],exp_args["prob_lims"][1])
                    
                if not single_plot:
                    ax.set_title(f"{cond_dict}")  #  {'length': 172}
                ax.set_xlabel(key_label)  # key_label = receive
                ax.set_ylabel("Tail probability")
                ax.grid()  # 启用图表的网格线
                ax.legend() # 启用图例，以显示每条曲线的标签

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
                    ax.set_ylim(exp_args["prob_lims"][0],exp_args["prob_lims"][1])

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
        tail_fig.tight_layout()  # 调整子图之间及周围的间距，确保所有元素紧凑排列且不重叠。
        if exp_args["logplot"]:
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