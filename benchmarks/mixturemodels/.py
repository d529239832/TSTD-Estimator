import os
import getopt
import json
import sys
import warnings
from pathlib import Path
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from .validate_pred import lookup_df
from pyspark.sql import SparkSession
from loguru import logger
from pr3d.common.evm import split_bulk_gpd, gpd_prob, gpd_tail_prob, mixture_prob, mixture_log_prob, mixture_tail_prob
from pr3d.de import ConditionalGaussianMixtureEVMW2W, ConditionalGaussianMixtureEVMWW, ConditionalGaussianMixtureEVMW, ConditionalGaussianMixtureEVM,ConditionalGaussianMM

warnings.filterwarnings("ignore")

tfd = tfp.distributions


def model_params_view_parse(argv: list[str]):
    # parse arguments to a dict
    args_dict = {}
    try:
        opts, args = getopt.getopt(
            argv,
            "hm:l:",
            [
                "models=",
                "label=",
            ],
        )
    except getopt.GetoptError:
        print('Wrong args')
        sys.exit(2)
    # parse the args
    for opt, arg in opts:
        if opt in ("-m", "--models"):  # 使用"."分割
            args_dict["models"] = [s.strip().split(".") for s in arg.split(",")]
        elif opt in ("-l", "--label"):
            args_dict["label"] = arg

    return args_dict


def model_params_view_function(exp_args: list):
    # this project folder setting
    p = Path(__file__).parents[0]  #p: /project/wireless-pr3d/benchmarks/mixturemodels
    main_path = str(p) + "/"

    model_project_name = exp_args["models"][0][0]
    model_conf_key = exp_args["models"][0][1]
    ensemble_num = exp_args["models"][0][2]

    # x = exp_args["label"]   # 0, 0.322, 0.661, 1
    x: npt.NDArray[np.float64] = np.array([exp_args["label"]], dtype=np.float64)
    model_path = (main_path + model_project_name + "_results/" + model_conf_key + "/")
    # model_path: /project/wireless-pr3d/benchmarks/mixturemodels/wifi/trained_length_dl_l_results/gmevm/
    with open(
            model_path + f"model_{ensemble_num}.json"
    ) as json_file:
        model_dict = json.load(json_file)

    if model_dict["type"] == "gmevmw2w":
        pr_model = ConditionalGaussianMixtureEVMW2W(
            h5_addr=model_path + f"model_{ensemble_num}.h5",
        )
        prediction_res = pr_model._params_model.predict(x, )

        # print("GMM分量的权重:", prediction_res[0][0])
        # print("GMM分量的均值:", prediction_res[1][0])
        # print("GMM分量的方差:", prediction_res[2][0])
        # print("GPD参数xi:", prediction_res[3][0])
        # print("GPD参数u:", prediction_res[4][0])
        # print("GPD参数beta:", prediction_res[5][0])
        # print("混合参数alpher:", prediction_res[6][0])
        # print(prediction_res)

        return prediction_res

    elif model_dict["type"] == "gmevmww":
        pr_model = ConditionalGaussianMixtureEVMWW(
            h5_addr=model_path + f"model_{ensemble_num}.h5",
        )
        prediction_res = pr_model._params_model.predict(x, )

        # print("GMM分量的权重:", prediction_res[0][0])
        # print("GMM分量的均值:", prediction_res[1][0])
        # print("GMM分量的方差:", prediction_res[2][0])
        # print("GPD参数xi:", prediction_res[3][0])
        # print("GPD参数u:", prediction_res[4][0])
        # print("GPD参数beta:", prediction_res[5][0])
        # print("混合参数alpher:", prediction_res[6][0])

        return prediction_res

    elif model_dict["type"] == "gmevmw":
        pr_model = ConditionalGaussianMixtureEVMW(
            h5_addr=model_path + f"model_{ensemble_num}.h5",
        )
        prediction_res = pr_model._params_model.predict(x, )

        # print("GMM分量的权重:", prediction_res[0][0])
        # print("GMM分量的均值:", prediction_res[1][0])
        # print("GMM分量的方差:", prediction_res[2][0])
        # print("GPD参数xi:", prediction_res[3][0])
        # print("GPD参数u:", prediction_res[4][0])
        # print("GPD参数beta:", prediction_res[5][0])
        # print("混合参数alpher:", prediction_res[6][0])

        return prediction_res

    elif model_dict["type"] == "gmevm":
        pr_model = ConditionalGaussianMixtureEVM(
            h5_addr=model_path + f"model_{ensemble_num}.h5",
        )
        prediction_res = pr_model._params_model.predict(x, )

        # print("GMM分量的权重:", prediction_res[0][0])
        # print("GMM分量的均值:", prediction_res[1][0])
        # print("GMM分量的方差:", prediction_res[2][0])
        # print("GPD参数xi:", prediction_res[3][0])
        # print("GPD参数u:", prediction_res[4][0])
        # print("GPD参数beta:", prediction_res[5][0])
        return prediction_res

    elif model_dict["type"] == "gmm":
        pr_model = ConditionalGaussianMM(
            h5_addr=model_path + f"model_{ensemble_num}.h5",
        )
        prediction_res = pr_model._params_model.predict(x, )

        # print("GMM分量的权重:", prediction_res[0][0])
        # print("GMM分量的均值:", prediction_res[1][0])
        # print("GMM分量的方差:", prediction_res[2][0])

        return prediction_res
    else:
        return


def model_params_plot_view(exp_args: list):

    # this project folder setting
    spark = (
        SparkSession.builder.master("local")
        .appName("LoadParquets")
        .config("spark.executor.memory", "6g")
        .config("spark.driver.memory", "70g")
        .config("spark.driver.maxResultSize", 0)
        .getOrCreate()
    )

    p = Path(__file__).parents[0]  #p: /project/wireless-pr3d/benchmarks/mixturemodels
    main_path = str(p) + "/"
    Graph_path = main_path + exp_args["label"] + "_plot_results/"
    os.makedirs(Graph_path, exist_ok=True)

    dataset_project_path = main_path + exp_args["dataset"] + "_results/"

    conditions = []
    cond_dataframes = []
    for cond_num in exp_args["condition_nums"]:
        cond_dict, cond_df = lookup_df(dataset_project_path, cond_num, spark)
        cond_dataframes.append(cond_df)
        conditions.append(cond_dict)

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

        max_point = max(np.array(cond_df_pandas['receive_scaled']))  # 设置间隔边界和间隔大小
        max_point = min(exp_args["y_points"][2], max_point)

        del cond_df_pandas
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
        emp_pdf = np.append(emp_pdf, [0]) * exp_args["y_points"][1] / (
                    exp_args["y_points"][2] - exp_args["y_points"][0])


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
            ax.plot(
                y_points,
                np.float64(1.00) - np.array(emp_cdf, dtype=np.float64),
                marker=exp_args["condition_markers"][idx],
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
        # for model_list in exp_args["models"]:

        model_project_name = exp_args["models"][0][0]
        model_conf_key = exp_args["models"][0][1]
        ensemble_num = exp_args["models"][0][2]

        # x = exp_args["label"]   # 0, 0.322, 0.661, 1
        # x: npt.NDArray[np.float64] = np.array([exp_args["label"]], dtype=np.float64)
        model_path = (main_path + model_project_name + "_results/" + model_conf_key + "/")
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
        y = np.array(y_points_standard, dtype=np.float64)

        if model_dict["type"] == "gmevmw2w":
            pr_model = ConditionalGaussianMixtureEVMW2W(
                h5_addr=model_path + f"model_{ensemble_num}.h5",
            )
            prediction_res = pr_model._params_model.predict(x, )

            # print("GMM分量的权重:", prediction_res[0][0])
            # print("GMM分量的均值:", prediction_res[1][0])
            # print("GMM分量的方差:", prediction_res[2][0])
            # print("GPD参数xi:", prediction_res[3][0])
            # print("GPD参数u:", prediction_res[4][0])
            # print("GPD参数beta:", prediction_res[5][0])
            # print("混合参数alpher:", prediction_res[6][0])

            # gmm_prob, gmm_logprob, gmm_cdf, mevm_pdf, mevm_logpdf, mevm_cdf = prob_distribution_plot_view(
            #     y_points_standard, prediction_res, model_dict["type"])
            mevmw2w_prob, mevmw2w_logprob, mevmw2w_cdf = pr_model.prob_batch(x, y)

            # return prediction_res

        elif model_dict["type"] == "gmevmww":
            pr_model = ConditionalGaussianMixtureEVMWW(
                h5_addr=model_path + f"model_{ensemble_num}.h5",
            )
            prediction_res = pr_model._params_model.predict(x, )

            # print("GMM分量的权重:", prediction_res[0][0])
            # print("GMM分量的均值:", prediction_res[1][0])
            # print("GMM分量的方差:", prediction_res[2][0])
            # print("GPD参数xi:", prediction_res[3][0])
            # print("GPD参数u:", prediction_res[4][0])
            # print("GPD参数beta:", prediction_res[5][0])
            # print("混合参数alpher:", prediction_res[6][0])

            # gmm_prob, gmm_logprob, gmm_cdf, mevm_pdf, mevm_logpdf, mevm_cdf = prob_distribution_plot_view(
            #     y_points_standard, prediction_res, model_dict["type"])
            mevmww_prob, mevmww_logprob, mevmww_cdf = pr_model.prob_batch(x, y)

            # return prediction_res
        elif model_dict["type"] == "gmevmw":
            pr_model = ConditionalGaussianMixtureEVMW(
                h5_addr=model_path + f"model_{ensemble_num}.h5",
            )
            prediction_res = pr_model._params_model.predict(x, )

            # print("GMM分量的权重:", prediction_res[0][0])
            # print("GMM分量的均值:", prediction_res[1][0])
            # print("GMM分量的方差:", prediction_res[2][0])
            # print("GPD参数xi:", prediction_res[3][0])
            # print("GPD参数u:", prediction_res[4][0])
            # print("GPD参数beta:", prediction_res[5][0])
            # print("混合参数alpher:", prediction_res[6][0])

            gmm_prob, gmm_logprob, gmm_cdf, mevm_pdf, mevm_logpdf, mevm_cdf = prob_distribution_plot_view(
                y_points_standard, prediction_res, model_dict["type"])
            mevmw_prob, mevmw_logprob, mevmw_cdf = pr_model.prob_batch(x, y)

            # return prediction_res

        elif model_dict["type"] == "gmevm":
            pr_model = ConditionalGaussianMixtureEVM(
                h5_addr=model_path + f"model_{ensemble_num}.h5",
            )
            prediction_res = pr_model._params_model.predict(x, )

            # print("GMM分量的权重:", prediction_res[0][0])
            # print("GMM分量的均值:", prediction_res[1][0])
            # print("GMM分量的方差:", prediction_res[2][0])
            # print("GPD参数xi:", prediction_res[3][0])
            # print("GPD参数u:", prediction_res[4][0])
            # print("GPD参数beta:", prediction_res[5][0])
            gmm_prob, gmm_logprob, gmm_cdf = prob_distribution_plot_view(y_points_standard, prediction_res, model_dict["type"])
            mevm_pdf, mevm_logpdf, mevm_cdf = pr_model.prob_batch(x, y)

            # return prediction_res

        elif model_dict["type"] == "gmm":
            pr_model = ConditionalGaussianMM(
                h5_addr=model_path + f"model_{ensemble_num}.h5",
            )
            prediction_res = pr_model._params_model.predict(x, )

            # print("GMM分量的权重:", prediction_res[0][0])
            # print("GMM分量的均值:", prediction_res[1][0])
            # print("GMM分量的方差:", prediction_res[2][0])

            return prediction_res
        else:
            return

        # if exp_args["plotcdf"]:
        #     if not single_plot:
        #         ax = cdf_axes[idx]
        #     else:
        #         ax = cdf_ax
        #     ax.plot(
        #         y_points,
        #         gmm_cdf,
        #         marker="",
        #         label="pred. " + model_project_name + "." + model_conf_key + "." + ensemble_num,
        #     )
        #     ax.plot(
        #         y_points,
        #         mevm_cdf,
        #         marker="",
        #         label="pred. " + model_project_name + "." + model_conf_key + "." + ensemble_num,
        #     )
        #     ax.plot(
        #         y_points,
        #         mevmw_cdf,
        #         marker="",
        #         label="pred. " + model_project_name + "." + model_conf_key + "." + ensemble_num,
        #     )
        #     if exp_args["logplot"]:
        #         ax.set_yscale('log')
        #     elif exp_args["loglogplot"]:
        #         ax.set_yscale('log')
        #         ax.set_xscale('log')
        #
        #     if exp_args["prob_lims"]:
        #         ax.set_ylim(exp_args["prob_lims"][0], exp_args["prob_lims"][1])
        #
        #     if not single_plot:
        #         ax.set_title(f"{cond_dict}")
        #     ax.set_xlabel(key_label)
        #     ax.set_ylabel("Success probability")
        #     ax.grid()
        #     ax.legend()

        if exp_args["plottail"]:
            if not single_plot:
                ax = tail_axes[idx]
            else:
                ax = tail_ax
            # ax.plot(
            #     y_points,
            #     np.float64(1.00) - np.array(gmm_cdf, dtype=np.float64),
            #     marker="",
            #     label="pred. " + model_project_name + "." + model_conf_key + "." + ensemble_num,
            # )
            # ax.plot(
            #     y_points,
            #     np.float64(1.00) - np.array(mevm_cdf, dtype=np.float64),
            #     marker="",
            #     label="pred. " + model_project_name + "." + model_conf_key + "." + ensemble_num,
            # )
            # ax.plot(
            #     y_points,
            #     np.float64(1.00) - np.array(mevmw_cdf, dtype=np.float64),
            #     marker="",
            #     label="pred. " + model_project_name + "." + model_conf_key + "." + ensemble_num,
            # )
            ax.plot(
                y_points,
                np.float64(1.00) - np.array(mevmww_cdf, dtype=np.float64),
                marker="",
                label="pred. " + model_project_name + "." + model_conf_key + "." + ensemble_num,
            )
            # ax.plot(
            #     y_points,
            #     np.float64(1.00) - np.array(mevmw2w_cdf, dtype=np.float64),
            #     marker="",
            #     label="pred. " + model_project_name + "." + model_conf_key + "." + ensemble_num,
            # )
            ux = prediction_res[4][0][0] + (key_mean * key_scale)
            ax.plot([ux, ux], [0, 1], color='black', linestyle='-')
            # print(prediction_res[4][0][0])
            # print(prediction_res[8][0][0])
            uxx = ux + prediction_res[8][0][0]
            ax.plot([uxx, uxx], [0, 1], color='red')

            if exp_args["logplot"]:
                ax.set_yscale('log')
            elif exp_args["loglogplot"]:
                ax.set_yscale('log')
                ax.set_xscale('log')

            ax.set_xlim(0, max_point)

            if exp_args["prob_lims"]:
                ax.set_ylim(exp_args["prob_lims"][0], exp_args["prob_lims"][1])

            if not single_plot:
                ax.set_title(f"{cond_dict}")
            ax.set_xlabel(key_label)
            ax.set_ylabel("Tail probability")
            ax.grid()
            ax.legend()

        # if exp_args["plotpdf"]:
        #     if not single_plot:
        #         ax = pdf_axes[idx]
        #     else:
        #         ax = pdf_ax
        #     ax.plot(
        #         y_points,
        #         prob,
        #         marker="",
        #         label="pred. " + model_project_name + "." + model_conf_key + "." + ensemble_num,
        #     )
        #     if exp_args["logplot"]:
        #         ax.set_yscale('log')
        #     elif exp_args["loglogplot"]:
        #         ax.set_yscale('log')
        #         ax.set_xscale('log')
        #
        #     if exp_args["prob_lims"]:
        #         ax.set_ylim(exp_args["prob_lims"][0], exp_args["prob_lims"][1])
        #
        #     if not single_plot:
        #         ax.set_title(f"{cond_dict}")
        #     ax.set_xlabel(key_label)
        #     ax.set_ylabel("probability")
        #     ax.grid()
        #     ax.legend()

    if exp_args["plotcdf"]:
        # cdf figure
        cdf_fig.tight_layout()
        if exp_args["logplot"]:
            cdf_fig.savefig(Graph_path + f"{key_label}_log_cdf.png")
        elif exp_args["loglogplot"]:
            cdf_fig.savefig(Graph_path + f"{key_label}_loglog_cdf.png")
        else:
            cdf_fig.savefig(Graph_path + f"{key_label}_cdf.png")

    if exp_args["plottail"]:
        # cdf figure
        tail_fig.tight_layout()
        if exp_args["logplot"]:
            tail_fig.savefig(Graph_path + f"{key_label}_log_tail_" + f"{ensemble_num}.png")
        elif exp_args["loglogplot"]:
            tail_fig.savefig(Graph_path + f"{key_label}_loglog_tail.png")
        else:
            tail_fig.savefig(Graph_path + f"{key_label}_tail.png")

    if exp_args["plotpdf"]:
        # pdf figure
        pdf_fig.tight_layout()
        if exp_args["logplot"]:
            pdf_fig.savefig(Graph_path + f"{key_label}_log_pdf.png")
        elif exp_args["loglogplot"]:
            pdf_fig.savefig(Graph_path + f"{key_label}_loglog_pdf.png")
        else:
            pdf_fig.savefig(Graph_path + f"{key_label}_pdf.png")


def model_params_average_plot_view(exp_args: list):

    # this project folder setting
    spark = (
        SparkSession.builder.master("local")
        .appName("LoadParquets")
        .config("spark.executor.memory", "6g")
        .config("spark.driver.memory", "70g")
        .config("spark.driver.maxResultSize", 0)
        .getOrCreate()
    )

    p = Path(__file__).parents[0]  #p: /project/wireless-pr3d/benchmarks/mixturemodels
    main_path = str(p) + "/"
    Graph_path = main_path + exp_args["label"] + "_plot_results/"
    os.makedirs(Graph_path, exist_ok=True)

    dataset_project_path = main_path + exp_args["dataset"] + "_results/"

    conditions = []
    cond_dataframes = []
    for cond_num in exp_args["condition_nums"]:
        cond_dict, cond_df = lookup_df(dataset_project_path, cond_num, spark)
        cond_dataframes.append(cond_df)
        conditions.append(cond_dict)

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

        max_point = max(np.array(cond_df_pandas['receive_scaled']))  # 设置间隔边界和间隔大小
        max_point = min(exp_args["y_points"][2], max_point)

        del cond_df_pandas
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
        emp_pdf = np.append(emp_pdf, [0]) * exp_args["y_points"][1] / (
                    exp_args["y_points"][2] - exp_args["y_points"][0])


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
            ax.plot(
                y_points,
                np.float64(1.00) - np.array(emp_cdf, dtype=np.float64),
                marker=exp_args["condition_markers"][idx],
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
        # for model_list in exp_args["models"]:

        model_project_name = exp_args["models"][0][0]
        model_conf_key = exp_args["models"][0][1]
        ensemble_num = exp_args["models"][0][2]

        # x = exp_args["label"]   # 0, 0.322, 0.661, 1
        # x: npt.NDArray[np.float64] = np.array([exp_args["label"]], dtype=np.float64)
        model_path = (main_path + model_project_name + "_results/" + model_conf_key + "/")
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
        y = np.array(y_points_standard, dtype=np.float64)

        if model_dict["type"] == "gmevmw2w":

            ensemble_num = 0
            pr_model = ConditionalGaussianMixtureEVMW2W(
                h5_addr=model_path + f"model_{ensemble_num}.h5",
            )

            mevmw2w_prob, mevmw2w_logprob, mevmw2w_cdf = pr_model.prob_batch(x, y)

            for ensemble_num in range(1, 9):
                pr_model = ConditionalGaussianMixtureEVMW2W(
                    h5_addr=model_path + f"model_{ensemble_num}.h5",
                )

                mevmw2w_prob_, mevmw2w_logprob_, mevmw2w_cdf_ = pr_model.prob_batch(x, y)

                mevmw2w_prob = mevmw2w_prob + mevmw2w_prob_
                mevmw2w_logprob = mevmw2w_logprob + mevmw2w_logprob_
                mevmw2w_cdf = mevmw2w_cdf + mevmw2w_cdf_

            mevmw2w_cdf = mevmw2w_cdf/9

        elif model_dict["type"] == "gmevmww":
            pr_model = ConditionalGaussianMixtureEVMWW(
                h5_addr=model_path + f"model_{ensemble_num}.h5",
            )
            prediction_res = pr_model._params_model.predict(x, )

            # print("GMM分量的权重:", prediction_res[0][0])
            # print("GMM分量的均值:", prediction_res[1][0])
            # print("GMM分量的方差:", prediction_res[2][0])
            # print("GPD参数xi:", prediction_res[3][0])
            # print("GPD参数u:", prediction_res[4][0])
            # print("GPD参数beta:", prediction_res[5][0])
            # print("混合参数alpher:", prediction_res[6][0])

            # gmm_prob, gmm_logprob, gmm_cdf, mevm_pdf, mevm_logpdf, mevm_cdf = prob_distribution_plot_view(
            #     y_points_standard, prediction_res, model_dict["type"])
            # mevmww_prob, mevmww_logprob, mevmww_cdf = pr_model.prob_batch(x, y)

            # return prediction_res
        elif model_dict["type"] == "gmevmw":
            pr_model = ConditionalGaussianMixtureEVMW(
                h5_addr=model_path + f"model_{ensemble_num}.h5",
            )
            prediction_res = pr_model._params_model.predict(x, )

            # print("GMM分量的权重:", prediction_res[0][0])
            # print("GMM分量的均值:", prediction_res[1][0])
            # print("GMM分量的方差:", prediction_res[2][0])
            # print("GPD参数xi:", prediction_res[3][0])
            # print("GPD参数u:", prediction_res[4][0])
            # print("GPD参数beta:", prediction_res[5][0])
            # print("混合参数alpher:", prediction_res[6][0])

            gmm_prob, gmm_logprob, gmm_cdf, mevm_pdf, mevm_logpdf, mevm_cdf = prob_distribution_plot_view(
                y_points_standard, prediction_res, model_dict["type"])
            mevmw_prob, mevmw_logprob, mevmw_cdf = pr_model.prob_batch(x, y)

            # return prediction_res

        elif model_dict["type"] == "gmevm":

            ensemble_num = 0
            pr_model = ConditionalGaussianMixtureEVMW2W(
                h5_addr=model_path + f"model_{ensemble_num}.h5",
            )

            mevm_pdf, mevm_logpdf, mevm_cdf = pr_model.prob_batch(x, y)

            for ensemble_num in range(1, 9):

                pr_model = ConditionalGaussianMixtureEVM(
                    h5_addr=model_path + f"model_{ensemble_num}.h5",
                )

                mevm_pdf_, mevm_logpdf_, mevm_cdf_ = pr_model.prob_batch(x, y)

                mevm_pdf = mevm_pdf + mevm_pdf_
                mevm_logpdf = mevm_logpdf + mevm_logpdf_
                mevm_cdf = mevm_cdf + mevm_cdf_

            mevm_cdf = mevm_cdf/9

        elif model_dict["type"] == "gmm":
            pr_model = ConditionalGaussianMM(
                h5_addr=model_path + f"model_{ensemble_num}.h5",
            )
            prediction_res = pr_model._params_model.predict(x, )

            # print("GMM分量的权重:", prediction_res[0][0])
            # print("GMM分量的均值:", prediction_res[1][0])
            # print("GMM分量的方差:", prediction_res[2][0])

            # return prediction_res
        else:
            return

        # if exp_args["plotcdf"]:
        #     if not single_plot:
        #         ax = cdf_axes[idx]
        #     else:
        #         ax = cdf_ax
        #     ax.plot(
        #         y_points,
        #         gmm_cdf,
        #         marker="",
        #         label="pred. " + model_project_name + "." + model_conf_key + "." + ensemble_num,
        #     )
        #     ax.plot(
        #         y_points,
        #         mevm_cdf,
        #         marker="",
        #         label="pred. " + model_project_name + "." + model_conf_key + "." + ensemble_num,
        #     )
        #     ax.plot(
        #         y_points,
        #         mevmw_cdf,
        #         marker="",
        #         label="pred. " + model_project_name + "." + model_conf_key + "." + ensemble_num,
        #     )
        #     if exp_args["logplot"]:
        #         ax.set_yscale('log')
        #     elif exp_args["loglogplot"]:
        #         ax.set_yscale('log')
        #         ax.set_xscale('log')
        #
        #     if exp_args["prob_lims"]:
        #         ax.set_ylim(exp_args["prob_lims"][0], exp_args["prob_lims"][1])
        #
        #     if not single_plot:
        #         ax.set_title(f"{cond_dict}")
        #     ax.set_xlabel(key_label)
        #     ax.set_ylabel("Success probability")
        #     ax.grid()
        #     ax.legend()

        if exp_args["plottail"]:
            if not single_plot:
                ax = tail_axes[idx]
            else:
                ax = tail_ax
            # ax.plot(
            #     y_points,
            #     np.float64(1.00) - np.array(gmm_cdf, dtype=np.float64),
            #     marker="",
            #     label="pred. " + model_project_name + "." + model_conf_key + "." + ensemble_num,
            # )
            # ax.plot(
            #     y_points,
            #     np.float64(1.00) - np.array(mevm_cdf, dtype=np.float64),
            #     marker="",
            #     label="pred. " + model_project_name + "." + model_conf_key + "." + ensemble_num,
            # )
            # ax.plot(
            #     y_points,
            #     np.float64(1.00) - np.array(mevmw_cdf, dtype=np.float64),
            #     marker="",
            #     label="pred. " + model_project_name + "." + model_conf_key + "." + ensemble_num,
            # )
            # ax.plot(
            #     y_points,
            #     np.float64(1.00) - np.array(mevm_cdf, dtype=np.float64),
            #     marker="",
            #     label="pred_mevm",
            # )
            ax.plot(
                y_points,
                np.float64(1.00) - np.array(mevmw2w_cdf, dtype=np.float64),
                marker="",
                label="pred_mevmw2w",
            )
            # ux = prediction_res[4][0][0] + (key_mean * key_scale)
            # ax.plot([ux, ux], [0, 1], color='black', linestyle='-')
            # print(prediction_res[4][0][0])
            # print(prediction_res[8][0][0])
            # uxx = ux + prediction_res[8][0][0]
            # ax.plot([uxx, uxx], [0, 1], color='red')

            ax.plot([15, 15], [0, 1], color='red')

            if exp_args["logplot"]:
                ax.set_yscale('log')
            elif exp_args["loglogplot"]:
                ax.set_yscale('log')
                ax.set_xscale('log')

            ax.set_xlim(0, max_point)

            if exp_args["prob_lims"]:
                ax.set_ylim(exp_args["prob_lims"][0], exp_args["prob_lims"][1])

            if not single_plot:
                ax.set_title(f"{cond_dict}")
            ax.set_xlabel(key_label)
            ax.set_ylabel("Tail probability")
            ax.grid()
            ax.legend()

        # if exp_args["plotpdf"]:
        #     if not single_plot:
        #         ax = pdf_axes[idx]
        #     else:
        #         ax = pdf_ax
        #     ax.plot(
        #         y_points,
        #         prob,
        #         marker="",
        #         label="pred. " + model_project_name + "." + model_conf_key + "." + ensemble_num,
        #     )
        #     if exp_args["logplot"]:
        #         ax.set_yscale('log')
        #     elif exp_args["loglogplot"]:
        #         ax.set_yscale('log')
        #         ax.set_xscale('log')
        #
        #     if exp_args["prob_lims"]:
        #         ax.set_ylim(exp_args["prob_lims"][0], exp_args["prob_lims"][1])
        #
        #     if not single_plot:
        #         ax.set_title(f"{cond_dict}")
        #     ax.set_xlabel(key_label)
        #     ax.set_ylabel("probability")
        #     ax.grid()
        #     ax.legend()

    if exp_args["plotcdf"]:
        # cdf figure
        cdf_fig.tight_layout()
        if exp_args["logplot"]:
            cdf_fig.savefig(Graph_path + f"{key_label}_log_cdf.png")
        elif exp_args["loglogplot"]:
            cdf_fig.savefig(Graph_path + f"{key_label}_loglog_cdf.png")
        else:
            cdf_fig.savefig(Graph_path + f"{key_label}_cdf.png")

    if exp_args["plottail"]:
        # cdf figure
        tail_fig.tight_layout()
        if exp_args["logplot"]:
            tail_fig.savefig(Graph_path + f"{key_label}_log_tail_" + f"{ensemble_num}.png")
        elif exp_args["loglogplot"]:
            tail_fig.savefig(Graph_path + f"{key_label}_loglog_tail.png")
        else:
            tail_fig.savefig(Graph_path + f"{key_label}_tail.png")

    if exp_args["plotpdf"]:
        # pdf figure
        pdf_fig.tight_layout()
        if exp_args["logplot"]:
            pdf_fig.savefig(Graph_path + f"{key_label}_log_pdf.png")
        elif exp_args["loglogplot"]:
            pdf_fig.savefig(Graph_path + f"{key_label}_loglog_pdf.png")
        else:
            pdf_fig.savefig(Graph_path + f"{key_label}_pdf.png")

def prob_distribution_plot_view(y_input, prob_dis_param, tag):
    if tag == "gmevm":

        weights = prob_dis_param[0]
        locs = prob_dis_param[1]
        scales = prob_dis_param[2]
        # weights = prob_dis_param[0][0]
        # locs = prob_dis_param[1][0]
        # scales = prob_dis_param[2][0]
        # tail_param = prob_dis_param[3][0]
        # tail_threshold = prob_dis_param[4][0]
        # tail_scale = prob_dis_param[5][0]

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

        return gmm_pdf, gmm_log_pdf, gmm_ecdf

    elif tag == "gmevmw":

        weights = prob_dis_param[0]
        locs = prob_dis_param[1]
        scales = prob_dis_param[2]
        tail_param = prob_dis_param[3]
        tail_threshold = prob_dis_param[4]
        tail_scale = prob_dis_param[5]

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

        gpd_prob_t = gpd_prob(  # 广义帕莱托分布
            tail_threshold=tail_threshold,
            tail_param=tail_param,
            tail_scale=tail_scale,
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

    else:
        return 0



