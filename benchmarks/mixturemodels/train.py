import getopt
import json
import multiprocessing as mp
import multiprocessing.context as ctx
import os
import signal
import sys
import warnings
from os.path import abspath, dirname
from pathlib import Path
import time
import multiprocessing.context as ctx
import multiprocessing as mp
import numpy as np
import pandas as pd
from loguru import logger
from petastorm.spark import SparkDatasetConverter
from pyspark.sql import SparkSession
from pyspark.sql.functions import rand

warnings.filterwarnings("ignore")
# very important line to make tensorflow run in sub processes
ctx._force_start_method("spawn")
# disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def get_length_mapping(dataset_path, spark_memory="4g"):
    """
    独立于抽样逻辑，直接从原始 parquet 文件中提取 length -> length_normed 的映射
    """
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F

    # 临时启动一个小的 SparkSession 来读取元数据（如果主流程没启动的话）
    spark = SparkSession.builder.appName("MappingExtractor") \
        .config("spark.driver.memory", spark_memory) \
        .getOrCreate()

    # 获取所有 parquet 文件
    files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".parquet")]
    if not files:
        return {}

    # 只读取需要的两列并去重
    mapping_df = spark.read.parquet(*files).select("length", "length_normed").distinct()
    mapping_rows = mapping_df.collect()

    # 构造映射表：{归一化值: 原始长度}
    # 使用 round(x, 3) 匹配你命令行传入的精度
    norm_to_length = {round(float(row.length_normed), 3): int(row.length) for row in mapping_rows}

    # 也可以反过来存一份方便查找 {原始长度: 归一化值}
    length_to_norm = {int(row.length): round(float(row.length_normed), 3) for row in mapping_rows}

    return {"norm_to_length": norm_to_length, "length_to_norm": length_to_norm}

def parse_train_args(argv: list[str]):
    # parse arguments to a dict
    args_dict = {}
    try:
        opts, args = getopt.getopt(
            argv,
            "hd:l:c:e:t:u:v:k:",
            ["dataset=", "label=", "config=", "ensembles=", "u1=", "u2=", "nol="],
        )
    except getopt.GetoptError:
        print('Wrong args, type "python -m models_benchmark train -h" for help')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print(
                "python -m models_benchmark train "
                + "-d <dataset label> -l <label> -c <config json file>",
            )
            sys.exit()
        elif opt in ("-d", "--dataset"):
            args_dict["dataset"] = arg
        elif opt in ("-l", "--label"):
            args_dict["label"] = arg
        elif opt in ("-c", "--config-file"):
            with open(arg) as json_file:
                data = json.load(json_file)
            args_dict["train_config"] = data
        elif opt in ("-e", "--ensembles"):
            args_dict["ensembles"] = int(arg)
        elif opt in ("-t", "--dataset-type"):
            args_dict["datatype"] = arg
        elif opt in ("-u", "--u1"):
            args_dict["u1"] = [float(x) for x in arg.split(",")]
        elif opt in ("-v", "--u2"):
            args_dict["u2"] = [float(x) for x in arg.split(",")]
        elif opt in ("-k", "--nol"):
            args_dict["normalize"] = [float(x) for x in arg.split(",")]
    return args_dict


def run_train_processes(exp_args: list):
    # 创建 Manager 实例
    manager = mp.Manager()
    result_queue = manager.Queue()  # 创建共享队列

    logger.info(
        "Prepare models benchmark experiment args "
        + f"with command line args: {exp_args}"
    )

    # project folder setting

    p = Path(__file__).parents[0]
    main_path = str(p) + "/"  # /pr3d/wireless-pr3d-1.0.0-globecomm-paper-2023/benchmarks/mixturemodels/
    project_path = str(p) + "/" + exp_args["label"] + "_results/"  # ....../trained_length_dl_l_results
    os.makedirs(project_path, exist_ok=True)
    parquet_folder = main_path + "__trainparquets__"
    os.makedirs(parquet_folder, exist_ok=True)

    train_configs = exp_args["train_config"]
    # 取消并行注释下面这段
    n_workers = 18 # 如果要计算单个模型的时间，一定要设置为1，因为并行会消耗资源，运行时间 = 9个模型执行完时间 = 单个模型用时（因为并行9个一起运行）
    logger.info(f"Initializng {n_workers} workers")
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = mp.Pool(n_workers)  # 进程池可以用来并行执行任务
    signal.signal(signal.SIGINT, original_sigint_handler)  # 保存原始的中断信号处理程序，并将中断信号处理程序设置为忽略

    # create params list for each run
    n_runs = len(train_configs.keys()) * exp_args["ensembles"]  # train_configs = {'gmevm':xx, 'gmm':xx}

    dataset_path = main_path + exp_args["dataset"] + "_results/"  # /prepped_length_results/
    logger.info("Extracting length mapping from dataset...")
    full_mapping = get_length_mapping(dataset_path)
    params_list = []
    for ensemble_num in range(exp_args["ensembles"]):  # range(ensembles == 9) 这个循环9次
        for model_conf_key in train_configs.keys():  # 这里循环2次
            model_conf = train_configs[model_conf_key]  # 分别赋值配置文件中gmevm、gmm的值

            # create and prepare the results directory
            records_path = project_path + model_conf_key + "/"  # ...../trained_length_dl_l_results/gmevm/
            os.makedirs(records_path, exist_ok=True)

            # save the json info
            jsoninfo = json.dumps(model_conf)  # dumps 用于将 Python 对象转换为 JSON 格式的字符串
            with open(records_path + "info.json", "w") as f:  # 将gmm、gmevm字典内容写入info.json文件
                f.write(jsoninfo)



            if isinstance(exp_args["u1"], str):
                exp_args["u1"] = [float(x) for x in exp_args["u1"].split(",")]

            if isinstance(exp_args["u2"], str):
                exp_args["u2"] = [float(x) for x in exp_args["u2"].split(",")]
            params = {
                "ensembles": exp_args["ensembles"],
                "order_seed": 9988334,
                "ensemble_num": ensemble_num,
                "sample_seed": ensemble_num * 101012,
                "dataset_path": dataset_path,
                "records_path": records_path,
                "model_conf": model_conf,  # 配置文件中gmevm、gmm的值
                "model_conf_key": model_conf_key,
                "spark_total_memory": "70g",
                "spark_subprocess_memory": "3g",
                "parquet_folder": parquet_folder,
                "trained_model_path": project_path + "gmevm/" + f"model_0.h5",
                # "project_path": project_path + "/gmevm/" + f"model_{ensemble_num}.h5",
                "datatype": exp_args["datatype"],
                # 加入传入的4个 u1、u2 阈值列表
                "u1": exp_args["u1"],
                "u2": exp_args["u2"],
                "normalize": exp_args["normalize"],
                "length_mapping": full_mapping
            }
            params_list.append(params)

    # load dataset and sample 是否抽样
    load_dataset_and_sample(params)

    try:
        logger.info(f"Starting {n_runs} jobs")
        # res = pool.map_async(train_model, params_list)
        res = pool.starmap(train_model, [(params, result_queue) for params in params_list])
        logger.info("Waiting for results")
        # res.get(100000)  # Without the timeout this blocking call ignores all signals.
        print("res's length is:",len(res))

        # # 不需要调用 get()，直接遍历 res 列表
        # for result in res:
        #     result_queue.put(result)  # 将每个结果放入队列，以便后续处理
        # logger.info("All results processed successfully.")

        # 计算并输出所有进程的全局平均时长
        total_training_time_all_processes = 0
        total_epoch_duration_all_processes = 0
        total_step_duration_all_processes = 0
        total_processes = result_queue.qsize()
        print("total_processes is :", total_processes)
        while not result_queue.empty():
            result = result_queue.get()
            total_training_time_all_processes += result["average_training_time"]
            total_epoch_duration_all_processes += result["average_epoch_duration"]
            total_step_duration_all_processes += result["average_step_duration"]

        if total_processes > 0:
            avg_training_time = total_training_time_all_processes / total_processes / 60  # 转换为分钟
            avg_epoch_duration = total_epoch_duration_all_processes / total_processes
            avg_step_duration = total_step_duration_all_processes / total_processes * 1000

            logger.info(f"Global Average Training Time: {avg_training_time:.4f} minutes")
            logger.info(f"Global Average Epoch Duration: {avg_epoch_duration:.4f} seconds")
            logger.info(f"Global Average Step Duration: {avg_step_duration:.4f} milliseconds")

            # 构造保存路径
            json_filename = os.path.join(project_path, f"train_time.json")

            # 保存到 JSON 文件
            results_dict = {
                "global_avg_training_time (minutes)": avg_training_time,
                "global_avg_epoch_duration (seconds)": avg_epoch_duration,
                "global_avg_step_duration (milliseconds)": avg_step_duration
            }
            with open(json_filename, "w") as json_file:
                json.dump(results_dict, json_file, indent=4)
            logger.info("Saved training results to training_results.json")
        else:
            logger.warning("No results were collected from the workers.")

    except KeyboardInterrupt:
        logger.info("Caught KeyboardInterrupt, terminating workers")
        pool.terminate()
    else:
        logger.info("Normal termination")
        pool.close()

    # n_workers = len(params_list)
    # logger.info(f"Initializng {n_workers} workers")
    # try:
    #     logger.info(f"Starting {n_runs} jobs")
    #     # 取消并行执行，改为顺序执行
    #     for params in params_list:
    #         train_model(params)  # 直接在主进程中调用train_model函数
    #     logger.info("All jobs completed")
    # except KeyboardInterrupt:
    #     logger.info("Caught KeyboardInterrupt, terminating workers")
    #     # 无需终止进程池
    #     # pool.terminate()
    # else:
    #     logger.info("Normal termination")
    #     # 无需关闭进程池
    #     # pool.close()


def load_dataset_and_sample(params):
    # load the dataset, take 'dataset' 'ensembles' times
    # save them in parquet files, send the address to the
    # sub-process

    logger.info("Load dataset and sample")

    # init Spark
    spark = (
        SparkSession.builder.appName("Training")
        .config("spark.driver.memory", params["spark_total_memory"])
        .config("spark.driver.maxResultSize", 0)
        .getOrCreate()
    )
    # sc = spark.sparkContext

    # Set a cache directory on DBFS FUSE for intermediate data.
    file_path = dirname(abspath(__file__))  # dirname 函数用于获取路径的目录部分
    # path = /pr3d/wireless-pr3d-1.0.0-globecomm-paper-2023/benchmarks/mixturemodels

    spark_cash_addr = "file://" + file_path + "/__sparkcache__/__main__"  # 构建Spark 缓存目录的地址
    spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, spark_cash_addr)
    logger.info(
        f"load_dataset_and_sample: Spark cache folder is set up at: {spark_cash_addr}"
    )

    # read all the files from the project
    files = []
    logger.info(f"Opening the path {params['dataset_path']}")  # dataset_path == /prepped_length_results/
    all_files = os.listdir(params["dataset_path"])  # os.listdir() 函数用于返回指定目录中的文件和子目录的名称列表
    # all_files == ['2_records.parquet', '2_conditions.json', '1_records.parquet', 'info.json'.....]
    for f in all_files:
        if f.endswith(".parquet"):  # 结尾为par的
            files.append(params["dataset_path"] + "/" + f)  # files存储所有pre里的文件包括了目录的列表，除了'info.json'文件

    # read all files into one Spark df
    main_df = spark.read.parquet(*files)

    # Absolutely necessary for randomizing the rows (bug fix)
    # first shuffle, then sample!
    main_df = main_df.orderBy(rand(seed=params["order_seed"]))  # "order_seed": 9988334 为啥是这个数
    training_params = params["model_conf"]["training_params"]  # 字典params中键model_conf的值training_params的值（字典中存字典）
    # training_params == 配置文件中gmevm、gmm的键值为batchsize等

    for ensemble_num in range(params["ensembles"]):  # ensembles == 命令行传入的9

        # take the desired number of records for learning  获取所需数量的记录进行学习
        df_train = main_df.sample(
            withReplacement=False,
            fraction=training_params["dataset_size"] / main_df.count(),  # training_params["dataset_size"] == 840000
            seed=params["sample_seed"],  # sample_seed": ensemble_num * 101012
        )
        # # 不抽样，使用100%数据集
        # df_train = main_df
        logger.info(
            f"{ensemble_num}: sample {training_params['dataset_size']} rows, result {df_train.count()} samples"
        )

        ensemble_parquet_addr = params[
                                    "parquet_folder"] + f"/{ensemble_num}.parquet"  # parquet_folder = main_path + "__trainparquets__"
        logger.info(f"{ensemble_num}: writing sub-dataset into {ensemble_parquet_addr}")
        pandas_df = df_train.toPandas()
        pandas_df.to_parquet(ensemble_parquet_addr,
                             compression="snappy")  # 保存为 Parquet 格式文件，使用 Snappy 压缩，存储地址为 ensemble_parquet_addr
        del df_train
        del pandas_df


def train_model(params, result_queue):
    from pyarrow.fs import LocalFileSystem
    import tensorflow as tf
    import tensorflow_addons as tfa
    from petastorm import TransformSpec
    from petastorm.spark import SparkDatasetConverter, make_spark_converter
    from pr3d.de import (ConditionalGaussianMixtureEVM, ConditionalGaussianMM, GaussianMM, GaussianMixtureEVM, ConditionalGaussianMixtureEVMW2WU,)
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import rand

    logger.info(
        f"{params['model_conf_key']}.{params['ensemble_num']}: starts with params {params}"
    )
    # set params
    model_conf = params["model_conf"]  # 配置文件中gmevm、gmm的值
    module_label = params["model_conf_key"]  # gmm或 gmevm
    predictors_path = params["records_path"]  # ...../trained_length_dl_l_results/gmevm || gmm
    training_params = model_conf["training_params"]  # "dataset_size": 840000, "batch_size": 100

    # set data types
    # npdtype = np.float64
    # tfdtype = tf.float64
    strdtype = "float64"

    logger.info(f"Opening predictors directory '{predictors_path}'")
    os.makedirs(predictors_path, exist_ok=True)

    # read sub-dataset into Pandas df
    ensemble_parquet_addr = (
            params["parquet_folder"] + f"/{params['ensemble_num']}.parquet"  # parquet_folder = "__trainparquets__"
    )
    df_train = pd.read_parquet(ensemble_parquet_addr)  # 抽样后的数据集之一
    logger.info(
        f"{module_label}.{params['ensemble_num']}: dataset loaded, train sampels: {len(df_train)}"  # ensemble_num = 0-8
    )

    # get parameters
    y_label = model_conf["y_label"]  # "y_label" : "receive_standard"
    model_type = model_conf["type"]  # "type": "gmevm"
    training_rounds = training_params["rounds"]  # 学习率和轮次
    batch_size = training_params["batch_size"]

    if "condition_labels" not in model_conf:  # 配置文件中gmevm、gmm的值

        # dataset pre process
        df_train = df_train[[y_label]]
        df_train["y_input"] = df_train[y_label]
        df_train = df_train.drop(columns=[y_label])

        # initiate the non conditional predictor
        if model_type == "gmm":
            model = GaussianMM(
                centers=model_conf["centers"],
                dtype=strdtype,
                bayesian=model_conf["bayesian"]
            )
        elif model_type == "gmevm":
            model = GaussianMixtureEVM(
                centers=model_conf["centers"],
                dtype=strdtype,
                bayesian=model_conf["bayesian"]
            )

        X = None
        Y = df_train.y_input

    else:

        condition_labels = model_conf["condition_labels"]
        # dataset pre process
        df_train = df_train[[y_label, *condition_labels]]  # 只要 "receive_standard"，"length_normed" 这两列
        df_train["y_input"] = df_train[y_label]
        df_train = df_train.drop(columns=[y_label])  # 删除名为 y_label 的列，并将结果重新赋给 df_train。（换了个名字为y_input？）

        # import pdb
        # pdb.set_trace()
        # initiate the non conditional predictor
        if model_type == "gmm":
            model = ConditionalGaussianMM(
                x_dim=condition_labels,  # length_normed
                centers=model_conf["centers"],  # 10
                hidden_sizes=model_conf["hidden_sizes"],  # [10, 50, 50, 40]
                dtype=strdtype,
                bayesian=model_conf["bayesian"],
                # batch_size = 1024,
            )
        elif model_type == "gmevm":
            model = ConditionalGaussianMixtureEVM(
                x_dim=condition_labels,  # length_normed
                centers=model_conf["centers"],
                hidden_sizes=model_conf["hidden_sizes"],
                dtype=strdtype,
                bayesian=model_conf["bayesian"],
                # batch_size = 1024,
            )
        elif model_type == "gmevmw2wu":
            model = ConditionalGaussianMixtureEVMW2WU(
                x_dim=condition_labels,  # length_normed
                centers=model_conf["centers"],
                hidden_sizes=model_conf["hidden_sizes"],
                dtype=strdtype,
                bayesian=model_conf["bayesian"],
                datatype=params["datatype"],
                u1=params["u1"],
                u2=params["u2"],
                normalize=params["normalize"]
                # batch_size = 1024,
            )
        X = df_train[condition_labels]
        Y = df_train.y_input

    steps_per_epoch = len(df_train) // batch_size
    total_training_time_all_rounds = 0
    total_epoch_duration_all_rounds = 0
    total_step_duration_all_rounds = 0

    for idx, round_params in enumerate(training_rounds):  # {"learning_rate": 1e-2, "epochs":200},
        start_time = time.time()
        logger.info(
            f"{module_label}.{params['ensemble_num']}: training session "
            + f"{idx + 1}/{len(training_rounds)} with {round_params}, "
            + f"steps_per_epoch: {steps_per_epoch}, batch size: {batch_size}"
        )

        # 编译 TensorFlow 模型的训练过程 compile 方法用于配置模型的训练过程。
        model.training_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=round_params["learning_rate"],
            ),
            loss=model.loss,
        )

        if X is None:
            Xnp = np.zeros(len(Y))
            Ynp = np.array(Y)
            model.training_model.fit(
                x=[Xnp, Ynp],
                y=Ynp,
                steps_per_epoch=steps_per_epoch,
                epochs=round_params["epochs"],
                verbose=1,
            )

        else:
            Xnp = np.array(X)
            Ynp = np.array(Y)
            training_data = tuple([Xnp[:, i] for i in range(len(condition_labels))]) + (
                Ynp,
            )
            # print("training_data is :",training_data)
            # import pdb
            # pdb.set_trace()
            model.training_model.fit(
                x=training_data,
                y=Ynp,
                steps_per_epoch=steps_per_epoch,
                epochs=round_params["epochs"],
                verbose=1,
            )

        end_time = time.time()
        # 计算总训练时长
        total_training_time = end_time - start_time       # 1/4 个rounds的时间
        print(f"round {idx} training time(minutes):",total_training_time / 60)
        total_training_time_all_rounds += total_training_time  # 累加四个rounds的时间

        # 计算每个epoch的平均时长
        epoch_duration = total_training_time / round_params["epochs"]   # 单个rounds中每个epoch时间
        total_epoch_duration_all_rounds += epoch_duration               # 累加四个

        # 计算每个step的平均时长
        step_duration = epoch_duration / steps_per_epoch
        total_step_duration_all_rounds += step_duration

    # 计算所有回合的平均值
    average_training_time = total_training_time_all_rounds # 计算整个模型时间
    print(f"total_training_time_all_rounds is : {total_training_time_all_rounds / 60} minutes")
    #average_training_time = total_training_time_all_rounds / len(training_rounds) # 计算每个rounds
    average_epoch_duration = total_epoch_duration_all_rounds / len(training_rounds)
    average_step_duration = total_step_duration_all_rounds / len(training_rounds)

    result = {
        "average_training_time": average_training_time,
        "average_epoch_duration": average_epoch_duration,
        "average_step_duration": average_step_duration
    }
    result_queue.put(result)

    # --- 最终保存逻辑：只保留结构化信息和模型文件 ---
    model_json_path = predictors_path + f"model_{params['ensemble_num']}.json"
    model_conf_to_save = dict(model_conf)

    # 1. 从 params 提取映射数据
    mapping_data = params.get("length_mapping", {})
    norm_to_length = mapping_data.get("norm_to_length", {})
    u1_list = params.get("u1", [])
    u2_list = params.get("u2", [])
    norm_list = params.get("normalize", [])

    # 2. 构造唯一的结构化配置（以原始长度为 Key）
    structured_configs = {}
    for i in range(len(norm_list)):
        current_norm = norm_list[i]
        # 自动匹配原始长度，匹配不上则标记未知
        raw_length = norm_to_length.get(round(float(current_norm), 3), f"unknown_{current_norm}")

        structured_configs[str(raw_length)] = {
            "length": raw_length,
            "length_normed": current_norm,
            "u1": u1_list[i] if i < len(u1_list) else None,
            "u2": u2_list[i] if i < len(u2_list) else None
        }

    # 3. 只将结构化映射存入 JSON，删掉冗余的 u1/u2/normalize 列表
    model_conf_to_save["length_specific_params"] = structured_configs

    with open(model_json_path, "w") as write_file:
        json.dump(model_conf_to_save, write_file, indent=4)

    # 4. 保存模型文件（绝对不能删）
    model.save(predictors_path + f"model_{params['ensemble_num']}.h5")

    logger.info(f"Model {params['ensemble_num']} training completed. Config saved to JSON (structured by length).")