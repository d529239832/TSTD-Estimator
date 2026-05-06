import getopt
import json
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
import shutil
from datetime import datetime
MAX_HISTORY = 10
warnings.filterwarnings("ignore")
# very important line to make tensorflow run in sub processes
ctx._force_start_method("spawn")
# disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def parse_train_args_Online(argv: list[str]):
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


def format_duration(seconds: float):
    if seconds < 1e-3:
        return seconds * 1e6, "µs"
    elif seconds < 1:
        return seconds * 1e3, "ms"
    elif seconds < 60:
        return seconds, "s"
    else:
        return seconds / 60, "min"


def run_train_processes_Online(exp_args: list):
    manager = mp.Manager()
    result_queue = manager.Queue()

    logger.info(
        "Prepare models benchmark experiment args "
        + f"with command line args: {exp_args}"
    )

    p = Path(__file__).parents[0]
    main_path = str(p) + "/"
    project_path = str(p) + "/" + exp_args["label"] + "_results/"
    os.makedirs(project_path, exist_ok=True)
    parquet_folder = main_path + "wifiOL/__Sampledata__"
    os.makedirs(parquet_folder, exist_ok=True)

    train_configs = exp_args["train_config"]
    n_workers = 1
    logger.info(f"Initializng {n_workers} workers")
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = mp.Pool(n_workers)
    signal.signal(signal.SIGINT, original_sigint_handler)

    n_runs = len(train_configs.keys()) * exp_args["ensembles"]

    params_list = []
    for ensemble_num in range(exp_args["ensembles"]):
        for model_conf_key in train_configs.keys():
            model_conf = train_configs[model_conf_key]

            records_path = project_path + model_conf_key + "/"
            os.makedirs(records_path, exist_ok=True)

            jsoninfo = json.dumps(model_conf)
            with open(records_path + "info.json", "w") as f:
                f.write(jsoninfo)

            dataset_path = main_path + exp_args["dataset"] + "_results/"
            if isinstance(exp_args["u1"], str):
                exp_args["u1"] = [float(x) for x in exp_args["u1"].split(",")]

            if isinstance(exp_args["u2"], str):
                exp_args["u2"] = [float(x) for x in exp_args["u2"].split(",")]

            if isinstance(exp_args["normalize"], str):
                exp_args["normalize"] = [float(x) for x in exp_args["normalize"].split(",")]

            params = {
                "ensembles": exp_args["ensembles"],
                "order_seed": 9988334,
                "ensemble_num": ensemble_num,
                "sample_seed": ensemble_num * 101012,
                "dataset_path": dataset_path,
                "records_path": records_path,
                "model_conf": model_conf,
                "model_conf_key": model_conf_key,
                "parquet_folder": parquet_folder,
                "trained_model_path": project_path + "gmevm/" + f"model_0.h5",
                "datatype": exp_args["datatype"],
                "u1": exp_args["u1"],
                "u2": exp_args["u2"],
                "normalize": exp_args["normalize"]
            }
            params_list.append(params)

    # 替换为无Spark的数据加载采样
    load_dataset_and_sample(params)

    try:
        logger.info(f"Starting {n_runs} jobs")
        res = pool.starmap(train_model, [(params, result_queue) for params in params_list])
        logger.info("Waiting for results")

        total_training_time_all_processes = 0
        total_epoch_duration_all_processes = 0
        total_step_duration_all_processes = 0
        total_processes = result_queue.qsize()

        while not result_queue.empty():
            result = result_queue.get()
            total_training_time_all_processes += result["average_training_time"]
            total_epoch_duration_all_processes += result["average_epoch_duration"]
            total_step_duration_all_processes += result["average_step_duration"]

        if total_processes > 0:
            avg_training_time = total_training_time_all_processes / total_processes
            avg_epoch_duration = total_epoch_duration_all_processes / total_processes
            avg_step_duration = total_step_duration_all_processes / total_processes

            train_val, train_unit = format_duration(avg_training_time)
            epoch_val, epoch_unit = format_duration(avg_epoch_duration)
            step_val, step_unit = format_duration(avg_step_duration)

            logger.info(f"Global Average Training Time: {train_val:.4f} {train_unit}")
            logger.info(f"Global Average Epoch Duration: {epoch_val:.4f} {epoch_unit}")
            logger.info(f"Global Average Step Duration: {step_val:.4f} {step_unit}")

            json_filename = os.path.join(project_path, "train_time.json")
            results_dict = {
                "global_avg_training_time": {"value": train_val, "unit": train_unit},
                "global_avg_epoch_duration": {"value": epoch_val, "unit": epoch_unit},
                "global_avg_step_duration": {"value": step_val, "unit": step_unit}
            }
            with open(json_filename, "w") as json_file:
                json.dump(results_dict, json_file, indent=4)
            logger.info("Saved training results to train_time.json")
        else:
            logger.warning("No results were collected from the workers.")
    except KeyboardInterrupt:
        logger.info("Caught KeyboardInterrupt, terminating workers")
        pool.terminate()
    else:
        logger.info("Normal termination")
        pool.close()


def load_dataset_and_sample(params):
    logger.info("Load dataset and sample (No-Spark Stable Version)")

    all_files = os.listdir(params["dataset_path"])
    files = []
    for f in all_files:
        if f.endswith(".parquet"):
            files.append(os.path.join(params["dataset_path"], f))

    if not files:
        logger.error(f"No parquet files found in {params['dataset_path']}")
        return

    # 读取所有 parquet 文件合并为 Pandas DataFrame
    logger.info(f"Reading {len(files)} files into Pandas...")
    df_list = [pd.read_parquet(f) for f in files]
    main_df = pd.concat(df_list, ignore_index=True)

    # 彻底打乱数据 (等效于 main_df.orderBy(rand(seed=...)))
    main_df = main_df.sample(frac=1, random_state=params["order_seed"]).reset_index(drop=True)
    total_count = len(main_df)
    logger.info(f"Total dataset size: {total_count}")

    # 设定抽样比例为 21%
    SAMPLING_FRAC = 0.21

    for ensemble_num in range(params["ensembles"]):
        # 无论数据量多大，统一抽取 20%
        # 建议种子加上 ensemble_num，确保每个集成子集的数据有所不同
        df_train = main_df.sample(frac=SAMPLING_FRAC, random_state=params["sample_seed"] + ensemble_num)

        logger.info(f"{ensemble_num}: Sampled {len(df_train)} rows (20% of total)")

        ensemble_parquet_addr = os.path.join(params["parquet_folder"], f"{ensemble_num}.parquet")
        logger.info(f"{ensemble_num}: Writing sub-dataset into {ensemble_parquet_addr}")
        df_train.to_parquet(ensemble_parquet_addr, compression="snappy", index=False)

    # 释放内存
    del main_df
    del df_list

def cleanup_history(history_root: Path, max_keep: int):
    if not history_root.exists():
        return

    # 按时间戳字符串排序（时间戳格式保证字典序等于时间序）
    runs = sorted(
        [d for d in history_root.iterdir() if d.is_dir()],
        key=lambda x: x.name
    )

    excess = runs[:-max_keep]

    for d in excess:
        try:
            logger.info(f"Removing old training history: {d}")
            shutil.rmtree(d)
        except Exception as e:
            logger.warning(f"Failed to remove {d}: {e}")

def train_model(params, result_queue):
    import tensorflow as tf
    from pr3d.de import (ConditionalGaussianMixtureEVM, ConditionalGaussianMM, GaussianMM, GaussianMixtureEVM,
                         ConditionalGaussianMixtureEVMW2WU)

    logger.info(f"{params['model_conf_key']}.{params['ensemble_num']}: training start")
    model_conf = params["model_conf"]
    predictors_path = params["records_path"]
    training_params = model_conf["training_params"]
    strdtype = "float64"

    os.makedirs(predictors_path, exist_ok=True)
    ensemble_parquet_addr = os.path.join(params["parquet_folder"], f"{params['ensemble_num']}.parquet")
    df_train = pd.read_parquet(ensemble_parquet_addr)

    y_label = model_conf["y_label"]
    model_type = model_conf["type"]
    training_rounds = training_params["rounds"]
    batch_size = training_params["batch_size"]

    if "condition_labels" not in model_conf:
        df_train["y_input"] = df_train[y_label]
        if model_type == "gmm":
            model = GaussianMM(centers=model_conf["centers"], dtype=strdtype, bayesian=model_conf["bayesian"])
        elif model_type == "gmevm":
            model = GaussianMixtureEVM(centers=model_conf["centers"], dtype=strdtype, bayesian=model_conf["bayesian"])
        X = None
        Y = df_train.y_input
    else:
        condition_labels = model_conf["condition_labels"]
        df_train["y_input"] = df_train[y_label]
        if model_type == "gmm":
            model = ConditionalGaussianMM(x_dim=condition_labels, centers=model_conf["centers"],
                                          hidden_sizes=model_conf["hidden_sizes"], dtype=strdtype,
                                          bayesian=model_conf["bayesian"])
        elif model_type == "gmevm":
            model = ConditionalGaussianMixtureEVM(x_dim=condition_labels, centers=model_conf["centers"],
                                                  hidden_sizes=model_conf["hidden_sizes"], dtype=strdtype,
                                                  bayesian=model_conf["bayesian"])
        elif model_type == "gmevmw2wu":
            model = ConditionalGaussianMixtureEVMW2WU(x_dim=condition_labels, centers=model_conf["centers"],
                                                      hidden_sizes=model_conf["hidden_sizes"], dtype=strdtype,
                                                      bayesian=model_conf["bayesian"], datatype=params["datatype"],
                                                      u1=params["u1"], u2=params["u2"], normalize=params["normalize"])
        X = df_train[condition_labels]
        Y = df_train.y_input

    steps_per_epoch = 16
    # steps_per_epoch = len(df_train) // batch_size
    total_training_time_all_rounds = 0
    total_epoch_duration_all_rounds = 0
    total_step_duration_all_rounds = 0

    for idx, round_params in enumerate(training_rounds):
        start_time = time.time()
        model.training_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=round_params["learning_rate"]),
                                     loss=model.loss)

        if X is None:
            Xnp, Ynp = np.zeros(len(Y)), np.array(Y)
            model.training_model.fit(x=[Xnp, Ynp], y=Ynp, steps_per_epoch=steps_per_epoch,
                                     epochs=round_params["epochs"], verbose=1)
        else:
            Xnp, Ynp = np.array(X), np.array(Y)
            training_data = tuple([Xnp[:, i] for i in range(len(condition_labels))]) + (Ynp,)
            model.training_model.fit(x=training_data, y=Ynp, steps_per_epoch=steps_per_epoch,
                                     epochs=round_params["epochs"], verbose=1)

        end_time = time.time()
        duration = end_time - start_time
        total_training_time_all_rounds += duration
        epoch_dur = duration / round_params["epochs"]
        total_epoch_duration_all_rounds += epoch_dur
        total_step_duration_all_rounds += (epoch_dur / steps_per_epoch)

    result_queue.put({
        "average_training_time": total_training_time_all_rounds,
        "average_epoch_duration": total_epoch_duration_all_rounds / len(training_rounds),
        "average_step_duration": total_step_duration_all_rounds / len(training_rounds)
    })


    # 保存模型 JSON 文件，并记录 u1, u2, normalize
    model_json_path = predictors_path + f"model_{params['ensemble_num']}.json"
    model_conf_to_save = dict(model_conf)  # 复制一份原始配置

    # 添加 u1, u2, normalize
    model_conf_to_save["u1"] = params["u1"]
    model_conf_to_save["u2"] = params["u2"]
    model_conf_to_save["normalize"] = params["normalize"]

    with open(model_json_path, "w") as write_file:
        json.dump(model_conf_to_save, write_file, indent=4)

    logger.info(f"Model {params['ensemble_num']} trained and saved with u1/u2/normalize in JSON.")

    # 保存模型 h5 文件
    model.save(predictors_path + f"model_{params['ensemble_num']}.h5")

    # ==============================
    # history 版本归档逻辑
    # ==============================

    current_model_dir = Path(predictors_path)
    history_root = current_model_dir.parent / "history"
    history_root.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    history_run_dir = history_root / timestamp
    history_run_dir.mkdir(parents=True, exist_ok=True)

    json_file = current_model_dir / f"model_{params['ensemble_num']}.json"
    h5_file = current_model_dir / f"model_{params['ensemble_num']}.h5"

    shutil.copy2(json_file, history_run_dir / json_file.name)
    shutil.copy2(h5_file, history_run_dir / h5_file.name)

    logger.info(f"Archived model to history folder: {history_run_dir}")

    # 自动清理旧历史，只保留最近 MAX_HISTORY 个
    cleanup_history(history_root, MAX_HISTORY)
