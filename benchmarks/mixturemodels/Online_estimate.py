import getopt
import json
import os
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

# --- 深度还原 LaTeX 学术风格字体配置 ---
# 强制使用衬线字体
matplotlib.rcParams['font.family'] = 'serif'
# 优先使用 STIXGeneral 或 Computer Modern 风格，这比 Times New Roman 更瘦、更尖锐
matplotlib.rcParams['font.serif'] = ['STIXGeneral', 'DejaVu Serif', 'Computer Modern Serif', 'serif']
# 关键：设置数学字体集为 cm (Computer Modern)，解决 10^-2 这种指数的样式问题
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['axes.unicode_minus'] = False
# 刻度线方向与镜像设置（匹配图片中四周都有刻度且向内的样式）
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['xtick.bottom'] = True
matplotlib.rcParams['ytick.left'] = True
matplotlib.rcParams['ytick.right'] = True
# ------------------------------------

import matplotlib.pyplot as plt
from loguru import logger
from pr3d.de import ConditionalGaussianMixtureEVMW2WU
import faulthandler
from datetime import datetime

warnings.filterwarnings("ignore")
# Web中 history 下最多只保留 ? 个时间戳目录
MAX_HISTORY = 50
faulthandler.enable()


# 设置图形边框的粗细
def adjust_plot_borders(ax):
    # 设置四个边框的粗细
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)  # 1是边框的粗细，可以根据需要调整


def parse_estimate_args_single(argv: list[str]):
    # parse arguments to a dict
    args_dict = {}
    try:
        opts, args = getopt.getopt(
            argv,
            "hd:t:x:n:m:l:r:c:y:z:f:i:p:g:o:s:q:u:v:k:",
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
                "round-number=",
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
    args_dict["round_number"] = None

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
        elif opt == "--round-number":
            args_dict["round_number"] = int(arg)

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


def cleanup_history(history_root: Path, max_keep: int):
    if not history_root.exists():
        return

    runs = sorted(
        [d for d in history_root.iterdir() if d.is_dir()],
        key=lambda x: x.name
    )

    excess = runs[:-max_keep]
    for d in excess:
        try:
            logger.info(f"Removing old history: {d}")
            for f in d.glob("*"):
                f.unlink()
            d.rmdir()
        except Exception as e:
            logger.warning(f"Failed to remove {d}: {e}")


def run_estimate_processes_single(exp_args: list):
    logger.info("Prepare models benchmark validate args " + f"with command line args: {exp_args}")

    p = Path(__file__).parents[0]
    main_path = str(p) + "/"
    project_path = main_path + exp_args["label"] + "_results/"
    os.makedirs(project_path, exist_ok=True)

    estimate_root = Path(project_path)
    latest_dir = estimate_root / "latest"
    round_str = f"_round_{exp_args['round_number']}" if exp_args["round_number"] is not None else ""
    history_dir = estimate_root / "history" / (
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + round_str
    )
    latest_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    dataset_project_path = main_path + exp_args["dataset"] + "_results/"

    # 加载条件数据
    conditions = []
    cond_dataframes = []
    for cond_num in exp_args["condition_nums"]:
        cond_dict, cond_df = lookup_df(dataset_project_path, cond_num)
        cond_dataframes.append(cond_df)
        conditions.append(cond_dict)

    # info.json
    json_path = os.path.join(dataset_project_path, "info.json")
    with open(json_path, "r") as file:
        info_json = json.load(file)

    key_label = exp_args["target"]
    key_mean = info_json[key_label]["mean"]
    key_scale = info_json[key_label]["scale"]

    # --- 新增逻辑：计算所有 parquet 文件中的全局最大延时 ---
    global_max_val = -1.0
    target_col = key_label + '_scaled'

    # 遍历目录下所有 parquet 文件
    for parquet_file in Path(dataset_project_path).glob("*.parquet"):
        try:
            # 只读取目标列以提高速度
            temp_df = pd.read_parquet(parquet_file, columns=[target_col])
            current_max = temp_df[target_col].max()
            if current_max > global_max_val:
                global_max_val = current_max
        except Exception as e:
            logger.warning(f"Failed to read {parquet_file} for max value check: {e}")

    # 判断并更新 exp_args 里的 y_points 参数
    # exp_args["y_points"][2] 是 stop 值，[1] 是 num 个数
    if global_max_val > exp_args["y_points"][2]:
        new_max_limit = global_max_val + 10
        exp_args["y_points"][2] = int(new_max_limit)
        exp_args["y_points"][1] = int(new_max_limit * 2)
        logger.info(f"Global max {global_max_val} found. Updating y_points to: {exp_args['y_points']}")
    else:
        logger.info(f"Global max {global_max_val} is within range. Using default y_points.")
    # --------------------------------------------------

    y_points = np.linspace(exp_args["y_points"][0], exp_args["y_points"][2], exp_args["y_points"][1])
    y_points_standard = np.linspace(exp_args["y_points"][0] - (key_mean * key_scale),
                                    exp_args["y_points"][2] - (key_mean * key_scale),
                                    exp_args["y_points"][1])

    single_plot = exp_args["rows"] == 0 and exp_args["columns"] == 0

    # 初始化 figure
    if exp_args["plotcdf"]:
        if not single_plot:
            cdf_fig, cdf_axes = plt.subplots(nrows=exp_args["rows"], ncols=exp_args["columns"],
                                             figsize=(7 * exp_args["columns"], 5 * exp_args["rows"]))
            cdf_axes = cdf_axes.flat
        else:
            cdf_fig, cdf_ax = plt.subplots(1, 1)

    if exp_args["plottail"]:
        if not single_plot:
            tail_fig, tail_axes = plt.subplots(nrows=exp_args["rows"], ncols=exp_args["columns"],
                                               figsize=(7 * exp_args["columns"], 5 * exp_args["rows"]))
            tail_axes = tail_axes.flat
        else:
            tail_fig, tail_ax = plt.subplots(1, 1)

    if exp_args["plotpdf"]:
        if not single_plot:
            pdf_fig, pdf_axes = plt.subplots(nrows=exp_args["rows"], ncols=exp_args["columns"],
                                             figsize=(7 * exp_args["columns"], 5 * exp_args["rows"]))
            pdf_axes = pdf_axes.flat
        else:
            pdf_fig, pdf_ax = plt.subplots(1, 1)

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "round_number": exp_args.get("round_number", None),
        "target": exp_args["target"],
        "conditions": []
    }

    # 遍历条件
    for idx, cond_dict in enumerate(conditions):
        logger.info(f"Plotting dataframe {idx} with conditions {cond_dict}")
        cond_df = cond_dataframes[idx]
        total_count = len(cond_df)

        # 使用 Pandas 计算经验分布
        emp_cdf = [(cond_df[key_label + '_scaled'] <= y).sum() / total_count for y in y_points]
        emp_pdf = np.diff(emp_cdf)
        emp_pdf = np.append(emp_pdf, [0]) * exp_args["y_points"][1] / (
                    exp_args["y_points"][2] - exp_args["y_points"][0])

        emp_max_x = cond_df[key_label + "_scaled"].max()
        emp_max_x = float(emp_max_x) if pd.notnull(emp_max_x) else 0.0
        x_max_auto = min(emp_max_x, exp_args["y_points"][2])

        emp_tail = np.float64(1.00) - np.array(emp_cdf, dtype=np.float64)

        # 绘制测量数据
        if exp_args["plotcdf"]:
            ax = cdf_ax if single_plot else cdf_axes[idx]
            ax.plot(y_points, emp_cdf, marker=exp_args["condition_markers"][idx], label=f"meas. {cond_dict}")
        if exp_args["plottail"]:
            ax = tail_ax if single_plot else tail_axes[idx]
            ax.plot(y_points, np.float64(1.00) - np.array(emp_cdf, dtype=np.float64), marker=".", markersize=4,
                    color='black', label="MEAS")
        if exp_args["plotpdf"]:
            ax = pdf_ax if single_plot else pdf_axes[idx]
            ax.plot(y_points, emp_pdf, marker=exp_args["condition_markers"][idx], label=f"meas. {cond_dict}")

        # 模型预测
        for model_list in exp_args["models"]:
            model_project_name, model_conf_key, ensemble_num = model_list
            model_path = main_path + model_project_name + "_results/" + model_conf_key + "/"

            with open(model_path + f"model_{ensemble_num}.json") as json_file:
                model_dict = json.load(json_file)

            cond_columns = model_dict["condition_labels"]

            sample_size = len(y_points_standard)
            if total_count > sample_size:
                rows = cond_df[cond_columns].sample(n=sample_size, random_state=0)
            else:
                rows = cond_df[cond_columns]

            x = rows.values
            y = np.array(y_points_standard, dtype=np.float64)

            if model_dict["type"] == "gmevmw2wu":
                pr_model = ConditionalGaussianMixtureEVMW2WU(
                    h5_addr=model_path + f"model_{ensemble_num}.h5",
                    datatype=exp_args["datatype"],
                    u1=exp_args["u1"],
                    u2=exp_args["u2"],
                    normalize=exp_args["normalize"],
                )
            prob, logprob, pred_cdf = pr_model.prob_batch(x, y)

            pred_tail = np.float64(1.00) - np.array(pred_cdf, dtype=np.float64)
            mae_tail = np.mean(np.abs(emp_tail - pred_tail))

            # 获取当前绘图轴
            if exp_args["plottail"]:
                ax = tail_ax if single_plot else tail_axes[idx]
            elif exp_args["plotcdf"]:
                ax = cdf_ax if single_plot else cdf_axes[idx]
            else:
                ax = pdf_ax if single_plot else pdf_axes[idx]

            if exp_args["logplot"]:
                ax.set_yscale('log')
            elif exp_args["loglogplot"]:
                ax.set_yscale('log')
                ax.set_xscale('log')
            if exp_args["prob_lims"]:
                ax.set_ylim(exp_args["prob_lims"][0], exp_args["prob_lims"][1])

            # 绘制预测
            if exp_args["plotcdf"]:
                ax = cdf_ax if single_plot else cdf_axes[idx]
                ax.plot(y_points, pred_cdf, marker="",
                        label="pred. " + model_project_name + "." + model_conf_key + "." + ensemble_num)
            if exp_args["plottail"]:
                ax = tail_ax if single_plot else tail_axes[idx]
                ax.plot(y_points, np.float64(1.00) - np.array(pred_cdf, dtype=np.float64), color="blue", label="TSTD",
                        linestyle='-', marker="")
                ax.set_xlim(0, x_max_auto + 10)

                ax.set_title(f"'length': {cond_dict['length']} bytes", fontsize=25)

                ax.set_xlabel("End-to-End delay [ms]", fontsize=26)
                ax.set_ylabel("Tail probability", fontsize=26)
                ax.grid(linestyle='-', linewidth=0.8, alpha=0.7)

                # 设置刻度细节
                ax.minorticks_on()
                legend = ax.legend(prop={'size': 22}, frameon=False)
                for line in legend.get_lines():
                    line.set_linewidth(2)

                # 严格匹配图中向内且四周显示的刻度参数
                ax.tick_params(axis='both', which='major', labelsize=22, width=1.2, length=6)
                ax.tick_params(axis='both', which='minor', labelsize=20, width=1.0, length=4)

                adjust_plot_borders(ax)

            if exp_args["plotpdf"]:
                ax = pdf_ax if single_plot else pdf_axes[idx]
                ax.plot(y_points, prob, marker="",
                        label="pred. " + model_project_name + "." + model_conf_key + "." + ensemble_num)

        metrics["conditions"].append({
            "condition": cond_dict,
            "total_samples": total_count,
            "u1": exp_args["u1"][idx] if "u1" in exp_args else None,
            "u2": exp_args["u2"][idx] if "u2" in exp_args else None,
            "normalize": exp_args["normalize"][idx] if "normalize" in exp_args else None,
            "empirical_max_delay": emp_max_x,
            "tail_mae": float(mae_tail)
        })

    def save_fig(fig, name):
        fig.tight_layout()
        fig.savefig(history_dir / name, dpi=150)
        fig.savefig(latest_dir / name, dpi=150)
        plt.close(fig)

    if exp_args["plotcdf"]:
        save_fig(cdf_fig, "cdf.png")
    if exp_args["plottail"]:
        save_fig(tail_fig, "tail.png")
    if exp_args["plotpdf"]:
        save_fig(pdf_fig, "pdf.png")

    with open(latest_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(history_dir / "metrics.json", "w") as f_hist:
        json.dump(metrics, f_hist, indent=2)

    cleanup_history(estimate_root / "history", MAX_HISTORY)