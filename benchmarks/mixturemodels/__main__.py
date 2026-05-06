import multiprocessing.context as ctx
import os
import sys


from .plot_evaluation import parse_plot_evaluation_args, plot_evaluation_main
from .train import parse_train_args, run_train_processes
from .Online_train import parse_train_args_Online, run_train_processes_Online
from .train_appendix import parse_train_app_args, run_train_app_processes
from .Online_prep_dataset import parse_Online_prep_dataset_args, run_Online_prep_dataset_processes
from .prep_dataset import parse_prep_dataset_args, run_prep_dataset_processes
from .Online_estimate import parse_estimate_args_single, run_estimate_processes_single
from .validate_pred import parse_validate_pred_args, run_validate_pred_processes
from .plot_prepped_dataset import parse_plot_prepped_dataset_args, run_plot_prepped_dataset_processes
from .evaluate_pred import parse_evaluate_pred_args, run_evaluate_pred_processes
from .plot_final_validate import parse_plot_final_validate_args, run_plot_final_validate_processes
from .model_params_view import model_params_view_parse, model_params_view_function
from .validate_meanpred import parse_validate_pred_args_mean, run_validate_pred_processes_mean
from .calcu_400_mean import parse_validate_pred_args_mean400, run_validate_pred_processes_mean400
from .calcu_400_maxmin_mean import parse_validate_pred_args_maxminmean400, run_validate_pred_processes_maxminmean400
from .phase_one95 import parse_validate_pred_args_one95, run_validate_pred_processes_one95
from .Online_phase_one import parse_online_phase_one_args, run_online_phase_one_processes
from .phase_one import parse_validate_pred_args_one, run_validate_pred_processes_one




# very important line to make tensorflow run in sub processes
ctx._force_start_method("spawn")
# disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == "__main__":

    argv = sys.argv[1:]
    if argv[0] == "Online_prep_dataset":
        validate_gym_args = parse_Online_prep_dataset_args(argv[1:])
        run_Online_prep_dataset_processes(validate_gym_args)
    elif argv[0] == "prep_dataset":
        validate_gym_args = parse_prep_dataset_args(argv[1:])
        run_prep_dataset_processes(validate_gym_args)
    elif argv[0] == "plot_prepped_dataset":
        validate_gym_args = parse_plot_prepped_dataset_args(argv[1:])
        run_plot_prepped_dataset_processes(validate_gym_args)
    elif argv[0] == "train":
        train_args = parse_train_args(argv[1:])
        run_train_processes(train_args)
    elif argv[0] == "Online_train":
        train_args = parse_train_args_Online(argv[1:])
        run_train_processes_Online(train_args)
    elif argv[0] == "train_app":
        train_args = parse_train_app_args(argv[1:])
        run_train_app_processes(train_args)
    elif argv[0] == "validate_pred":
        validate_args = parse_validate_pred_args(argv[1:])
        run_validate_pred_processes(validate_args)
    elif argv[0] == "Online_estimate":
        validate_args = parse_estimate_args_single(argv[1:])
        run_estimate_processes_single(validate_args)
    elif argv[0] == "validate_meanpred":
        validate_args = parse_validate_pred_args_mean(argv[1:])
        run_validate_pred_processes_mean(validate_args)
    elif argv[0] == "calcu_400_mean":
        validate_args = parse_validate_pred_args_mean400(argv[1:])
        run_validate_pred_processes_mean400(validate_args)
    elif argv[0] == "calcu_400_maxmin_mean":
        validate_args = parse_validate_pred_args_maxminmean400(argv[1:])
        run_validate_pred_processes_maxminmean400(validate_args)
    elif argv[0] == "phase_one":
        validate_args = parse_validate_pred_args_one(argv[1:])
        run_validate_pred_processes_one(validate_args)
    elif argv[0] == "Online_phase_one":
        validate_args = parse_online_phase_one_args(argv[1:])
        run_online_phase_one_processes(validate_args)
    elif argv[0] == "phase_one95":
        validate_args = parse_validate_pred_args_one95(argv[1:])
        run_validate_pred_processes_one95(validate_args)
    elif argv[0] == "plot_final_validate":
        plot_val_args = parse_plot_final_validate_args(argv[1:])
        run_plot_final_validate_processes(plot_val_args)
    elif argv[0] == "evaluate_pred":
        train_args = parse_evaluate_pred_args(argv[1:])
        run_evaluate_pred_processes(train_args)
    elif argv[0] == "plot_evaluation":
        plot_args = parse_plot_evaluation_args(argv[1:])
        plot_evaluation_main(plot_args)
    elif argv[0] == "model_params_view":
        view_args = model_params_view_parse(argv[1:])
        model_params_view_function(view_args)

    else:
        raise Exception("wrong command line option")
