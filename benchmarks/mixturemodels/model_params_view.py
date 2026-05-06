import getopt
import json
import pdb
import sys
import warnings
from pathlib import Path
import numpy as np
import numpy.typing as npt
from pr3d.de import (ConditionalGaussianMixtureEVM,GMEVMU2_params_output,ConditionalGaussianMixtureEVMW2WU)

warnings.filterwarnings("ignore")


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
    for model_list in exp_args["models"]:
        model_project_name = model_list[0]
        model_conf_key = model_list[1]
        ensemble_num = model_list[2]
        # import pdb
        # pdb.set_trace()
        # x = exp_args["label"]   # 0, 0.322, 0.661, 1
        x: npt.NDArray[np.float64] = np.array([exp_args["label"]], dtype=np.float64)
        model_path = (main_path + model_project_name + "_results/" + model_conf_key + "/")
        # model_path: /project/wireless-pr3d/benchmarks/mixturemodels/wifi/trained_length_dl_l_results/gmevm/
        with open(
                model_path + f"model_{ensemble_num}.json"
        ) as json_file:
            model_dict = json.load(json_file)

        if model_dict["type"] == "gmevm":
            pr_model = ConditionalGaussianMixtureEVM(
                h5_addr=model_path + f"model_{ensemble_num}.h5",
            )
            prediction_res = pr_model._params_model.predict(x, )
            # import pdb
            # pdb.set_trace()
            print("GMEVM分量的权重:", prediction_res[0][0])
            print("GMEVM分量的均值:", prediction_res[1][0])
            print("GMEVM分量的方差:", prediction_res[2][0])
            print("GMEVM参数xi:", prediction_res[3][0])
            print("GMEVM参数u:", prediction_res[4][0])
            print("GMEVM参数beta:", prediction_res[5][0])
        elif model_dict["type"] == "gmevmu2":
            pr_model_1 = ConditionalGaussianMixtureEVM(
                h5_addr=main_path + "wifi/trained_length_dl_l_results/gmevm/model_0.h5",
            )
            prediction_res_1 = pr_model_1._params_model.predict(x, )

            # 抽取GMEVMW2模型的 weight_alpha 参数
            pr_model_2 = GMEVMU2_params_output(
                h5_addr=main_path + "wifi/trained_length_dl_l_results/gmevmu2/model_0.h5",
            )
            prediction_res_2 = pr_model_2._params_model.predict(x, )

            print("GEU2分量的权重:", prediction_res_1[0][0])
            print("GEU2分量的均值:", prediction_res_1[1][0])
            print("GEU2分量的方差:", prediction_res_1[2][0])
            print("GEU2参数xi:", prediction_res_1[3][0])
            print("GEU2参数u:", prediction_res_1[4][0])
            print("GEU2参数beta:", prediction_res_1[5][0])
            # import pdb
            # pdb.set_trace()
            print("GEU2_U:", prediction_res_2[0][0])
        elif model_dict["type"] == "gmevmw2wu":
            pr_model = ConditionalGaussianMixtureEVMW2WU(
                h5_addr=model_path + f"model_{ensemble_num}.h5",
            )
            prediction_res = pr_model._params_model.predict(x, )
            # import pdb
            # pdb.set_trace()
            print("Gw分量的weights:", prediction_res[0][0])
            print("Gw分量的locs:", prediction_res[1][0])
            print("Gw分量的scales:", prediction_res[2][0])
            print("Gw参数tail_param:", prediction_res[3][0])
            #print("Gw参数tail_threshold:", prediction_res[4][0])
            print("Gw参数tail_scale:", prediction_res[4][0])
            print("tail_weights:", prediction_res[5][0])
            print("tail_param2:", prediction_res[6][0])
            #print("tail_threshold_delta:", prediction_res[8][0])
            print("tail_scale2:", prediction_res[7][0])
            print("tail_weights2:", prediction_res[8][0])
            print("Gw参数tail_threshold:", prediction_res[9][0])
            print("tail_threshold_delta:", prediction_res[10][0])
            #print("tail_threshold_delta2:", prediction_res[11][0])




