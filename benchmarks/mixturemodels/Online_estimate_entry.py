# Online_estimate_entry.py

import sys
import os
import multiprocessing.context as ctx

# 必须在最前面
ctx._force_start_method("spawn")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from Online_estimate import (
    parse_estimate_args_single,
    run_estimate_processes_single,
)

def main():
    exp_args = parse_estimate_args_single(sys.argv[1:])
    run_estimate_processes_single(exp_args)

if __name__ == "__main__":
    main()
