import getopt
import json
import os
import sys
import itertools
import warnings
import random
from pathlib import Path
import numpy as np
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.functions import rand,min as spark_min, max as spark_max


warnings.filterwarnings("ignore")


def parse_Online_prep_dataset_args(argv: list[str]):
    # parse arguments to a dict  （getopt.getopt() 函数用于解析命令行参数）
    args_dict = {}
    try:
        opts, args = getopt.getopt(
            argv,
            "hd:t:x:l:r:c:y:n:s:f:p:g:w:o:",  # 第二个参数是一个字符串，表示短选项，每个字母后面的冒号表示该选项需要接受一个值。
            ["dataset=", "target=", "conditions=", "label=", "rows=", "columns=", "y-points=", "normalize=", "size=",
             "plot-pdf", "plot-cdf", "log", "preview",],
            # 第三个参数是一个列表，表示长选项，每个选项以双破折线开头，其中有些选项后面也可以跟一个等号和一个值。
            # 如命令 python -m mixturemodels prep_dataset -d wifi/measurement_loc --preview
        )
    except getopt.GetoptError:
        print('Wrong args, type "python -m models_benchmark validate -h" for help')
        sys.exit(2)

    # default values
    args_dict["y_points"] = [0, 100, 400]
    args_dict["plotcdf"] = False
    args_dict["plotpdf"] = False
    args_dict["logplot"] = False
    args_dict["preview"] = False
    args_dict["normalize"] = None
    args_dict["cond_ds_size"] = None
    args_dict["conditions"] = None

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
        elif opt in ("-x", "--conditions"):
            args_dict["conditions"] = json.loads(arg)
        elif opt in ("-l", "--label"):
            args_dict["label"] = arg
        elif opt in ("-r", "--rows"):
            args_dict["rows"] = int(arg)
        elif opt in ("-c", "--columns"):
            args_dict["columns"] = int(arg)
        elif opt in ("-y", "--y-points"):
            args_dict["y_points"] = [int(s.strip()) for s in arg.split(",")]
        elif opt in ("-n", "--normalize"):
            args_dict["normalize"] = [s.strip() for s in arg.split(",")]
        elif opt in ("-s", "--size"):
            args_dict["cond_ds_size"] = int(arg)
        elif opt in ("-f", "--plot-cdf"):
            args_dict["plotcdf"] = True
        elif opt in ("-p", "--plot-pdf"):
            args_dict["plotpdf"] = True
        elif opt in ("-g", "--log"):
            args_dict["logplot"] = True
        elif opt in ("-w", "--preview"):
            args_dict["preview"] = True
    return args_dict


def run_Online_prep_dataset_processes(exp_args: list):
    # 使用日志记录器记录了正在准备数据集的操作
    logger.info(
        "Prepare models benchmark validate args "
        + f"with command line args: {exp_args}"  # exp_args == 解析行返回的字典
    )

    # 创建了一个 SparkSession 对象，以关系型数据库中表的形式生成DataFrame，用于处理大规模数据的框架，指定配置参数
    spark = (
        SparkSession.builder.master("local")
        .appName("LoadParquets")
        .config("spark.executor.memory", "6g")  # 执行器内存
        .config("spark.driver.memory", "8g")  # 驱动程序内存
        .config("spark.driver.maxResultSize", 0)
        .config("spark.driver.extraJavaOptions", "-XX:-TieredCompilation")
        .getOrCreate()
    )
    try:

        # bulk plot axis， linspace()函数用于生成一组均匀分布的数值，作为绘制图表时的Y轴坐标点，由命令行参数 ["y_points"] 决定
        y_points = np.linspace(
            start=exp_args["y_points"][0],
            stop=exp_args["y_points"][2],
            num=exp_args["y_points"][1],
        )

        """获取所有以 parquet 结尾的文件,决定是否预览数据集"""
        # folder setting
        p = Path(__file__).parents[0]  # 获取当前文件的父目录路径
        main_path = str(p) + "/"  # 构建数据集主目录的路径
        # dataset project folder setting
        dataset_project_path = main_path + exp_args["dataset"] + "_results/"  # d = measurement_length

        # open the empirical dataset
        all_files = os.listdir(dataset_project_path)  # 获取路径下的所有文件和文件夹。
        files = []
        for f in all_files:
            if f.endswith(".parquet"):
                files.append(dataset_project_path + f)
        # 以 ".parquet" 结尾的文件将其路径添加到 files 列表中
        # files[0] = wifi/measurement_loc_results/dataset_9_172.parquet
        df = spark.read.parquet(*files)  # 使用SparkSession的read.parquet()方法读取所有Parquet文件，并将它们合并为一个DataFrame。
        total_count = df.count()  # 赋值 DataFrame 中的总样本数
        logger.info(f"Parquet files {files} are loaded.")  # 记录了加载 Parquet 文件。
        logger.info(f"Total number of samples in this empirical dataset: {total_count}")  # 记录数据集中的总样本数

        # Get All column names and it's types
        logger.info("Dataset preview:")
        df.printSchema()  # 打印了 DataFrame 数据集的结构信息

        if exp_args["preview"]:  # 判断命令行是否需要预览数据集。
            df.summary().show()  # 调用pyspark API summary() 方法获取数据集的摘要信息，并通过 show() 方法显示在终端上
            return

        # if exp_args["scale"]:
        """ 决定是否对数据集中的特定列进行缩放和添加噪声
        如果需要对数据集进行缩放和添加噪声，则计算了平均值并设置了缩放因子和噪声参数，并对指定列进行了处理"""
        from pyspark.sql.functions import col, mean, randn
        # col（用于引用 DataFrame 中的列）、mean（用于计算列的平均值）、 randn（用于生成随机噪声）
        noise_seed = random.randint(0, 1000)  # 初始0到1000之间随机数
        send_mean = df.select(mean('send')).collect()[0][0]  # 只有一个平均数
        # 计算 send 列的平均值。使用 .collect() 将结果收集到本地，最后取得第一个元素并获取其值。
        send_scale = 1e-6  # 定义缩放因子和噪声参数
        send_noise_variance = 1  # 论文中添加方差为1毫秒的随机高斯噪声
        send_noise_highvariance = 3
        df = df.withColumn('send_scaled', (col('send') * send_scale))
        # df.withColumn 是 PySpark DataFrame 对象的方法，用于添加新的列或替换现有的列。它接受两个参数：列名和要添加到 DataFrame 中的表达式
        df = df.withColumn('send_standard', ((col('send') - send_mean) * send_scale))
        df = df.withColumn('send_noisy', col('send_standard') + (randn(seed=noise_seed) * send_noise_variance))
        df = df.withColumn('send_verynoisy', col('send_standard') + (randn(seed=noise_seed) * send_noise_highvariance))

        receive_mean = df.select(mean('receive')).collect()[0][0]
        receive_scale = 1e-6
        receive_noise_variance = 1
        receive_noise_highvariance = 3
        df = df.withColumn('receive_scaled', (col('receive') * receive_scale))
        df = df.withColumn('receive_standard', ((col('receive') - receive_mean) * receive_scale))
        df = df.withColumn('receive_noisy', col('receive_standard') + (randn(seed=noise_seed) * receive_noise_variance))
        df = df.withColumn('receive_verynoisy',
                           col('receive_standard') + (randn(seed=noise_seed) * receive_noise_highvariance))

        rtt_mean = df.select(mean('rtt')).collect()[0][0]
        rtt_scale = 1e-6
        rtt_noise_variance = 1
        rtt_noise_highvariance = 3
        df = df.withColumn('rtt_scaled', (col('rtt') * rtt_scale))
        df = df.withColumn('rtt_standard', ((col('rtt') - rtt_mean) * rtt_scale))
        df = df.withColumn('rtt_noisy', col('rtt_standard') + (randn(seed=noise_seed) * rtt_noise_variance))
        df = df.withColumn('rtt_verynoisy', col('rtt_standard') + (randn(seed=noise_seed) * rtt_noise_highvariance))

        means_dict = {
            "send": {
                "mean": send_mean,  # send列的平均值
                "scale": send_scale  # 缩放信息
            },
            "receive": {
                "mean": receive_mean,
                "scale": receive_scale
            },
            "rtt": {
                "mean": rtt_mean,
                "scale": rtt_scale
            }
        }

        """ 如果需要归一化处理，则使用 PySpark 的 ML 包中的 MinMaxScaler 对指定列进行最大-最小归一化处理。"""
        if exp_args["normalize"]:
            from pyspark.ml.feature import MinMaxScaler  # 用于最大-最小归一化
            from pyspark.ml.feature import VectorAssembler  # 用于将列转换为向量
            from pyspark.ml import Pipeline
            from pyspark.sql.functions import udf
            from pyspark.sql.types import DoubleType
            #from pyspark.sql.functions import min, max

            # UDF for converting column type from vector to double type
            # 用于将列类型从向量转换为双精度类型的UDF
            unlist = udf(lambda x: round(float(list(x)[0]), 3), DoubleType())
            # x转化为列表再转换为浮点数，然后保留三位小数，最后返回一个双精度浮点数类型
            # round(..., 3) 对浮点数进行四舍五入，并保留三位小数
            # udf: 用于创建用户自定义函数。它接受两个参数：第一个参数是一个 Python 函数，第二个参数是返回值的数据类型。

            # Iterating over columns to be scaled 迭代要缩放的列
            for i in exp_args["normalize"]:

                assembler = VectorAssembler(inputCols=[i], outputCol=i + "_vect")
                # assembler 是一个 VectorAssembler 对象的实例化 用于将指定的输入列拼接成一个向量，并将结果存储到指定的输出列中。
                # VectorAssembler():将数组(即列表)列转换为向量列

                # MinMaxScaler Transformation  最小最大标量变换
                scaler = MinMaxScaler(inputCol=i + "_vect", outputCol=i + "_normed")
                # MinMaxScaler 是 Spark ML 中用于进行最大-最小归一化的转换器。它将数据的特征缩放到一个指定的范围内，通常是 [0, 1]
                # MinMaxScaler 会将输入列的特征值进行缩放，使得它们在输出列中的值范围在指定的范围内。

                # Pipeline of VectorAssembler and MinMaxScaler 创建一个处理流水线，包括向量转换和归一化处理
                pipeline = Pipeline(stages=[assembler, scaler])
                # Pipeline 是 Spark ML 中用于构建机器学习工作流的类。它允许将多个数据转换器（transformers）和估计器（estimators）组合在一起，形成一个连续的处理流程。
                # Pipeline 可以确保这些转换器和估计器按照指定的顺序依次执行，以完成数据的预处理和模型训练等任务。
                # Pipeline 包含两个阶段（stages），分别是 assembler 和 scaler。这意味着数据将首先通过 assembler 转换器进行特征向量的拼装，然后再通过 scaler 转换器进行最大-最小归一化处理。

                # Fitting pipeline on dataframe
                df = pipeline.fit(df).transform(df).withColumn(i + "_normed", unlist(i + "_normed")).drop(i + "_vect")

                imin = df.agg(spark_min(i)).collect()[0][0]
                imax = df.agg(spark_max(i)).collect()[0][0]

                if i == "length" and exp_args["conditions"]:
                    used_lengths = exp_args["conditions"]["length"]

                    length_norm_map = {}
                    for l in used_lengths:
                        normed = (l - imin) / (imax - imin)
                        length_norm_map[str(l)] = round(normed, 3)

                    means_dict.setdefault("normalize", {})
                    means_dict["normalize"]["length"] = {
                        "global_range": {
                            "min": imin,
                            "max": imax,
                        },
                        "used_lengths": used_lengths,
                        "norm_map": length_norm_map,
                    }

            logger.info("Dataset after normalization:")
            df.summary().show()

        logger.info("Dataset preview after scales and normalizations:")
        # df.summary().show() 打印了经过缩放和归一化处理后的数据集的摘要信息，便于查看数据预处理的效果

        # this project folder setting
        project_path = main_path + exp_args["label"] + "_results/"
        os.makedirs(project_path, exist_ok=True)

        # save json file
        with open(
                project_path + f"info.json",
                "w",
                encoding="utf-8",
        ) as f:
            f.write(json.dumps(means_dict, indent=4))  # json.dumps() 函数用于将 Python 对象转换为 JSON 格式的字符串，四个空格缩进

        if exp_args["conditions"] is None:
            # append the conditional df to the list
            logger.info(f"Dataframe with no conditions has {df.count()} samples.")

            if exp_args["cond_ds_size"]:
                # shuffle samples
                df = df.orderBy(rand())  # 对 DataFrame 进行随机排序，打乱样本顺序
                # take the desired number of records
                df = df.sample(  # sample()随机抽样一定比例的样本
                    withReplacement=False,  # 不进行重复抽样，即每个样本只会被抽取一次；
                    fraction=exp_args["cond_ds_size"] / df.count(),  # 抽样的比例
                    seed=12345,  # 种子？
                )

            logger.info(f"Writing {df.count()} samples to the parquet file.")

            # save the parquet file
            pd_df = df.toPandas()  # 将PySpark DataFrame df 转换为 Pandas DataFrame，方便后续使用 Pandas 提供的功能操作数据
            pd_df.to_parquet(  # to_parquet() 方法用于将 DataFrame 保存为 Parquet 格式的文件
                project_path + "0_records.parquet",
            )
            return

        # create conditional dataframes
        empt_dict = {"cond_dataset_num": None}
        dim_tuple = ()
        for cond_label in exp_args["conditions"]:
            empt_dict = {**empt_dict, cond_label: None}
            # 解包** 将原始的 dict 中的所有键值对添加到新的字典中
            dim_tuple = dim_tuple + (len(exp_args["conditions"][cond_label]),)
            # 表示length 条件标签conditions对应的可能取值分别是 [172, 3440, 6880, 10320]，则 dim_tuple 的值应该是 (4,)。

        logger.info(f"Chosen conditions dimensions: {dim_tuple}")
        dim_list = list(dim_tuple)
        cond_dataframes = []
        conditions = []
        for idx in itertools.product(*[range(s) for s in dim_list]):  # 这个for循环用于筛选出命令行中符合x的数据形成单独文件
            # 使用 itertools.product 生成所有可能的条件组合
            # idx would be (0,0,1) or (1,2,3) in case of having 3 conditions
            # i would be "idx[0]", j "idx[1]" and so on...
            condition_dict = {}  # 用于存储当前条件组合的条件标签和对应的取值。如"length": 10320
            # copy the original df
            cond_df = df.alias(f"cond_df_{len(cond_dataframes)}")  # 复制原始数据框架，并给副本起一个新的别名cond_df
            # dim_tuple 的值是 (4,)，因此，在循环中，len(cond_dataframes) 的值将是每个条件组合的DataFrame的索引，范围是从 0 到 3。
            for jdx, cond_label in enumerate(exp_args["conditions"]):

                # enumerate 是 Python 中的一个内置函数，用于将一个可迭代对象组合为一个索引序列，同时返回索引和对应的值。
                cond = exp_args["conditions"][cond_label][idx[jdx]]  # 获取"length": 和 10320 的索引为 3
                condition_dict[cond_label] = cond  # 键值对"length": 10320 存进字典
                if isinstance(cond, list):  # isinstance(object, classinfo) 判断 cond 是否属于指定的类型或类 list
                    cond_df = cond_df.filter(  # filter()函数用于过滤序列
                        cond_df[cond_label].between(cond[0], cond[1]),  # 对该列中的值进行了范围判断，检查每个值是否在 cond[0] 和 cond[1] 之间
                    )
                else:
                    cond_df = cond_df.filter(
                        df[cond_label] == cond,  # 使用等于号进行过滤
                    )

            if cond_df.count():
                # append the conditional df to the list 说明生成的 DataFrame 所对应的条件以及该 DataFrame 中包含的样本数
                logger.info(
                    f"Dataframe {len(cond_dataframes)} for conditions {condition_dict} has {cond_df.count()} samples.")
                # Dataframe 0 for conditions {'length': 172} has 1256295 samples.
                # cond_df.summary().show()

                # save json file
                with open(
                        project_path + f"{len(cond_dataframes)}_conditions.json",
                        "w",
                        encoding="utf-8",
                ) as f:
                    f.write(json.dumps(condition_dict, indent=4))  # json.dumps() 函数用于将 Python 对象转换为 JSON 格式的字符串，四个空格缩进

                if exp_args["cond_ds_size"]:
                    # shuffle samples 随机抽样
                    cond_df = cond_df.orderBy(rand())  # 对 DataFrame 进行随机排序，打乱样本顺序
                    # take the desired number of records 获取所需数量的记录
                    cond_df = cond_df.sample(
                        withReplacement=False,  # 不进行重复抽样，即每个样本只会被抽取一次；
                        fraction=exp_args["cond_ds_size"] / cond_df.count(),  # 抽样的比例
                        seed=12345,
                    )

                logger.info(f"Writing {cond_df.count()} samples to the parquet file.")

                # save the parquet file
                pd_cond_df = cond_df.toPandas()
                pd_cond_df.to_parquet(
                    project_path + f"{len(cond_dataframes)}_records.parquet",
                )

                # append the conditions dict and dataframe to the lists
                cond_dataframes.append(cond_df)
                conditions.append(condition_dict)
            else:
                logger.info(f"No samples were found for conditions {condition_dict}.")
    finally:
        logger.info("Stopping SparkSession for Online_prep_dataset")
        spark.stop()