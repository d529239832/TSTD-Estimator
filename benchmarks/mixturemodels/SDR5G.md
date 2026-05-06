# SDR5G

Download the parquet files, store them in `mixturemodels/oai5g/measurement_results`.

We analyze uplink delays, conditioned on the uplink MCS indices: 3,5, and 7. It contains 5M samples in total.我们分析上行链路延迟，条件是上行链路MCS索引:3、5和7。它总共包含500万个样本。
```
python -m mixturemodels prep_dataset -d oai5g/measurement -x '{"mcs_UL":[3,5,7]}' -l oai5g/prepped_ulmcs_norm_no1 -n mcs_UL
python -m mixturemodels plot_prepped_dataset -d oai5g/prepped_ulmcs_norm_no1 -x 0,1,2 -m .,*,2 -t send_scaled --plot-pdf --plot-tail --log -r 1 -c 3 -y 0,400,50 -l 1e-6,1
python -m mixturemodels plot_prepped_dataset -d oai5g/prepped_ulmcs_norm_no1 -x 0,1,2 -m .,*,2 -t send_scaled --plot-tail --log -r 0 -c 0 -y 0,400,50 -l 1e-6,1
```

Create another dataset, without MCS=5. This dataset is used for evaluation of the ML model's generalization.
```
创建另一个数据集，不使用MCS=5。该数据集用于评估ML模型的泛化能力。
python -m mixturemodels prep_dataset -d oai5g/measurement -x '{"mcs_UL":[3,7]}' -l oai5g/prepped_ulmcs_norm_no1a5 -n mcs_UL
```

默认培训配置:
Default training config:
```
{
    "gmevm" : {
        "type": "gmevm",
        "bayesian": false,
        "centers": 15,
        "hidden_sizes": [10, 100, 100, 80],
        "condition_labels" : ["mcs_UL_normed"],
        "y_label" : "send_noisy",
        "training_params": {
            "dataset_size": 1000000,
            "batch_size": 128000,
            "rounds" : [
                {"learning_rate": 1e-2, "epochs":200},
                {"learning_rate": 1e-3, "epochs":200},
                {"learning_rate": 1e-4, "epochs":200},
                {"learning_rate": 1e-5, "epochs":200}
            ]
        }
    }
}
```
在具有高斯噪声的3级训练样本上训练预测器。大型:1M样本(20%)，中型:256k样本(5%)，小型:64k样本(1.3%)。然后对比一下。批量大小始终是训练样本的1/8。
Train predictors on 3 levels of training samples with Gaussian noise. Large: 1M samples (20%), Medium: 256k samples (5%), Small: 64k samples (1.3%). Then compare them. Batch size is always 1/8 of the training samples.
```
python -m mixturemodels train -d oai5g/prepped_ulmcs_norm_no1 -l oai5g/trained_ulmcs_l_noisy -c mixturemodels/oai5g/train_conf_l_noisy.json -e 5 -t SDR
python -m mixturemodels train -d oai5g/prepped_ulmcs_norm_no1 -l oai5g/trained_ulmcs_l_noisy -c mixturemodels/oai5g/train_conf_lw_noisy.json -e 5 -t SDR
python -m mixturemodels train -d oai5g/prepped_ulmcs_norm_no1 -l oai5g/trained_ulmcs_l -c mixturemodels/oai5g/train_conf_lw.json -e 5 -t SDR
python -m mixturemodels train -d oai5g/prepped_ulmcs_norm_no1 -l oai5g/trained_ulmcs_l_verynoisy -c mixturemodels/oai5g/train_conf_lw_verynoisy.json -e 5 -t SDR

无噪声
python -m mixturemodels train -d oai5g/prepped_ulmcs_norm_no1 -l oai5g/trained_ulmcs_l -c mixturemodels/oai5g/train_conf_l.json -e 5 -t SDR

原数据评估单个画图
python -m mixturemodels evaluate_pred -d oai5g/prepped_ulmcs_norm_no1 -t send -x 0,1,2 -m oai5g/trained_ulmcs_l_noisy.gmm,oai5g/trained_ulmcs_l_noisy.gmevm -l oai5g/evaluate_pred_ulmcs_l_noisy -y 0,400,50
python -m mixturemodels plot_evaluation -p oai5g/evaluate_pred_ulmcs_l_noisy -m oai5g/trained_ulmcs_l_noisy.gmevm,oai5g/trained_ulmcs_l_noisy.gmm -x 0,1,2 -y 0,50 -t tail -u 1e-6,1 --single

单个画图
python -m mixturemodels validate_pred -d oai5g/prepped_ulmcs_norm_no1 -t send -x 0,1,2 -m oai5g/trained_ulmcs_l.gmm.0,oai5g/trained_ulmcs_l.gmevm.0,oai5g/trained_ulmcs_l.gmevmw2wu.0 -l oai5g/validate_mean_doubleuw2wnonoisy --plot-tail --log -r 1 -c 3 -y 0,400,80
画图平均
python -m mixturemodels validate_meanpred_SDR -d oai5g/prepped_ulmcs_norm_no1 -t send -x 0,1,2 -m oai5g/trained_ulmcs_l_noisy.gmm.0,oai5g/trained_ulmcs_l_noisy.gmevm.0,oai5g/trained_ulmcs_l_noisy.gmevmw2wu.0 -l oai5g/validate_mean_100w --plot-tail --log -r 1 -c 3 -y 0,400,80 -q SDR
python -m mixturemodels validate_meanpred_SDR -d oai5g/prepped_ulmcs_norm_no1 -t send -x 1,2 -m oai5g/trained_ulmcs_l_noisy.gmm.0,oai5g/trained_ulmcs_l_noisy.gmevm.0,oai5g/trained_ulmcs_l_noisy.gmevmw2wu.0 -l oai5g/validate_mean_100w --plot-tail --log -r 1 -c 2 -y 0,400,80 -q SDR
3ms画图
python -m mixturemodels validate_meanpred_SDR -d oai5g/prepped_ulmcs_norm_no1 -t send -x 0,1,2 -m oai5g/trained_ulmcs_l_verynoisy.gmm.0,oai5g/trained_ulmcs_l_verynoisy.gmevm.0,oai5g/trained_ulmcs_l_verynoisy.gmevmw2wu.0 -l oai5g/validate_mean_100w --plot-tail --log -r 1 -c 3 -y 0,400,80 -q SDR
python -m mixturemodels validate_meanpred_SDR -d oai5g/prepped_ulmcs_norm_no1 -t send -x 1,2 -m oai5g/trained_ulmcs_l_verynoisy.gmm.0,oai5g/trained_ulmcs_l_verynoisy.gmevm.0,oai5g/trained_ulmcs_l_verynoisy.gmevmw2wu.0 -l oai5g/validate_mean_100w --plot-tail --log -r 1 -c 2 -y 0,400,80 -q SDR
非log画图pdf
python -m mixturemodels validate_meanpred_SDR -d oai5g/prepped_ulmcs_norm_no1 -t send -x 0,1,2 -m oai5g/trained_ulmcs_l_noisy.gmm.0,oai5g/trained_ulmcs_l_noisy.gmevm.0,oai5g/trained_ulmcs_l_noisy.gmevmw2wu.0 -l oai5g/validate_mean_100w-no-log --plot-pdf -r 1 -c 3 -y 0,400,80 -q SDR
无噪声平均画图
python -m mixturemodels validate_meanpred_SDR -d oai5g/prepped_ulmcs_norm_no1 -t send -x 0,1,2 -m oai5g/trained_ulmcs_l.gmm.0,oai5g/trained_ulmcs_l.gmevm.0,oai5g/trained_ulmcs_l.gmevmw2wu.0 -l oai5g/validate_mean_100withnonoisy --plot-tail --log -r 1 -c 3 -y 0,400,80 -q SDR
python -m mixturemodels validate_meanpred_SDR -d oai5g/prepped_ulmcs_norm_no1 -t send -x 1,2 -m oai5g/trained_ulmcs_l.gmm.0,oai5g/trained_ulmcs_l.gmevm.0,oai5g/trained_ulmcs_l.gmevmw2wu.0 -l oai5g/validate_mean_100withnonoisy --plot-tail --log -r 1 -c 2 -y 0,400,80 -q SDR
测量1ms+3ms噪声一起画
python -m mixturemodels validate_meanpred_SDR -d oai5g/prepped_ulmcs_norm_no1 -t send -x 0,1,2 -m oai5g/trained_ulmcs_l.gmm.0,oai5g/trained_ulmcs_l.gmevm.0,oai5g/trained_ulmcs_l.gmevmw2wu.0 -l oai5g/validate_meas_noisywithnoisy --plot-tail --log -r 1 -c 3 -y 0,400,80 -q SDR

查看参数
python -m mixturemodels model_params_view -m oai5g/trained_ulmcs_l.gmevmw2wu.2 -l 0.333
选阈值
python -m mixturemodels phase_one -d oai5g/prepped_ulmcs_norm_no1 -t send -x 0,1,2 -l oai5g/validate_length_l --plot-tail --log -r 2 -c 2 -y 0,160,80
次主峰选u1+u2（pdf图）
python -m mixturemodels phase_one95 -d oai5g/prepped_ulmcs_norm_no1 -t send -x 0,1,2 -l oai5g/validate_find_peaks --plot-pdf --log -r 1 -c 3 -y 0,80,20 -q SDR
python -m mixturemodels phase_one95 -d oai5g/prepped_ulmcs_norm_no1 -t send -x 1,2 -l oai5g/validate_find_peaks --plot-pdf --log -r 1 -c 3 -y 0,80,20 -q SDR





python -m mixturemodels train -d oai5g/prepped_ulmcs_norm_no1 -l oai5g/trained_ulmcs_m_noisy -c mixturemodels/oai5g/train_conf_m_noisy.json -e 9 -t SDR
python -m mixturemodels evaluate_pred -d oai5g/prepped_ulmcs_norm_no1 -t send -x 0,1,2 -m oai5g/trained_ulmcs_m_noisy.gmm,oai5g/trained_ulmcs_m_noisy.gmevm -l oai5g/evaluate_pred_ulmcs_m_noisy -y 0,400,50
python -m mixturemodels plot_evaluation -p oai5g/evaluate_pred_ulmcs_m_noisy -m oai5g/trained_ulmcs_m_noisy.gmevm,oai5g/trained_ulmcs_m_noisy.gmm -x 0,1,2 -y 0,50 -t tail -u 1e-6,1 --single

python -m mixturemodels train -d oai5g/prepped_ulmcs_norm_no1 -l oai5g/trained_ulmcs_s_noisy -c mixturemodels/oai5g/train_conf_s_noisy.json -e 9
python -m mixturemodels evaluate_pred -d oai5g/prepped_ulmcs_norm_no1 -t send -x 0,1,2 -m oai5g/trained_ulmcs_s_noisy.gmm,oai5g/trained_ulmcs_s_noisy.gmevm -l oai5g/evaluate_pred_ulmcs_s_noisy -y 0,400,50
python -m mixturemodels plot_evaluation -p oai5g/evaluate_pred_ulmcs_s_noisy -m oai5g/trained_ulmcs_s_noisy.gmevm,oai5g/trained_ulmcs_s_noisy.gmm -x 0,1,2 -y 0,50 -t tail -u 1e-6,1 --single
```
用大量样本训练无附加高斯白噪声的预测器。在培训配置中，设置y_label到send_standard. 
Train predictors without additional white Gaussian noise with the large number of samples. In training config, set `y_label` to `send_standard`.
```
python -m mixturemodels train -d oai5g/prepped_ulmcs_norm_no1 -l oai5g/trained_ulmcs_l -c mixturemodels/oai5g/train_conf_l.json -e 5
python -m mixturemodels evaluate_pred -d oai5g/prepped_ulmcs_norm_no1 -t send -x 0,1,2 -m oai5g/trained_ulmcs_l.gmm,oai5g/trained_ulmcs_l.gmevm -l oai5g/evaluate_pred_ulmcs_l -y 0,400,50
python -m mixturemodels plot_evaluation -p oai5g/evaluate_pred_ulmcs_l -m oai5g/trained_ulmcs_l.gmevm,oai5g/trained_ulmcs_l.gmm -x 0,1,2 -y 0,50 -t tail -u 1e-6,1 --single
```
用中等样本数的方差为3ms的附加高斯白噪声训练预测器。在培训配置中，设置y_label到send_verynoisy.
Train predictors with additional white Gaussian noise with variance 3ms with the medium number of samples. In training config, set `y_label` to `send_verynoisy`.
```
python -m mixturemodels train -d oai5g/prepped_ulmcs_norm_no1 -l oai5g/trained_ulmcs_m_verynoisy -c mixturemodels/oai5g/train_conf_m_verynoisy.json -e 9
python -m mixturemodels evaluate_pred -d oai5g/prepped_ulmcs_norm_no1 -t send -x 0,1,2 -m oai5g/trained_ulmcs_m_verynoisy.gmm,oai5g/trained_ulmcs_m_verynoisy.gmevm -l oai5g/evaluate_pred_ulmcs_m_verynoisy -y 0,400,50
python -m mixturemodels plot_evaluation -p oai5g/evaluate_pred_ulmcs_m_verynoisy -m oai5g/trained_ulmcs_m_verynoisy.gmevm,oai5g/trained_ulmcs_m_verynoisy.gmm -x 0,1,2 -y 0,50 -t tail -u 1e-6,1 --single
```

没有MCS=5时训练预测器。
Train predictors without MCS=5.
```
python -m mixturemodels train -d oai5g/prepped_ulmcs_norm_no1a5 -l oai5g/trained_ulmcs_l_no5 -c mixturemodels/oai5g/train_conf_l_noisy_no5.json -e 7
python -m mixturemodels evaluate_pred -d oai5g/prepped_ulmcs_norm_no1 -t send -x 0,1,2 -m oai5g/trained_ulmcs_l_no5.gmm,oai5g/trained_ulmcs_l_no5.gmevm -l oai5g/evaluate_pred_ulmcs_l_no5 -y 0,400,50
python -m mixturemodels plot_evaluation -p oai5g/evaluate_pred_ulmcs_l_no5 -m oai5g/evaluate_pred_ulmcs_l_no5.gmevm,oai5g/evaluate_pred_ulmcs_l_no5.gmm -x 0,1,2 -y 0,50 -t tail -u 1e-6,1 --single
```
没有MCS=5的训练预测器非常高的噪声。
Train predictors without MCS=5 very noisy.
```
python -m mixturemodels train -d oai5g/prepped_ulmcs_norm_no1a5 -l oai5g/trained_ulmcs_l_verynoisy_no5 -c mixturemodels/oai5g/train_conf_l_verynoisy_no5.json -e 6
python -m mixturemodels evaluate_pred -d oai5g/prepped_ulmcs_norm_no1 -t send -x 0,1,2 -m oai5g/trained_ulmcs_l_verynoisy_no5.gmm,oai5g/trained_ulmcs_l_verynoisy_no5.gmevm -l oai5g/evaluate_pred_ulmcs_l_verynoisy_no5 -y 0,400,50
python -m mixturemodels plot_evaluation -p oai5g/evaluate_pred_ulmcs_l_verynoisy_no5 -m oai5g/trained_ulmcs_l_verynoisy_no5.gmevm,oai5g/trained_ulmcs_l_verynoisy_no5.gmm -x 0,1,2 -y 0,50 -t tail -u 1e-6,1 --single
```
