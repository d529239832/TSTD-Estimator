# Data-Driven Estimation of End-to-End Delay Probability Density Function for Time-Sensitive WiFi Networks

Time-sensitive applications require the End-to-End (E2E) delay of wireless networks to be deterministic. For example, control signals in industrial automation, intelligent transportation, and telemedicine must be transmitted to their destinations within the millisecond range, with delay jitter controlled within the microsecond range. To formulate effective policies for maintaining E2E delay within a small deterministic range, it is essential to estimate the probability density function (PDF) of E2E delay. Data-driven methods based on mixture density networks have been employed to estimate the PDF of E2E delay in wireless networks. However, in WiFi networks, the estimation results produced by existing methods exhibit significant discrepancies and fluctuations when compared to actual measurements. Motivated by this, an improved estimation method is proposed, where the delay PDF is divided into three segments with different functional expressions that are coupled together. Moreover, the parameter estimation process is implemented in two stages. First, the two division thresholds for the three segments of the PDF are calculated based on the variation trend of E2E delay measurements. Second, the remaining parameters are obtained through training using an improved mixture density network. Experimental results indicate that the E2E delay PDF obtained by the proposed method exhibits a smaller gap compared to actual measurements than existing methods. Specifically, the mean absolute errors and average fluctuation amplitudes of tail probabilities at certain delay values decrease by at least one order of magnitude. Moreover, the multiple-segmentation feature of the proposed method enhances its robustness in situations where measurement data are affected by low levels of Gaussian noise

对于延迟预测任务，使用项目改进的pr3d。为了重现论文结果，您需要下载数据集，并使用它们来训练或评估在pr3d中实现的潜伏期预测器。Pr3d用Python，Tensorflow，Keras。
原版作者的pr3d在此处[pr3d](https://github.com/samiemostafavi/pr3d).

## 论文
该仓库包含以下论文的模型、评估方案和数值:数据驱动的无线网络延迟概率预测: [here](https://www.mdpi.com/2079-9292/14/12/2324)).


## 引用
如果您在研究中使用这项工作的成果，请引用以下论文:
```
@Article{electronics14122324,
AUTHOR = {Cao, Jianyu and Dai, Yujun and Huang, Shuping and Zhang, Minghe},
TITLE = {Data-Driven Estimation of End-to-End Delay Probability Density Function for Time-Sensitive WiFi Networks},
JOURNAL = {Electronics},
VOLUME = {14},
YEAR = {2025},
NUMBER = {12},
ARTICLE-NUMBER = {2324},
URL = {https://www.mdpi.com/2079-9292/14/12/2324},
ISSN = {2079-9292},
DOI = {10.3390/electronics14122324}
}


```

