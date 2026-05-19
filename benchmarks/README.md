[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# 开始

使用“virtualenv”创建Python 3.9虚拟环境

```
cd benchmarks
python -m virtualenv --python=python3.9 ./venv
source venv/bin/activate
```

要使用pySpark Java开发工具包(JDK ),必须在您的机器上安装
```
sudo apt-get install openjdk-8-jdk
```

安装python依赖项
```
pip install -r requirements.txt
```

要使用SciencePlot，必须在您的计算机上安装latex
```
sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super
```

# 访问测量数据集


Samie Mostafavi 的测量结果公布在Kaggle上: [wireless-pr3d MEAS](https://www.kaggle.com/datasets/samiemostafavi/wireless-pr3d).

[TSTD MEAS]https://www.kaggle.com/datasets/daiyujun/wifi-network-delay-dataset/data

在服务器中，如果您安装Kaggle Python包并添加您的令牌，您可以下载它们
```
pip install kaggle
vim /home/wlab/.kaggle/kaggle.json
```

然后通过运行以下命令下载数据集
```
cd benchmarks
kaggle datasets download -d samiemostafavi/wireless-pr3d
unzip wireless-pr3d.zip
```

准备 COTS 5G dataset:
```
mv COTS-5G ./mixturemodels/ep5g/measurement_validation_results
```

准备 SDR 5G dataset:
```
mv SDR-5G ./mixturemodels/oai5g/measurement_results
```

准备 IEEE802.11g dataset: 
```
mkdir ./mixturemodels/wifi/measurement_loc_results
find ./IEEE80211g -type f -name 'dataset_*_172*' -exec mv -t ./mixturemodels/wifi/measurement_loc_results/ {} +
mv IEEE80211g/ ./mixturemodels/wifi/measurement_length_results
for i in {0..5}; do
  find ./mixturemodels/wifi/measurement_loc_results/ -type f -name "dataset_${i}_172*" -exec mv -t ./mixturemodels/wifi/measurement_length_results/ {} +
  find test -type f -name "dataset_${i}_172*" -exec cp -t test2 {} +
done
```



