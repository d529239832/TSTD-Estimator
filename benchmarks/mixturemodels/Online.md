# Online
# STAGES = ["UPLOAD", "PROCESS", "PREP", "THRESHOLDS", "TRAIN"， ”ESTIMATE“]

服务器环境启动
```
cd /home/dyj/wireless-pr3d/benchmarks/
python -m virtualenv --python=python3.9 ./venv
source venv/bin/activate
cd /home/dyj/wireless-pr3d/wlab/
python Onlinepred.py
```

在服务器上启动web访问最新预测结果
```
cd/home/dyj/wireless-pr3d/wlab/
uvicorn Online_dashboard:app --host 0.0.0.0 --port 8080
# 本地浏览器打开
http://<服务器IP>:8080
http://10.60.81.180:8080
```

树莓派准备chronyc操作
```
# 查看同步源
chronyc sources
# 重启 chronyc
sudo systemctl restart chrony

# 之前没有正常推出端口已占用
#kill irtt
sudo lsof -i :2112
sudo kill #ID

# 多端口启动irtt监听
irtt server -b :2112 & irtt server -b :2113 & irtt server -b :2114 


```
oenwrt操作观察采集执行情况
```
观察采集执行情况
ps | grep Online_measurement.py
kill -TERM #ID
```

——————————————————手动输入代码测试——————————————————

数据集预处理  "PREP"
```
训练集+评估集
python -m mixturemodels Online_prep_dataset -d wifiOL/measurement_length -x '{"length":[172,3440,6880,10320]}' -l wifiOL/prepped_length -n length 

```

选阈值  "THRESHOLDS"
```
训练集
python -m mixturemodels Online_phase_one -d wifiOL/prepped_length -t receive -x 0,1,2,3 -l wifiOL/thresholds_length_l -r 2 -c 2 -y 0,400,200 
评估集（含泛化）
python -m mixturemodels Online_phase_one -d wifiOL/validate_length -t receive -x 0,1,2,3 -l wifiOL/thresholds2_length_l -r 2 -c 2 -y 0,400,200 

```

训练  "TRAIN"
```
python -m mixturemodels Online_train -d wifiOL/prepped_length -l wifiOL/trained_length_dl_l -c mixturemodels/wifiOL/train_conf_dl_l.json -e 1 -t WIFI
#手动 输入阈值方式 测试
python -m mixturemodels Online_train -d wifiOL/prepped_length -l wifiOL/trained_length_dl_l -c mixturemodels/wifiOL/train_conf_dl_l.json -e 1 -t WIFI -u 1,5,5,5 -v 5,10,5.5,5.5 -k 0,405,0.83,1
```

估计
```
#手动 输入阈值方式 测试
#LENGTH_ALL = [172, 3440, 6880, 8250]
python -m mixturemodels Online_estimate -d wifiOL/validate_length -t receive -x 0,1,2,3 -m wifiOL/trained_length_dl_l.gmevmw2wu.0 -l wifiOL/estimate_length --plot-tail --log -r 2 -c 2 -y 0,400,200 -q WIFI -u 1,5,5,5 -v 6,8,5.5,5.5 -k 0,0.405,0.83,1
#LENGTH_ALL = [172, 3440, 6880, 10320]
python -m mixturemodels Online_estimate -d wifiOL/validate_length -t receive -x 0,1,2,3 -m wifiOL/trained_length_dl_l.gmevmw2wu.0 -l wifiOL/estimate_length --plot-tail --log -r 2 -c 2 -y 0,400,200 -q WIFI -u 1,5,5,5 -v 3.5,7,5.5,5.5 -k 0,0.322,0.622,1
#LENGTH_ALL = [172, 3440, 6880, 9620]
python -m mixturemodels Online_estimate -d wifiOL/validate_length -t receive -x 0,1,2,3 -m wifiOL/trained_length_dl_l.gmevmw2wu.0 -l wifiOL/estimate_length --plot-tail --log -r 2 -c 2 -y 0,400,200 -q WIFI -u 1,5,5,5 -v 3.5,7,5.5,5.5 -k 0,0.346,0.71,1
#LENGTH_ALL = [172, 3440, 6880, 8610]
python -m mixturemodels Online_estimate -d wifiOL/validated_length -t receive -x 0,1,2,3 -m wifiOL/trained_length_dl_l.gmevmw2wu.0 -l wifiOL/estimate_length --plot-tail --log -r 2 -c 2 -y 0,400,200 -q WIFI -u 1,5,5,5 -v 3.5,7,5.5,5.5 -k 0,0.387,0.795,1
#LENGTH_ALL = [172, 3440, 5660, 7680]
python -m mixturemodels Online_estimate -d wifiOL/validate_length -t receive -x 0,1,2,3 -m wifiOL/trained_length_dl_l.gmevmw2wu.0 -l wifiOL/estimate_length --plot-tail --log -r 2 -c 2 -y 0,400,200 -q WIFI -u 1,5,5,5 -v 3.5,7,5.5,5.5 -k 0,0.435,0.731,1

```
