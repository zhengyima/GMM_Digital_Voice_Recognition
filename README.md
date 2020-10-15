# GMM_Digital_Voice_Recognition

基于GMM与MFCC特征进行数字0-9的语音识别，GMM，MFCC，语音识别，中文数据，sklearn，scikit-learn，Digital Voice Recognition。


## Preinstallation
```
 conda create -n GMM -c anaconda python=3.6 numpy pyaudio scipy #也可以使用pip
 conda activate GMM
 pip install -r requirements.txt
```

数据链接: https://pan.baidu.com/s/124TiAs8m7Ioa2_3dUrxGSg 提取码: xsfe

以下命令假设下载数据至/tmp/dataset.zip


## Launch the script
```
  git clone https://github.com/zhengyima/GMM_Digital_Voice_Recognition.git GMM_DVR
  cd GMM_DVR
  unzip /tmp/dataset.zip -d ./  # dataset.zip是从百度网盘下载的数据
  python speaker-recognition.py -t enroll -i  "./data_zh_1/*/"  -m model.out
  python speaker-recognition.py -t predict  -i  "./data_zh_test_1/*/"  -m model.out
  
```

[HMM实现](https://github.com/zhengyima/HMM_Digital_Voice_Recognition)

[DTW实现](https://github.com/zhengyima/DTW_Digital_Voice_Recognition)
