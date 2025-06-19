# 实时语音唤醒pipeline

这是一个基于funasr的实时语音唤醒系统，使用了语音活动检测（VAD）和关键词唤醒（KWS）技术。

## 系统架构图

```text
+-------------------+     +--------------+     +----------------+
|   音频采集        | --> |  VAD检测     | --> | 唤醒词检测     |
|  (FFmpeg/Pulse)   |     |  (FSMN模型)  |     | (Character-Ctc)|
+-------------------+     +--------------+     +----------------+
```
## 实现原理

### 核心组件
1. **音频采集**：使用FFmpeg从默认音频输入设备采集16kHz单声道音频
2. **语音活动检测(VAD)**：使用fsmn-vad模型检测语音活动
3. **关键词唤醒(KWS)**：使用speech_charctc_kws_phone-xiaoyun模型检测预定义唤醒词

### 关键技术参数
- VAD处理块大小：200ms
- 唤醒模型处理步长：480ms
- 采样率：16000Hz
- 支持多组唤醒词：
  - 主唤醒词：小云小云，你好小云，你好问问，嗨小问
  - 场景命令词：音乐控制、地图导航、家电控制等

## 运行要求

### 硬件要求
- CPU或GPU支持
- 麦克风设备
- 至少4GB内存

### 软件依赖
- Python 3.7+
- FFmpeg
- PulseAudio
- ModelScope库
- FunASR库

## 安装指南

### 1. 安装系统依赖
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg pulseaudio libpulse-dev
```

## 使用说明

### 启动系统
```bash
python3 wakeup.py
```

### 停止系统
按下 `Ctrl+C` 终止程序

## 自定义配置

### 修改唤醒词
在代码顶部修改[KEYWORDS](file:///home/media/nero/modelscope/wakeup.py#L12-L52)列表，添加或删除唤醒词：
```python
KEYWORDS = [
    "新唤醒词",
    # 其他命令词...
]
```

### 调整音频参数
根据需要修改以下常量：
```python
VAD_CHUNK_SIZE_MS = 200    # VAD处理块大小
KWS_CHUNK_STRIDE_MS = 480  # 唤醒模型处理步长
SAMPLE_RATE = 16000        # 采样率
```

## 故障排查

### 音频采集失败
- 检查FFmpeg是否安装正确
- 确认PulseAudio服务正在运行
- 检查麦克风权限

## 已知问题
- 耐久测试未通过，程序跑一天会出现cpu和内存增长


