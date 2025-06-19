import time
import numpy as np
import subprocess as sp
import threading
from funasr import AutoModel


# 主唤醒词：小云小云，你好小云，你好问问，嗨小问
# 音乐场景命令词：播放音乐，增大音量，减小音量，继续播放，暂停播放，上一首，下一首，单曲循环，随机模式，列表循环
# 地图场景命令词：取消导航，退出导航，放大地图，查看全程，缩小地图，不走高速，躲避拥堵，避免收费，高速优先
# 家电场景命令词：返回桌面，睡眠模式，蓝牙模式，打开灯光，关闭灯光，打开空调，关闭空调，拍照拍照，我要拍照
# 通用场景命令词：上一页，下一页，上一个，下一个，换一批，打开录音，关闭录音
KEYWORDS = [
    "小云小云",
    "你好小云",
    "你好问问",
    "嗨小问",
    "播放音乐",
    "增大音量",
    "减小音量",
    "继续播放",
    "暂停播放",
    "上一首",
    "下一首",
    "单曲循环",
    "随机模式",
    "列表循环",
    "取消导航",
    "退出导航",
    "放大地图",
    "查看全程",
    "缩小地图",
    "不走高速",
    "躲避拥堵",
    "避免收费",
    "高速优先",
    "返回桌面",
    "睡眠模式",
    "蓝牙模式",
    "打开灯光",
    "关闭灯光",
    "打开空调",
    "关闭空调",
    "拍照拍照",
    "我要拍照",
    "上一页",
    "下一页",
    "上一个",
    "下一个",
    "换一批", 
    "打开录音",
    "关闭录音"
]
# 系统配置参数
VAD_CHUNK_SIZE_MS = 200    # VAD处理块大小
KWS_CHUNK_STRIDE_MS = 480  # 唤醒模型处理步长
SAMPLE_RATE = 16000        # 采样率
FORMAT = np.int16          # 音频格式
CHANNELS = 1               # 声道数

class RealTimeVoiceSystem:
    def __init__(self, wakeup_word="小云小云"):
        # 系统状态控制
        self.running = True
        self.is_speaking = False
        
        # 音频采集相关
        self.proc = None
        self.proc_initialized = threading.Event()
        # KEYWORDS列表转成字符串，用逗号分隔
        wakeup_word = ','.join(KEYWORDS) if isinstance(wakeup_word, list) else wakeup_word
        # 模型初始化
        self._init_models(wakeup_word)
        
        # 音频缓冲区管理
        self._init_audio_buffers()
        
        # 线程同步锁
        self.buffer_lock = threading.Lock()

    def _init_models(self, wakeup_word):
        """初始化语音处理模型"""
        try:
            # 语音活动检测模型
            self.vad_model = AutoModel(
                model="fsmn-vad",
                model_revision="v2.0.4",
                disable_pbar=True
            )
            
            # 唤醒词检测模型
            self.kws_model = AutoModel(
                model="iic/speech_charctc_kws_phone-xiaoyun",
                keywords=wakeup_word,
                output_dir="./outputs/debug",
                device='cpu',
                disable_pbar=True
            )
            
            # 模型缓存状态
            self.vad_cache = {}
            self.kws_cache = {}
            
            # 计算块参数
            self.vad_chunk_samples = int(VAD_CHUNK_SIZE_MS * SAMPLE_RATE / 1000)
            self.kws_chunk_stride = int(KWS_CHUNK_STRIDE_MS * SAMPLE_RATE / 1000)
            
        except Exception as e:
            raise RuntimeError(f"模型初始化失败: {str(e)}")

    def _init_audio_buffers(self):
        """初始化音频缓冲区"""
        self.raw_buffer = np.array([], dtype=np.float32)  # 原始音频缓冲区
        self.wakeup_queue = []                            # 唤醒模型处理队列
        self.MAX_QUEUE_SIZE = 10                          # 最大队列长度

    def start(self):
        """启动系统"""
        # 启动音频采集线程
        self.capture_thread = threading.Thread(
            target=self._capture_audio, 
            daemon=True
        )
        self.capture_thread.start()
        
        # 等待音频采集初始化
        if not self.proc_initialized.wait(timeout=5):
            print("音频采集初始化超时")
            return False
            
        # 启动处理线程
        self.process_thread = threading.Thread(
            target=self._process_audio, 
            daemon=True
        )
        self.process_thread.start()
        
        return True

    def _capture_audio(self):
        """音频采集线程"""
        try:
            FFMPEG_PATH = "ffmpeg"
            cmd = [
                FFMPEG_PATH,
                '-loglevel', '0',
                '-ar', str(SAMPLE_RATE),
                '-ac', str(CHANNELS),
                '-f', 'pulse',
                '-i', 'default',
                '-fflags', 'nobuffer',
                '-flags', 'low_delay',
                '-f', 's16le',
                '-'
            ]
            
            self.proc = sp.Popen(
                cmd,
                stdout=sp.PIPE,
                stderr=sp.DEVNULL,
                bufsize=1024*64
            )
            self.proc_initialized.set()
            
            while self.running and self.proc.poll() is None:
                time.sleep(0.1)
                
        except Exception as e:
            print(f"音频采集失败: {str(e)}")
            self.running = False
        finally:
            self.proc_initialized.set()

    def _process_audio(self):
        """主处理循环"""
        try:
            while self.running:
                # 读取原始音频数据
                pcm_data = self.proc.stdout.read(self.vad_chunk_samples * 2)
                if not pcm_data:
                    break
                
                # 转换并缓存音频数据
                audio_array = self._convert_audio(pcm_data)
                self._update_buffers(audio_array)
                
                # 执行VAD检测
                while self._buffer_ready():
                    chunk = self._get_audio_chunk()
                    self._vad_detection(chunk)
                    
                    # 如果检测到语音活动，处理唤醒模型
                    if self.is_speaking:
                        self._process_wakeup(chunk)
                        
        except Exception as e:
            print(f"处理错误: {str(e)}")
        finally:
            self._cleanup()

    def _convert_audio(self, pcm_data):
        """转换音频格式"""
        audio_array = np.frombuffer(pcm_data, dtype=FORMAT)
        return audio_array.astype(np.float32) / np.iinfo(FORMAT).max

    def _update_buffers(self, audio_array):
        """更新音频缓冲区"""
        with self.buffer_lock:
            self.raw_buffer = np.concatenate((self.raw_buffer, audio_array))

    def _buffer_ready(self):
        """检查缓冲区是否就绪"""
        return len(self.raw_buffer) >= self.vad_chunk_samples

    def _get_audio_chunk(self):
        """获取处理块"""
        with self.buffer_lock:
            chunk = self.raw_buffer[:self.vad_chunk_samples]
            self.raw_buffer = self.raw_buffer[self.vad_chunk_samples:]
            return chunk

    def _vad_detection(self, chunk):
        """执行VAD检测"""
        vad_result = self.vad_model.generate(
            input=chunk,
            cache=self.vad_cache,
            is_final=False,
            chunk_size=VAD_CHUNK_SIZE_MS
        )
        
        if vad_result and len(vad_result[0]["value"]) > 0:
            for value in vad_result[0]['value']:
                if value[1] == -1:  # 语音开始
                    self.is_speaking = True
                    print(f"检测到语音开始: {value[0]}ms")
                else:  # 语音结束
                    self.is_speaking = False
                    print(f"检测到语音结束: {value[1]}ms")

    def _process_wakeup(self, chunk):
        """处理唤醒词检测"""
        # 添加数据到队列
        self.wakeup_queue.append(chunk)
        
        # 维护队列长度
        if len(self.wakeup_queue) > self.MAX_QUEUE_SIZE:
            self.wakeup_queue.pop(0)
            
        # 执行唤醒检测
        if len(self.wakeup_queue) >= 3:  # 至少1.5秒数据
            input_data = np.concatenate(self.wakeup_queue)
            res = self.kws_model.generate(
                input=input_data,
                cache=self.kws_cache,
                chunk_size=KWS_CHUNK_STRIDE_MS
            )
            
            if res and res[0]['text'] != 'rejected':
                print(f"唤醒成功: {res[0]}")
                self.wakeup_queue.clear()

    def _cleanup(self):
        """资源清理"""
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=1)
        if hasattr(self, 'process_thread'):
            self.process_thread.join(timeout=1)

    def stop(self):
        """停止系统"""
        self.running = False
        self._cleanup()

if __name__ == "__main__":
    system = RealTimeVoiceSystem()
    
    try:
        if system.start():
            print("系统已启动，正在监听...")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在停止系统...")
    finally:
        system.stop()
        print("系统已关闭")
