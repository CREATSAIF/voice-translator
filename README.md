# Voice Translator - 实时语音翻译工具

基于小模型的实时语音翻译工具，使用 FunASR 进行语音识别。

## 功能特点

- 实时麦克风音频捕获
- 基于能量的语音活动检测 (VAD)
- 使用 FunASR (paraformer-zh) 进行中文语音识别
- 静音检测自动结束语音段
- WebSocket 支持实时流式处理

## 项目结构

```
voice-translator/
├── voice_translator.py      # 主程序
├── requirements.txt         # Python 依赖
├── README.md               # 本文件
└── src/                     # Rust 源码 (开发中)
```

## 依赖

- Python 3.8+
- numpy
- sounddevice 或 pyaudio
- requests

## 安装

```bash
pip install numpy sounddevice requests
```

## 运行

1. 确保 FunASR 服务运行在 `http://localhost:8765`

2. 启动语音翻译:

```bash
python voice_translator.py
```

3. 参数选项:

```bash
# 指定 FunASR API 地址
python voice_translator.py --api-url http://localhost:8765

# 调整 VAD 灵敏度
python voice_translator.py --threshold 0.03

# 调整静音结束时间
python voice_translator.py --silence-ms 800
```

## FunASR 服务安装

如果还没有 FunASR 服务，可以这样安装:

```bash
# 克隆 FunASR
cd /home/clow
git clone https://github.com/modelscope/FunASR.git

# 安装依赖
cd FunASR
pip install -e .

# 启动服务
python funasr_api/funasr_server.py
```

## 技术架构

1. **AudioCapture**: 使用 sounddevice/pyaudio 捕获麦克风输入
2. **VAD**: 基于能量的语音活动检测
3. **TranscriptionService**: 调用 FunASR API 进行语音转文字
4. **VoiceTranslator**: 协调各模块,管理语音段处理流程

## 开发路线

- [x] Python 版本基础功能
- [ ] Rust GUI 版本 (使用 eframe/egui)
- [ ] 实时翻译功能
- [ ] 多语言支持
- [ ] 语音合成输出