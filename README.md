| Voice Translator - 实时语音翻译工具

基于小模型的实时语音翻译工具，使用 FunASR 进行语音识别。

## 功能特点

- 实时麦克风音频捕获
- 基于能量的语音活动检测 (VAD)
- 使用 FunASR (paraformer-zh) 进行中文语音识别
- 静音检测自动结束语音段
- WebSocket 支持实时流式处理
- **Rust + eframe 桌面 GUI** (v0.2.0+): `voice-translator-gui` 二进制，开箱即用
  - Mock 后端：无需麦克风即可看演示
  - 实时后端：驱动完整的 `pipeline` 模块
  - 多语言 (zh / en / ja / ko / fr / de / es / pt / ru / it / ar / th / tr)
  - 后台 worker + mpsc channel 异步事件流，UI 不阻塞

## 项目结构

```
voice-translator/
├── voice_translator.py      # 主程序 (Python)
├── requirements.txt         # Python 依赖
├── README.md               # 本文件
├── src/                     # Rust 源码
│   ├── audio.rs            # 麦克风捕获 (cpal)
│   ├── vad.rs              # 能量 VAD
│   ├── transcription.rs    # FunASR HTTP 客户端
│   ├── translation.rs      # 翻译抽象 + CloudTranslator
│   ├── pipeline.rs         # 端到端 orchestrator
│   ├── ui.rs               # eframe GUI (Backend trait, MockBackend, RealtimeBackend)
│   └── bin/
│       └── voice-translator-gui.rs   # 桌面 GUI 入口
├── tests/                   # Python 测试
└── .github/workflows/       # CI (ci.yml) + release.yml (tag → 跨平台构建)
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

## Rust 桌面 GUI (v0.2.0+)

### 下载预编译二进制

从 [Releases 页面](https://github.com/CREATSAIF/voice-translator/releases) 下载对应平台的二进制:

- **Linux**: `voice-translator-gui-linux-x86_64`
- **Windows**: `voice-translator-gui-windows-x86_64.exe`
- **macOS**: 目前需从源码编译（cross-compile 在 Linux runner 上跑不动 cpal/egui_glow 的 macOS 框架依赖；见 #6）

```bash
# Linux 示例
chmod +x voice-translator-gui-linux-x86_64
./voice-translator-gui-linux-x86_64 --mock        # 默认：演示模式
./voice-translator-gui-linux-x86_64 --realtime    # 实时麦克风
./voice-translator-gui-linux-x86_64 --source en --target zh
./voice-translator-gui-linux-x86_64 --help        # 全部选项
```

### 从源码编译

```bash
# 需要 Rust stable + 系统音频依赖
# Linux:
sudo apt-get install -y libasound2-dev pkg-config

# macOS:
# (无需额外步骤，cargo 即可)

# Windows:
# (建议使用 MSVC build tools)
cargo build --release --bin voice-translator-gui
./target/release/voice-translator-gui --mock
```

## 切一个 release (维护者)

```bash
git tag -a v0.X.Y -m "v0.X.Y — ..."
git push origin v0.X.Y
```

`.github/workflows/release.yml` 会自动：1) 跨平台构建 (linux/windows x86_64)，2) 上传 artifacts，3) 创建带自动生成 release notes 的 GitHub Release。

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
- [x] VAD 模块（能量阈值） — `#1`
- [x] Translation 模块 + 端到端 pipeline — `98c8456`
- [x] Rust eframe 桌面 GUI — `#4`
- [x] 跨平台 release workflow — `#5`, `#6`
- [x] 多语言支持（13 种语言） — `LanguageCode` 枚举
- [ ] macOS 二进制发布（需要 `runs-on: macos-latest` 工作机）
- [ ] 语音合成输出
