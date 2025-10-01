# FOSS TTS Tool [Coqui Kokoro Bark]
Free and Open Source text to speech app for unlimited reading and audio generation offline. 

Runs on your own machine with no limits, no API keys, and no tracking.

---

## ✨ Features

- **Multiple backends**:  
  - [Coqui TTS](https://github.com/coqui-ai/TTS)  
  - [Kokoro](https://github.com/hexgrad/kokoro-onnx)  
  - [Suno Bark](https://github.com/suno-ai/bark)  

- **Offline, unlimited usage** (no quotas, no cloud).  
- **Multi-speaker voice models** (choose different speakers).  
- **Realtime feedback**:  
  - "Synthesizing…" indicator  
  - Playback progress bar  
- **Direct numeric input** for speed, temperature, and sample rate (fine control).  
- **Replay without re-synthesizing**.  
- **Export audio** to WAV.  
- **Cross-platform** (Linux, macOS, Windows).  

---

## 📦 Requirements

Python 3.9 – 3.11 recommended.  

Install dependencies into a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows

pip install --upgrade pip
pip install numpy soundfile simpleaudio
pip install TTS            # Coqui
pip install kokoro         # or: pip install kokoro-onnx onnxruntime
pip install suno-bark      # Bark
```

---

## 🚀 Usage

Run the Tkinter GUI:

    python tts_lab_tk.py

Steps:
1. Select a backend (Coqui, Kokoro, Bark).
2. Enter text in the input box.
3. Adjust speaker, speed, temperature, or sample rate.
4. Click "Synthesize & Play".

You can then:
- Replay the last output
- Stop playback
- Save audio as a WAV file

---

## ⚠️ Notes

- The first time you use Coqui or Bark, models may be downloaded automatically.
- Some models are multi-speaker: you must select a voice from the dropdown.
- Audio export is currently WAV only, but you can convert with ffmpeg if needed.

---

## 📜 License

This project is licensed under the GNU General Public License (GPL).  
Each backend model follows its own license (Apache 2.0, MIT, etc).  
Check upstream repositories for details.

---

## 🙌 Credits

- Coqui TTS: https://github.com/coqui-ai/TTS
- Kokoro:    https://github.com/hexgrad/kokoro-onnx
- Suno Bark: https://github.com/suno-ai/bark
