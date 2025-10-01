#!/usr/bin/env python3
"""
TTS Lab (Tk) — Offline desktop GUI for Coqui, Kokoro, Bark
with synth progress, playback progress, numeric inputs, replay, and export.

Reqs (in your venv):
  pip install numpy soundfile simpleaudio
  pip install TTS
  pip install suno-bark
  pip install kokoro          # or: pip install kokoro-onnx onnxruntime
"""

import threading
import time
import traceback
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import soundfile as sf  # for saving WAV

# Playback
try:
    import simpleaudio as sa
except Exception:
    sa = None

# -------- Optional backend imports (soft fail) --------
_HAS_COQUI = False
try:
    from TTS.api import TTS as CoquiTTS
    _HAS_COQUI = True
except Exception:
    pass

_HAS_BARK = False
try:
    from bark import SAMPLE_RATE as BARK_SR
    from bark import generate_audio as bark_generate_audio
    from bark.generation import preload_models as bark_preload_models
    _HAS_BARK = True
except Exception:
    pass

_HAS_KOKORO = False
_KOKORO_ONNX = False
try:
    import kokoro   # preferred
    _HAS_KOKORO = True
except Exception:
    try:
        import kokoro_onnx as kokoro
        _HAS_KOKORO = True
        _KOKORO_ONNX = True
    except Exception:
        pass


# ---------- Audio utils ----------
def to_float_mono(wave: np.ndarray) -> np.ndarray:
    if wave is None:
        return np.zeros(1, dtype=np.float32)
    if not isinstance(wave, np.ndarray):
        wave = np.asarray(wave)
    wave = wave.astype(np.float32, copy=False)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    mx = np.max(np.abs(wave)) if wave.size else 1.0
    if mx > 1.0:
        wave = wave / (mx + 1e-9)
    return wave

def to_pcm16(wave: np.ndarray) -> np.ndarray:
    wave = to_float_mono(wave)
    mx = max(np.max(np.abs(wave)), 1e-9)
    return (wave / mx * 0.95 * 32767.0).astype(np.int16)

def play_pcm16_blocking(wave_int16: np.ndarray, sample_rate: int, stop_event: threading.Event):
    if sa is None:
        return
    audio = wave_int16.tobytes()
    play_obj = sa.play_buffer(audio, 1, 2, sample_rate)
    while play_obj.is_playing():
        if stop_event.is_set():
            play_obj.stop()
            break
        stop_event.wait(0.05)


# ---------- Coqui helpers ----------
_coqui_cache = {}

def _coqui_load(model: str):
    """Create a Coqui TTS instance regardless of arg name differences."""
    try:
        return CoquiTTS(model_name=model, progress_bar=False, gpu=False)
    except TypeError:
        return CoquiTTS(model_id=model, progress_bar=False, gpu=False)

def coqui_list_voices(model_id: str):
    if not _HAS_COQUI:
        return []
    try:
        tts = _coqui_cache.get(model_id)
        if tts is None:
            tts = _coqui_load(model_id)
            _coqui_cache[model_id] = tts
        spk = getattr(tts, "speakers", None) or []
        if isinstance(spk, dict):
            spk = list(spk.keys())
        return list(spk)
    except Exception:
        return ["p225", "p226", "p227", "p228", "p229", "p230", "p231"]

def coqui_tts(text, model_id, speaker, speed, temperature):
    tts = _coqui_cache.get(model_id)
    if tts is None:
        tts = _coqui_load(model_id)
        _coqui_cache[model_id] = tts

    speakers = getattr(tts, "speakers", None) or []
    if isinstance(speakers, dict):
        speakers = list(speakers.keys())
    speakers = list(speakers)

    if speakers:  # multi-speaker
        if not speaker or speaker not in speakers:
            speaker = speakers[0]  # safe default
    else:
        speaker = None

    wav = tts.tts(
        text=text,
        speaker=speaker,
        speed=float(speed),
        temperature=float(temperature),
    )
    sr = int(getattr(tts, "output_sample_rate", 22050))
    return to_float_mono(np.asarray(wav)), sr


# ---------- Bark helpers ----------
_bark_preloaded = False
_bark_lock = threading.Lock()

def bark_list_presets():
    return [
        "v2/en_speaker_1","v2/en_speaker_2","v2/en_speaker_3",
        "v2/en_speaker_4","v2/en_speaker_5","v2/en_speaker_6",
    ]

def bark_tts(text, preset, temperature):
    global _bark_preloaded
    with _bark_lock:
        if not _bark_preloaded:
            bark_preload_models()
            _bark_preloaded = True
    presets = set(bark_list_presets())
    if preset not in presets:
        preset = "v2/en_speaker_6"
    wav = bark_generate_audio(
        text,
        history_prompt=preset,
        text_temp=float(temperature),
        waveform_temp=float(temperature),
    )
    return to_float_mono(wav), int(BARK_SR)


# ---------- Kokoro helpers ----------
def kokoro_list_voices():
    return ["af_sky","af_alloy","am_adam","bf_emma","bm_george"]

def kokoro_tts(text, speaker, speed, sample_rate):
    if speaker not in set(kokoro_list_voices()):
        speaker = "af_sky"
    wav = kokoro.tts(text, speaker=speaker, speed=float(speed), sr=int(sample_rate))
    return to_float_mono(np.asarray(wav)), int(sample_rate)


# ---------- GUI ----------
class TTSLabTk:
    def __init__(self, root):
        self.root = root
        root.title("TTS Lab — Coqui / Kokoro / Bark")
        root.geometry("820x640")

        # State
        self.stop_event = threading.Event()
        self.synth_thread = None
        self.play_thread = None
        self.last_wav = None
        self.last_sr = None
        self.play_start_ts = 0.0
        self.play_duration = 0.0
        self._playing_flag = False

        # Available backends
        self.backends = []
        if _HAS_COQUI:  self.backends.append("Coqui")
        if _HAS_KOKORO: self.backends.append("Kokoro")
        if _HAS_BARK:   self.backends.append("Bark")
        if not self.backends:
            messagebox.showerror("Error", "No TTS backends available. Install TTS, kokoro, or suno-bark.")
            root.destroy()
            return

        # Top row: backend
        frm_top = ttk.Frame(root)
        frm_top.pack(fill="x", padx=10, pady=8)

        ttk.Label(frm_top, text="Backend:").pack(side="left")
        self.backend_var = tk.StringVar(value=self.backends[0])
        self.backend_dd = ttk.Combobox(frm_top, textvariable=self.backend_var, values=self.backends, width=12, state="readonly")
        self.backend_dd.pack(side="left", padx=6)
        self.backend_dd.bind("<<ComboboxSelected>>", self.on_backend_changed)

        # --- Coqui settings ---
        self.coqui_frame = ttk.LabelFrame(root, text="Coqui Settings")
        self.coqui_model_var = tk.StringVar(value="tts_models/en/vctk/vits")
        self.coqui_speaker_var = tk.StringVar(value="p225")
        self.coqui_speed_var = tk.DoubleVar(value=1.0)
        self.coqui_temp_var = tk.DoubleVar(value=0.8)

        ttk.Label(self.coqui_frame, text="model_id:").grid(row=0, column=0, sticky="e", padx=5, pady=4)
        self.coqui_model_entry = ttk.Entry(self.coqui_frame, textvariable=self.coqui_model_var, width=36)
        self.coqui_model_entry.grid(row=0, column=1, sticky="w")
        self.btn_refresh_coqui = ttk.Button(self.coqui_frame, text="Refresh Voices", command=self.refresh_coqui_voices)
        self.btn_refresh_coqui.grid(row=0, column=2, padx=5)

        ttk.Label(self.coqui_frame, text="Speaker:").grid(row=1, column=0, sticky="e", padx=5, pady=4)
        self.coqui_speaker_dd = ttk.Combobox(self.coqui_frame, textvariable=self.coqui_speaker_var, values=[], width=18, state="readonly")
        self.coqui_speaker_dd.grid(row=1, column=1, sticky="w")

        # speed slider + entry
        ttk.Label(self.coqui_frame, text="Speed:").grid(row=2, column=0, sticky="e", padx=5, pady=4)
        row2 = ttk.Frame(self.coqui_frame); row2.grid(row=2, column=1, sticky="we")
        self.coqui_speed_scale = ttk.Scale(row2, from_=0.5, to=1.5, orient="horizontal", variable=self.coqui_speed_var)
        self.coqui_speed_scale.pack(side="left", fill="x", expand=True)
        self.coqui_speed_entry = ttk.Entry(row2, width=6)
        self.coqui_speed_entry.insert(0, "1.0")
        self.coqui_speed_entry.pack(side="left", padx=6)

        ttk.Label(self.coqui_frame, text="Temperature:").grid(row=3, column=0, sticky="e", padx=5, pady=4)
        row3 = ttk.Frame(self.coqui_frame); row3.grid(row=3, column=1, sticky="we")
        self.coqui_temp_scale = ttk.Scale(row3, from_=0.1, to=1.5, orient="horizontal", variable=self.coqui_temp_var)
        self.coqui_temp_scale.pack(side="left", fill="x", expand=True)
        self.coqui_temp_entry = ttk.Entry(row3, width=6)
        self.coqui_temp_entry.insert(0, "0.8")
        self.coqui_temp_entry.pack(side="left", padx=6)

        self.coqui_frame.columnconfigure(1, weight=1)

        # --- Bark settings ---
        self.bark_frame = ttk.LabelFrame(root, text="Bark Settings")
        self.bark_preset_var = tk.StringVar(value="v2/en_speaker_6")
        self.bark_temp_var = tk.DoubleVar(value=0.7)

        ttk.Label(self.bark_frame, text="Preset:").grid(row=0, column=0, sticky="e", padx=5, pady=4)
        self.bark_preset_dd = ttk.Combobox(self.bark_frame, textvariable=self.bark_preset_var, values=bark_list_presets() if _HAS_BARK else [], width=22, state="readonly")
        self.bark_preset_dd.grid(row=0, column=1, sticky="w")

        ttk.Label(self.bark_frame, text="Temperature:").grid(row=1, column=0, sticky="e", padx=5, pady=4)
        rowb = ttk.Frame(self.bark_frame); rowb.grid(row=1, column=1, sticky="we")
        self.bark_temp_scale = ttk.Scale(rowb, from_=0.1, to=1.5, orient="horizontal", variable=self.bark_temp_var)
        self.bark_temp_scale.pack(side="left", fill="x", expand=True)
        self.bark_temp_entry = ttk.Entry(rowb, width=6)
        self.bark_temp_entry.insert(0, "0.7")
        self.bark_temp_entry.pack(side="left", padx=6)

        self.bark_frame.columnconfigure(1, weight=1)

        # --- Kokoro settings ---
        self.kokoro_frame = ttk.LabelFrame(root, text="Kokoro Settings")
        self.kokoro_speaker_var = tk.StringVar(value="af_sky")
        self.kokoro_speed_var   = tk.DoubleVar(value=1.0)
        self.kokoro_sr_var      = tk.StringVar(value="24000")  # editable

        ttk.Label(self.kokoro_frame, text="Speaker:").grid(row=0, column=0, sticky="e", padx=5, pady=4)
        self.kokoro_speaker_dd = ttk.Combobox(self.kokoro_frame, textvariable=self.kokoro_speaker_var,
                                              values=kokoro_list_voices() if _HAS_KOKORO else [], width=18, state="readonly")
        self.kokoro_speaker_dd.grid(row=0, column=1, sticky="w")

        ttk.Label(self.kokoro_frame, text="Speed:").grid(row=1, column=0, sticky="e", padx=5, pady=4)
        rowk1 = ttk.Frame(self.kokoro_frame); rowk1.grid(row=1, column=1, sticky="we")
        self.kokoro_speed_scale = ttk.Scale(rowk1, from_=0.5, to=1.5, orient="horizontal", variable=self.kokoro_speed_var)
        self.kokoro_speed_scale.pack(side="left", fill="x", expand=True)
        self.kokoro_speed_entry = ttk.Entry(rowk1, width=6)
        self.kokoro_speed_entry.insert(0, "1.0")
        self.kokoro_speed_entry.pack(side="left", padx=6)

        ttk.Label(self.kokoro_frame, text="Sample rate:").grid(row=2, column=0, sticky="e", padx=5, pady=4)
        rowk2 = ttk.Frame(self.kokoro_frame); rowk2.grid(row=2, column=1, sticky="we")
        self.kokoro_sr_dd = ttk.Combobox(rowk2, textvariable=self.kokoro_sr_var, values=["16000", "22050", "24000", "44100"], width=10, state="normal")  # editable
        self.kokoro_sr_dd.pack(side="left")
        self.kokoro_frame.columnconfigure(1, weight=1)

        # --- Text input ---
        self.text_frame = ttk.LabelFrame(root, text="Input Text")
        self.text_frame.pack(fill="both", expand=True, padx=10, pady=8)
        self.txt = tk.Text(self.text_frame, height=7, wrap="word")
        self.txt.pack(fill="both", expand=True, padx=6, pady=6)

        # --- Action buttons ---
        frm_btns = ttk.Frame(root)
        frm_btns.pack(fill="x", padx=10, pady=4)
        self.btn_synth = ttk.Button(frm_btns, text="🎙️  Synthesize & Play", command=self.on_synth)
        self.btn_replay = ttk.Button(frm_btns, text="🔁  Replay", command=self.on_replay, state="disabled")
        self.btn_stop  = ttk.Button(frm_btns, text="⏹  Stop", command=self.on_stop)
        self.btn_save  = ttk.Button(frm_btns, text="💾  Save…", command=self.on_save, state="disabled")
        self.btn_synth.pack(side="left", padx=4)
        self.btn_replay.pack(side="left", padx=4)
        self.btn_stop.pack(side="left", padx=4)
        self.btn_save.pack(side="left", padx=4)

        # --- Progress bars + status ---
        frm_prog = ttk.Frame(root)
        frm_prog.pack(fill="x", padx=10, pady=(2, 0))

        ttk.Label(frm_prog, text="Synthesis:").pack(side="left")
        self.pb_synth = ttk.Progressbar(frm_prog, mode="indeterminate", length=180)
        self.pb_synth.pack(side="left", padx=6)

        ttk.Label(frm_prog, text="Playback:").pack(side="left", padx=(16, 0))
        self.pb_play = ttk.Progressbar(frm_prog, mode="determinate", length=280, maximum=1000)
        self.pb_play.pack(side="left", padx=6)

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(root, textvariable=self.status_var, anchor="w").pack(fill="x", padx=10, pady=6)

        # Show current backend pane
        self.on_backend_changed()

        # Preload Coqui voice list initially (if available)
        if _HAS_COQUI:
            self.refresh_coqui_voices()

        # Link numeric entries <-> sliders (two-way)
        self._wire_numeric_entries()

    # ---- UI helpers ----
    def _wire_numeric_entries(self):
        # keep slider -> entry in sync
        def bind_two_way(scale, entry, var, fmt="{:.2f}", clamp=None):
            def scale_changed(_=None):
                entry.delete(0, "end")
                entry.insert(0, fmt.format(var.get()))
            scale.configure(command=lambda v: scale_changed())

            def entry_changed(event=None):
                try:
                    val = float(entry.get().strip())
                    if clamp:
                        lo, hi = clamp
                        val = max(lo, min(hi, val))
                    var.set(val)
                    scale_changed()
                except Exception:
                    pass  # ignore bad keystrokes
            entry.bind("<FocusOut>", entry_changed)
            entry.bind("<Return>", entry_changed)
            # initialize
            scale_changed()

        bind_two_way(self.coqui_speed_scale, self.coqui_speed_entry, self.coqui_speed_var, fmt="{:.2f}", clamp=(0.5, 1.5))
        bind_two_way(self.coqui_temp_scale,  self.coqui_temp_entry,  self.coqui_temp_var,  fmt="{:.2f}", clamp=(0.1, 1.5))
        bind_two_way(self.bark_temp_scale,   self.bark_temp_entry,   self.bark_temp_var,   fmt="{:.2f}", clamp=(0.1, 1.5))
        bind_two_way(self.kokoro_speed_scale,self.kokoro_speed_entry,self.kokoro_speed_var,fmt="{:.2f}", clamp=(0.5, 1.5))

    def show_frame(self, frame, show: bool):
        if show:
            frame.pack(fill="x", padx=10, pady=6)
        else:
            frame.pack_forget()

    def on_backend_changed(self, *_):
        b = self.backend_var.get()
        self.show_frame(self.coqui_frame, b == "Coqui")
        self.show_frame(self.kokoro_frame, b == "Kokoro")
        self.show_frame(self.bark_frame,   b == "Bark")

    def refresh_coqui_voices(self):
        model_id = self.coqui_model_var.get().strip()
        if not model_id:
            return
        self.status_var.set("Loading Coqui voices… (first time may download weights)")
        self.root.update_idletasks()
        try:
            voices = coqui_list_voices(model_id)
            self.coqui_speaker_dd["values"] = voices
            current = self.coqui_speaker_var.get().strip()
            if not voices:
                self.coqui_speaker_var.set("")  # single-speaker models
            else:
                if current not in voices:
                    self.coqui_speaker_var.set(voices[0])
            self.status_var.set(f"Coqui voices: {len(voices)} found.")
        except Exception as e:
            self.status_var.set(f"Coqui voice list error: {e}")

    # ---- Actions ----
    def on_stop(self):
        self.stop_event.set()
        self._playing_flag = False
        self.pb_play["value"] = 0
        self.status_var.set("Stopping…")

    def on_replay(self):
        if self.last_wav is None or self.last_sr is None:
            return
        self._play_array(self.last_wav, self.last_sr, meta="Replay")

    def on_save(self):
        if self.last_wav is None or self.last_sr is None:
            messagebox.showinfo("Nothing to save", "Synthesize something first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV audio", "*.wav")],
            title="Save audio as…",
        )
        if not path:
            return
        try:
            sf.write(path, self.last_wav, self.last_sr)
            self.status_var.set(f"Saved: {path}")
        except Exception as e:
            messagebox.showerror("Save error", f"{type(e).__name__}: {e}")

    def on_synth(self):
        text = self.txt.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("Info", "Type some text first.")
            return
        if self.synth_thread and self.synth_thread.is_alive():
            messagebox.showinfo("Busy", "Synthesis already running. Please wait or click Stop to cancel playback.")
            return

        # Read numeric fields (entries may be more up to date than vars)
        def safe_float(widget, default):
            try:
                return float(widget.get().strip())
            except Exception:
                return default

        self.coqui_speed_var.set(safe_float(self.coqui_speed_entry, self.coqui_speed_var.get()))
        self.coqui_temp_var.set(safe_float(self.coqui_temp_entry,  self.coqui_temp_var.get()))
        self.bark_temp_var.set( safe_float(self.bark_temp_entry,   self.bark_temp_var.get()))
        self.kokoro_speed_var.set(safe_float(self.kokoro_speed_entry, self.kokoro_speed_var.get()))
        try:
            int(self.kokoro_sr_var.get().strip())
        except Exception:
            self.kokoro_sr_var.set("24000")

        # UI feedback
        self.btn_synth.configure(state="disabled")
        self.btn_replay.configure(state="disabled")
        self.btn_save.configure(state="disabled")
        self.pb_synth.start(10)   # spinner
        self.status_var.set("Synthesizing…")

        self.stop_event.clear()
        self.synth_thread = threading.Thread(target=self._synth_worker, args=(text,), daemon=True)
        self.synth_thread.start()

    # ---- Background work ----
    def _synth_worker(self, text: str):
        backend = self.backend_var.get()
        try:
            if backend == "Coqui":
                if not _HAS_COQUI:
                    raise RuntimeError("Coqui not installed.")
                model_id = self.coqui_model_var.get().strip()
                speaker  = self.coqui_speaker_var.get().strip()
                speed    = float(self.coqui_speed_var.get())
                temp     = float(self.coqui_temp_var.get())
                wav, sr  = coqui_tts(text, model_id, speaker, speed, temp)
                meta = f"Coqui | {model_id} | {speaker or '(single)'} | speed={speed:.2f} temp={temp:.2f}"

            elif backend == "Kokoro":
                if not _HAS_KOKORO:
                    raise RuntimeError("Kokoro not installed.")
                speaker = self.kokoro_speaker_var.get().strip()
                speed   = float(self.kokoro_speed_var.get())
                sr      = int(self.kokoro_sr_var.get().strip())
                wav, sr = kokoro_tts(text, speaker, speed, sr)
                meta = f"Kokoro | {speaker} | speed={speed:.2f} sr={sr}"

            elif backend == "Bark":
                if not _HAS_BARK:
                    raise RuntimeError("Bark not installed.")
                preset = self.bark_preset_var.get().strip()
                temp   = float(self.bark_temp_var.get())
                wav, sr = bark_tts(text, preset, temp)
                meta = f"Bark | {preset} | temp={temp:.2f}"

            else:
                raise RuntimeError(f"Unknown backend: {backend}")

            self.last_wav, self.last_sr = wav, sr
            self.root.after(0, self._on_synth_done_ui, meta)
            # Auto-play
            self._play_array(wav, sr, meta=meta)

        except Exception as e:
            tb = traceback.format_exc()
            print("\n--- TTSLab ERROR (full traceback) ---\n", tb, "\n---------------------------\n")
            self.root.after(0, self._on_synth_fail_ui, f"{type(e).__name__}: {e}")

    def _on_synth_done_ui(self, meta: str):
        self.pb_synth.stop()
        self.status_var.set(f"Generated audio: {meta}. Playing…")
        self.btn_synth.configure(state="normal")
        self.btn_replay.configure(state="normal")
        self.btn_save.configure(state="normal")

    def _on_synth_fail_ui(self, msg: str):
        self.pb_synth.stop()
        self.btn_synth.configure(state="normal")
        self.status_var.set(f"Error: {msg}")

    # ---- Playback (with progress) ----
    def _play_array(self, wav: np.ndarray, sr: int, meta: str = ""):
        if sa is None:
            messagebox.showerror("Playback error", "simpleaudio not installed.")
            return
        # Stop any current playback
        self.on_stop()
        self._playing_flag = True
        self.play_start_ts = time.time()
        self.play_duration = float(len(wav)) / float(sr) if len(wav) else 0.0
        self.pb_play["value"] = 0

        wav_i16 = to_pcm16(wav)
        self.stop_event.clear()
        self.status_var.set(f"Playing… {meta}")
        self.play_thread = threading.Thread(
            target=play_pcm16_blocking, args=(wav_i16, sr, self.stop_event), daemon=True
        )
        self.play_thread.start()
        self._tick_play_progress()

    def _tick_play_progress(self):
        if not self._playing_flag:
            self.pb_play["value"] = 0
            return
        if self.stop_event.is_set():
            self._playing_flag = False
            self.pb_play["value"] = 0
            self.status_var.set("Stopped.")
            return

        elapsed = max(0.0, time.time() - self.play_start_ts)
        total = max(0.001, self.play_duration)
        frac = min(1.0, elapsed / total)
        self.pb_play["value"] = int(frac * 1000)

        # If thread finished naturally
        if self.play_thread and not self.play_thread.is_alive():
            self._playing_flag = False
            self.pb_play["value"] = 1000
            self.status_var.set("Done.")
            return

        # Schedule next update
        self.root.after(100, self._tick_play_progress)


def main():
    root = tk.Tk()
    TTSLabTk(root)
    root.mainloop()

if __name__ == "__main__":
    main()

