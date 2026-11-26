"""
Simple app that plays a sound when a button is pressed.

The GUI button mirrors the quick-test ergonomics of ``recognise_clicks.py``.
If a serial-connected Arduino sends ``CLICK`` events, those hardware presses
also trigger the same sound instead of generating keyboard events.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import wave
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import simpleaudio as sa  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    sa = None

try:
    import winsound  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    winsound = None

try:
    import serial  # type: ignore
    import serial.tools.list_ports  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    serial = None  # type: ignore


# -------------- Config ----------------
DEFAULT_SOUND: Optional[Path] = None
# Example: DEFAULT_SOUND = Path(__file__).with_name("click.wav")
WINDOW_TITLE = "Spectral Ninja Button Sound"
BUTTON_LABEL = "Play sound"

BAUD = 9600
TRIGGER_WORDS = {"CLICK"}
IGNORE_WORDS = {"HELLO_UNITY", "HEARTBEAT", "PING", ""}
DEBOUNCE_MS = 120
RESCAN_INTERVAL_SEC = 1.0
DEFAULT_SERIAL_PORT: Optional[str] = None
# --------------------------------------


StatusCallback = Callable[[str], None]


@dataclass(frozen=True)
class WaveMetadata:
    channels: int
    sample_rate: int
    duration_s: float


class SoundPlayer:
    """Load a WAV file once and play it on demand."""

    def __init__(self, wav_path: Path):
        self.wav_path = wav_path.expanduser().resolve()
        self.metadata = self._load_metadata(self.wav_path)
        self._lock = threading.Lock()
        self._play_obj = None
        self._wave_obj = self._prepare_backend()

    @staticmethod
    def _load_metadata(path: Path) -> WaveMetadata:
        if not path.exists():
            raise FileNotFoundError(f"WAV file not found: {path}")
        if path.suffix.lower() != ".wav":
            raise ValueError(f"Unsupported file type (expected .wav): {path}")

        with wave.open(str(path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            frames = wav_file.getnframes()

        duration_s = frames / float(sample_rate) if sample_rate else 0.0
        return WaveMetadata(channels=channels, sample_rate=sample_rate, duration_s=duration_s)

    def _prepare_backend(self):
        if sa is not None:
            return sa.WaveObject.from_wave_file(str(self.wav_path))
        if winsound is not None:
            return None
        raise RuntimeError(
            "No audio backend available. Install `simpleaudio` (pip install simpleaudio) "
            "or run on Windows with winsound support."
        )

    def play(self):
        with self._lock:
            self.stop()
            if sa is not None and self._wave_obj is not None:
                self._play_obj = self._wave_obj.play()
            elif winsound is not None:
                winsound.PlaySound(str(self.wav_path), winsound.SND_FILENAME | winsound.SND_ASYNC)

    def stop(self):
        if sa is not None and self._play_obj is not None:
            self._play_obj.stop()
            self._play_obj = None
        elif winsound is not None:
            winsound.PlaySound(None, winsound.SND_PURGE)


class SerialSoundWorker(threading.Thread):
    """Background worker that plays audio when serial trigger words arrive."""

    def __init__(
        self,
        port_name: str,
        player: SoundPlayer,
        status_callback: Optional[StatusCallback],
        *,
        baud: int = BAUD,
    ):
        if serial is None:
            raise RuntimeError("pyserial is required for serial integration (pip install pyserial).")
        super().__init__(daemon=True)
        self.port_name = port_name
        self.player = player
        self.status_callback = status_callback
        self.baud = baud
        self._running = True
        self._ser: Optional[serial.Serial] = None  # type: ignore[attr-defined]
        self._last_press = datetime.min

    def run(self):
        self._notify(f"Listening on {self.port_name} for trigger words: {', '.join(sorted(TRIGGER_WORDS))}")
        while self._running:
            if self._ser is None or not self._ser.is_open:
                self._open_serial()
                if self._ser is None:
                    time.sleep(RESCAN_INTERVAL_SEC)
                    continue

            try:
                raw = self._ser.readline()
            except Exception as exc:
                self._notify(f"Serial error ({self.port_name}): {exc}")
                self._close_serial()
                time.sleep(RESCAN_INTERVAL_SEC)
                continue

            if not raw:
                time.sleep(0.02)
                continue

            line = raw.decode(errors="ignore").strip()
            if line in IGNORE_WORDS:
                continue

            if line in TRIGGER_WORDS:
                now = datetime.now()
                if now - self._last_press >= timedelta(milliseconds=DEBOUNCE_MS):
                    self.player.play()
                    self._notify(f"Playing {self.player.wav_path.name} (triggered by {self.port_name})")
                    self._last_press = now
            # else: ignore unexpected lines unless debugging is needed

    def stop(self):
        self._running = False
        self._close_serial()

    def _open_serial(self):
        try:
            self._ser = serial.Serial(  # type: ignore[arg-type,attr-defined]
                self.port_name,
                self.baud,
                timeout=0.2,
                write_timeout=0.2,
            )
            time.sleep(1.5)  # Nanos often reset on open; give them time
            self._notify(f"Opened {self.port_name} at {self.baud} baud.")
        except Exception as exc:
            self._notify(f"Failed to open {self.port_name}: {exc}")
            self._ser = None

    def _close_serial(self):
        try:
            if self._ser and self._ser.is_open:
                self._ser.close()
        except Exception:
            pass
        finally:
            self._ser = None

    def _notify(self, message: str):
        if self.status_callback:
            self.status_callback(message)
        else:
            print(message)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play a WAV file when the GUI button or Arduino trigger fires."
    )
    parser.add_argument(
        "wav_file",
        nargs="?",
        type=Path,
        help="Path to the WAV file. If omitted, a file dialog prompts for one (unless --no-gui).",
    )
    parser.add_argument(
        "--port",
        help="Serial port providing CLICK messages from the Arduino (auto-detects if omitted).",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=BAUD,
        help=f"Serial baud rate (default={BAUD}).",
    )
    parser.add_argument(
        "--no-serial",
        action="store_true",
        help="Disable serial listening; only the GUI button will play the sound.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run headless (no Tk window). Requires a WAV path and active serial trigger.",
    )
    return parser.parse_args(argv)


def _resolve_wav_path(arg_path: Optional[Path], *, allow_dialog: bool) -> Path:
    if arg_path is not None:
        return arg_path.expanduser()

    if DEFAULT_SOUND is not None:
        return DEFAULT_SOUND.expanduser()

    if not allow_dialog:
        raise SystemExit("No WAV file provided and dialogs are disabled (--no-gui).")

    root = tk.Tk()
    root.withdraw()
    chosen = filedialog.askopenfilename(
        title="Select WAV file",
        filetypes=(("WAV files", "*.wav"), ("All files", "*.*")),
    )
    root.destroy()

    if not chosen:
        raise SystemExit("No WAV file selected; exiting.")
    return Path(chosen).expanduser()


def _make_status_text(path: Path, meta: WaveMetadata) -> str:
    duration = f"{meta.duration_s:.2f}s" if meta.duration_s else "unknown length"
    return f"Loaded {path.name}  |  {meta.channels} ch @ {meta.sample_rate} Hz  |  {duration}"


def _make_status_setter(root: tk.Tk, status_var: tk.StringVar) -> StatusCallback:
    def set_status(message: str) -> None:
        root.after(0, lambda: status_var.set(message))

    return set_status


def build_app(player: SoundPlayer) -> tuple[tk.Tk, StatusCallback]:
    root = tk.Tk()
    root.title(WINDOW_TITLE)

    status_var = tk.StringVar(value=_make_status_text(player.wav_path, player.metadata))
    set_status = _make_status_setter(root, status_var)

    main_frame = ttk.Frame(root, padding=16)
    main_frame.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    button = ttk.Button(
        main_frame,
        text=BUTTON_LABEL,
        command=lambda: _safe_play(player, set_status),
    )
    button.grid(row=0, column=0, pady=(0, 12))
    button.focus_set()

    status_label = ttk.Label(main_frame, textvariable=status_var, wraplength=360, justify="center")
    status_label.grid(row=1, column=0)

    return root, set_status


def _safe_play(player: SoundPlayer, set_status: StatusCallback) -> None:
    try:
        player.play()
        set_status(f"Playing {player.wav_path.name} ...")
    except Exception as exc:  # pragma: no cover - GUI feedback path
        set_status(f"Playback failed: {exc}")
        messagebox.showerror("Playback error", str(exc))


def list_candidate_ports():
    if serial is None:
        raise RuntimeError("pyserial not available; install it with `pip install pyserial`.")
    ports = serial.tools.list_ports.comports()

    def score(p):
        s = " ".join(
            [
                (p.description or "").lower(),
                (p.manufacturer or "").lower(),
                (p.hwid or "").lower(),
            ]
        )
        hits = 0
        for key in [
            "arduino",
            "wch",
            "ch340",
            "ftdi",
            "usb serial",
            "usb-serial",
            "usbmodem",
            "usbserial",
            "1a86",
            "2341",
            "2a03",
            "0403",
        ]:
            if key in s:
                hits += 1
        return hits

    return sorted(ports, key=score, reverse=True)


def resolve_serial_port(requested: Optional[str]) -> Optional[str]:
    if requested:
        return requested
    if DEFAULT_SERIAL_PORT:
        return DEFAULT_SERIAL_PORT
    try:
        candidates = list_candidate_ports()
    except RuntimeError as exc:
        print(f"[serial] {exc}")
        return None

    if not candidates:
        print("[serial] No serial ports detected. Plug the Arduino in and restart.")
        return None

    if len(candidates) == 1:
        port = candidates[0].device
        print(f"[serial] Auto-selected {port}")
        return port

    print("\nDetected serial ports:")
    for idx, port in enumerate(candidates):
        print(f"[{idx}] {port.device:>12}  {port.description}  (mfg={port.manufacturer}, hwid={port.hwid})")

    while True:
        try:
            choice = input("Select index for the Arduino button: ").strip()
            idx = int(choice)
            if 0 <= idx < len(candidates):
                return candidates[idx].device
        except Exception:
            pass
        print("Invalid choice. Try again.")


def start_serial_worker(
    port: Optional[str],
    player: SoundPlayer,
    set_status: Optional[StatusCallback],
    *,
    baud: int,
) -> Optional[SerialSoundWorker]:
    if port is None:
        if set_status:
            set_status("Serial disabled or not detected; GUI button still works.")
        else:
            print("Serial disabled or not detected; GUI button still works.")
        return None

    worker = SerialSoundWorker(port, player, set_status, baud=baud)
    worker.start()
    return worker


def run_headless(player: SoundPlayer, port: Optional[str], baud: int) -> int:
    def print_status(message: str) -> None:
        print(message)

    worker = start_serial_worker(port, player, print_status, baud=baud)
    if worker is None:
        print("Nothing to do: provide a serial port or enable the GUI for manual playback.")
        return 1

    print("Headless mode running. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        worker.stop()
        player.stop()
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    allow_dialog = not args.no_gui

    try:
        wav_path = _resolve_wav_path(args.wav_file, allow_dialog=allow_dialog)
        player = SoundPlayer(wav_path)
    except Exception as exc:
        if allow_dialog:
            messagebox.showerror("Setup error", str(exc))
        print(f"Failed to initialise button sound: {exc}", file=sys.stderr)
        return 1

    port: Optional[str] = None
    if not args.no_serial:
        port = resolve_serial_port(args.port)

    if args.no_gui:
        return run_headless(player, port, args.baud)

    app, set_status = build_app(player)
    serial_worker: Optional[SerialSoundWorker] = None

    if not args.no_serial and serial is None:
        set_status("pyserial not installed; install it to enable the Arduino button.")
    elif not args.no_serial:
        serial_worker = start_serial_worker(port, player, set_status, baud=args.baud)

    def on_close() -> None:
        if serial_worker:
            serial_worker.stop()
        player.stop()
        app.destroy()

    app.protocol("WM_DELETE_WINDOW", on_close)

    try:
        app.mainloop()
    finally:
        if serial_worker:
            serial_worker.stop()
        player.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
