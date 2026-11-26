"""
Spectral Ninja - Arduino Hardware Click Controller.

Provides a cyberpunk GUI matching the Spectral Ninja neon aesthetic:
- Single-Button Setup (1 Nano mapped to Space, Enter, or any custom key)
- Dual-Button Setup (2 Nanos mapped to individual keys, e.g. Space and Enter)
- Auto port detection with one-click rescan
- Emulates keyboard keypresses using pydirectinput for external Unity/DirectX games
- Reliable DirectX keypress hold time (30ms) to ensure game engine detection
- Live Cyber Terminal console with real-time CLICK telemetry monitoring
- Window 'Stay on Top' toggle for in-game overlay convenience
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import pydirectinput as pdi
import serial
import serial.tools.list_ports
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# ----------------- Configuration Defaults -----------------
BAUD = 9600
TRIGGER_WORDS = {"CLICK", "PRESS", "DOWN", "BUTTON_DOWN"}
IGNORE_WORDS = {"HELLO_UNITY", "HEARTBEAT", "PING", ""}

DEFAULT_DEBOUNCE_MS = 45
RESCAN_INTERVAL_SEC = 1.0
REPEAT_INTERVAL_S = 0.03
KEY_HOLD_DURATION_S = 0.025

# DirectInput scan code settings
pdi.PAUSE = 0
pdi.FAILSAFE = False

COMMON_GAME_KEYS = [
    "SPACE",
    "ENTER",
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "W",
    "A",
    "S",
    "D",
    "Z",
    "X",
    "C",
    "SHIFT",
    "ESCAPE",
    "TAB",
    "E",
    "Q",
    "F",
]

KEY_ALIASES = {
    "spacebar": "space",
    "space": "space",
    "enter": "enter",
    "return": "enter",
    "esc": "escape",
    "escape": "escape",
    "shift": "shiftleft",
    "lshift": "shiftleft",
    "rshift": "shiftright",
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
}

LogCallback = Callable[[str, str], None]
TriggerCallback = Callable[[str], None]


def get_asset_path(filename: str) -> Path:
    """Resolve asset path whether running raw or from PyInstaller bundle."""
    if hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    else:
        base = Path(__file__).resolve().parent
    return base / filename


def list_candidate_ports():
    ports = serial.tools.list_ports.comports()

    def score(p):
        s = " ".join([
            (p.description or "").lower(),
            (p.manufacturer or "").lower(),
            (p.hwid or "").lower(),
        ])
        keywords = [
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
        ]
        return sum(1 for k in keywords if k in s)

    return sorted(ports, key=score, reverse=True)


class PortWorker(threading.Thread):
    """Background worker that listens on a serial port and triggers keypresses on CLICK."""

    def __init__(
        self,
        port_name: str,
        action_name: str,
        multiplier: int,
        log_fn: Optional[LogCallback] = None,
        trigger_fn: Optional[TriggerCallback] = None,
        baud: int = BAUD,
        debounce_ms: int = DEFAULT_DEBOUNCE_MS,
    ):
        super().__init__(daemon=True, name=f"Worker-{port_name}")
        self.port_name = port_name
        self.action_name = action_name.strip()
        self.multiplier = max(1, int(multiplier))
        self.log_fn = log_fn
        self.trigger_fn = trigger_fn
        self.baud = baud
        self.debounce_ms = debounce_ms
        self.ser: Optional[serial.Serial] = None
        self.last_press = datetime.min
        self.last_interval_s = 0.40  # Default 150 BPM baseline
        self._running = True

    def _log(self, msg: str, level: str = "info"):
        if self.log_fn:
            self.log_fn(msg, level)
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [{self.port_name}] {msg}")

    def run(self):
        self._log(f"Listening on {self.port_name} for {self.action_name.upper()} (x{self.multiplier})", "cyan")
        while self._running:
            if self.ser is None or not self.ser.is_open:
                self._open_serial()
                if self.ser is None:
                    time.sleep(RESCAN_INTERVAL_SEC)
                    continue

            try:
                raw = self.ser.readline()
            except Exception as e:
                self._log(f"Serial error: {e}", "red")
                self._close_serial()
                time.sleep(RESCAN_INTERVAL_SEC)
                continue

            if not raw:
                time.sleep(0.01)
                continue

            line = raw.decode(errors="ignore").strip().upper()
            if not line or line in IGNORE_WORDS:
                continue

            if line in TRIGGER_WORDS:
                now = datetime.now()
                if now - self.last_press >= timedelta(milliseconds=self.debounce_ms):
                    self._do_action()
                    self.last_press = now

    def stop(self):
        self._running = False
        self._close_serial()

    def _open_serial(self):
        try:
            self.ser = serial.Serial(self.port_name, self.baud, timeout=0.2, write_timeout=0.2)
            time.sleep(1.2)  # USB settling time
            self._log(f"Opened {self.port_name} at {self.baud} baud.", "green")
        except Exception as e:
            self._log(f"Could not open {self.port_name}: {e}", "yellow")
            self.ser = None

    def _close_serial(self):
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass
        self.ser = None

    def _press_single(self, key: str):
        try:
            pdi.keyDown(key)
            time.sleep(KEY_HOLD_DURATION_S)
            pdi.keyUp(key)
        except Exception as exc:
            self._log(f"pydirectinput error ({key}): {exc}", "red")

    def _do_action(self):
        raw_key = self.action_name.strip().lower()
        key = KEY_ALIASES.get(raw_key, raw_key)
        now = datetime.now()

        # Track interval between consecutive physical taps
        if self.last_press != datetime.min:
            delta = (now - self.last_press).total_seconds()
            if 0.08 < delta < 2.0:
                self.last_interval_s = delta

        # Keypress 1: Immediate on physical tap
        self._press_single(key)

        # Multiplier > 1: Distribute extra keypresses evenly across tap interval
        if self.multiplier > 1:
            step = self.last_interval_s / self.multiplier
            for i in range(1, self.multiplier):
                delay = max(0.035, min(step * i, 1.0))
                threading.Timer(delay, self._press_single, args=[key]).start()

        self._log(f"CLICK on {self.port_name} → {self.action_name.upper()} x{self.multiplier}", "trigger")
        if self.trigger_fn:
            self.trigger_fn(self.action_name)


class SpectralNinjaApp:
    """Cyberpunk GUI for Spectral Ninja Arduino Click Bridge."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SPECTRAL NINJA // HARDWARE CLICK BRIDGE")
        self.root.geometry("520x730")
        self.root.minsize(480, 680)
        self.root.configure(bg="#060913")

        # Set taskbar/window icon
        ico_file = get_asset_path("spectral_ninja.ico")
        if ico_file.exists():
            try:
                self.root.iconbitmap(str(ico_file))
            except Exception:
                pass

        self._workers: list[PortWorker] = []
        self._is_active = False
        self._hover_btn = False

        # Mode: "SINGLE" or "DUAL"
        self.mode_var = tk.StringVar(value="SINGLE")
        self.topmost_var = tk.BooleanVar(value=False)

        # Port detection cache
        self.ports_list: list[serial.tools.list_ports_common.ListPortInfo] = []

        # Single mode vars
        self.single_port_var = tk.StringVar()
        self.single_key_var = tk.StringVar(value="SPACE")
        self.single_multiplier_var = tk.IntVar(value=1)

        # Dual mode vars
        self.dual_space_port_var = tk.StringVar()
        self.dual_space_key_var = tk.StringVar(value="SPACE")
        self.dual_space_mult_var = tk.IntVar(value=1)

        self.dual_enter_port_var = tk.StringVar()
        self.dual_enter_key_var = tk.StringVar(value="ENTER")
        self.dual_enter_mult_var = tk.IntVar(value=1)

        # Global settings
        self.debounce_var = tk.IntVar(value=DEFAULT_DEBOUNCE_MS)

        self._init_ui()
        self._refresh_ports()

    def _init_ui(self):
        # 1. Header Banner
        banner_path = get_asset_path("spectral_ninja_banner.png")
        if banner_path.exists():
            try:
                im = Image.open(banner_path)
                target_w = 480
                target_h = int(im.height * (target_w / im.width))
                im = im.resize((target_w, target_h), Image.Resampling.LANCZOS)
                self.banner_tk = ImageTk.PhotoImage(im)
                banner_lbl = tk.Label(self.root, image=self.banner_tk, bg="#060913")
                banner_lbl.pack(pady=(6, 4))
            except Exception:
                self._draw_fallback_banner()
        else:
            self._draw_fallback_banner()

        # 2. Segmented Mode Switcher (Cyan vs Red)
        mode_frame = tk.Frame(self.root, bg="#060913")
        mode_frame.pack(fill=tk.X, padx=20, pady=(4, 8))

        self.btn_single_mode = tk.Button(
            mode_frame,
            text="⚡ SINGLE BUTTON SETUP",
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT,
            bd=0,
            cursor="hand2",
            padx=16,
            pady=8,
            command=lambda: self._set_mode("SINGLE"),
        )
        self.btn_single_mode.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        self.btn_dual_mode = tk.Button(
            mode_frame,
            text="⚔ DUAL BUTTON SETUP",
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT,
            bd=0,
            cursor="hand2",
            padx=16,
            pady=8,
            command=lambda: self._set_mode("DUAL"),
        )
        self.btn_dual_mode.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(6, 0))

        # 3. Main Configuration Card
        self.card = tk.Frame(
            self.root,
            bg="#0c1222",
            highlightbackground="#00e5ff",
            highlightthickness=1,
            padx=16,
            pady=12,
        )
        self.card.pack(fill=tk.X, padx=20, pady=4)

        # Container for Single Button fields
        self.single_frame = tk.Frame(self.card, bg="#0c1222")
        self._build_single_controls()

        # Container for Dual Button fields
        self.dual_frame = tk.Frame(self.card, bg="#0c1222")
        self._build_dual_controls()

        # Rescan Ports Row
        scan_row = tk.Frame(self.card, bg="#0c1222")
        scan_row.pack(fill=tk.X, pady=(10, 0))

        self.port_status_lbl = tk.Label(
            scan_row,
            text="Scanning serial ports...",
            font=("Segoe UI", 8),
            fg="#94a3b8",
            bg="#0c1222",
        )
        self.port_status_lbl.pack(side=tk.LEFT)

        btn_scan = tk.Button(
            scan_row,
            text="🔄 RESCAN PORTS",
            font=("Segoe UI", 8, "bold"),
            bg="#17223b",
            fg="#38bdf8",
            activebackground="#1e293b",
            activeforeground="#00e5ff",
            relief=tk.FLAT,
            bd=0,
            padx=10,
            pady=4,
            cursor="hand2",
            command=self._refresh_ports,
        )
        btn_scan.pack(side=tk.RIGHT)

        # 4. Big Action Button (Cyber Canvas)
        self.action_canvas = tk.Canvas(
            self.root,
            height=64,
            bg="#060913",
            highlightthickness=0,
            cursor="hand2",
        )
        self.action_canvas.pack(fill=tk.X, padx=20, pady=8)

        self.action_canvas.bind("<Enter>", lambda e: self._on_btn_hover(True))
        self.action_canvas.bind("<Leave>", lambda e: self._on_btn_hover(False))
        self.action_canvas.bind("<Button-1>", lambda e: self.toggle_active())
        self.action_canvas.bind("<Configure>", lambda e: self._draw_action_button())

        # 5. Live Cyber Console / Terminal
        console_frame = tk.Frame(
            self.root,
            bg="#03050a",
            highlightbackground="#131b2e",
            highlightthickness=1,
            padx=8,
            pady=6,
        )
        console_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(2, 6))

        console_header = tk.Frame(console_frame, bg="#03050a")
        console_header.pack(fill=tk.X, pady=(0, 4))

        tk.Label(
            console_header,
            text="CYBER HUD // LIVE TELEMETRY LOG",
            font=("Consolas", 8, "bold"),
            fg="#00e5ff",
            bg="#03050a",
        ).pack(side=tk.LEFT)

        btn_clear = tk.Button(
            console_header,
            text="CLEAR",
            font=("Consolas", 7),
            fg="#64748b",
            bg="#03050a",
            relief=tk.FLAT,
            bd=0,
            cursor="hand2",
            command=self._clear_log,
        )
        btn_clear.pack(side=tk.RIGHT)

        self.log_text = tk.Text(
            console_frame,
            height=6,
            bg="#03050a",
            fg="#94a3b8",
            font=("Consolas", 8),
            relief=tk.FLAT,
            bd=0,
            wrap=tk.WORD,
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Log color tags
        self.log_text.tag_config("cyan", foreground="#38bdf8")
        self.log_text.tag_config("green", foreground="#4ade80")
        self.log_text.tag_config("yellow", foreground="#facc15")
        self.log_text.tag_config("red", foreground="#f87171")
        self.log_text.tag_config("trigger", foreground="#00e5ff", font=("Consolas", 8, "bold"))

        # 6. Bottom Status & Controls Bar
        bottom_bar = tk.Frame(self.root, bg="#060913")
        bottom_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=(0, 8))

        self.status_pill = tk.Label(
            bottom_bar,
            text="READY • Configure buttons, start listening, then focus your game window",
            font=("Segoe UI", 8),
            fg="#94a3b8",
            bg="#060913",
        )
        self.status_pill.pack(side=tk.LEFT)

        chk_top = tk.Checkbutton(
            bottom_bar,
            text="📌 Stay on top",
            variable=self.topmost_var,
            command=self._toggle_topmost,
            font=("Segoe UI", 8),
            fg="#94a3b8",
            bg="#060913",
            selectcolor="#0c1222",
            activebackground="#060913",
            activeforeground="#00e5ff",
            cursor="hand2",
        )
        chk_top.pack(side=tk.RIGHT)

        # Style TTK Comboboxes dark
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            "TCombobox",
            fieldbackground="#141a29",
            background="#1e293b",
            foreground="#f8fafc",
            darkcolor="#1e293b",
            lightcolor="#1e293b",
        )

        self._update_mode_styling()

    def _toggle_topmost(self):
        self.root.wm_attributes("-topmost", self.topmost_var.get())

    def _draw_fallback_banner(self):
        hdr = tk.Frame(self.root, bg="#060913", pady=15)
        hdr.pack(fill=tk.X)
        tk.Label(
            hdr,
            text="SPECTRAL",
            font=("Segoe UI", 26, "bold"),
            fg="#00e5ff",
            bg="#060913",
        ).pack()
        tk.Label(
            hdr,
            text="NINJA",
            font=("Segoe UI", 26, "bold"),
            fg="#ff1744",
            bg="#060913",
        ).pack()

    def _build_single_controls(self):
        # Port Row
        row1 = tk.Frame(self.single_frame, bg="#0c1222")
        row1.pack(fill=tk.X, pady=(2, 6))

        tk.Label(
            row1,
            text="BUTTON PORT:",
            font=("Segoe UI", 9, "bold"),
            fg="#00e5ff",
            bg="#0c1222",
            width=14,
            anchor="w",
        ).pack(side=tk.LEFT)

        self.combo_single_port = ttk.Combobox(
            row1,
            textvariable=self.single_port_var,
            state="readonly",
            font=("Segoe UI", 9),
        )
        self.combo_single_port.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Key Mapping Row
        row2 = tk.Frame(self.single_frame, bg="#0c1222")
        row2.pack(fill=tk.X, pady=6)

        tk.Label(
            row2,
            text="ACTION / KEY:",
            font=("Segoe UI", 9, "bold"),
            fg="#38bdf8",
            bg="#0c1222",
            width=14,
            anchor="w",
        ).pack(side=tk.LEFT)

        combo_key = ttk.Combobox(
            row2,
            textvariable=self.single_key_var,
            values=COMMON_GAME_KEYS,
            font=("Segoe UI", 9),
            width=12,
        )
        combo_key.pack(side=tk.LEFT)

        tk.Label(
            row2,
            text="MULTIPLIER:",
            font=("Segoe UI", 9, "bold"),
            fg="#94a3b8",
            bg="#0c1222",
            padx=10,
        ).pack(side=tk.LEFT)

        spin_mult = tk.Spinbox(
            row2,
            from_=1,
            to=10,
            textvariable=self.single_multiplier_var,
            width=4,
            font=("Segoe UI", 9),
            bg="#141a29",
            fg="#f8fafc",
            buttonbackground="#1e293b",
        )
        spin_mult.pack(side=tk.LEFT)

    def _build_dual_controls(self):
        # Button 1 (Cyan / Space default)
        row_btn1 = tk.Frame(self.dual_frame, bg="#0c1222")
        row_btn1.pack(fill=tk.X, pady=(2, 6))

        tk.Label(
            row_btn1,
            text="BUTTON 1 (CYAN):",
            font=("Segoe UI", 8, "bold"),
            fg="#00e5ff",
            bg="#0c1222",
            width=15,
            anchor="w",
        ).pack(side=tk.LEFT)

        self.combo_dual_space = ttk.Combobox(
            row_btn1,
            textvariable=self.dual_space_port_var,
            state="readonly",
            font=("Segoe UI", 8),
            width=20,
        )
        self.combo_dual_space.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        combo_b1_key = ttk.Combobox(
            row_btn1,
            textvariable=self.dual_space_key_var,
            values=COMMON_GAME_KEYS,
            font=("Segoe UI", 8),
            width=8,
        )
        combo_b1_key.pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(row_btn1, text="x", font=("Segoe UI", 8, "bold"), fg="#94a3b8", bg="#0c1222").pack(side=tk.LEFT)
        spin_space_mult = tk.Spinbox(
            row_btn1,
            from_=1,
            to=10,
            textvariable=self.dual_space_mult_var,
            width=3,
            font=("Segoe UI", 8),
            bg="#141a29",
            fg="#f8fafc",
            buttonbackground="#1e293b",
        )
        spin_space_mult.pack(side=tk.LEFT)

        # Button 2 (Red / Enter default)
        row_btn2 = tk.Frame(self.dual_frame, bg="#0c1222")
        row_btn2.pack(fill=tk.X, pady=6)

        tk.Label(
            row_btn2,
            text="BUTTON 2 (RED):",
            font=("Segoe UI", 8, "bold"),
            fg="#ff1744",
            bg="#0c1222",
            width=15,
            anchor="w",
        ).pack(side=tk.LEFT)

        self.combo_dual_enter = ttk.Combobox(
            row_btn2,
            textvariable=self.dual_enter_port_var,
            state="readonly",
            font=("Segoe UI", 8),
            width=20,
        )
        self.combo_dual_enter.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        combo_b2_key = ttk.Combobox(
            row_btn2,
            textvariable=self.dual_enter_key_var,
            values=COMMON_GAME_KEYS,
            font=("Segoe UI", 8),
            width=8,
        )
        combo_b2_key.pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(row_btn2, text="x", font=("Segoe UI", 8, "bold"), fg="#94a3b8", bg="#0c1222").pack(side=tk.LEFT)
        spin_enter_mult = tk.Spinbox(
            row_btn2,
            from_=1,
            to=10,
            textvariable=self.dual_enter_mult_var,
            width=3,
            font=("Segoe UI", 8),
            bg="#141a29",
            fg="#f8fafc",
            buttonbackground="#1e293b",
        )
        spin_enter_mult.pack(side=tk.LEFT)

    def _set_mode(self, mode: str):
        if self._is_active:
            self.toggle_active()
        self.mode_var.set(mode)
        self._update_mode_styling()
        self._log(f"Switched mode to: {mode} BUTTON SETUP", "cyan")

    def _update_mode_styling(self):
        is_single = self.mode_var.get() == "SINGLE"

        if is_single:
            self.btn_single_mode.config(
                bg="#00e5ff",
                fg="#060913",
                activebackground="#38bdf8",
                activeforeground="#060913",
            )
            self.btn_dual_mode.config(
                bg="#141a29",
                fg="#94a3b8",
                activebackground="#1e293b",
                activeforeground="#ffffff",
            )
            self.card.config(highlightbackground="#00e5ff")
            self.dual_frame.pack_forget()
            self.single_frame.pack(fill=tk.X)
        else:
            self.btn_single_mode.config(
                bg="#141a29",
                fg="#94a3b8",
                activebackground="#1e293b",
                activeforeground="#ffffff",
            )
            self.btn_dual_mode.config(
                bg="#ff1744",
                fg="#ffffff",
                activebackground="#ff5252",
                activeforeground="#ffffff",
            )
            self.card.config(highlightbackground="#ff1744")
            self.single_frame.pack_forget()
            self.dual_frame.pack(fill=tk.X)

        self._draw_action_button()

    def _refresh_ports(self):
        self.ports_list = list_candidate_ports()
        entries = [f"{p.device}  •  {p.description}" for p in self.ports_list]

        self.combo_single_port["values"] = entries
        self.combo_dual_space["values"] = entries
        self.combo_dual_enter["values"] = entries

        count = len(self.ports_list)
        if count == 0:
            self.port_status_lbl.config(
                text="⚠️ No serial devices detected. Plug your Arduino Nano in.",
                fg="#f87171",
            )
            self.single_port_var.set("")
            self.dual_space_port_var.set("")
            self.dual_enter_port_var.set("")
        elif count == 1:
            self.port_status_lbl.config(
                text=f"Detected 1 device: {self.ports_list[0].device}",
                fg="#38bdf8",
            )
            self.single_port_var.set(entries[0])
            self.dual_space_port_var.set(entries[0])
            self.dual_enter_port_var.set("")
        else:
            self.port_status_lbl.config(
                text=f"Detected {count} devices ({', '.join(p.device for p in self.ports_list)})",
                fg="#4ade80",
            )
            self.single_port_var.set(entries[0])
            self.dual_space_port_var.set(entries[0])
            self.dual_enter_port_var.set(entries[1])

        self._log(f"Scanned ports: found {count} serial device(s).", "info")

    def _parse_selected_device(self, val_str: str) -> Optional[str]:
        if not val_str:
            return None
        return val_str.split()[0].strip()

    def _on_btn_hover(self, hover: bool):
        self._hover_btn = hover
        self._draw_action_button()

    def _draw_action_button(self):
        self.action_canvas.delete("all")
        w = self.action_canvas.winfo_width() or 480
        h = 64
        r = 14

        is_single = self.mode_var.get() == "SINGLE"

        if self._is_active:
            # Active Stop state (Red)
            bg_color = "#dc2626" if not self._hover_btn else "#ef4444"
            glow_color = "#ff1744"
            title = "⏹ STOP LISTENING"
            sub = "Click to disconnect hardware bridge"
            fg_title = "#ffffff"
            fg_sub = "#fee2e2"
        else:
            # Idle Start state (Cyan if single, Red if dual)
            if is_single:
                bg_color = "#00e5ff" if not self._hover_btn else "#38bdf8"
                glow_color = "#00e5ff"
                title = "▶ START LISTENING"
                sub = f"Single [{self.single_key_var.get()} x{self.single_multiplier_var.get()}] • Focus your game window to play"
                fg_title = "#060913"
                fg_sub = "#060913"
            else:
                bg_color = "#ff1744" if not self._hover_btn else "#ff5252"
                glow_color = "#ff1744"
                title = "▶ START LISTENING"
                sub = f"Dual [{self.dual_space_key_var.get()} & {self.dual_enter_key_var.get()}] • Focus your game window to play"
                fg_title = "#ffffff"
                fg_sub = "#ffebee"

        # Outer rounded pill
        self.action_canvas.create_oval(0, 0, r * 2, h, fill=bg_color, outline=glow_color, width=2)
        self.action_canvas.create_oval(w - r * 2, 0, w, h, fill=bg_color, outline=glow_color, width=2)
        self.action_canvas.create_rectangle(r, 0, w - r, h, fill=bg_color, outline="")
        self.action_canvas.create_line(r, 0, w - r, 0, fill=glow_color, width=2)
        self.action_canvas.create_line(r, h, w - r, h, fill=glow_color, width=2)

        self.action_canvas.create_text(w / 2, 22, text=title, font=("Segoe UI", 12, "bold"), fill=fg_title)
        self.action_canvas.create_text(w / 2, 44, text=sub, font=("Segoe UI", 8), fill=fg_sub)

    def toggle_active(self):
        if self._is_active:
            self._stop_workers()
        else:
            self._start_workers()

    def _start_workers(self):
        self._stop_workers()
        mode = self.mode_var.get()

        if mode == "SINGLE":
            port = self._parse_selected_device(self.single_port_var.get())
            if not port:
                messagebox.showerror("Port Error", "Please select a serial port for the single button.")
                return
            key = self.single_key_var.get().strip() or "SPACE"
            mult = self.single_multiplier_var.get()

            worker = PortWorker(
                port_name=port,
                action_name=key,
                multiplier=mult,
                log_fn=self._log_safe,
                trigger_fn=self._on_trigger,
                debounce_ms=self.debounce_var.get(),
            )
            self._workers.append(worker)

        else:  # DUAL MODE
            b1_port = self._parse_selected_device(self.dual_space_port_var.get())
            b2_port = self._parse_selected_device(self.dual_enter_port_var.get())

            if not b1_port or not b2_port:
                messagebox.showerror(
                    "Dual Port Error",
                    "Dual-button mode requires two serial ports.\n\n"
                    "If you only have 1 button plugged in, click 'SINGLE BUTTON SETUP'.",
                )
                return

            if b1_port == b2_port:
                messagebox.showerror(
                    "Port Conflict",
                    f"You selected {b1_port} for both Button 1 and Button 2.\n"
                    "Please choose two different ports or use Single Button mode.",
                )
                return

            b1_key = self.dual_space_key_var.get().strip() or "SPACE"
            b2_key = self.dual_enter_key_var.get().strip() or "ENTER"

            w1 = PortWorker(
                port_name=b1_port,
                action_name=b1_key,
                multiplier=self.dual_space_mult_var.get(),
                log_fn=self._log_safe,
                trigger_fn=self._on_trigger,
                debounce_ms=self.debounce_var.get(),
            )
            w2 = PortWorker(
                port_name=b2_port,
                action_name=b2_key,
                multiplier=self.dual_enter_mult_var.get(),
                log_fn=self._log_safe,
                trigger_fn=self._on_trigger,
                debounce_ms=self.debounce_var.get(),
            )
            self._workers.extend([w1, w2])

        # Start background threads
        for w in self._workers:
            w.start()

        self._is_active = True
        self._draw_action_button()
        self.status_pill.config(
            text="🟢 ACTIVE • Listening for clicks! Focus your game window to play.",
            fg="#4ade80",
        )
        self._log("Bridge ACTIVE. Now focus your game window.", "green")

    def _stop_workers(self):
        for w in self._workers:
            w.stop()
        self._workers.clear()

        self._is_active = False
        self._draw_action_button()
        self.status_pill.config(
            text="READY • Stopped. Configure buttons, start listening, then focus game.",
            fg="#94a3b8",
        )
        self._log("Bridge STOPPED.", "yellow")

    def _log(self, msg: str, level: str = "info"):
        self._log_safe(msg, level)

    def _log_safe(self, msg: str, level: str = "info"):
        def _append():
            ts = datetime.now().strftime("%H:%M:%S")
            self.log_text.insert(tk.END, f"[{ts}] {msg}\n", level)
            self.log_text.see(tk.END)

        self.root.after(0, _append)

    def _on_trigger(self, key_name: str):
        def _flash():
            orig_bg = self.card.cget("highlightbackground")
            flash_color = "#00e5ff" if key_name.upper() in ("SPACE", "W", "UP") else "#ff1744"
            self.card.config(highlightbackground=flash_color, highlightthickness=2)
            self.root.after(120, lambda: self.card.config(highlightbackground=orig_bg, highlightthickness=1))

        self.root.after(0, _flash)

    def _clear_log(self):
        self.log_text.delete("1.0", tk.END)

    def on_close(self):
        self._stop_workers()
        self.root.destroy()


def parse_arguments(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spectral Ninja - Arduino Hardware Click Controller"
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run in headless interactive CLI mode instead of GUI.",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "dual"],
        default="single",
        help="Mode for headless startup (default: single).",
    )
    parser.add_argument(
        "--port",
        help="Serial port for single button in headless mode.",
    )
    return parser.parse_args(argv)


def run_headless_cli(args: argparse.Namespace) -> int:
    plist = list_candidate_ports()
    if not plist:
        print("No serial ports detected. Plug your Arduino Nano in and restart.")
        return 1

    print("\n--- SPECTRAL NINJA (HEADLESS MODE) ---")
    if args.mode == "single":
        port = args.port or plist[0].device
        print(f"Running Single-Button mode on {port} (Space x1)")
        worker = PortWorker(port, "SPACE", 1, log_fn=None)
        worker.start()
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopping...")
            worker.stop()
        return 0
    else:
        if len(plist) < 2:
            print(f"Dual mode requires 2 ports, but only found {len(plist)} ({plist[0].device}).")
            return 1
        w1 = PortWorker(plist[0].device, "SPACE", 1)
        w2 = PortWorker(plist[1].device, "ENTER", 1)
        w1.start()
        w2.start()
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopping...")
            w1.stop()
            w2.stop()
        return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_arguments(argv)
    if args.no_gui:
        return run_headless_cli(args)

    root = tk.Tk()
    app = SpectralNinjaApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)

    try:
        root.mainloop()
    finally:
        app._stop_workers()

    return 0


if __name__ == "__main__":
    sys.exit(main())
