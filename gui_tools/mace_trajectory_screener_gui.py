"""MACE Trajectory Screener - Desktop Graphical User Interface.

Provides intuitive graphical screening of combined .xyz trajectories for MACE
neural network potential training:
- Trajectory partitioning with endpoint preservation (initial t0 & final t_final).
- High-force barrier state retention (Fmax >= 5.0 eV/A).
- Geometric similarity screening (all-atom pair distance RMSD >= 0.08 A).
- Standardized REF_energy and REF_forces tagging.
- Real-time structure count reduction and memory savings dashboard.
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Determine runtime directory (frozen executable vs normal script)
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    BUNDLE_DIR = Path(sys._MEIPASS)
else:
    BUNDLE_DIR = Path(__file__).resolve().parent

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from screener_core import (  # noqa: E402
    DEFAULT_HIGH_ENERGY_PER_ATOM_EV,
    DEFAULT_HIGH_FORCE_EV_A,
    DEFAULT_MAX_INTERMEDIATE_FRAMES,
    DEFAULT_MIN_ENERGY_CHANGE_PER_ATOM_EV,
    DEFAULT_MIN_PAIR_DISTANCE_RMSD_A,
    DEFAULT_SIMILARITY_THRESHOLD,
    inspect_dataset,
    run_screening_workflow,
)


class ScreenerApp(tk.Tk):
    """Modern Tkinter desktop application for MACE trajectory screening."""

    def __init__(self) -> None:
        super().__init__()
        self.title("MACE Trajectory Screener")
        self.geometry("740x860")
        self.minsize(680, 750)

        # Set window icon if available
        ico_candidates = [BUNDLE_DIR / "icon.ico", SCRIPT_DIR / "icon.ico"]
        png_candidates = [BUNDLE_DIR / "icon.png", SCRIPT_DIR / "icon.png"]
        ico_path = next((p for p in ico_candidates if p.is_file()), None)
        png_path = next((p for p in png_candidates if p.is_file()), None)

        if ico_path:
            try:
                self.iconbitmap(default=str(ico_path))
            except Exception:
                pass
        elif png_path:
            try:
                img = tk.PhotoImage(file=str(png_path))
                self.iconphoto(True, img)
            except Exception:
                pass
            except Exception:
                pass

        self.msg_queue: queue.Queue = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.current_dataset_info: dict | None = None

        self._configure_styles()
        self._build_ui()
        self._check_queue()

    def _configure_styles(self) -> None:
        """Configure clean modern TTK styling."""
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        # Color scheme
        self.bg_color = "#f4f6f9"
        self.card_bg = "#ffffff"
        self.accent_color = "#1976d2"
        self.accent_hover = "#1565c0"
        self.success_color = "#2e7d32"
        self.text_color = "#212529"
        self.muted_text = "#6c757d"
        self.border_color = "#dee2e6"

        self.configure(bg=self.bg_color)

        self.style.configure(".", background=self.bg_color, foreground=self.text_color, font=("Segoe UI", 9))
        self.style.configure("TFrame", background=self.bg_color)
        self.style.configure("Card.TFrame", background=self.card_bg, relief="solid", borderwidth=1)
        self.style.configure("CardLabel.TLabel", background=self.card_bg, foreground=self.text_color, font=("Segoe UI", 9))
        self.style.configure("CardMuted.TLabel", background=self.card_bg, foreground=self.muted_text, font=("Segoe UI", 8))
        self.style.configure("CardBold.TLabel", background=self.card_bg, foreground=self.text_color, font=("Segoe UI", 9, "bold"))
        self.style.configure("Header.TLabel", background=self.bg_color, foreground="#0d47a1", font=("Segoe UI", 16, "bold"))
        self.style.configure("SubHeader.TLabel", background=self.bg_color, foreground=self.muted_text, font=("Segoe UI", 9))

        self.style.configure(
            "Primary.TButton",
            font=("Segoe UI", 10, "bold"),
            background=self.accent_color,
            foreground="#ffffff",
            padding=(12, 6),
        )
        self.style.map(
            "Primary.TButton",
            background=[("active", self.accent_hover), ("disabled", "#b0bec5")],
            foreground=[("disabled", "#eceff1")],
        )

        self.style.configure(
            "Success.TButton",
            font=("Segoe UI", 9, "bold"),
            background=self.success_color,
            foreground="#ffffff",
            padding=(10, 4),
        )
        self.style.map(
            "Success.TButton",
            background=[("active", "#1b5e20"), ("disabled", "#b0bec5")],
        )

        self.style.configure("Horizontal.TProgressbar", thickness=14)

    def _build_ui(self) -> None:
        """Construct user interface layout."""
        main_container = ttk.Frame(self, padding="16 12 16 16")
        main_container.pack(fill=tk.BOTH, expand=True)

        # 1. Header
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(header_frame, text="MACE Trajectory Screener", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            header_frame,
            text="High-force barrier retention & similarity screening for neural network potential training",
            style="SubHeader.TLabel",
        ).pack(anchor="w")

        # 2. File Selection Card
        file_card = ttk.Frame(main_container, style="Card.TFrame", padding="12")
        file_card.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(file_card, text="1. Select Input Trajectory Dataset or Folder", style="CardBold.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 6)
        )

        self.input_path_var = tk.StringVar()
        entry_input = ttk.Entry(file_card, textvariable=self.input_path_var, font=("Segoe UI", 9))
        entry_input.grid(row=1, column=0, sticky="ew", padx=(0, 8), pady=2)

        btn_box = ttk.Frame(file_card, style="Card.TFrame")
        btn_box.grid(row=1, column=1, columnspan=2, sticky="e", pady=2)
        btn_browse_file = ttk.Button(btn_box, text="Browse File...", command=self._browse_input_file)
        btn_browse_file.pack(side=tk.LEFT, padx=(0, 4))
        btn_browse_folder = ttk.Button(btn_box, text="Browse Folder...", command=self._browse_input_folder)
        btn_browse_folder.pack(side=tk.LEFT)

        # Dataset stats banner
        self.stats_label_var = tk.StringVar(value="Select an .xyz, .extxyz, .traj, or .db file, or a folder filled with trajectories.")
        lbl_stats = ttk.Label(file_card, textvariable=self.stats_label_var, style="CardMuted.TLabel", wraplength=640)
        lbl_stats.grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 0))

        # Output file row
        ttk.Label(file_card, text="Output Screened Dataset:", style="CardLabel.TLabel").grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(10, 2)
        )
        self.output_path_var = tk.StringVar()
        entry_output = ttk.Entry(file_card, textvariable=self.output_path_var, font=("Segoe UI", 9))
        entry_output.grid(row=4, column=0, columnspan=2, sticky="ew", padx=(0, 8), pady=2)
        btn_browse_out = ttk.Button(file_card, text="Change...", command=self._browse_output)
        btn_browse_out.grid(row=4, column=2, sticky="e", pady=2)

        file_card.columnconfigure(0, weight=1)

        # 3. Screening Parameters Card
        param_card = ttk.Frame(main_container, style="Card.TFrame", padding="12")
        param_card.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(param_card, text="2. Screening Parameters", style="CardBold.TLabel").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 6)
        )

        # Parameter variables
        self.var_high_force = tk.DoubleVar(value=DEFAULT_HIGH_FORCE_EV_A)
        self.var_similarity_threshold = tk.DoubleVar(value=DEFAULT_SIMILARITY_THRESHOLD)
        self.var_rmsd = tk.DoubleVar(value=DEFAULT_MIN_PAIR_DISTANCE_RMSD_A)
        self.var_energy_change = tk.DoubleVar(value=DEFAULT_MIN_ENERGY_CHANGE_PER_ATOM_EV)
        self.var_max_intermediate = tk.IntVar(value=DEFAULT_MAX_INTERMEDIATE_FRAMES)
        self.var_dynamic_threshold = tk.BooleanVar(value=True)
        self.var_audit_csv = tk.BooleanVar(value=True)

        # Col 0 & 1: Force & REMatch Similarity
        ttk.Label(param_card, text="High Force Barrier Threshold (eV/Å):", style="CardLabel.TLabel").grid(
            row=1, column=0, sticky="w", pady=3
        )
        spn_force = ttk.Spinbox(param_card, from_=1.0, to=20.0, increment=0.5, textvariable=self.var_high_force, width=8)
        spn_force.grid(row=1, column=1, sticky="w", padx=(6, 20), pady=3)

        ttk.Label(param_card, text="Max REMatch Similarity (fallback):", style="CardLabel.TLabel").grid(
            row=2, column=0, sticky="w", pady=3
        )
        spn_sim = ttk.Spinbox(param_card, from_=0.50, to=0.999, increment=0.005, textvariable=self.var_similarity_threshold, width=8)
        spn_sim.grid(row=2, column=1, sticky="w", padx=(6, 20), pady=3)

        # Col 2 & 3: Energy diff & Max frames
        ttk.Label(param_card, text="Energy Variance Cutoff (eV/atom):", style="CardLabel.TLabel").grid(
            row=1, column=2, sticky="w", pady=3
        )
        spn_de = ttk.Spinbox(param_card, from_=0.005, to=0.20, increment=0.005, textvariable=self.var_energy_change, width=8)
        spn_de.grid(row=1, column=3, sticky="w", padx=(6, 0), pady=3)

        ttk.Label(param_card, text="Max Intermediate Frames / Traj:", style="CardLabel.TLabel").grid(
            row=2, column=2, sticky="w", pady=3
        )
        spn_max = ttk.Spinbox(param_card, from_=1, to=50, increment=1, textvariable=self.var_max_intermediate, width=8)
        spn_max.grid(row=2, column=3, sticky="w", padx=(6, 0), pady=3)

        # Checkboxes
        chk_dyn = ttk.Checkbutton(
            param_card,
            text="Dynamic per-trajectory threshold (elbow heuristic / spread)",
            variable=self.var_dynamic_threshold,
        )
        chk_dyn.grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 2))

        chk_csv = ttk.Checkbutton(
            param_card,
            text="Generate detailed audit CSV report",
            variable=self.var_audit_csv,
        )
        chk_csv.grid(row=3, column=2, columnspan=2, sticky="w", pady=(8, 2))

        # 4. Action & Progress Area
        action_frame = ttk.Frame(main_container)
        action_frame.pack(fill=tk.X, pady=(0, 10))

        self.btn_run = ttk.Button(action_frame, text="⚡ Run Screening", style="Primary.TButton", command=self._start_screening)
        self.btn_run.pack(side=tk.LEFT, padx=(0, 10))

        self.status_var = tk.StringVar(value="Ready.")
        self.lbl_status = ttk.Label(action_frame, textvariable=self.status_var, font=("Segoe UI", 9, "italic"))
        self.lbl_status.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.progress_bar = ttk.Progressbar(main_container, orient=tk.HORIZONTAL, mode="determinate")
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))

        # 5. Results & Savings Dashboard Card
        self.results_card = ttk.Frame(main_container, style="Card.TFrame", padding="14")
        self.results_card.pack(fill=tk.BOTH, expand=True)

        ttk.Label(self.results_card, text="3. Screening Results & Memory Savings", style="CardBold.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8)
        )

        # Metrics display grid
        self.metric_initial_var = tk.StringVar(value="-")
        self.metric_final_var = tk.StringVar(value="-")
        self.metric_reduction_var = tk.StringVar(value="-")
        self.metric_filesize_var = tk.StringVar(value="-")
        self.metric_breakdown_var = tk.StringVar(value="Awaiting screening run...")

        self._build_metric_tile(self.results_card, "Initial Structures", self.metric_initial_var, row=1, col=0)
        self._build_metric_tile(self.results_card, "Screened Structures", self.metric_final_var, row=1, col=1)
        self._build_metric_tile(self.results_card, "Structure Reduction", self.metric_reduction_var, row=1, col=2)

        # Detailed breakdown & Memory savings banner
        self.savings_banner_var = tk.StringVar(value="")
        lbl_savings = ttk.Label(
            self.results_card,
            textvariable=self.savings_banner_var,
            style="CardBold.TLabel",
            foreground=self.success_color,
            font=("Segoe UI", 11, "bold"),
        )
        lbl_savings.grid(row=2, column=0, columnspan=3, sticky="w", pady=(12, 4))

        lbl_breakdown = ttk.Label(
            self.results_card,
            textvariable=self.metric_breakdown_var,
            style="CardLabel.TLabel",
            wraplength=640,
        )
        lbl_breakdown.grid(row=3, column=0, columnspan=3, sticky="w", pady=(0, 10))

        # Open Output Folder Button
        self.btn_open_folder = ttk.Button(
            self.results_card,
            text="📁 Open Output Folder",
            style="Success.TButton",
            command=self._open_output_folder,
            state="disabled",
        )
        self.btn_open_folder.grid(row=4, column=0, sticky="w", pady=(6, 0))

        self.results_card.columnconfigure(0, weight=1)
        self.results_card.columnconfigure(1, weight=1)
        self.results_card.columnconfigure(2, weight=1)

    def _build_metric_tile(self, parent: ttk.Frame, title: str, var: tk.StringVar, row: int, col: int) -> None:
        """Create clean metric card tile."""
        tile = ttk.Frame(parent, style="Card.TFrame", padding="8")
        tile.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)
        ttk.Label(tile, text=title, style="CardMuted.TLabel").pack(anchor="center")
        ttk.Label(tile, textvariable=var, style="CardBold.TLabel", font=("Segoe UI", 14, "bold")).pack(anchor="center", pady=(2, 0))

    def _browse_input_file(self) -> None:
        """Open file dialog for input dataset (.xyz, .extxyz, .traj, .db) and inspect properties."""
        path = filedialog.askopenfilename(
            title="Select Trajectory Dataset or Database",
            filetypes=[
                ("All Supported Datasets (*.xyz, *.extxyz, *.traj, *.db)", "*.xyz *.extxyz *.traj *.db"),
                ("Extended XYZ / XYZ Trajectories (*.xyz, *.extxyz)", "*.xyz *.extxyz"),
                ("ASE Trajectories (*.traj)", "*.traj"),
                ("ASE SQLite Databases (*.db)", "*.db"),
                ("All Files", "*.*"),
            ],
        )
        if not path:
            return
        self._set_input_path(path)

    def _browse_input_folder(self) -> None:
        """Open folder dialog to select directory filled with trajectories."""
        folder = filedialog.askdirectory(title="Select Folder Filled with Trajectories (.traj, .db, .xyz, .extxyz)")
        if not folder:
            return
        self._set_input_path(folder)

    def _set_input_path(self, path: str) -> None:
        """Set input path and trigger async inspection."""
        self.input_path_var.set(path)
        p = Path(path)
        if p.is_dir():
            out_suggested = p / f"{p.name}_screened.xyz"
            self.stats_label_var.set("Inspecting directory of trajectories...")
        else:
            out_suggested = p.parent / f"{p.stem}_screened.xyz"
            self.stats_label_var.set("Inspecting dataset...")
        self.output_path_var.set(str(out_suggested))
        threading.Thread(target=self._async_inspect, args=(path,), daemon=True).start()

    def _async_inspect(self, path: str) -> None:
        """Inspect dataset properties asynchronously."""
        try:
            info = inspect_dataset(path)
            self.msg_queue.put(("inspect_done", info))
        except Exception as e:
            self.msg_queue.put(("inspect_error", str(e)))

    def _browse_output(self) -> None:
        """Open save dialog for output dataset."""
        init_dir = None
        curr = self.output_path_var.get()
        if curr:
            init_dir = str(Path(curr).parent)
        path = filedialog.asksaveasfilename(
            title="Save Screened Dataset",
            initialdir=init_dir,
            filetypes=[
                ("Extended XYZ File (*.xyz, *.extxyz)", "*.xyz *.extxyz"),
                ("ASE Trajectory (*.traj)", "*.traj"),
                ("ASE SQLite Database (*.db)", "*.db"),
                ("All Files", "*.*"),
            ],
            defaultextension=".xyz",
        )
        if path:
            self.output_path_var.set(path)

    def _start_screening(self) -> None:
        """Validate inputs and launch screening worker thread."""
        in_path = self.input_path_var.get().strip()
        out_path = self.output_path_var.get().strip()

        p_in = Path(in_path)
        if not in_path or not (p_in.is_file() or p_in.is_dir()):
            messagebox.showerror("Error", "Please select a valid input dataset file or folder filled with trajectories.")
            return

        if not out_path:
            messagebox.showerror("Error", "Please specify an output file path.")
            return

        if Path(in_path).resolve() == Path(out_path).resolve():
            messagebox.showerror("Error", "Output path cannot be the same as input path.")
            return

        # Disable controls during run
        self.btn_run.configure(state="disabled")
        self.btn_open_folder.configure(state="disabled")
        self.progress_bar.configure(value=0)
        self.status_var.set("Starting screening...")

        audit_path = None
        if self.var_audit_csv.get():
            p_out = Path(out_path)
            audit_path = str(p_out.parent / f"{p_out.stem}_audit.csv")

        kwargs = {
            "input_path": in_path,
            "output_path": out_path,
            "high_force_ev_a": float(self.var_high_force.get()),
            "similarity_threshold": "dynamic" if self.var_dynamic_threshold.get() else float(self.var_similarity_threshold.get()),
            "min_pair_distance_rmsd_a": float(self.var_rmsd.get()),
            "min_energy_change_per_atom_ev": float(self.var_energy_change.get()),
            "max_intermediate_frames": int(self.var_max_intermediate.get()),
            "tag_ref_properties": True,
            "audit_csv_path": audit_path,
        }

        self.worker_thread = threading.Thread(target=self._worker_run, kwargs=kwargs, daemon=True)
        self.worker_thread.start()

    def _worker_run(self, **kwargs) -> None:
        """Worker thread executing the screening workflow."""
        def progress_cb(current: int, total: int, status: str) -> None:
            pct = (current / max(1, total)) * 100.0
            self.msg_queue.put(("progress", (pct, status)))

        try:
            results = run_screening_workflow(progress_callback=progress_cb, **kwargs)
            self.msg_queue.put(("screening_done", results))
        except Exception as e:
            self.msg_queue.put(("screening_error", str(e)))

    def _check_queue(self) -> None:
        """Periodic GUI event loop polling the message queue."""
        try:
            while True:
                msg_type, payload = self.msg_queue.get_nowait()
                if msg_type == "inspect_done":
                    self.current_dataset_info = payload
                    self._update_inspect_ui(payload)
                elif msg_type == "inspect_error":
                    self.stats_label_var.set(f"Dataset inspection error: {payload}")
                elif msg_type == "progress":
                    pct, status = payload
                    self.progress_bar.configure(value=pct)
                    self.status_var.set(status)
                elif msg_type == "screening_done":
                    self._on_screening_finished(payload)
                elif msg_type == "screening_error":
                    self.btn_run.configure(state="normal")
                    self.status_var.set("Error occurred.")
                    messagebox.showerror("Screening Error", f"Screening failed with error:\n{payload}")
        except queue.Empty:
            pass
        finally:
            self.after(100, self._check_queue)

    def _update_inspect_ui(self, info: dict) -> None:
        """Update dataset statistics UI after inspection."""
        count = info.get("total_frames", 0)
        size_mb = info.get("file_size_mb", 0.0)
        symbols = ", ".join(info.get("elements", [])) or "None"
        if info.get("is_directory"):
            file_cnt = info.get("file_count", 0)
            self.stats_label_var.set(
                f"Folder: {file_cnt} trajectory file(s) (.traj, .db, .xyz, .extxyz) | {count:,} total structures ({size_mb:.2f} MB) | Elements: [{symbols}]"
            )
        else:
            formulas = info.get("sample_formulas", [])
            form_str = formulas[0] if formulas else "Unknown"
            self.stats_label_var.set(
                f"Dataset: {count:,} structures ({size_mb:.2f} MB) | Elements: [{symbols}] | Sample: {form_str}"
            )
        self.metric_initial_var.set(f"{count:,}")

    def _on_screening_finished(self, res: dict) -> None:
        """Display screening results and update savings metrics."""
        self.btn_run.configure(state="normal")
        self.btn_open_folder.configure(state="normal")
        self.progress_bar.configure(value=100)
        self.status_var.set("Screening complete! ✨")

        in_cnt = res["initial_count"]
        out_cnt = res["final_count"]
        reduction_pct = res["structure_savings_pct"]
        size_pct = res["file_savings_pct"]
        in_mb = res["initial_size_mb"]
        out_mb = res["final_size_mb"]
        saved_mb = in_mb - out_mb

        self.metric_initial_var.set(f"{in_cnt:,}")
        self.metric_final_var.set(f"{out_cnt:,}")
        self.metric_reduction_var.set(f"-{reduction_pct:.1f}%")

        self.savings_banner_var.set(
            f"🎉 Saved {reduction_pct:.1f}% structures ({saved_mb:.2f} MB / {size_pct:.1f}% disk & memory savings)"
        )

        high_f = res.get("high_force_kept", 0)
        endpts = res.get("endpoints_kept", 0)
        interm = res.get("intermediate_kept", 0)
        trajs = res.get("trajectories_count", 0)
        files_cnt = res.get("files_count", 1)

        rej_sim = res.get("rejected_similarity", 0)
        rej_cap = res.get("rejected_cap", 0)
        rej_e = res.get("rejected_energy", 0)
        rej_f = res.get("rejected_force", 0)
        total_rej = res.get("total_rejected", in_cnt - out_cnt)

        breakdown_text = (
            f"Analyzed {files_cnt} file(s) across {trajs} optimization trajectory bins.\n"
            f"• Retained: {out_cnt:,} structures ({endpts:,} endpoints, {high_f:,} high-force barriers, {interm:,} distinct intermediates)\n"
            f"• Rejected: {total_rej:,} structures:\n"
            f"    - {rej_sim:,} rejected by similarity / geometric redundancy\n"
            f"    - {rej_cap:,} rejected by intermediate quota cap\n"
            f"    - {rej_e:,} rejected by energy (outliers / non-finite / extreme)\n"
            f"    - {rej_f:,} rejected by force divergence (Fmax > 100 eV/Å)\n"
            f"• Calculator forces and REF_forces/REF_energy attached for tracking: {res['output_path']}"
        )
        if res.get("audit_csv_path"):
            breakdown_text += f"\n• Detailed audit CSV saved to: {res['audit_csv_path']}"

        self.metric_breakdown_var.set(breakdown_text)

        messagebox.showinfo(
            "Screening Completed",
            f"Successfully screened dataset!\n\n"
            f"Initial: {in_cnt:,} structures ({in_mb:.2f} MB)\n"
            f"Retained: {out_cnt:,} structures ({out_mb:.2f} MB)\n"
            f"Structure Reduction: -{reduction_pct:.1f}%\n"
            f"Memory/Disk Savings: -{size_pct:.1f}% ({saved_mb:.2f} MB)\n\n"
            f"Rejections breakdown ({total_rej:,} total):\n"
            f"  • {rej_sim:,} by similarity / redundancy\n"
            f"  • {rej_cap:,} by intermediate quota cap\n"
            f"  • {rej_e:,} by energy outliers/non-finite\n"
            f"  • {rej_f:,} by force divergence\n\n"
            f"Output saved to:\n{res['output_path']}",
        )

    def _open_output_folder(self) -> None:
        """Open output file directory in OS file manager."""
        out_path = self.output_path_var.get().strip()
        if not out_path:
            return
        folder = Path(out_path).resolve().parent
        if folder.is_dir():
            try:
                if sys.platform == "win32":
                    os.startfile(folder)
                elif sys.platform == "darwin":
                    subprocess.run(["open", str(folder)], check=False)
                else:
                    subprocess.run(["xdg-open", str(folder)], check=False)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open folder:\n{e}")


def main() -> None:
    """Launch MACE Trajectory Screener GUI."""
    app = ScreenerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
