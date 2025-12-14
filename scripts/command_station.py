#!/usr/bin/env python3
"""
Local Command Station GUI to manage a worker node and block/cluster state:
- Start/stop the local worker process
- View node/model status from /worker_status
- View and manage the global training block queue
- View and manage cluster nodes
- Run starter-block smoke test via /run_starter_test
- Build experience blocks from prepared data
- Send quick test prompts to the live inference API

Adjust TRAINER_URL, INFERENCE_URL, WORKER_CMD, and expected API schemas to your environment.
"""
import queue
import subprocess
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import requests

# ---------------------------
# Config – adjust to your setup
# ---------------------------

TRAINER_URL = "http://localhost:5001"  # trainer/server base URL
INFERENCE_URL = "http://localhost:5000/generate"  # inference API for live test calls
WORKER_CMD = ["python", "-m", "actors.worker", "--trainer-url", TRAINER_URL]

STATUS_ENDPOINT = "/worker_status"
STARTER_TEST_ENDPOINT = "/run_starter_test"

# New endpoints for queue + cluster management.
QUEUE_STATUS_ENDPOINT = "/queue_status"
ENQUEUE_BLOCKS_ENDPOINT = "/enqueue_blocks"
CLUSTER_STATUS_ENDPOINT = "/cluster_status"
DISABLE_NODE_ENDPOINT = "/disable_node"
ENABLE_NODE_ENDPOINT = "/enable_node"


class CommandStation(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Training Command Station")
        self.geometry("1200x700")

        self.worker_process = None
        self.log_queue = queue.Queue()

        self._build_layout()

        # Periodic background polling.
        self.after(2000, self._poll_status_loop)
        self.after(250, self._poll_log_queue)

    # Layout -------------------------------------------------------------
    def _build_layout(self):
        # ---------------- Top: node + model + controls -----------------
        top_frame = ttk.Frame(self)
        top_frame.pack(fill="x", padx=10, pady=10)

        # Node status
        node_frame = ttk.LabelFrame(top_frame, text="Node Status")
        node_frame.pack(side="left", fill="both", expand=True, padx=5)

        self.node_status_var = tk.StringVar(value="Stopped")
        self.node_id_var = tk.StringVar(value="unknown")
        self.trainer_url_var = tk.StringVar(value=TRAINER_URL)
        self.inference_url_var = tk.StringVar(value=INFERENCE_URL)

        ttk.Label(node_frame, text="Node ID:").grid(row=0, column=0, sticky="w")
        ttk.Label(node_frame, textvariable=self.node_id_var).grid(row=0, column=1, sticky="w")

        ttk.Label(node_frame, text="Status:").grid(row=1, column=0, sticky="w")
        ttk.Label(node_frame, textvariable=self.node_status_var).grid(row=1, column=1, sticky="w")

        ttk.Label(node_frame, text="Trainer URL:").grid(row=2, column=0, sticky="w")
        trainer_entry = ttk.Entry(node_frame, textvariable=self.trainer_url_var, width=40)
        trainer_entry.grid(row=2, column=1, sticky="w")

        ttk.Label(node_frame, text="Inference URL:").grid(row=3, column=0, sticky="w")
        infer_entry = ttk.Entry(node_frame, textvariable=self.inference_url_var, width=40)
        infer_entry.grid(row=3, column=1, sticky="w")

        # Model status
        model_frame = ttk.LabelFrame(top_frame, text="Model / Adapter")
        model_frame.pack(side="left", fill="both", expand=True, padx=5)

        self.base_version_var = tk.StringVar(value="-")
        self.adapter_name_var = tk.StringVar(value="-")
        self.quantization_var = tk.StringVar(value="-")
        self.flash_var = tk.StringVar(value="-")
        self.compile_var = tk.StringVar(value="-")
        self.blocks_processed_var = tk.StringVar(value="0")

        row = 0
        ttk.Label(model_frame, text="Base version:").grid(row=row, column=0, sticky="w")
        ttk.Label(model_frame, textvariable=self.base_version_var).grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Label(model_frame, text="Adapter:").grid(row=row, column=0, sticky="w")
        ttk.Label(model_frame, textvariable=self.adapter_name_var).grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Label(model_frame, text="Quantization:").grid(row=row, column=0, sticky="w")
        ttk.Label(model_frame, textvariable=self.quantization_var).grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Label(model_frame, text="Flash Attn:").grid(row=row, column=0, sticky="w")
        ttk.Label(model_frame, textvariable=self.flash_var).grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Label(model_frame, text="torch.compile:").grid(row=row, column=0, sticky="w")
        ttk.Label(model_frame, textvariable=self.compile_var).grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Label(model_frame, text="Blocks processed:").grid(row=row, column=0, sticky="w")
        ttk.Label(model_frame, textvariable=self.blocks_processed_var).grid(row=row, column=1, sticky="w")
        row += 1

        # Control buttons
        control_frame = ttk.Frame(top_frame)
        control_frame.pack(side="right", fill="y", padx=5)

        ttk.Button(control_frame, text="Start Worker", command=self.start_worker).pack(fill="x", pady=2)
        ttk.Button(control_frame, text="Stop Worker", command=self.stop_worker).pack(fill="x", pady=2)
        ttk.Button(control_frame, text="Refresh Status", command=self.refresh_status).pack(fill="x", pady=2)
        ttk.Button(control_frame, text="Run Starter Block Test", command=self.run_starter_test).pack(
            fill="x", pady=8
        )

        # ---------------- Middle: assigned blocks + logs ----------------
        middle_frame = ttk.PanedWindow(self, orient="horizontal")
        middle_frame.pack(fill="both", expand=True, padx=10, pady=5)

        blocks_frame = ttk.LabelFrame(middle_frame, text="Assigned Blocks (this node)")
        middle_frame.add(blocks_frame, weight=3)

        columns = ("block_id", "status", "loss", "updated_at")
        style = ttk.Style(self)
        style.configure("Blocks.Treeview", rowheight=24)

        self.blocks_tree = ttk.Treeview(
            blocks_frame, columns=columns, show="headings", height=8, style="Blocks.Treeview"
        )
        for col in columns:
            self.blocks_tree.heading(col, text=col)
            self.blocks_tree.column(col, width=140)
        self.blocks_tree.pack(fill="both", expand=True)

        log_frame = ttk.LabelFrame(middle_frame, text="Logs")
        middle_frame.add(log_frame, weight=2)

        self.log_text = scrolledtext.ScrolledText(log_frame, wrap="word", state="disabled")
        self.log_text.pack(fill="both", expand=True)

        # ---------------- Cluster Nodes panel ----------------
        cluster_frame = ttk.LabelFrame(self, text="Cluster Nodes")
        cluster_frame.pack(fill="both", expand=False, padx=10, pady=5)

        cluster_inner = ttk.Frame(cluster_frame)
        cluster_inner.pack(fill="both", expand=True)

        cluster_columns = ("node_id", "state", "blocks_done", "trust", "last_seen")
        self.cluster_tree = ttk.Treeview(
            cluster_inner, columns=cluster_columns, show="headings", height=6
        )
        for col in cluster_columns:
            self.cluster_tree.heading(col, text=col)
            width = 140 if col != "last_seen" else 200
            self.cluster_tree.column(col, width=width)
        self.cluster_tree.pack(side="left", fill="both", expand=True)

        cluster_btns = ttk.Frame(cluster_inner)
        cluster_btns.pack(side="right", fill="y", padx=5)

        ttk.Button(cluster_btns, text="Refresh Cluster", command=self.refresh_cluster_status).pack(
            fill="x", pady=2
        )
        ttk.Button(cluster_btns, text="Disable Node", command=self.disable_selected_node).pack(
            fill="x", pady=2
        )
        ttk.Button(cluster_btns, text="Enable Node", command=self.enable_selected_node).pack(
            fill="x", pady=2
        )

        # ---------------- Bottom: block builder + queue + inference -----
        bottom_frame = ttk.Frame(self)
        bottom_frame.pack(fill="both", expand=False, padx=10, pady=5)

        # Block builder
        builder_frame = ttk.LabelFrame(bottom_frame, text="Block Builder (local)")
        builder_frame.pack(side="left", fill="both", expand=True, padx=5)

        self.input_path_var = tk.StringVar(value="data/raw/ebooks")
        self.output_path_var = tk.StringVar(
            value="data/processed/experience_blocks.jsonl"
        )
        self.dataset_label_var = tk.StringVar(value="dev_dataset")
        self.block_adapter_var = tk.StringVar(value="")

        ttk.Label(builder_frame, text="Input path/file:").grid(row=0, column=0, sticky="w")
        ttk.Entry(builder_frame, textvariable=self.input_path_var, width=40).grid(
            row=0, column=1, sticky="we"
        )
        ttk.Button(builder_frame, text="Browse", command=self.browse_dataset).grid(
            row=0, column=2, sticky="we", padx=2
        )

        ttk.Label(builder_frame, text="Output blocks file:").grid(row=1, column=0, sticky="w")
        ttk.Entry(builder_frame, textvariable=self.output_path_var, width=40).grid(
            row=1, column=1, sticky="we"
        )

        ttk.Label(builder_frame, text="Dataset label:").grid(row=2, column=0, sticky="w")
        ttk.Entry(builder_frame, textvariable=self.dataset_label_var, width=25).grid(
            row=2, column=1, sticky="w"
        )

        ttk.Label(builder_frame, text="Target adapter (optional):").grid(
            row=3, column=0, sticky="w"
        )
        ttk.Entry(builder_frame, textvariable=self.block_adapter_var, width=25).grid(
            row=3, column=1, sticky="w"
        )

        ttk.Button(builder_frame, text="Build Blocks Locally", command=self.run_block_builder).grid(
            row=4, column=0, columnspan=2, sticky="we", pady=4
        )

        # Block queue controls (talks to trainer)
        queue_frame = ttk.LabelFrame(bottom_frame, text="Global Block Queue")
        queue_frame.pack(side="left", fill="both", expand=True, padx=5)

        # Summary labels
        self.queue_total_var = tk.StringVar(value="0")
        self.queue_pending_var = tk.StringVar(value="0")
        self.queue_assigned_var = tk.StringVar(value="0")
        self.queue_completed_var = tk.StringVar(value="0")
        self.queue_failed_var = tk.StringVar(value="0")

        qrow = 0
        ttk.Label(queue_frame, text="Total:").grid(row=qrow, column=0, sticky="w")
        ttk.Label(queue_frame, textvariable=self.queue_total_var).grid(row=qrow, column=1, sticky="w")
        ttk.Label(queue_frame, text="Pending:").grid(row=qrow, column=2, sticky="w")
        ttk.Label(queue_frame, textvariable=self.queue_pending_var).grid(row=qrow, column=3, sticky="w")
        qrow += 1

        ttk.Label(queue_frame, text="Assigned:").grid(row=qrow, column=0, sticky="w")
        ttk.Label(queue_frame, textvariable=self.queue_assigned_var).grid(row=qrow, column=1, sticky="w")
        ttk.Label(queue_frame, text="Completed:").grid(row=qrow, column=2, sticky="w")
        ttk.Label(queue_frame, textvariable=self.queue_completed_var).grid(row=qrow, column=3, sticky="w")
        qrow += 1

        ttk.Label(queue_frame, text="Failed:").grid(row=qrow, column=0, sticky="w")
        ttk.Label(queue_frame, textvariable=self.queue_failed_var).grid(row=qrow, column=1, sticky="w")
        qrow += 1

        ttk.Button(queue_frame, text="Refresh Queue", command=self.refresh_queue_status).grid(
            row=qrow, column=0, columnspan=2, sticky="we", pady=2
        )
        ttk.Button(queue_frame, text="Enqueue Output Blocks", command=self.enqueue_blocks).grid(
            row=qrow, column=2, columnspan=2, sticky="we", pady=2
        )
        qrow += 1

        queue_columns = ("block_id", "task", "status", "assigned_to", "updated_at")
        self.queue_tree = ttk.Treeview(
            queue_frame, columns=queue_columns, show="headings", height=6
        )
        for col in queue_columns:
            self.queue_tree.heading(col, text=col)
            width = 120 if col != "updated_at" else 160
            self.queue_tree.column(col, width=width)
        self.queue_tree.grid(row=qrow, column=0, columnspan=4, sticky="nsew")
        queue_frame.rowconfigure(qrow, weight=1)
        queue_frame.columnconfigure(1, weight=1)

        # Live inference test
        infer_frame = ttk.LabelFrame(bottom_frame, text="Live Model Test")
        infer_frame.pack(side="right", fill="both", expand=True, padx=5)

        self.test_persona = tk.StringVar(value="tester_npc")
        self.test_context = tk.StringVar(value="town square")
        self.test_state = tk.StringVar(value="idle")
        self.test_player = tk.StringVar(value="Hello there")
        self.test_adapter_name = tk.StringVar(value="")
        self.test_audience = tk.StringVar(value="adult")

        ttk.Label(infer_frame, text="Adapter name:").grid(row=0, column=0, sticky="w")
        ttk.Entry(infer_frame, textvariable=self.test_adapter_name, width=25).grid(
            row=0, column=1, sticky="we"
        )
        ttk.Label(infer_frame, text="Audience (adult/minor):").grid(row=1, column=0, sticky="w")
        ttk.Entry(infer_frame, textvariable=self.test_audience, width=15).grid(
            row=1, column=1, sticky="w"
        )

        ttk.Label(infer_frame, text="Persona:").grid(row=2, column=0, sticky="w")
        ttk.Entry(infer_frame, textvariable=self.test_persona, width=25).grid(
            row=2, column=1, sticky="we"
        )

        ttk.Label(infer_frame, text="Context:").grid(row=3, column=0, sticky="w")
        ttk.Entry(infer_frame, textvariable=self.test_context, width=25).grid(
            row=3, column=1, sticky="we"
        )

        ttk.Label(infer_frame, text="State:").grid(row=4, column=0, sticky="w")
        ttk.Entry(infer_frame, textvariable=self.test_state, width=25).grid(
            row=4, column=1, sticky="we"
        )

        ttk.Label(infer_frame, text="Player input:").grid(row=5, column=0, sticky="w")
        ttk.Entry(infer_frame, textvariable=self.test_player, width=40).grid(
            row=5, column=1, sticky="we"
        )

        ttk.Button(infer_frame, text="Send to Model", command=self.run_live_inference).grid(
            row=6, column=0, columnspan=2, sticky="we", pady=4
        )
        self.infer_output = scrolledtext.ScrolledText(infer_frame, wrap="word", height=8, state="disabled")
        self.infer_output.grid(row=7, column=0, columnspan=2, sticky="nsew", padx=2, pady=2)
        infer_frame.rowconfigure(7, weight=1)

    # Worker control -----------------------------------------------------
    def start_worker(self):
        if self.worker_process and self.worker_process.poll() is None:
            messagebox.showinfo("Worker", "Worker is already running.")
            return
        self._log("Starting worker process...")
        try:
            self.worker_process = subprocess.Popen(
                WORKER_CMD,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            threading.Thread(target=self._stream_worker_logs, daemon=True).start()
            self.node_status_var.set("Starting...")
        except Exception as e:
            self._log(f"Failed to start worker: {e}")
            messagebox.showerror("Error", f"Failed to start worker:\n{e}")

    def stop_worker(self):
        if self.worker_process and self.worker_process.poll() is None:
            self._log("Stopping worker process...")
            self.worker_process.terminate()
            try:
                self.worker_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._log("Worker did not exit, killing...")
                self.worker_process.kill()
            self.node_status_var.set("Stopped")
            self._log("Worker stopped.")
        else:
            messagebox.showinfo("Worker", "Worker is not running.")

    def _stream_worker_logs(self):
        if not self.worker_process or not self.worker_process.stdout:
            return
        for line in self.worker_process.stdout:
            if not line:
                break
            self.log_queue.put(line.rstrip("\n"))

    # Status polling -----------------------------------------------------
    def _poll_status_loop(self):
        # Called periodically; fan out to all status refreshers.
        self.refresh_status(background=True)
        self.refresh_queue_status(background=True)
        self.refresh_cluster_status(background=True)
        self.after(4000, self._poll_status_loop)

    def refresh_status(self, background=False):
        def _task():
            try:
                url = self.trainer_url_var.get().rstrip("/") + STATUS_ENDPOINT
                resp = requests.get(url, timeout=3)
                resp.raise_for_status()
                status = resp.json()
                self._update_status_from_json(status)
            except Exception as e:
                if not background:
                    messagebox.showerror("Status Error", f"Failed to fetch status:\n{e}")
                self._log(f"Status fetch failed: {e}")

        threading.Thread(target=_task, daemon=True).start()

    def _update_status_from_json(self, status):
        self.node_id_var.set(status.get("node_id", "unknown"))
        self.node_status_var.set(status.get("state", "unknown"))
        self.base_version_var.set(str(status.get("base_model_version", "-")))
        self.adapter_name_var.set(status.get("adapter_name", "-"))
        self.quantization_var.set(status.get("quantization", "-"))
        self.flash_var.set(str(status.get("use_flash_attn", False)))
        self.compile_var.set(str(status.get("compile_model", False)))
        self.blocks_processed_var.set(str(status.get("blocks_processed", 0)))

        # Assigned blocks for this node.
        for row in self.blocks_tree.get_children():
            self.blocks_tree.delete(row)
        for blk in status.get("blocks", []):
            self.blocks_tree.insert(
                "",
                "end",
                values=(
                    blk.get("block_id", ""),
                    blk.get("status", ""),
                    blk.get("loss", ""),
                    blk.get("updated_at", ""),
                ),
            )

    # Queue status -------------------------------------------------------
    def refresh_queue_status(self, background=False):
        def _task():
            try:
                url = self.trainer_url_var.get().rstrip("/") + QUEUE_STATUS_ENDPOINT
                resp = requests.get(url, timeout=5)
                resp.raise_for_status()
                data = resp.json()
                self._update_queue_from_json(data)
            except Exception as e:
                if not background:
                    messagebox.showerror("Queue Error", f"Failed to fetch queue status:\n{e}")
                self._log(f"Queue status fetch failed: {e}")

        threading.Thread(target=_task, daemon=True).start()

    def _update_queue_from_json(self, data):
        self.queue_total_var.set(str(data.get("total", 0)))
        self.queue_pending_var.set(str(data.get("pending", 0)))
        self.queue_assigned_var.set(str(data.get("assigned", 0)))
        self.queue_completed_var.set(str(data.get("completed", 0)))
        self.queue_failed_var.set(str(data.get("failed", 0)))

        for row in self.queue_tree.get_children():
            self.queue_tree.delete(row)
        for blk in data.get("blocks", []):
            self.queue_tree.insert(
                "",
                "end",
                values=(
                    blk.get("block_id", ""),
                    blk.get("task", ""),
                    blk.get("status", ""),
                    blk.get("assigned_to", ""),
                    blk.get("updated_at", ""),
                ),
            )

    def enqueue_blocks(self):
        """
        Enqueue the blocks file produced by run_block_builder into the trainer's global queue.
        Trainer-side /enqueue_blocks should accept:
        {
          "dataset_label": "...",
          "blocks_path": "...",
          "target_adapter": "optional"
        }
        """
        def _task():
            payload = {
                "dataset_label": self.dataset_label_var.get(),
                "blocks_path": self.output_path_var.get(),
                "target_adapter": self.block_adapter_var.get() or None,
            }
            self._log(f"Enqueuing blocks to trainer: {payload}")
            try:
                url = self.trainer_url_var.get().rstrip("/") + ENQUEUE_BLOCKS_ENDPOINT
                resp = requests.post(url, json=payload, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                msg = data.get("message", "Blocks enqueued.")
                self._log(msg)
                messagebox.showinfo("Enqueue Blocks", msg)
                # Refresh queue after enqueue
                self.refresh_queue_status(background=True)
            except Exception as e:
                self._log(f"Enqueue failed: {e}")
                messagebox.showerror("Enqueue Error", f"{e}")

        threading.Thread(target=_task, daemon=True).start()

    # Cluster status -----------------------------------------------------
    def refresh_cluster_status(self, background=False):
        def _task():
            try:
                url = self.trainer_url_var.get().rstrip("/") + CLUSTER_STATUS_ENDPOINT
                resp = requests.get(url, timeout=5)
                resp.raise_for_status()
                data = resp.json()
                self._update_cluster_from_json(data)
            except Exception as e:
                if not background:
                    messagebox.showerror("Cluster Error", f"Failed to fetch cluster status:\n{e}")
                self._log(f"Cluster status fetch failed: {e}")

        threading.Thread(target=_task, daemon=True).start()

    def _update_cluster_from_json(self, data):
        for row in self.cluster_tree.get_children():
            self.cluster_tree.delete(row)
        for node in data.get("nodes", []):
            self.cluster_tree.insert(
                "",
                "end",
                values=(
                    node.get("node_id", ""),
                    node.get("state", ""),
                    node.get("blocks_completed", 0),
                    node.get("trust_score", 0.0),
                    node.get("last_seen", ""),
                ),
            )

    def _get_selected_cluster_node_id(self):
        sel = self.cluster_tree.selection()
        if not sel:
            messagebox.showinfo("Cluster", "Select a node first.")
            return None
        item = self.cluster_tree.item(sel[0])
        values = item.get("values", [])
        if not values:
            return None
        return values[0]

    def disable_selected_node(self):
        node_id = self._get_selected_cluster_node_id()
        if not node_id:
            return

        def _task():
            self._log(f"Disabling node {node_id}...")
            try:
                url = self.trainer_url_var.get().rstrip("/") + DISABLE_NODE_ENDPOINT
                resp = requests.post(url, json={"node_id": node_id}, timeout=5)
                resp.raise_for_status()
                data = resp.json()
                msg = data.get("message", f"Node {node_id} disabled.")
                self._log(msg)
                messagebox.showinfo("Disable Node", msg)
                self.refresh_cluster_status(background=True)
            except Exception as e:
                self._log(f"Disable node failed: {e}")
                messagebox.showerror("Disable Node Error", f"{e}")

        threading.Thread(target=_task, daemon=True).start()

    def enable_selected_node(self):
        node_id = self._get_selected_cluster_node_id()
        if not node_id:
            return

        def _task():
            self._log(f"Enabling node {node_id}...")
            try:
                url = self.trainer_url_var.get().rstrip("/") + ENABLE_NODE_ENDPOINT
                resp = requests.post(url, json={"node_id": node_id}, timeout=5)
                resp.raise_for_status()
                data = resp.json()
                msg = data.get("message", f"Node {node_id} enabled.")
                self._log(msg)
                messagebox.showinfo("Enable Node", msg)
                self.refresh_cluster_status(background=True)
            except Exception as e:
                self._log(f"Enable node failed: {e}")
                messagebox.showerror("Enable Node Error", f"{e}")

        threading.Thread(target=_task, daemon=True).start()

    # Starter block test -------------------------------------------------
    def run_starter_test(self):
        def _task():
            self._log("Running starter-block smoke test...")
            try:
                url = self.trainer_url_var.get().rstrip("/") + STARTER_TEST_ENDPOINT
                resp = requests.post(url, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                ok = data.get("ok", False)
                msg = data.get("message", "no message")
                if ok:
                    self._log(f"Starter test OK: {msg}")
                    messagebox.showinfo("Starter Test", f"Success:\n{msg}")
                else:
                    self._log(f"Starter test FAILED: {msg}")
                    messagebox.showwarning("Starter Test", f"FAILED:\n{msg}")
            except Exception as e:
                self._log(f"Starter test error: {e}")
                messagebox.showerror("Starter Test Error", f"Error:\n{e}")

        threading.Thread(target=_task, daemon=True).start()

    # Logging ------------------------------------------------------------
    def _log(self, msg: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_queue.put(f"[{timestamp}] {msg}")

    def _poll_log_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self._append_log(msg)
        except queue.Empty:
            pass
        self.after(250, self._poll_log_queue)

    def _append_log(self, msg: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    # Block builder ------------------------------------------------------
    def run_block_builder(self):
        def _task():
            cmd = [
                "python",
                "data/make_experience_blocks.py",
                "--input-path",
                self.input_path_var.get(),
                "--output",
                self.output_path_var.get(),
            ]
            self._log(f"Building blocks: {' '.join(cmd)}")
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
                out = proc.stdout.strip() or "Block build complete."
                self._log(out)
                messagebox.showinfo("Block Builder", out)
            except subprocess.CalledProcessError as e:
                err = e.stderr or str(e)
                self._log(f"Block build failed: {err}")
                messagebox.showerror("Block Build Error", err)

        threading.Thread(target=_task, daemon=True).start()

    # File picker -------------------------------------------------------
    def browse_dataset(self):
        path = filedialog.askopenfilename(
            title="Select dataset file",
            filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")],
        )
        if path:
            self.input_path_var.set(path)

    # Live inference -----------------------------------------------------
    def run_live_inference(self):
        def _task():
            payload = {
                "persona": self.test_persona.get(),
                "context": self.test_context.get(),
                "state": self.test_state.get(),
                "player_input": self.test_player.get(),
                "audience": self.test_audience.get() if hasattr(self, "test_audience") else "adult",
            }
            if self.test_adapter_name.get():
                payload["adapter_name"] = self.test_adapter_name.get()
            try:
                url = self.inference_url_var.get().rstrip("/")
                start = time.time()
                resp = requests.post(url, json=payload, timeout=20)
                elapsed = (time.time() - start) * 1000.0
                resp.raise_for_status()
                data = resp.json()
                self._log(f"Inference response ({elapsed:.1f} ms): {data}")
                self._append_infer_output(f"Latency: {elapsed:.1f} ms\n{data}\n")
            except Exception as e:
                self._log(f"Inference error: {e}")
                messagebox.showerror("Inference Error", f"{e}")

        threading.Thread(target=_task, daemon=True).start()

    def _append_infer_output(self, msg: str):
        if not hasattr(self, "infer_output"):
            return
        self.infer_output.configure(state="normal")
        self.infer_output.insert("end", msg + "\n")
        self.infer_output.see("end")
        self.infer_output.configure(state="disabled")


def main():
    app = CommandStation()
    app.mainloop()


if __name__ == "__main__":
    main()
