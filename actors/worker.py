"""
Dummy worker that requests tasks from the trainer and submits synthetic updates.
"""
import argparse
import random
import time
import uuid

import requests


def parse_args():
    p = argparse.ArgumentParser(description="Dummy worker to exercise trainer endpoints.")
    p.add_argument("--trainer-url", default="http://localhost:5001")
    p.add_argument("--num-tasks", type=int, default=3)
    p.add_argument("--sleep", type=float, default=0.5, help="Sleep between tasks (seconds).")
    return p.parse_args()


def main():
    args = parse_args()
    for _ in range(args.num_tasks):
        r = requests.get(f"{args.trainer_url}/get_task", timeout=5)
        r.raise_for_status()
        task = r.json()
        task_id = task["task_id"]
        model_version = task["model_version"]
        delta_scale = random.uniform(0.5, 1.5)
        payload = {
            "task_id": task_id,
            "base_model_version": model_version,
            "delta_scale": delta_scale,
        }
        resp = requests.post(f"{args.trainer_url}/submit_update", json=payload, timeout=5)
        resp.raise_for_status()
        print(f"Submitted update for task {task_id}, base_version {model_version}, delta {delta_scale}")
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
