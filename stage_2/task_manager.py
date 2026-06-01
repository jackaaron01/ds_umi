#!/usr/bin/env python3
"""
Task manager for Stage 2 data collection.

Loads task definitions from YAML, provides task lookup by index, and
exports to LeRobot-compatible tasks.parquet format.

Usage:
    from stage_2.task_manager import TaskManager

    tm = TaskManager("tasks/example_tasks.yaml")
    task = tm.get_task(0)
    print(task["description"])

    # Export for LeRobot dataset
    tm.export_tasks_parquet("/path/to/dataset/meta/tasks.parquet")
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import yaml
import pandas as pd


@dataclass
class TaskDefinition:
    index: int
    name: str
    category: str
    description: str
    objects: list = field(default_factory=list)
    tolerance_mm: float = 20.0


class TaskManager:
    """Manages task definitions for teleoperation data collection."""

    def __init__(self, yaml_path: str = None):
        self._tasks: dict = {}  # index → TaskDefinition
        self._default_index: int = 0
        if yaml_path is not None:
            self.load(yaml_path)

    def load(self, yaml_path: str):
        """Load task definitions from a YAML file."""
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        self._tasks.clear()
        for t in data.get("tasks", []):
            td = TaskDefinition(
                index=t["index"],
                name=t["name"],
                category=t["category"],
                description=t["description"],
                objects=t.get("objects", []),
                tolerance_mm=t.get("tolerance_mm", 20.0),
            )
            self._tasks[td.index] = td
        self._default_index = data.get("default_task_index", 0)

    @staticmethod
    def default():
        """Factory: create a TaskManager with a built-in minimal task set."""
        tm = TaskManager()
        tm._tasks = {
            0: TaskDefinition(0, "free_teleop", "freeform", "Freeform teleoperation (no specific task)", [], 50),
            1: TaskDefinition(1, "pick_place", "pick_and_place", "Pick and place object", [], 20),
            2: TaskDefinition(2, "peg_insert", "peg_insertion", "Insert peg into hole", ["peg", "hole"], 2),
        }
        return tm

    @property
    def default_task_index(self) -> int:
        return self._default_index

    @property
    def num_tasks(self) -> int:
        return len(self._tasks)

    def get_task(self, index: int) -> Optional[TaskDefinition]:
        return self._tasks.get(index)

    def list_tasks(self) -> list:
        """Return all tasks sorted by index."""
        return [self._tasks[i] for i in sorted(self._tasks)]

    def to_dataframe(self) -> pd.DataFrame:
        """Export tasks as a DataFrame (LeRobot tasks.parquet format)."""
        rows = []
        for idx in sorted(self._tasks):
            t = self._tasks[idx]
            rows.append({"task_index": t.index, "task": t.description})
        return pd.DataFrame(rows)

    def export_tasks_parquet(self, path: str):
        """Write tasks.parquet compatible with LeRobot v3.0."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)

    def print_task_list(self):
        """Pretty-print the task catalog for operator reference."""
        header = f"{'Idx':>3}  {'Category':<18} {'Name':<28} {'Description'}"
        print(header)
        print("-" * len(header))
        for t in self.list_tasks():
            print(f"{t.index:>3}  {t.category:<18} {t.name:<28} {t.description}")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Task manager for UMI data collection")
    parser.add_argument("--tasks", "-t", default=None, help="Path to tasks YAML file")
    parser.add_argument("--list", action="store_true", help="List all tasks")
    parser.add_argument("--export", default=None, help="Export tasks.parquet to this path")
    args = parser.parse_args()

    tm = TaskManager(args.tasks) if args.tasks else TaskManager.default()

    if args.list:
        tm.print_task_list()

    if args.export:
        tm.export_tasks_parquet(args.export)
        print(f"Exported {tm.num_tasks} tasks to {args.export}")

    if not args.list and args.export is None:
        tm.print_task_list()


if __name__ == "__main__":
    main()
