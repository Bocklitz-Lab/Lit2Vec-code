# src/validation_comparator.py

import json
from rich import print
from rich.console import Console

console = Console()


class ValidationComparator:
    def __init__(self, ground_truth_path, test_path):
        self.gt_path = ground_truth_path
        self.test_path = test_path

    def load_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def extract_hashes(self, data):
        """Extracts field -> hash mapping from record_hash section."""
        try:
            return {
                field: info["hash"]
                for field, info in data["validation_info"]["checks"]["record_hash"]["field_hashes"].items()
                if "hash" in info
            }
        except KeyError as e:
            print(f"[bold red]❌ Missing expected structure: {e}[/bold red]")
            return {}

    def compare(self):
        gt = self.load_json(self.gt_path)
        test = self.load_json(self.test_path)

        gt_hashes = self.extract_hashes(gt)
        test_hashes = self.extract_hashes(test)

        all_fields = sorted(set(gt_hashes) | set(test_hashes))
        mismatches = []

        for field in all_fields:
            gt_val = gt_hashes.get(field)
            test_val = test_hashes.get(field)
            if gt_val != test_val:
                mismatches.append((field, gt_val, test_val))

        if mismatches:
            print(f"[bold red]❌ Hash mismatch: {self.test_path.name}[/bold red]")
            for field, gt_val, test_val in mismatches:
                print(f"[yellow]- {field}[/yellow]:")
                print(f"  🟥 expected: {gt_val}")
                print(f"  🟦 found:    {test_val}")
            return False

        print(f"[bold green]✅ Hashes match: {self.test_path.name}[/bold green]")
        return True
