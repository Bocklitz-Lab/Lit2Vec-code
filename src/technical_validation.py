# src/ValidatorPipeline.py

import os
import glob
import json
import gc
import torch
from datetime import datetime
from tqdm import tqdm
import logging

from src.MetadataValidator import MetadataValidator
from src.LicenseValidator import LicenseValidator
from src.ChunkValidator import ChunkValidator
from src.EmbeddingValidator import EmbeddingValidator
from src.RecordHasher import RecordHasher
from src.SummaryEvaluator import SummaryEvaluator
from src.SchemaValidator import SchemaValidator
from src.TextValidator import TextValidator
from src.SubfieldValidator import SubfieldValidator
from src.IDValidator import IDValidator

logging.getLogger("absl").setLevel(logging.WARNING)

class ValidatorPipeline:
    def __init__(self, config: dict):
        self.input_folder = config["input_folder"]
        self.output_folder = config["output_folder"]
        self.chunk_size = config.get("chunk_size", 1000)
        self.hash_exclude_keys = config.get("hash_exclude_keys", [])
        self.summary_config = config.get("summary_evaluator", {})
        self.schema_config = config.get("schema_validator", {})
        self.validator_version = config.get("validator_version", "unknown")
        self.schema_version = config.get("schema_version", "unknown")

        os.makedirs(self.output_folder, exist_ok=True)

        self._init_validators()

    def _init_validators(self):
        self.metadata_validator = MetadataValidator()
        self.license_validator = LicenseValidator()
        self.chunk_validator = ChunkValidator()
        self.embedding_validator = EmbeddingValidator()
        self.hasher = RecordHasher(exclude_keys=self.hash_exclude_keys)
        self.summary_evaluator = SummaryEvaluator(
            model_type=self.summary_config.get("model_type", "bert-base-uncased"),
            backend=self.summary_config.get("backend", "evaluate")
        )
        self.schema_validator = SchemaValidator(**self.schema_config)
        self.text_validator = TextValidator()
        self.subfield_validator = SubfieldValidator()
        self.id_validator = IDValidator()

    def _chunked(self, iterable, size):
        for i in range(0, len(iterable), size):
            yield iterable[i:i + size]

    def _determine_overall_status(self, checks: dict) -> str:
        for result in checks.values():
            if isinstance(result, dict) and result.get("status") == "fail":
                return "fail"
            if isinstance(result, list):
                if any(r.get("status") == "fail" for r in result if isinstance(r, dict)):
                    return "fail"
        return "pass"

    def run(self):
        all_json_paths = sorted(glob.glob(os.path.join(self.input_folder, "*.json")))
        all_output_paths = {
            os.path.basename(p): os.path.join(self.output_folder, os.path.basename(p))
            for p in all_json_paths
        }
        remaining_paths = [
            p for p in all_json_paths if not os.path.exists(all_output_paths[os.path.basename(p)])
        ]

        print(f"📄 Found {len(all_json_paths)} JSON files.")
        print(f"✅ {len(all_json_paths) - len(remaining_paths)} already processed.")
        print(f"🚧 {len(remaining_paths)} files will be processed.")

        with tqdm(total=len(remaining_paths), desc="Validating files", unit="file") as progress_bar:
            for batch_idx, path_chunk in enumerate(self._chunked(remaining_paths, self.chunk_size)):
                records, filenames = self._load_records(path_chunk)
                if not records:
                    print(f"⚠️ No valid records in chunk {batch_idx + 1}, skipping.")
                    continue

                self._validate_and_save(records, filenames, progress_bar)

    def _load_records(self, path_chunk):
        records = []
        filenames = []
        for path in path_chunk:
            try:
                with open(path, encoding="utf-8") as f:
                    record = json.load(f)
                    records.append(record)
                    filenames.append(os.path.basename(path))
            except Exception as e:
                print(f"⚠️ Skipping {path}: {e}")
        return records, filenames

    def _validate_and_save(self, records, filenames, progress_bar):
        schema_results = self.schema_validator.validate_batch(records, parallel=True, use_threads=True)
        metadata_results = self.metadata_validator.validate_batch(records)
        license_results = self.license_validator.validate_batch(records)
        chunk_results = self.chunk_validator.validate_batch(records, use_threads=True)
        embedding_results = self.embedding_validator.validate_batch(records, n_compare=1, use_threads=True)
        hash_results = [self.hasher.hash_fields_from_dict(r) for r in records]
        text_results = self.text_validator.validate_batch(records, parallel=True)
        subfield_results = self.subfield_validator.validate_batch(records, parallel=True)

        filename_ids = [
            int(fname.split('.')[0]) if fname.split('.')[0].isdigit() else None
            for fname in filenames
        ]
        id_results = self.id_validator.validate_batch(records, filename_ids=filename_ids, parallel=True)

        tldr_pairs = [(r.get("abstract", ""), r.get("tldr", "")) for r in records]
        try:
            tldr_results = self.summary_evaluator.evaluate_batch(tldr_pairs)
        except Exception as e:
            print(f"❌ Batch summary evaluation failed: {e}")
            tldr_results = [{} for _ in records]

        for i, filename in enumerate(filenames):
            output_path = os.path.join(self.output_folder, filename)
            checks = {
                "schema_validation": schema_results[i],
                "metadata_validation": metadata_results[i],
                "license_validation": license_results[i],
                "text_validation": text_results[i],
                "chunk_validation": chunk_results[i],
                "embedding_validation": embedding_results[i],
                "record_hash": hash_results[i],
                "summary_evaluation": tldr_results[i],
                "subfield_validation": subfield_results[i],
                "id_validation": id_results[i],
            }

            overall_status = self._determine_overall_status(checks)
            combined = {
                "validation_info": {
                    "schema_version": self.schema_version,
                    "overall_status": overall_status,
                    "last_validated": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "validator_version": self.validator_version,
                    "checks": checks
                }
            }

            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(combined, f, indent=2)
                progress_bar.set_postfix(file=filename)
                progress_bar.update(1)
            except Exception as e:
                print(f"❌ Failed to write {filename}: {e}")

        del records, metadata_results, license_results, schema_results
        del chunk_results, embedding_results, hash_results
        del tldr_results, text_results, subfield_results, id_results
        gc.collect()
        torch.cuda.empty_cache()
