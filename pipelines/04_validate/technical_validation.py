import os
import glob
import json
import gc
import torch
import logging
from tqdm import tqdm

from MetadataValidator import MetadataValidator
from LicenseValidator import LicenseValidator
from ChunkValidator import ChunkValidator
from EmbeddingValidator import EmbeddingValidator
from RecordHasher import RecordHasher
from SummaryEvaluator import SummaryEvaluator  # Updated batch-capable evaluator
from SchemaValidator import SchemaValidator
from TextValidator import TextValidator
from SubfieldValidator import SubfieldValidator
from IDValidator import IDValidator

# --------------------------
# Logging Configuration
# --------------------------
logging.getLogger("absl").setLevel(logging.WARNING)

# --------------------------
# Configuration
# --------------------------
input_folder = "/mnt/d/data/final_database_republishable"
output_folder = "/mnt/d/data/validation_results"
os.makedirs(output_folder, exist_ok=True)

CHUNK_SIZE = 1000

# --------------------------
# Utility: Chunk generator
# --------------------------
def chunked(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i+size]

def determine_overall_status(checks: dict) -> str:
    for key, result in checks.items():
        if isinstance(result, dict):
            if result.get("status") == "fail":
                return "fail"
        elif isinstance(result, list):  # Some validators might return a list of issues
            for r in result:
                if isinstance(r, dict) and r.get("status") == "fail":
                    return "fail"
    return "pass"

# --------------------------
# Load all input JSON paths
# --------------------------
all_json_paths = sorted(glob.glob(os.path.join(input_folder, "*.json")))
all_output_paths = {os.path.basename(p): os.path.join(output_folder, os.path.basename(p)) for p in all_json_paths}

# Filter out files that are already processed
remaining_paths = [p for p in all_json_paths if not os.path.exists(all_output_paths[os.path.basename(p)])]
print(f"📄 Found {len(all_json_paths)} JSON files.")
print(f"✅ {len(all_json_paths) - len(remaining_paths)} already processed.")
print(f"🚧 {len(remaining_paths)} files will be processed.")

# --------------------------
# Initialize validators once
# --------------------------
metadata_validator = MetadataValidator()
license_validator = LicenseValidator()
chunk_validator = ChunkValidator()
embedding_validator = EmbeddingValidator()
hasher = RecordHasher(exclude_keys=[
    "abstract_embedding", "corpus_id", "embeddings", "schema_version", "validation_info"
])
summary_evaluator = SummaryEvaluator(
    model_type="bert-base-uncased",
    backend="evaluate"
)
schema_validator = SchemaValidator(max_errors=50) 
text_validator = TextValidator()
subfield_validator = SubfieldValidator()
id_validator = IDValidator()

# --------------------------
# Global progress bar over all files
# --------------------------
with tqdm(total=len(remaining_paths), desc="Validating files", unit="file") as progress_bar:
    for batch_idx, path_chunk in enumerate(chunked(remaining_paths, CHUNK_SIZE)):
        # --------------------------
        # Load records
        # --------------------------
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

        if not records:
            print(f"⚠️ No valid records in chunk {batch_idx + 1}, skipping.")
            continue

        # --------------------------
        # Run validators
        # --------------------------
        schema_results = schema_validator.validate_batch(records, parallel=True, use_threads=True)
        metadata_results   = metadata_validator.validate_batch(records)
        license_results    = license_validator.validate_batch(records)
        chunk_results      = chunk_validator.validate_batch(records, use_threads=True)
        embedding_results  = embedding_validator.validate_batch(records, n_compare=1, use_threads=True)
        hash_results       = [hasher.hash_fields_from_dict(r) for r in records]
        text_results = text_validator.validate_batch(records, parallel=True)
        subfield_results = subfield_validator.validate_batch(records, parallel=True)
        filename_ids = [
            int(os.path.splitext(fname)[0]) if fname.split('.')[0].isdigit() else None
            for fname in filenames
        ]
        id_results = id_validator.validate_batch(records, filename_ids=filename_ids, parallel=True)

        # --------------------------
        # Batch summary evaluation
        # --------------------------
        tldr_pairs = [(r.get("abstract", ""), r.get("tldr", "")) for r in records]
        try:
            tldr_results = summary_evaluator.evaluate_batch(tldr_pairs)
        except Exception as e:
            print(f"❌ Batch summary evaluation failed: {e}")
            tldr_results = [_ZERO_SCORES.copy() for _ in records]

        # --------------------------
        # Write results per file
        # --------------------------
        for i, filename in enumerate(filenames):
            output_path = os.path.join(output_folder, filename)
            from datetime import datetime
            # Compose the checks dictionary
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

            # Compute overall status
            overall_status = determine_overall_status(checks)

            # Wrap in final structure
            combined = {
                "validation_info": {
                    "schema_version": "1.0",
                    "overall_status": overall_status,
                    "last_validated": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "validator_version": "lit2vec-qc@0.4.0",
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

        # --------------------------
        # Clean memory after batch
        # --------------------------
        del records, metadata_results, license_results, schema_results
        del chunk_results, embedding_results, hash_results, tldr_results, text_results, subfield_results,id_results
        gc.collect()
        torch.cuda.empty_cache()
