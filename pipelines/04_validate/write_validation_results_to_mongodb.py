import os
import json
from glob import glob
from pymongo import MongoClient, ASCENDING
from pymongo.errors import BulkWriteError
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# -------------------- CONFIG --------------------
JSON_DIR = "/mnt/d/data/validation_results_license_updated/"
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "validation_db"
COLLECTION_NAME = "document_validations"
NUM_WORKERS = 32

# ------------------------------------------------

def get_file_id(filepath):
    """Extract numeric ID from filename like '14631137.json' -> 14631137"""
    return int(os.path.splitext(os.path.basename(filepath))[0])

def load_json_and_prepare_doc(filepath):
    """Read JSON and prepare for MongoDB insertion."""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
            if isinstance(data, dict) and "validation_info" in data:
                data["_id"] = get_file_id(filepath)
                return data
            else:
                return {"_error": "Missing or malformed validation_info", "file": filepath}
    except Exception as e:
        return {"_error": str(e), "file": filepath}

def is_valid_mongo_document(doc):
    """Reject documents with invalid field names (dots, $) recursively."""
    def check_keys(d):
        if isinstance(d, dict):
            for key, value in d.items():
                if key.startswith("$") or "." in key:
                    return False
                if not check_keys(value):
                    return False
        elif isinstance(d, list):
            for item in d:
                if not check_keys(item):
                    return False
        return True

    return isinstance(doc, dict) and check_keys(doc)

from pymongo import ReplaceOne

BATCH_SIZE = 1000

def bulk_insert_mongodb(docs):
    """Insert documents into MongoDB using safe bulk upsert."""
    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]

    skipped = 0
    inserted_total = 0
    batch = []

    def flush_batch():
        nonlocal inserted_total, batch
        if not batch:
            return
        try:
            result = collection.bulk_write(batch, ordered=False)
            inserted_total += result.upserted_count + result.modified_count
        except BulkWriteError as bwe:
            print(f"[ERROR] Bulk write failed with {len(bwe.details.get('writeErrors', []))} writeErrors.")
        finally:
            batch = []

    for doc in docs:
        if "_error" in doc:
            print(f"[WARN] Failed to parse {doc.get('file')}: {doc.get('_error')}")
            skipped += 1
            continue
        if not isinstance(doc, dict) or "_id" not in doc:
            print(f"[WARN] Skipping malformed doc: {doc}")
            skipped += 1
            continue
        if not is_valid_mongo_document(doc):
            print(f"[WARN] Skipping doc with invalid MongoDB keys: _id={doc.get('_id')}")
            skipped += 1
            continue

        batch.append(ReplaceOne(
            filter={"_id": doc["_id"]},
            replacement=doc,
            upsert=True
        ))

        if len(batch) >= BATCH_SIZE:
            flush_batch()

    flush_batch()
    client.close()

    print(f"✅ {inserted_total} documents inserted/updated.")
    if skipped:
        print(f"⚠️  {skipped} documents skipped due to errors or invalid structure.")
    return inserted_total



def parallel_ingest(json_dir, num_workers=32):
    json_files = glob(os.path.join(json_dir, "*.json"))
    total_files = len(json_files)
    print(f"📄 Found {total_files} JSON files")

    docs = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(load_json_and_prepare_doc, f): f for f in json_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="🔄 Reading JSON"):
            doc = future.result()
            docs.append(doc)

    print(f"✅ Parsed {len(docs)} documents. Inserting to MongoDB...")
    inserted_count = bulk_insert_mongodb(docs)
    print(f"✅ Inserted {inserted_count} documents into MongoDB.")

def create_indexes():
    print("⚙️  Creating indexes...")
    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]

    index_fields = [
        "validation_info.overall_status",
        "validation_info.checks.schema_validation.status",
        "validation_info.checks.embedding_validation.status",
        "validation_info.checks.subfield_validation.status",
        "validation_info.checks.metadata_validation.metrics.warning_count",
        "validation_info.checks.text_validation.flags.abstract_embedding_missing_or_invalid",
        "validation_info.checks.chunk_validation.metrics.chunks_too_long",
        "validation_info.summary_evaluation.ROUGE-1",
    ]

    for field in index_fields:
        try:
            collection.create_index([(field, ASCENDING)])
            print(f"✅ Indexed: {field}")
        except Exception as e:
            print(f"[ERROR] Indexing {field} failed: {e}")

    client.close()
    print("✅ Indexing complete.")

def print_summary():
    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]
    print("\n📊 Status Summary:")
    pipeline = [
        {"$group": {"_id": "$validation_info.overall_status", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    try:
        for row in collection.aggregate(pipeline):
            print(f"  {row['_id']}: {row['count']}")
    except Exception as e:
        print(f"[ERROR] Summary query failed: {e}")
    finally:
        client.close()

def main():
    parallel_ingest(JSON_DIR, num_workers=NUM_WORKERS)
    create_indexes()
    print_summary()

if __name__ == "__main__":
    main()
