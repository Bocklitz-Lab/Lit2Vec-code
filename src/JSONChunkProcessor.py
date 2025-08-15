# src/json_chunk_processor.py

import os
import json
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count
from transformers import AutoTokenizer
from src.recursive_token_chunker import RecursiveTokenChunker


class JSONChunkProcessor:
    def __init__(self,
                 input_folder="data/processed_json",
                 output_folder="data/chunked_json",
                 chunk_size=200,
                 chunk_overlap=20,
                 token_threshold=100,
                 tokenizer_name="intfloat/e5-large"):
        self.INPUT_FOLDER = input_folder
        self.OUTPUT_FOLDER = output_folder
        self.CHUNK_SIZE = chunk_size
        self.CHUNK_OVERLAP = chunk_overlap
        self.TOKEN_THRESHOLD = token_threshold

        os.makedirs(self.OUTPUT_FOLDER, exist_ok=True)

        self.chunker = RecursiveTokenChunker(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP
        )

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def merge_small_chunks(self, chunks):
        merged_chunks = []
        buffer_chunk = ""

        for chunk in chunks:
            if buffer_chunk:
                buffer_chunk += " " + chunk
                if self.count_tokens(buffer_chunk) >= self.TOKEN_THRESHOLD:
                    merged_chunks.append(buffer_chunk)
                    buffer_chunk = ""
            else:
                if self.count_tokens(chunk) < self.TOKEN_THRESHOLD:
                    buffer_chunk = chunk
                else:
                    merged_chunks.append(chunk)

        if buffer_chunk:
            merged_chunks.append(buffer_chunk)

        return merged_chunks

    def process_file(self, file):
        input_path = os.path.join(self.INPUT_FOLDER, file)
        output_path = os.path.join(self.OUTPUT_FOLDER, file)

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        fulltext = data.get("fulltext", "").strip()
        if not fulltext:
            return f"Skipping {file}: no fulltext found."

        chunks = self.chunker.split_text(fulltext)
        merged_chunks = self.merge_small_chunks(chunks)

        paragraphs = {}
        file_stem = os.path.splitext(file)[0]

        for idx, chunk in enumerate(merged_chunks):
            paragraphs[f"{file_stem}P{idx}"] = chunk

        data["paragraphs"] = paragraphs

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        return f"Processed {file}: {len(merged_chunks)} paragraphs."

    def run(self):
        input_files = [f for f in os.listdir(self.INPUT_FOLDER) if f.endswith(".json")]
        print(f"Found {len(input_files)} JSON files to process.")

        num_workers = min(48, cpu_count() - 4)

        with ThreadPool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap_unordered(self.process_file, input_files),
                                total=len(input_files),
                                desc="Processing files"))

        print("\n".join(results))
        print("Processing complete!")
