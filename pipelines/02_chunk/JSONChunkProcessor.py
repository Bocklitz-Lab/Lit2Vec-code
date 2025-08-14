import os
import json
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count
import sys

sys.path.append(os.getcwd())

from recursive_token_chunker import RecursiveTokenChunker
from transformers import AutoTokenizer


class JSONChunkProcessor:
    def __init__(self,
                 input_folder=".",
                 output_folder=".",
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
        """Returns the number of tokens in a given text."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def merge_small_chunks(self, chunks):
        """Merges chunks if their token count is below the threshold."""
        merged_chunks = []
        buffer_chunk = ""

        for chunk in chunks:
            if buffer_chunk:
                buffer_chunk += " " + chunk
                token_count = self.count_tokens(buffer_chunk)

                if token_count >= self.TOKEN_THRESHOLD:
                    merged_chunks.append(buffer_chunk)
                    buffer_chunk = ""
            else:
                token_count = self.count_tokens(chunk)
                if token_count < self.TOKEN_THRESHOLD:
                    buffer_chunk = chunk
                else:
                    merged_chunks.append(chunk)

        if buffer_chunk:
            merged_chunks.append(buffer_chunk)

        return merged_chunks

    def process_file(self, file):
        """Processes a single JSON file: chunks its fulltext, writes paragraphs."""
        input_path = os.path.join(self.INPUT_FOLDER, file)
        output_path = os.path.join(self.OUTPUT_FOLDER, file)  # keep same filename

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        fulltext = data.get("fulltext", "").strip()
        if not fulltext:
            return f"Skipping {file}: no fulltext found."

        chunks = self.chunker.split_text(fulltext)
        merged_chunks = self.merge_small_chunks(chunks)

        # Build paragraphs dict
        paragraphs = {}
        file_stem = os.path.splitext(file)[0]  # removes .json

        for idx, chunk in enumerate(merged_chunks):
            chunk_key = f"{file_stem}P{idx}"
            paragraphs[chunk_key] = chunk

        data["paragraphs"] = paragraphs

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        return f"Processed {file}: {len(merged_chunks)} paragraphs."

    def run(self):
        """Main function to run parallel processing."""
        input_files = [f for f in os.listdir(self.INPUT_FOLDER) if f.endswith(".json")]
        print(f"Found {len(input_files)} JSON files to process.")

        num_workers = min(48, cpu_count() - 4)

        with ThreadPool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap_unordered(self.process_file, input_files),
                                total=len(input_files),
                                desc="Processing files"))

        print("\n".join(results))
        print("Processing complete!")

