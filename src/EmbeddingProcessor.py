import os
import json
import time
import torch
import pynvml
import random
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingProcessor:
    def __init__(
        self,
        input_folder,
        output_folder,
        model_name="intfloat/e5-large-v2",
        batch_size=32,
        normalize_embeddings=True,
        min_file_size_kb=5
    ):
        # Fix seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Initialize NVIDIA Management Library
        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        # Load the embedding model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.eval()

        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.MIN_FILE_SIZE = min_file_size_kb * 1024  # Convert KB to bytes

        self.input_folder = input_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

        self.processed_count = 0

    def get_gpu_temperature(self):
        return pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)

    def process_files(self):
        for filename in os.listdir(self.input_folder):
            if not filename.endswith(".json"):
                continue

            input_path = os.path.join(self.input_folder, filename)
            output_path = os.path.join(self.output_folder, filename)

            if os.path.getsize(input_path) < self.MIN_FILE_SIZE:
                print(f"Skipping {filename} (size < {self.MIN_FILE_SIZE / 1024:.1f}KB)")
                continue

            try:
                with open(input_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                paragraphs = data.get("paragraphs", {})
                if not paragraphs:
                    print(f"Skipping {filename} (no paragraphs)")
                    continue

                keys = list(paragraphs.keys())
                texts = [f"passage: {paragraphs[k]}" for k in keys]

                embeddings = self.model.encode(
                    texts,
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=self.normalize_embeddings
                )

                embeddings_dict = {k: emb.tolist() for k, emb in zip(keys, embeddings)}
                data["embeddings"] = embeddings_dict

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)

                self.processed_count += 1
                print(f"Processed: {filename} (Total: {self.processed_count})")

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

            gpu_temp = self.get_gpu_temperature()
            print(f"GPU Temperature: {gpu_temp}°C")

            if gpu_temp > 85:
                print("GPU too hot! Cooling down...")
                while self.get_gpu_temperature() > 70:
                    print(f"Waiting... GPU Temp: {self.get_gpu_temperature()}°C")
                    time.sleep(10)

        pynvml.nvmlShutdown()
        print("All files processed.")
