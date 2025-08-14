import os
import json
import time
import torch
import pynvml
import random
import numpy as np
from sentence_transformers import SentenceTransformer

class AbstractEmbeddingProcessor:
    def __init__(self, input_folder, output_folder):
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
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0

        # Load the embedding model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer("intfloat/e5-large-v2", device=self.device)
        self.model.eval()

        self.input_folder = input_folder
        self.output_folder = output_folder

        os.makedirs(self.output_folder, exist_ok=True)

        self.MIN_FILE_SIZE = 1000
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
                print(f"Skipping {filename} (size < 1KB)")
                continue

            try:
                with open(input_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                abstract_text = data.get("abstract", "").strip()
                if not abstract_text:
                    print(f"Skipping {filename} (missing or empty 'abstract')")
                    continue

                # Prepare input
                abstract_input = f"passage: {abstract_text}"

                # Generate embedding
                embedding = self.model.encode(
                    abstract_input,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )

                # Add to existing JSON structure
                data["abstract_embedding"] = embedding.tolist()

                # Save updated JSON to output folder
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)

                self.processed_count += 1
                print(f"Processed: {filename} (Total: {self.processed_count})")

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

            # Optional GPU temperature guard
            gpu_temp = self.get_gpu_temperature()
            print(f"GPU Temperature: {gpu_temp}°C")

            if gpu_temp > 85:
                print("GPU temperature too high! Waiting to cool down...")
                while self.get_gpu_temperature() > 70:
                    print(f"Waiting... Current GPU Temp: {self.get_gpu_temperature()}°C")
                    time.sleep(10)

        pynvml.nvmlShutdown()
        print("✅ All eligible files processed successfully.")

