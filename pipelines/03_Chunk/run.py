# run.py

import argparse
import yaml
from src import JSONChunkProcessor  # Adjust this import if needed

def main():
    parser = argparse.ArgumentParser(description="Run JSONChunkProcessor with config file.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML config file (default: config.yaml)"
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    processor_config = config.get("JSONChunkProcessor", {})

    processor = JSONChunkProcessor(
        input_folder=processor_config.get("input_folder", "data/processed_json"),
        output_folder=processor_config.get("output_folder", "data/chunked_json"),
        chunk_size=processor_config.get("chunk_size", 200),
        chunk_overlap=processor_config.get("chunk_overlap", 20),
        token_threshold=processor_config.get("token_threshold", 100),
        tokenizer_name=processor_config.get("tokenizer_name", "intfloat/e5-large")
    )

    processor.run()

if __name__ == "__main__":
    main()
