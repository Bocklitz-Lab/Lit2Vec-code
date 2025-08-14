# run.py
import argparse
import yaml
from src import MetadataProcessor  # Assuming you'll place the class in src/metadata_processor.py

def main():
    parser = argparse.ArgumentParser(description="Run metadata processing pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)"
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    mp_config = config.get("METADATA_PROCESSOR", {})
    processor = MetadataProcessor(config=mp_config)
    processor.process()

if __name__ == "__main__":
    main()
