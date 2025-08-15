import argparse
import yaml
from src.PaperProcessor import PaperProcessor  # Ensure this path is correct

def main():
    parser = argparse.ArgumentParser(description="Run the PaperProcessor with a config file.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML config file (default: config.yaml)"
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    json2md_config = config.get("JSON2MD", {})

    input_dir = json2md_config.get("input_dir", "data/raw_json")
    output_dir = json2md_config.get("output_dir", "data/processed_json")
    common_headers = json2md_config.get("common_headers", [])
    min_words_per_section = json2md_config.get("min_words_per_section", 10)

    processor = PaperProcessor(
        input_dir=input_dir,
        output_dir=output_dir,
        common_headers=common_headers,
        min_words_per_section=min_words_per_section
    )
    processor.process_all_json_files_in_memory()

if __name__ == "__main__":
    main()
