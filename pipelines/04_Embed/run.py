import argparse
import yaml
from src.EmbeddingProcessor import EmbeddingProcessor
from src.AbstractEmbeddingProcessor import AbstractEmbeddingProcessor

def main():
    parser = argparse.ArgumentParser(description="Run the selected embedding processor.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML config file (default: config.yaml)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["abstract", "paragraph", "both"],
        required=True,
        help="Which embedding processor to run: 'abstract', 'paragraph', or 'both'"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.mode in ["paragraph", "both"]:
        print("\n🧠 Running Paragraph Embedding Processor...\n")
        para_config = config.get("PARAGRAPH_EMBEDDINGS", {})
        paragraph_processor = EmbeddingProcessor(
            input_folder=para_config.get("input_folder", "data/paragraphs"),
            output_folder=para_config.get("output_folder", "data/embeddings/paragraphs"),
            model_name=para_config.get("model_name", "intfloat/e5-large-v2"),
            batch_size=para_config.get("batch_size", 32),
            normalize_embeddings=para_config.get("normalize_embeddings", True),
            min_file_size_kb=para_config.get("min_file_size_kb", 5),
        )
        paragraph_processor.process_files()

    if args.mode in ["abstract", "both"]:
        print("\n🧠 Running Abstract Embedding Processor...\n")
        abs_config = config.get("ABSTRACT_EMBEDDINGS", {})
        abstract_processor = AbstractEmbeddingProcessor(
            input_folder=abs_config.get("input_folder", "data/json_with_abstracts"),
            output_folder=abs_config.get("output_folder", "data/embeddings/abstracts"),
            model_name=abs_config.get("model_name", "intfloat/e5-large-v2"),
            normalize_embeddings=abs_config.get("normalize_embeddings", True),
            min_file_size_kb=abs_config.get("min_file_size_kb", 1),
        )
        abstract_processor.process_files()

if __name__ == "__main__":
    main()
