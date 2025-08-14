# run.py

import argparse
import yaml
from src import ValidatorPipeline  # <- This will come from the refactored main script

def main():
    parser = argparse.ArgumentParser(description="Run the lit2vec validator pipeline.")
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to the YAML config file"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    pipeline = ValidatorPipeline(config["VALIDATOR_CONFIG"])
    pipeline.run()

if __name__ == "__main__":
    main()
