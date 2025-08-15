# src/metadata_processor.py
import os
import json
from pathlib import Path

class MetadataProcessor:
    def __init__(self, config):
        self.base_input_folder = Path(config.get("base_input_folder", "./../example_docs/"))
        self.output_folder = Path(config.get("output_folder", "./../results/pipeline/"))

        input_folders = config.get("input_folders", {})
        self.metadata_folder = self.base_input_folder / input_folders.get("metadata", "semanticscholar_metadata")
        self.abstracts_folder = self.base_input_folder / input_folders.get("abstracts", "semanticscholar_abstracts")
        self.licenses_folder = self.base_input_folder / input_folders.get("licenses", "licences")

        self.license_sources = config.get("license_sources", ["unpaywall", "crossref", "openalex"])
        self.license_folders = {
            source: self.licenses_folder / source for source in self.license_sources
        }

        self.output_folder.mkdir(parents=True, exist_ok=True)

    def load_license(self, source_folder, file_name):
        license_file = source_folder / file_name
        if license_file.exists():
            with open(license_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def process(self):
        for input_file in self.metadata_folder.glob("*.json"):
            with open(input_file, "r", encoding="utf-8") as f:
                metadata_content = json.load(f)

            corpus_id = input_file.stem

            # Load abstract
            abstract_file = self.abstracts_folder / input_file.name
            abstract = ""
            if abstract_file.exists():
                with open(abstract_file, "r", encoding="utf-8") as f:
                    abstract_content = json.load(f)
                    abstract = abstract_content.get("abstract", "")

            # Load licenses
            licenses = {
                f"{source}_license": self.load_license(folder, input_file.name)
                for source, folder in self.license_folders.items()
            }

            output_data = {
                "schema_version": "1.0",
                "corpus_id": corpus_id,
                "metadata": metadata_content,
                "abstract": abstract,
                "fulltext": "",
                "paragraphs": {},
                "embeddings": {},
                **licenses,
                "tldr": "",
                "abstract_embedding": [],
                "predicted_subfield": ""
            }

            output_file = self.output_folder / input_file.name
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

        print("Processing complete!")
