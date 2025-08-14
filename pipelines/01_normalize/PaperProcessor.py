import os
import re
import json

class PaperProcessor:
    COMMON_CHEMISTRY_HEADERS = {
        "abstract", "introduction", "methods", "experimental methods",
        "materials and methods", "results", "results and discussion",
        "discussion", "conclusion", "conclusions", "references", "supporting information"
    }

    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def process_all_json_files_in_memory(self):
        """Process all JSON files: inject fulltext into output JSON."""
        for file_name in os.listdir(self.input_folder):
            if file_name.endswith(".json"):
                input_path = os.path.join(self.input_folder, file_name)
                output_path = os.path.join(self.output_folder, file_name)

                raw_md = self.process_json_file(input_path)
                formatted_md = self.format_headers(raw_md)
                cleaned_md = self.clean_markdown(formatted_md)

                # Load the existing output JSON
                if os.path.exists(output_path):
                    with open(output_path, "r", encoding="utf-8") as f:
                        output_json = json.load(f)
                else:
                    output_json = {}

                output_json["fulltext"] = cleaned_md

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output_json, f, indent=2, ensure_ascii=False)

                print(f"✅ Processed: {input_path} → {output_path} (fulltext updated)")

    def process_json_file(self, input_path):
        """Return structured paper as Markdown string."""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        content = data.get("content", {}).get("text", "")
        annotations = self._parse_annotations(data.get("content", {}).get("annotations", {}))

        titles = self._extract_text(content, annotations, "title")
        abstracts = self._extract_text(content, annotations, "abstract")
        section_headers = self._extract_text(content, annotations, "sectionheader")
        paragraphs = self._extract_text(content, annotations, "paragraph")

        structured_paper = self._build_paper(titles, abstracts, section_headers, paragraphs)
        return "\n\n".join(structured_paper)

    def _parse_annotations(self, annotations):
        """Ensure annotations are a dict with lists."""
        if isinstance(annotations, str):
            try:
                annotations = json.loads(annotations)
            except json.JSONDecodeError:
                print("❌ ERROR: Annotations could not be converted.")
                return {}

        for key in annotations.keys():
            if isinstance(annotations[key], str):
                try:
                    annotations[key] = json.loads(annotations[key])
                except json.JSONDecodeError:
                    print(f"⚠ WARNING: '{key}' annotation could not be converted.")
                    annotations[key] = []

        return annotations

    def _extract_text(self, content, annotations, annotation_type):
        """Extract text snippets based on annotation type."""
        extracted_texts = []
        if annotation_type in annotations and isinstance(annotations[annotation_type], list):
            for item in annotations[annotation_type]:
                if isinstance(item, dict) and "start" in item and "end" in item:
                    try:
                        start = int(item["start"])
                        end = int(item["end"])
                        snippet = self._convert_to_markdown(content[start:end].strip())
                        extracted_texts.append((start, snippet))
                    except (ValueError, TypeError):
                        print(f"⚠ WARNING: Invalid start/end in {annotation_type}: {item}")
        return sorted(extracted_texts)

    def _convert_to_markdown(self, text):
        """Convert notations to Markdown."""
        text = text.replace("_", "<sub>").replace("^", "<sup>")
        return text

    def _build_paper(self, titles, abstracts, section_headers, paragraphs):
        """Construct structured paper as Markdown."""
        structured_paper = []

        if titles:
            structured_paper.append(f"# {titles[0][1]}\n")

        if abstracts:
            structured_paper.append(f"## Abstract\n{abstracts[0][1]}\n")

        last_section_index = 0
        for i, (section_pos, section_text) in enumerate(section_headers):
            structured_paper.append(f"## {section_text}\n")
            while (last_section_index < len(paragraphs) and
                   paragraphs[last_section_index][0] < (section_headers[i + 1][0] if i + 1 < len(section_headers) else float("inf"))):
                structured_paper.append(paragraphs[last_section_index][1])
                last_section_index += 1

        return structured_paper

    def format_headers(self, markdown_text, min_words=10):
        """Format headers and filter short sections."""
        lines = markdown_text.splitlines(keepends=True)
        if not lines:
            return ""

        cleaned_lines = []
        section_buffer = []
        inside_section = False
        current_header = None
        normalized_header = None

        first_line = lines[0].strip()
        if not first_line.startswith("# "):
            first_line = "# " + first_line
        cleaned_lines.append(first_line + "\n")

        for line in lines[1:]:
            header_match = re.match(r'^(#{2,})\s+(.+)', line)
            if header_match:
                if inside_section:
                    content = ''.join(section_buffer[1:]).strip()
                    word_count = len(content.split())
                    if word_count >= min_words or normalized_header in self.COMMON_CHEMISTRY_HEADERS:
                        cleaned_lines.extend(section_buffer)

                current_header = header_match.group(2).strip()
                normalized_header = current_header.lower()

                if normalized_header in self.COMMON_CHEMISTRY_HEADERS:
                    formatted_header = f"## {current_header}"
                else:
                    formatted_header = f"### {current_header}"

                section_buffer = [formatted_header + "\n"]
                inside_section = True
            else:
                section_buffer.append(line)

        if inside_section:
            content = ''.join(section_buffer[1:]).strip()
            word_count = len(content.split())
            if word_count >= min_words or normalized_header in self.COMMON_CHEMISTRY_HEADERS:
                cleaned_lines.extend(section_buffer)

        return ''.join(cleaned_lines)

    def clean_markdown(self, markdown_text):
        """Clean up line breaks inside sections."""
        lines = markdown_text.splitlines()
        cleaned_lines = []
        buffer = []

        for line in lines:
            stripped = line.strip()
            if re.match(r'^\s*#+\s+', line):
                if buffer:
                    cleaned_lines.append(" ".join(buffer))
                    buffer = []
                cleaned_lines.append(line.rstrip())
            elif stripped == "":
                if buffer:
                    cleaned_lines.append(" ".join(buffer))
                    buffer = []
            elif re.match(r'^\[\s*\]$', stripped) or re.match(r'^Equ\.\s*\d+', stripped, re.IGNORECASE):
                if buffer:
                    cleaned_lines.append(" ".join(buffer))
                    buffer = []
                cleaned_lines.append(stripped)
            else:
                buffer.append(stripped)

        if buffer:
            cleaned_lines.append(" ".join(buffer))

        return "\n".join(cleaned_lines) + "\n"
