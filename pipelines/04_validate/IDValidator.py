
# Consistency validation
# To ensure that each record in the dataset is internally consistent and correctly linked to external bibliographic resources, we implemented an automated identifier validation pipeline. This process verifies the consistency of internal corpus identifiers, cross-checks Digital Object Identifiers (DOIs) across multiple metadata sources, and ensures that textual content is properly aligned with its corresponding vector embeddings.
# For every record, the validator extracts three classes of identifiers. The first consists of internal corpus-level identifiers, including the filename_id provided at ingest, the corpus_id field, and the corpusid value stored within the metadata subfield. The second consists of DOIs, which are collected from three distinct metadata sources: the dataset’s own metadata (externalids), Unpaywall records, and OpenAlex records. The third relates to structural content alignment, specifically paragraph identifiers and embedding identifiers, which are compared to ensure that every paragraph has a matching embedding and that no embeddings exist without an associated paragraph.
# The DOI values retrieved from each source are normalized to a canonical form by removing URL prefixes and converting all characters to lowercase. This normalization allows the validator to determine whether different sources refer to the same DOI even if they use different formatting conventions. After normalization, the validator compares the DOIs across the three sources. If all provided DOIs are identical, the DOI field is considered consistent. If two or more sources provide different DOIs, the record is flagged as having a DOI mismatch. If no DOI is present in any source, the record is flagged as lacking DOI coverage altogether.
# The corpus identifier check follows a similar agreement-based approach. If the filename_id, corpus_id, and metadata.corpusid values are all present and agree, the identifiers are considered valid. If any two disagree, the record is flagged as having an identifier mismatch. A missing metadata.corpusid value is recorded separately, as it indicates incomplete metadata even if other identifiers agree.
# The structural alignment check focuses on the relationship between paragraphs and embeddings. The validator first counts the number of paragraphs and the number of embeddings. It then identifies any paragraphs that lack embeddings, as well as any embeddings that do not correspond to a known paragraph. Missing embeddings indicate incomplete vector representation of text, whereas orphan embeddings suggest indexing or preprocessing errors. For transparency, the validator records up to five example IDs for both cases.
# The status of each record is determined by a hierarchical decision process. A corpus identifier mismatch automatically results in a “fail” status, since it indicates a fundamental linking error that could compromise the integrity of the dataset. If no such mismatch is found but other problems are detected — such as missing metadata identifiers, DOI mismatches, missing embeddings, orphan embeddings, or missing DOIs in all sources — the record is given a “warn” status, signaling that it is usable but potentially incomplete or inconsistent. Only records that pass all checks without warnings are assigned a “pass” status.
# The validator operates in both single-record and batch-processing modes. For large-scale validation, multiprocessing is used to distribute work across CPU cores, ensuring scalability without sacrificing reproducibility. The output for each record includes its validation status, a set of diagnostic flags, detailed metrics on identifiers and content alignment, and a UTC timestamp indicating when the validation was performed. This structured and timestamped reporting enables reproducible re-validation, downstream filtering by status, and targeted investigation of problematic records.


from typing import Dict, List, Any
from datetime import datetime, timezone
import re


# -----------------------
# Safe multiprocessing entry point
# -----------------------
def _validate_with_args(args):
    validator, doc, filename_id = args
    return validator.validate(doc, filename_id=filename_id)


class IDValidator:
    def __init__(self):
        pass

    @staticmethod
    def _normalize_doi(doi: str | None) -> str | None:
        if not doi:
            return None
        doi = str(doi).lower().strip()
        return re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)

    def validate(self, doc: Dict[str, Any], *, filename_id: int | None = None) -> Dict[str, Any]:
        checked_at = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

        # Extract corpus IDs
        corpus_id = doc.get("corpus_id")
        metadata = doc.get("metadata", {})
        metadata_id = metadata.get("corpusid") if isinstance(metadata, dict) else None
        externalids = metadata.get("externalids") if isinstance(metadata, dict) else None

        # Extract DOIs safely
        doi_meta = self._normalize_doi(externalids.get("DOI") if isinstance(externalids, dict) else None)
        unpaywall = doc.get("unpaywall_license")
        doi_unpay = self._normalize_doi(unpaywall.get("doi") if isinstance(unpaywall, dict) else None)
        openalex = doc.get("openalex_license")
        doi_openalex = self._normalize_doi(openalex.get("doi") if isinstance(openalex, dict) else None)

        doi_sources_present = {
            "metadata_doi": doi_meta is not None,
            "unpaywall_doi": doi_unpay is not None,
            "openalex_doi": doi_openalex is not None
        }

        # Extract paragraphs and embeddings
        para_ids = set(doc.get("paragraphs", {}).keys())
        embed_ids = set(doc.get("embeddings", {}).keys())

        flags = {
            "missing_metadata_id": metadata_id is None,
            "corpusid_mismatch": False,
            "doi_mismatch": False,
            "missing_embeddings": False,
            "orphan_embeddings": False,
            "doi_missing_sources": False,
        }

        metrics = {
            "corpus_ids": {
                "filename_id": filename_id,
                "corpus_id": corpus_id,
                "metadata_id": metadata_id
            },
            "doi_set": sorted({d for d in [doi_meta, doi_unpay, doi_openalex] if d}),
            "para_count": len(para_ids),
            "embed_count": len(embed_ids),
            "missing_para_ids": [],
            "extra_embed_ids": [],
            "doi_sources_present": doi_sources_present
        }

        # ID checks
        ids = {i for i in [filename_id, corpus_id, metadata_id] if i is not None}
        if len(ids) > 1:
            flags["corpusid_mismatch"] = True

        # DOI consistency
        if len(metrics["doi_set"]) > 1:
            flags["doi_mismatch"] = True
        if not any([doi_meta, doi_unpay, doi_openalex]):
            flags["doi_missing_sources"] = True

        # Paragraph ↔ Embedding alignment
        missing_embeds = para_ids - embed_ids
        extra_embeds = embed_ids - para_ids
        if missing_embeds:
            flags["missing_embeddings"] = True
            metrics["missing_para_ids"] = sorted(list(missing_embeds))[:5]
        if extra_embeds:
            flags["orphan_embeddings"] = True
            metrics["extra_embed_ids"] = sorted(list(extra_embeds))[:5]

        # Status decision
        if flags["corpusid_mismatch"]:
            status = "fail"
        elif any([
            flags["missing_metadata_id"],
            flags["doi_mismatch"],
            flags["missing_embeddings"],
            flags["orphan_embeddings"],
            flags["doi_missing_sources"]
        ]):
            status = "warn"
        else:
            status = "pass"

        return {
            "status": status,
            "flags": flags,
            "metrics": metrics,
            "checked_at": checked_at
        }

    def validate_batch(
        self,
        docs: List[Dict[str, Any]],
        filename_ids: List[int | None] | None = None,
        *,
        parallel: bool = False,
        max_workers: int | None = None
    ) -> List[Dict[str, Any]]:
        if filename_ids is None:
            filename_ids = [None] * len(docs)
        inputs = list(zip([self] * len(docs), docs, filename_ids))

        if not parallel or len(docs) <= 1:
            return [self.validate(doc, filename_id=fid) for doc, fid in zip(docs, filename_ids)]

        import concurrent.futures as cf
        with cf.ProcessPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(_validate_with_args, inputs))
