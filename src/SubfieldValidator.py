# Subfield validation
# To ensure that predicted disciplinary subfields in the dataset are accurate, standardized, and suitable for downstream analysis, we implemented an automated subfield validation pipeline. This process addresses the common issue of inconsistent label formats, incomplete predictions, and invalid probability scores, which can arise from machine learning–based classification outputs or heterogeneous annotation sources.
# For each record, the validator examines the predicted_subfield field, which is expected to be a dictionary mapping recognized subfield labels to confidence scores between 0.0 and 1.0. The allowed subfields are drawn from a fixed controlled vocabulary covering major branches of chemistry and related disciplines, including specialized areas such as Catalysis, Supramolecular Chemistry, Forensic Chemistry, and Radiochemistry & Nuclear Chemistry. Predictions that do not match one of the allowed subfields are considered invalid and are flagged for further review.
# The validation process begins by confirming that the predicted_subfield field is present. If the field is missing entirely, the record is immediately marked as a failure. If the field is present but is not structured as a dictionary, it is similarly marked as a failure. If the field is a dictionary, the validator counts the number of predicted labels and checks whether it is empty. For each label–score pair, the label is checked against the controlled vocabulary, and the score is verified to be a floating-point value within the permissible range. Any deviation from these requirements is recorded as an error condition.
# The validator determines the overall status of each record based on the severity of the detected issues. Records with missing predictions or structural errors receive a “fail” status. Records that contain otherwise valid predictions but include one or more invalid labels or scores receive a “warn” status, indicating that the record may still be usable but requires caution. Records with no detected issues receive a “pass” status, confirming that the predicted subfield assignments conform to both semantic and numerical constraints.
# The pipeline supports both single-record and batch validation. For large-scale datasets, parallel processing is available, enabling validation to be performed efficiently while maintaining consistent results across runs. Each validation output includes the overall status, a set of diagnostic flags indicating the specific issues encountered, a collection of metrics such as the number of predicted labels and lists of invalid values, and a UTC timestamp marking when the validation was performed. This detailed audit trail ensures that the subfield prediction quality can be assessed, reproduced, and traced back to individual records, thereby supporting transparent data curation practices.


from typing import Dict, List, Any
from datetime import datetime, timezone

VALID_SUBFIELDS = {
    "Catalysis",
    "Organic Chemistry",
    "Polymer Chemistry",
    "Inorganic Chemistry",
    "Materials Science",
    "Analytical Chemistry",
    "Physical Chemistry",
    "Biochemistry",
    "Environmental Chemistry",
    "Energy Chemistry",
    "Medicinal Chemistry",
    "Chemical Engineering",
    "Supramolecular Chemistry",
    "Radiochemistry & Nuclear Chemistry",
    "Forensic & Legal Chemistry",
    "Food Chemistry",
    "Chemical Education",
    "Others"
}


class SubfieldValidator:
    def __init__(
        self,
        *,
        require_nonempty: bool = True,
        score_range: tuple[float, float] = (0.0, 1.0),
        allowed_labels: set[str] = VALID_SUBFIELDS
    ):
        self.require_nonempty = require_nonempty
        self.score_range = score_range
        self.allowed_labels = allowed_labels

    def validate(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        checked_at = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
        subfield = doc.get("predicted_subfield")

        flags = {
            "missing_predicted_subfield": subfield is None,
            "not_a_dict": False,
            "contains_invalid_labels": False,
            "contains_invalid_scores": False,
            "empty_predictions": False,
        }
        metrics = {
            "label_count": 0,
            "invalid_labels": [],
            "invalid_scores": [],
        }

        if subfield is None:
            status = "fail"
        elif not isinstance(subfield, dict):
            flags["not_a_dict"] = True
            status = "fail"
        else:
            metrics["label_count"] = len(subfield)

            if self.require_nonempty and len(subfield) == 0:
                flags["empty_predictions"] = True

            for label, score in subfield.items():
                if label not in self.allowed_labels:
                    metrics["invalid_labels"].append(label)
                    flags["contains_invalid_labels"] = True

                if not isinstance(score, float) or not (self.score_range[0] <= score <= self.score_range[1]):
                    metrics["invalid_scores"].append({label: score})
                    flags["contains_invalid_scores"] = True

            if any(flags.values()):
                status = "warn" if not flags["missing_predicted_subfield"] else "fail"
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
        *,
        parallel: bool = False,
        max_workers: int | None = None
    ) -> List[Dict[str, Any]]:
        if not parallel or len(docs) <= 1:
            return [self.validate(doc) for doc in docs]

        import concurrent.futures as cf
        with cf.ProcessPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(self.validate, docs))

