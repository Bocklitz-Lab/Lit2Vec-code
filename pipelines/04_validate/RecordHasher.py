# Hashing based verification
# To guarantee the reproducibility and integrity of the dataset, we implemented a hashing-based verification process for individual records. The primary objective of this process is to ensure that any subsequent use or redistribution of the dataset can be accompanied by a robust and reproducible integrity check, allowing users to confirm that the records have not been altered from their original form. This validation is performed by computing stable, deterministic cryptographic hashes over the record’s contents.
# For each record, the hasher processes all top-level fields except those explicitly excluded from hashing. The exclusion mechanism allows non-essential or dynamic fields, such as timestamps or local annotations, to be ignored so that they do not trigger false change detections. Before hashing, each field’s value is serialized into a canonical JSON representation that sorts all keys and removes extraneous whitespace. This canonicalization step ensures that the same logical content always produces identical serialized text, regardless of environmental differences such as dictionary ordering or formatting conventions.
# The canonicalized representation of each field is then hashed using a user-specified cryptographic hash function, such as SHA-256, SHA-1, or MD5. The hashing algorithm is applied to the UTF-8–encoded form of the canonicalized value, and the resulting digest is recorded as a fixed-length hexadecimal string. Alongside the hash, the process records the length of the canonicalized representation, providing an additional simple integrity metric. All computed field-level hashes, together with a UTC timestamp marking the moment of validation, are returned in a structured format that indicates whether the operation was successful.
# If a record is later rehashed using the same configuration and hash function, identical content will produce identical digests. Any modification to a field’s content, including changes in whitespace, ordering, or structure, will result in a different digest, immediately flagging a potential alteration. This property allows the hasher to serve both as a reproducibility safeguard during dataset preparation and as a long-term integrity verification tool for downstream users. The deterministic design of the canonicalization and hashing steps ensures that integrity checks remain consistent across different computing environments, guaranteeing that verification results can be independently reproduced by other researchers.


import json
import hashlib
from datetime import datetime
from typing import Any, Dict


class RecordHasher:
    def __init__(self, hash_func: str = "sha256", exclude_keys: list[str] | None = None):
        """
        :param hash_func: Hashing algorithm name (e.g., sha256, sha1, md5)
        :param exclude_keys: List of top-level keys to exclude from hashing
        """
        try:
            self.hash_func = getattr(hashlib, hash_func)
        except AttributeError:
            raise ValueError(f"Unsupported hash function: {hash_func}")

        self.exclude_keys = set(exclude_keys or [])

    def _canonicalize(self, value: Any) -> str:
        """
        JSON-stringify the value in a canonical, deterministic way.
        Ensures that hashes are consistent across environments.
        """
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    def hash_fields_from_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hash each top-level field of the given JSON-like dictionary,
        excluding specified keys.

        :param data: JSON-like dict to hash
        :return: Dictionary with hash and metadata per field
        """
        try:
            result = {
                "status": "success",
                "checked_at": datetime.utcnow().isoformat(),
                "field_hashes": {}
            }

            for key, value in data.items():
                if key in self.exclude_keys:
                    continue

                canonical = self._canonicalize(value)
                digest = self.hash_func(canonical.encode("utf-8")).hexdigest()

                result["field_hashes"][key] = {
                    "hash": digest,
                    "length": len(canonical),
                }

            return result

        except Exception as e:
            return {
                "status": "fail",
                "error": str(e),
                "checked_at": datetime.utcnow().isoformat()
            }


#-----------------------------------
# data = {
#     "title": "A Study of Hashing",
#     "metadata": {"authors": ["Alice", "Bob"], "year": 2024},
#     "ignore_this": "unimportant"
# }

# hasher = RecordHasher(exclude_keys=["ignore_this"])
# result = hasher.hash_fields_from_dict(data)

# import json
# print(json.dumps(result, indent=2))
