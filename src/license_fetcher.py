"""Fetch DOI-level license metadata from Crossref, Unpaywall, and OpenAlex."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


class LicenseFetchError(RuntimeError):
    """Raised when a remote license source cannot be queried successfully."""


@dataclass(slots=True)
class LicenseFetcher:
    """Download and persist source-specific license payloads for a DOI."""

    user_agent: str = "Lit2Vec/1.0 (mailto:unknown@example.org)"
    timeout_seconds: int = 30
    unpaywall_email: str | None = None
    openalex_email: str | None = None

    def fetch_all(self, doi: str) -> dict[str, dict[str, Any]]:
        normalized_doi = self.normalize_doi(doi)
        return {
            "crossref": self.fetch_crossref(normalized_doi),
            "unpaywall": self.fetch_unpaywall(normalized_doi),
            "openalex": self.fetch_openalex(normalized_doi),
        }

    def fetch_crossref(self, doi: str) -> dict[str, Any]:
        response = self._get_json(
            f"https://api.crossref.org/works/{quote(doi, safe='')}",
        )
        message = response.get("message", {})
        return {
            "doi": doi,
            "license": message.get("license", []),
            "fetched_at": self._timestamp(),
            "source": "crossref",
        }

    def fetch_unpaywall(self, doi: str) -> dict[str, Any]:
        email = self.unpaywall_email or self._get_env_value("UNPAYWALL_EMAIL", "EMAIL")
        if not email:
            raise LicenseFetchError(
                "Unpaywall requires an email address. Set it in .env as UNPAYWALL_EMAIL or EMAIL, "
                "set the environment variable, or pass --email."
            )

        response = self._get_json(
            f"https://api.unpaywall.org/v2/{quote(doi, safe='')}?email={quote(email, safe='')}",
        )
        response["fetched_at"] = self._timestamp()
        response["source"] = "unpaywall"
        return response

    def fetch_openalex(self, doi: str) -> dict[str, Any]:
        url = f"https://api.openalex.org/works?filter=doi:{quote(doi, safe='')}"
        email = self.openalex_email or self._get_env_value("OPENALEX_EMAIL", "UNPAYWALL_EMAIL", "EMAIL")
        if email:
            url = f"{url}&mailto={quote(email, safe='')}"

        response = self._get_json(url)
        results = response.get("results", [])
        if not results:
            raise LicenseFetchError(f"OpenAlex returned no works for DOI '{doi}'.")

        work = results[0]
        work["fetched_at"] = self._timestamp()
        work["source"] = "openalex"
        return work

    @staticmethod
    def normalize_doi(doi: str) -> str:
        normalized = doi.strip()
        normalized = normalized.removeprefix("https://doi.org/")
        normalized = normalized.removeprefix("http://doi.org/")
        normalized = normalized.removeprefix("doi:")
        return normalized.strip()

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _get_env_value(*keys: str) -> str | None:
        env_values = _load_dotenv()
        for key in keys:
            value = os.getenv(key) or env_values.get(key)
            if value:
                return value
        return None

    def _get_json(self, url: str) -> dict[str, Any]:
        request = Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/json",
            },
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.load(response)
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise LicenseFetchError(
                f"Request failed for {url} with HTTP {exc.code}: {details}"
            ) from exc
        except URLError as exc:
            raise LicenseFetchError(f"Could not reach {url}: {exc.reason}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download DOI license metadata from Crossref, Unpaywall, and OpenAlex."
    )
    parser.add_argument("--doi", required=True, help="DOI to query.")
    parser.add_argument(
        "--email",
        help="Contact email used for Unpaywall and, if OPENALEX_EMAIL is unset, OpenAlex polite pool.",
    )
    parser.add_argument(
        "--user-agent",
        default="Lit2Vec/1.0 (mailto:unknown@example.org)",
        help="HTTP user agent sent to upstream services.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=30,
        help="Per-request timeout in seconds.",
    )
    return parser


def _load_dotenv(env_path: str | Path = ".env") -> dict[str, str]:
    path = Path(env_path)
    if not path.exists():
        return {}

    env_values: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key:
                env_values[key] = value

    return env_values


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    fetcher = LicenseFetcher(
        user_agent=args.user_agent,
        timeout_seconds=args.timeout_seconds,
        unpaywall_email=args.email,
        openalex_email=args.email,
    )
    payloads = fetcher.fetch_all(args.doi)
    print(json.dumps(payloads, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
