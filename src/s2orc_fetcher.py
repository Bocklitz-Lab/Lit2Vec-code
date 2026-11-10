"""Fetch abstract and S2ORC full text for a given DOI and print the result."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import zstandard as zstd

GRAPH_API_BASE = "https://api.semanticscholar.org/graph/v1"
DATASETS_API_BASE = "https://api.semanticscholar.org/datasets/v1"


class S2OrcFetchError(RuntimeError):
    """Raised when the DOI cannot be resolved or the S2ORC record cannot be found."""


def _load_dotenv(env_path: str | Path = ".env") -> dict[str, str]:
    path = Path(env_path)
    if not path.exists():
        print(f"Warning: .env file not found at {path}. Skipping loading environment variables from .env.", file=sys.stderr)
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


@dataclass(slots=True)
class S2OrcFetcher:
    """Download a single paper's abstract and S2ORC full text by DOI."""

    timeout_seconds: int = 120
    api_key: str | None = None
    max_retries: int = 5
    backoff_seconds: float = 2.0

    def fetch(
        self,
        doi: str,
        *,
        release: str = "latest",
        require_fulltext: bool = False,
    ) -> dict[str, Any]:
        normalized_doi = self.normalize_doi(doi)
        paper = self._fetch_paper_metadata(normalized_doi)
        fulltext = ""
        corpus_id = paper.get("corpusId")

        try:
            s2orc_record = self._find_s2orc_record(
                normalized_doi,
                corpus_id,
                release=release,
            )
        except S2OrcFetchError as exc:
            if require_fulltext or not self._should_fallback_without_fulltext(exc):
                raise
            self._warn(
                "S2ORC full text unavailable; returning abstract only with empty "
                f"'fulltext'. Reason: {exc}"
            )
        else:
            fulltext = s2orc_record.get("content", {}).get("text", "")
            corpus_id = s2orc_record.get("corpusid") or corpus_id

        return {
            "doi": normalized_doi,
            "corpusid": corpus_id,
            "abstract": paper.get("abstract") or "",
            "fulltext": fulltext,
        }

    @staticmethod
    def normalize_doi(doi: str) -> str:
        normalized = doi.strip()
        normalized = normalized.removeprefix("https://doi.org/")
        normalized = normalized.removeprefix("http://doi.org/")
        normalized = normalized.removeprefix("doi:")
        return normalized.strip().lower()

    def _fetch_paper_metadata(self, doi: str) -> dict[str, Any]:
        fields = "corpusId,abstract,externalIds,title"
        url = f"{GRAPH_API_BASE}/paper/DOI:{quote(doi, safe='')}"
        url = f"{url}?fields={quote(fields, safe=',')}"
        payload = self._get_json(url, allow_unauthenticated_fallback=True)

        if not isinstance(payload, dict) or "corpusId" not in payload:
            raise S2OrcFetchError(f"Could not resolve DOI '{doi}' via Semantic Scholar Graph API.")

        return payload

    def _find_s2orc_record(self, doi: str, corpus_id: Any, *, release: str) -> dict[str, Any]:
        shard_urls = self._get_s2orc_shard_urls(release)
        target_corpusid = str(corpus_id) if corpus_id is not None else None

        for shard_url in shard_urls:
            for record in self._iter_s2orc_records(shard_url):
                if self._record_matches(record, doi=doi, corpus_id=target_corpusid):
                    return record

        raise S2OrcFetchError(
            f"Could not find DOI '{doi}' in the S2ORC dataset release '{release}'."
        )

    def _get_s2orc_shard_urls(self, release: str) -> list[str]:
        url = f"{DATASETS_API_BASE}/release/{quote(release, safe='')}/dataset/s2orc"
        try:
            payload = self._get_json(url)
        except S2OrcFetchError as exc:
            message = str(exc)
            if "HTTP 401" in message or "HTTP 403" in message:
                raise S2OrcFetchError(
                    "Semantic Scholar denied access to the S2ORC dataset endpoint. "
                    "DOI metadata can still be fetched from the Graph API, but S2ORC "
                    "full text requires a Semantic Scholar API key with dataset access. "
                    f"Original response: {message}"
                ) from exc
            raise
        files = payload.get("files")
        if not isinstance(files, list) or not files:
            raise S2OrcFetchError(f"No downloadable S2ORC files returned for release '{release}'.")
        return [str(item) for item in files]

    def _iter_s2orc_records(self, shard_url: str) -> Iterator[dict[str, Any]]:
        request = self._build_request(shard_url)

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                stream = self._open_text_stream(response, shard_url)
                for raw_line in stream:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        yield payload
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise S2OrcFetchError(
                f"Request failed for {shard_url} with HTTP {exc.code}: {details}"
            ) from exc
        except URLError as exc:
            raise S2OrcFetchError(f"Could not reach {shard_url}: {exc.reason}") from exc

    def _build_request(self, url: str, *, include_api_key: bool = True) -> Request:
        headers = {"Accept": "application/json"}
        api_key = self._resolve_api_key() if include_api_key else None
        if api_key:
            headers["x-api-key"] = api_key
        return Request(url, headers=headers)

    @staticmethod
    def _record_matches(record: dict[str, Any], *, doi: str, corpus_id: str | None) -> bool:
        record_corpusid = record.get("corpusid")
        if corpus_id is not None and str(record_corpusid) == corpus_id:
            return True

        external_ids = record.get("externalids", {})
        if isinstance(external_ids, dict):
            record_doi = external_ids.get("doi") or external_ids.get("DOI")
            if isinstance(record_doi, str) and record_doi.strip().lower() == doi:
                return True

        return False

    @staticmethod
    def _open_text_stream(response: Any, shard_url: str) -> Iterator[str]:
        if shard_url.endswith(".gz"):
            with gzip.GzipFile(fileobj=response) as handle:
                yield from io.TextIOWrapper(handle, encoding="utf-8")
            return

        if shard_url.endswith(".zst"):
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(response) as reader:
                yield from io.TextIOWrapper(reader, encoding="utf-8")
            return

        yield from io.TextIOWrapper(response, encoding="utf-8")

    def _get_json(
        self,
        url: str,
        *,
        allow_unauthenticated_fallback: bool = False,
    ) -> dict[str, Any]:
        api_key = self._resolve_api_key()

        for attempt in range(self.max_retries + 1):
            request = self._build_request(url, include_api_key=bool(api_key))
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    return json.load(response)
            except HTTPError as exc:
                details = exc.read().decode("utf-8", errors="replace")
                if exc.code == 429 and attempt < self.max_retries:
                    retry_after = exc.headers.get("Retry-After")
                    delay = self._compute_retry_delay(attempt, retry_after)
                    time.sleep(delay)
                    continue
                if exc.code in {401, 403} and api_key and allow_unauthenticated_fallback:
                    api_key = None
                    continue
                if exc.code == 401:
                    raise S2OrcFetchError(
                        "Semantic Scholar rejected the request without a usable API key. "
                        "The fetcher will use SEMANTIC_SCHOLAR_API_KEY, SEMANTICSCHOLAR_API_KEY, "
                        "or S2_API_KEY from .env if present, but it does not require one for Graph calls. "
                        f"Original response: {details}"
                    ) from exc
                raise S2OrcFetchError(
                    f"Request failed for {url} with HTTP {exc.code}: {details}"
                ) from exc
            except URLError as exc:
                raise S2OrcFetchError(f"Could not reach {url}: {exc.reason}") from exc

        raise S2OrcFetchError(f"Request failed for {url} after repeated retries.")

    def _compute_retry_delay(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return max(float(retry_after), 0.0)
            except ValueError:
                pass
        return self.backoff_seconds * (2 ** attempt)

    @staticmethod
    def _should_fallback_without_fulltext(exc: S2OrcFetchError) -> bool:
        message = str(exc)
        if "datasets/v1/release/" not in message:
            return False
        return "HTTP 401" in message or "HTTP 403" in message

    @staticmethod
    def _warn(message: str) -> None:
        print(f"Warning: {message}", file=sys.stderr)

    def _resolve_api_key(self) -> str | None:
        value = self.api_key or self._get_env_value(
            "SEMANTIC_SCHOLAR_API_KEY",
            "SEMANTICSCHOLAR_API_KEY",
            "S2_API_KEY",
        )
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @staticmethod
    def _get_env_value(*keys: str) -> str | None:
        env_values = _load_dotenv()
        for key in keys:
            value = os.getenv(key) or env_values.get(key)
            if value:
                return value
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download abstract and S2ORC full text for a given DOI."
    )
    parser.add_argument("--doi", required=True, help="DOI to look up.")
    parser.add_argument(
        "--release",
        default="latest",
        help="Semantic Scholar dataset release to search. Default: latest",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=120,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Number of retries for rate-limited API requests.",
    )
    parser.add_argument(
        "--backoff-seconds",
        type=float,
        default=2.0,
        help="Base backoff in seconds for rate-limited API requests.",
    )
    parser.add_argument(
        "--require-fulltext",
        action="store_true",
        help=(
            "Fail instead of falling back to empty 'fulltext' when the S2ORC "
            "dataset endpoint is unavailable or access is denied."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    fetcher = S2OrcFetcher(
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        backoff_seconds=args.backoff_seconds,
    )
    try:
        payload = fetcher.fetch(
            args.doi,
            release=args.release,
            require_fulltext=args.require_fulltext,
        )
    except S2OrcFetchError as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
