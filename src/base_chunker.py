# Copied from the chunking_evaluation project (https://github.com/brandonstarxel/chunking_evaluation).
# Third-party code: see THIRD_PARTY_NOTICES.md for the license and copyright notice.
from abc import ABC, abstractmethod
from typing import List

class BaseChunker(ABC):
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        pass
