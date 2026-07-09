#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

from abc import ABC, abstractmethod
from typing import List

from dotenv import load_dotenv

load_dotenv()


class BaseLLM(ABC):
    """Abstract base class for all LLM models."""

    @abstractmethod
    def evaluate(self, prompts: List[str]) -> List[str]:
        """
        Method to send a list of prompts to the model and get the responses.

        Args:
            prompts (List[str]): List of prompts that need evaluation.

        Returns:
            List[str]: List of model-generated responses for each prompt.
        """
        pass
