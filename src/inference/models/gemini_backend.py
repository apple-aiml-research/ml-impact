#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import vertexai
import asyncio
import grpc
from http import HTTPStatus
from .base import BaseLLM
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)
from vertexai.preview.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
)
from typing import List

from google.api_core.exceptions import ResourceExhausted
from google.auth import exceptions as google_auth_exceptions


class GeminiEvaluatorBase(BaseLLM):
    """Base class for any Gemini-based evaluator. Doesn't define specific methods to call LLM"""

    def __init__(
        self,
        model_name: str,
        max_tokens: int,
        top_k: int,
        temperature: float,
        top_p: float,
        response_mime_type: str,
        candidate_count: int,
        system_instructions: str | None,
        timeout_seconds: int,
        max_retries: int,
        project: str,
        max_requests_per_second: int,
        region: str,
        think_mode: bool | None,
    ):
        vertexai.init(project=project, location=region)

        self._model_name = model_name
        self.gpc_model = GenerativeModel(
            model_name=self._model_name, system_instruction=system_instructions
        )
        self.parameters = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "response_mime_type": response_mime_type,
            "candidate_count": candidate_count,
        }

        self.think_mode = think_mode
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        self.force_timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.max_requests_per_second = max_requests_per_second
        self.semaphore = asyncio.Semaphore(max_requests_per_second)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(
            max=30
        ),  # Exponential backoff with jitter, between 1 and 30 seconds
        retry=retry_if_exception_type((grpc.RpcError, ResourceExhausted)),
    )
    async def send_prompt_to_llm(
        self, gcp_prompt: str, task_n: int, tot_tasks: int
    ) -> str:
        """
        Method that executes the actual asynchronous call to the LLM in GCP.
        Assumes a prompt in str format that will always be used as input.

        Args:
            gcp_prompt (str): Any prompt to be used in the LLM call

        Raises:
            RuntimeError: If max retries exceeded, will raise Runtime exception
            and this will be propagated back to user in `evaluate` method response

        Returns:
            str: Response of LLM in string format (not processed yet)
        """
        try:
            async with self.semaphore:
                llm_response = await asyncio.wait_for(
                    self.gpc_model.generate_content_async(
                        gcp_prompt,
                        safety_settings=self.safety_settings,
                        generation_config=GenerationConfig(**self.parameters),
                    ),
                    timeout=self.force_timeout_seconds,
                )
                return llm_response.text
        except asyncio.TimeoutError:
            raise

        except google_auth_exceptions.GoogleAuthError:
            raise
        except grpc.RpcError as e:
            if (
                e.code == grpc.StatusCode.RESOURCE_EXHAUSTED
                or e.code == HTTPStatus.TOO_MANY_REQUESTS
            ):
                raise
            else:
                raise
        except Exception:
            raise

    def evaluate(self, prompts: List[str]) -> List[dict]:
        if not isinstance(prompts, List):
            raise TypeError(
                f"Your input should be a list of prompts to evaluate. Got {type(prompts)}"
            )

        # run LLM calls asynchronously
        llm_responses = asyncio.run(self.run_async_llm_calls(prompts))
        final_responses: List[str | BaseException] = []
        for task_n, resp in enumerate(llm_responses):
            if isinstance(resp, BaseException):
                error_info = {
                    "success": False,
                    "error_message": str(resp),
                    "error_type": type(resp).__name__,
                }
                final_responses.append(error_info)
            else:
                final_responses.append(
                    {
                        "success": True,
                        "response": resp,
                    }
                )
        return final_responses

    async def run_async_llm_calls(
        self, prompts: List[str]
    ) -> List[str | BaseException]:
        tasks = [
            self.send_prompt_to_llm(prompt, task_n, len(prompts))
            for task_n, prompt in enumerate(prompts, 1)
        ]
        llm_responses = await asyncio.gather(*tasks, return_exceptions=True)
        return llm_responses
