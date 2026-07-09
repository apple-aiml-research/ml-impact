#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import os
import asyncio
import traceback
from typing import List, Dict, Any

from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)
import httpx


from .base import BaseLLM

load_dotenv(override=True)
token = os.getenv("vLLM_TOKEN")


class VLLMClient(BaseLLM):
    """Base class for a model deployed by vLLM woth OpenAI-compatible server."""

    def __init__(
        self,
        api_base: str,
        system_instructions: str | None,
        model_name: str,
        max_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        think_mode: bool | None,
    ):
        self.model_name = model_name
        self.api_base = api_base
        self.parameters = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
        }
        self.think_mode = think_mode
        self.system_instructions = system_instructions
        self.headers = {"Content-Type": "application/json", "Cookie": f"acack={token}"}

    def prepare_system_message(self):
        if self.system_instructions:
            return [{"role": "system", "content": self.system_instructions}]
        else:
            return []

    def prepare_user_message(self, prompt: str):
        return [{"role": "user", "content": prompt}]

    @retry(
        stop=stop_after_attempt(15),
        wait=wait_exponential_jitter(
            max=30
        ),  # Exponential backoff with jitter, between 1 and 30 seconds
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
    )
    async def send_payload_to_llm(self, payload, headers):
        url = f"{self.api_base}/chat/completions"
        async with httpx.AsyncClient(timeout=20000.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()

    async def run_async_llm_calls(
        self, payloads: List[Dict[str, Any]], headers: Dict[str, str]
    ):
        tasks = [self.send_payload_to_llm(payload, headers) for payload in payloads]
        return await asyncio.gather(*tasks)

    def evaluate(self, prompts: List[str]) -> List[Dict[str, Any]]:
        if not isinstance(prompts, List):
            raise TypeError(f"Expected a list of prompts, got {type(prompts)}")

        sys_message = self.prepare_system_message()
        payloads = [
            {
                "model": self.model_name,
                "messages": sys_message + self.prepare_user_message(prompt),
                **self.parameters,
            }
            for prompt in prompts
        ]

        try:
            llm_responses = asyncio.run(
                self.run_async_llm_calls(payloads, self.headers)
            )
        except Exception as e:
            raise RuntimeError(f"Async execution failed: {str(e)}") from e

        final_responses = []
        for result in llm_responses:
            if isinstance(result, Exception):
                final_responses.append(
                    {
                        "success": False,
                        "error": str(result),
                        "trace": traceback.format_exception(
                            type(result), result, result.__traceback__
                        ),
                    }
                )
            else:
                try:
                    response_dict = result["choices"][0]["message"]
                    message = response_dict["content"]
                    output_dict = {"success": True, "response": message}
                    if self.think_mode and "reasoning_content" in response_dict:
                        output_dict["reasoning_content"] = response_dict[
                            "reasoning_content"
                        ]
                    final_responses.append(output_dict)
                except Exception as e:
                    final_responses.append(
                        {
                            "success": False,
                            "error": "Invalid response structure",
                            "details": str(e),
                            "raw": result,
                        }
                    )

        return final_responses
