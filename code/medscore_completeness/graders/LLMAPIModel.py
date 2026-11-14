"""Basic methods for LLM API calls"""
from typing import List, Any, Dict, Optional, Iterable, Tuple, override, Union
from itertools import islice
import json
import os
import asyncio
from abc import ABC, abstractmethod
import logging
import random
import time

import openai
from openai.types.chat.chat_completion import ChatCompletion, Choice, ChatCompletionMessage
from tqdm.asyncio import tqdm as async_tqdm
from transformers import AutoTokenizer

from medscore_completeness import configure_logging

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)


# --- Base Class for API Interaction ---
class APIModel(ABC):
    """Base class for handling asynchronous LLM API calls with robust error handling and backoff."""
    def __init__(
            self,
            model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            model_server: str = "https://api.together.xyz/v1",
            api_key: str = os.environ.get("TOGETHER_API_KEY"),
            random_seed: int = 42,
            max_concurrent_requests: int = 10,
            max_completion_tokens: int = 4096,
            skip_connection_test: bool = False,
    ):
        self.max_concurrent_requests = int(max_concurrent_requests)
        self.random_seed = random_seed
        self.max_completion_tokens = max_completion_tokens

        if not api_key:
            raise ValueError("API key is not set. Please set the corresponding environment variable.")

        # VLLM and other OpenAI-compatible servers expect API calls under the /v1 prefix.
        # This safeguard ensures the base URL is correctly formatted.
        if not model_server.endswith(("/v1", "/v1/")):
            original_server_path = model_server
            # Append /v1 to the path, removing any trailing slashes first
            model_server = model_server.rstrip('/') + "/v1"
            logger.warning(
                f"Server path '{original_server_path}' did not end with /v1. "
                f"It has been automatically corrected to '{model_server}'."
            )

        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=model_server,
            max_retries=0,  # Disable the default retry mechanism to rely on our custom backoff
        )
        self.model_name = model_name

        if not skip_connection_test:
            logger.debug(f"Testing connection to API server at {model_server}...")
            self._test_connection(api_key=api_key, base_url=model_server)
            logger.debug("API server connection successful.")

    def _test_connection(self, api_key: str, base_url: str):
        """Makes a synchronous test call to the API server to verify connectivity."""
        try:
            # Use a synchronous client for the one-off test in the constructor
            sync_client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
                max_retries=0,
                timeout=10,  # 10-second timeout for the test
            )
            # A lightweight call to list models checks for a valid connection and API key
            sync_client.models.list()
        except openai.APIConnectionError as e:
            raise ConnectionError(
                f"Failed to connect to the API server at '{base_url}'. "
                "Please check the server path and ensure it is running and accessible."
            ) from e
        except openai.AuthenticationError as e:
            raise ValueError(
                "Authentication failed. Please check if the provided API key is correct."
            ) from e
        except Exception as e:
            raise ConnectionError(f"An unexpected error occurred while trying to connect to the server: {e}") from e

    async def __call__(
            self,
            inputs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Processes inputs in batches and returns the aggregated output."""
        all_output = []
        n_iter = (len(inputs) + self.max_concurrent_requests - 1) // self.max_concurrent_requests
        batcher = self._get_batches(inputs, self.max_concurrent_requests)

        async for batch in async_tqdm(batcher, desc=f"Running {self.__class__.__name__}", total=n_iter, ncols=0):
            completions = await self._batch_with_backoff(batch)
            output = self._process_completions(batch, completions)
            all_output.extend(output)
        return all_output

    async def _batch_with_backoff(self, batch: List[Dict[str, Any]]) -> List[Union[ChatCompletion, None]]:
        """Creates and gathers tasks with an exponential backoff retry mechanism."""
        tasks = [self._individual_call_with_backoff(b) for b in batch]
        return await asyncio.gather(*tasks)

    async def _individual_call_with_backoff(self, item: Dict[str, Any], retries: int = 5, delay: int = 2) -> Optional[ChatCompletion]:
        """
        Wrapper for an individual API call with exponential backoff.
        """
        retryable_exceptions = (openai.APITimeoutError, openai.APIConnectionError)
        # 429: Rate limit, 5xx: Server errors
        retryable_status_codes = {429, 500, 502, 503, 504}

        last_exception = None
        item_id = str(item.get('id', 'N/A'))

        for i in range(retries):
            try:
                logger.debug(f"Attempt {i + 1}/{retries} for item '{item_id}'.")
                return await self._individual_call(item)
            except openai.APIStatusError as e:
                last_exception = e
                if e.status_code in retryable_status_codes:
                    wait_time = delay * (2 ** i)
                    logger.warning(
                        f"API error on item '{item_id}' with status {e.status_code}. Retrying in {wait_time} seconds..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Non-retryable API error on item '{item_id}' with status {e.status_code}: {e}")
                    return self._create_failed_completion(str(last_exception))
            except retryable_exceptions as e:
                last_exception = e
                wait_time = delay * (2 ** i)
                logger.warning(
                    f"{type(e).__name__} on item '{item_id}'. Retrying in {wait_time} seconds..."
                )
                await asyncio.sleep(wait_time)
            except Exception as e:
                last_exception = e
                logger.error(f"An unexpected error occurred on item '{item_id}': {e}", exc_info=True)
                return self._create_failed_completion(str(last_exception))

        logger.error(f"Max retries reached for item '{item_id}'. API call failed. Last error: {last_exception}")
        return self._create_failed_completion(str(last_exception))

    def _create_failed_completion(self, error_message: str) -> ChatCompletion:
        """
        Creates a valid ChatCompletion object containing the error message.
        This prevents downstream code from crashing on 'None'.
        """
        return ChatCompletion(
            id="error-generated-locally",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(
                        content=f"API_ERROR: {error_message}",
                        role="assistant"
                    )
                )
            ],
            created=int(time.time()),
            model=self.model_name,
            object="chat.completion"
        )

    @abstractmethod
    async def _individual_call(self, item: Dict[str, Any]):
        """Abstract method for a single API call, to be implemented by subclasses."""
        raise NotImplementedError()

    def _get_system_prompt(self) -> str:
        return "You are a helpful assistant."

    @staticmethod
    def _get_batches(iterable: Iterable, n: int):
        """Yield successive n-sized chunks from an iterable."""
        it = iter(iterable)
        while True:
            chunk = tuple(islice(it, n))
            if not chunk:
                return
            yield chunk

    def _process_completions(
            self,
            batch: List[Dict[str, Any]],
            completions: List[Union[ChatCompletion, None]]
    ) -> List[Union[ChatCompletion, None]]:
        """Default completion processing, returns the raw completions."""
        return completions
