from __future__ import annotations

from typing import Any

from beartype import beartype

from ..llm import LLMClient
from ..utils import load_prompts


@beartype
class DatasetTopicGenerator:
    """
    Generate a short topic few words describing topic of a dataset
    """

    def __init__(self, client: LLMClient | Any, model_name: str, temperature: float = 0.0) -> None:
        # Support both LLMClient and legacy OpenAI clients
        if isinstance(client, LLMClient):
            self.llm_client = client
        else:
            # Legacy support: wrap OpenAI-compatible client
            from ..llm import OpenAICompatibleClient

            self.llm_client = OpenAICompatibleClient(client)
        self.model = model_name
        self.temperature = float(temperature)
        prompts = load_prompts()["topic_generation"]
        self._system_message = prompts["system_message"].strip()
        self._user_prompt = prompts["user_prompt"]

    def _build_prompt(
        self,
        title: str,
        original_description: str | None,
        dataset_sample: str,
        tags: list[str] | None = None,
    ) -> str:
        description_block = (
            f"Original Description: {original_description}\n" if original_description else ""
        )
        prompt = self._user_prompt.format(
            title=title,
            description_block=description_block,
            dataset_sample=dataset_sample,
        )
        if tags:
            prompt = f"{prompt}\nTags: {', '.join(tags)}"

        prompt = f"{prompt}\n\nTopic (2-3 words):"
        return prompt

    def generate_topic(
        self,
        title: str,
        original_description: str | None,
        dataset_sample: str,
        tags: list[str] | None = None,
    ) -> str:
        """
        Return a trimmed topic string for a dataset

        Args:
            title: Dataset title
            original_description: Existing description if available
            dataset_sample: CSV text sample
            tags: Optional list of tags associated with the dataset. When
                provided, they are appended to the prompt as extra context.

        Returns:
            Topic string
        """

        prompt = self._build_prompt(title, original_description, dataset_sample, tags)
        response = self.llm_client.chat_completions_create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )
        return response["choices"][0]["message"]["content"].strip()
