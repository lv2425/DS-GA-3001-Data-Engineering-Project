from __future__ import annotations

from typing import Any

from beartype import beartype

from ..llm import LLMClient, LocalLLMClient, OpenAICompatibleClient
from ..utils import load_prompts


@beartype
class QuestionDatasetEvaluator:
    """Evaluate if a dataset could help solve a given question using LLM"""

    def __init__(
        self,
        client: LLMClient | Any,
        model_name: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        # Support both LLMClient and legacy OpenAI clients
        if isinstance(client, LLMClient):
            self.llm_client = client
        else:
            # Legacy support: wrap OpenAI-compatible client
            self.llm_client = OpenAICompatibleClient(client)

        # Auto-detect model name from LocalLLMClient; require explicit name otherwise
        if model_name is None:
            if isinstance(client, LocalLLMClient):
                model_name = client.model_name
            else:
                raise ValueError(
                    "model_name must be provided when not using a LocalLLMClient"
                )

        self.model = model_name
        self.temperature = float(temperature)
        
        # Load question-dataset-specific prompts
        prompts = load_prompts()["question_dataset_evaluation"]
        self._system_message = prompts["system_message"].strip()
        self._prompt_config = prompts  # Store full config to access templates

    def eval_build_content(
        self,
        question: str,
        description: str | None = None,
        data_topic: str | None = None,
        data_sample: str | None = None,
        dataset_profile: str | None = None,
        semantic_profile: str | None = None,
        data_dict: dict[str, dict[str, Any]] | None = None,
    ) -> str:
        blocks = []

        if description:
            desc_template = self._prompt_config.get("description_template", "Dataset Description: {description}")
            blocks.append(desc_template.format(description=description))

        if data_dict:
            extra_template = self._prompt_config.get("extra_info_template", "Extra Info: {data_dict}")
            blocks.append(extra_template.format(data_dict=data_dict))

        if data_topic:
            topic_template = self._prompt_config.get("topic_template", "Data Topic: {data_topic}")
            blocks.append(topic_template.format(data_topic=data_topic))

        if data_sample:
            sample_template = self._prompt_config.get("sample_template", "Data Sample:\n{data_sample}")
            blocks.append(sample_template.format(data_sample=data_sample))

        if dataset_profile:
            profile_template = self._prompt_config.get("profile_template", "Dataset Profile: {dataset_profile}")
            blocks.append(profile_template.format(dataset_profile=dataset_profile))

        if semantic_profile:
            sem_template = self._prompt_config.get("semantic_profile_template", "Semantic Profile: {semantic_profile}")
            blocks.append(sem_template.format(semantic_profile=semantic_profile))
        
        # Join blocks with newlines
        description_block = "\n\n".join(blocks)
        
        # Get the main prompt template and fill in placeholders
        user_template = self._prompt_config.get("user_prompt", "")
        content = user_template.format(
            question=question,
            content_blocks=description_block
        )
        
        return content

    def evaluate_relevance(
        self,
        question: str,
        description: str | None = None,
        data_topic: str | None = None,
        data_sample: str | None = None,
        dataset_profile: str | None = None,
        semantic_profile: str | None = None,
        data_dict: dict[str, dict[str, Any]] | None = None,
    ) -> str:
        """
        Evaluate if the dataset could help solve the question

        Args:
            question: The question to evaluate (required)
            description: Description of the dataset (optional)
            data_topic: Short topic string (optional)
            data_sample: CSV text sample (optional)
            dataset_profile: Structural profile (optional)
            semantic_profile: Semantic profile (optional)
            data_dict: Data dictionary mapping column names to metadata (optional)

        Returns:
            LLM response indicating relevance (YES/NO with explanation)
        """
        content = self.eval_build_content(
            question=question,
            description=description,
            data_topic=data_topic,
            data_sample=data_sample,
            dataset_profile=dataset_profile,
            semantic_profile=semantic_profile,
            data_dict=data_dict,
        )
        
        response = self.llm_client.chat_completions_create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_message},
                {"role": "user", "content": content},
            ],
            temperature=self.temperature,
        )
        
        return response["choices"][0]["message"]["content"]