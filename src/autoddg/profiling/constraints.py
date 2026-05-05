from __future__ import annotations

from typing import Any

from beartype import beartype

from ..llm import LLMClient
from ..utils import load_prompts


@beartype
class ConstraintExtractor:
    """Infer column constraints from a dataset's data dictionary and sample using an LLM"""

    def __init__(
        self,
        client: LLMClient | Any,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ) -> None:
        if isinstance(client, LLMClient):
            self.llm_client = client
        else:
            from ..llm import OpenAICompatibleClient

            self.llm_client = OpenAICompatibleClient(client)
        self.model = model_name
        self.temperature = float(temperature)
        prompts = load_prompts()["constraint_extraction"]
        self._system_message = prompts["system_message"].strip()
        self._introduction = prompts["introduction"]
        self._sample_instruction = prompts["sample_instruction"]
        self._closing_instruction = prompts["closing_instruction"]

    @staticmethod
    def _format_data_dict(data_dict: dict[str, dict[str, Any]]) -> str:
        lines: list[str] = []
        for column, entry in data_dict.items():
            if isinstance(entry, dict):
                # The first field is assumed to be the column description so drop it and keep the rest.
                kept = dict(list(entry.items())[1:])
                if not kept:
                    continue  # omit columns with no additional metadata
                lines.append(f"- {column}:")
                for key, value in kept.items():
                    lines.append(f"    {key}: {value}")
            else:
                lines.append(f"- {column}:")
                lines.append(f"    {entry}")
        return "\n".join(lines)

    def _build_prompt(
        self,
        data_dict: dict[str, dict[str, Any]],
        dataset_sample: str | None,
    ) -> str:
        parts: list[str] = [
            self._introduction.format(data_dict=self._format_data_dict(data_dict))
        ]
        if dataset_sample:
            parts.append(self._sample_instruction.format(dataset_sample=dataset_sample))
        parts.append(self._closing_instruction.format())
        return "\n".join(parts)

    def extract_constraints(
        self,
        data_dict: dict[str, dict[str, Any]],
        dataset_sample: str | None = None,
    ) -> tuple[str, str]:
        """Call the LLM and return the prompt and the natural-language constraints

        Args:
            data_dict: Mapping of column name to metadata fields
            dataset_sample: Sample of rows for the LLM to confirm or refine constraints.

        Returns:
            (prompt, constraints_text)
        """

        prompt = self._build_prompt(data_dict, dataset_sample)

        response = self.llm_client.chat_completions_create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )

        constraints = response["choices"][0]["message"]["content"]
        return prompt, constraints
