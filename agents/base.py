"""
Base agent class shared by all specialized agents.
Provides the LLM client and a structured call helper.
"""

import json
from typing import Any, Dict, List, Optional

from openai import OpenAI

from config import settings


class BaseAgent:
    """
    Base class for all DEPI AI Mentor agents.
    Wraps the OpenAI client and provides a common _call_llm interface.
    """

    def __init__(self, name: str):
        self.name = name
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.LLM_MODEL

    def _call_llm(
        self,
        system_prompt: str,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        json_mode: bool = False,
        temperature: float = 0.4,
        max_tokens: int = 2000,
    ) -> str:
        """
        Send a prompt to the LLM and return the text response.

        Args:
            system_prompt: The system instruction for this agent.
            user_message: The current user turn message.
            history: Optional prior conversation turns for context.
            json_mode: If True, instruct the model to respond in JSON.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.

        Returns:
            The assistant's text response (parsed JSON string if json_mode=True).
        """
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": user_message})

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse a JSON string returned by the LLM, stripping markdown fences if present."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            cleaned = cleaned.rsplit("```", 1)[0]
        return json.loads(cleaned)
