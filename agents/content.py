"""
Content Recommendation Agent.

Uses RAG to retrieve relevant learning materials from the knowledge base
and recommends resources tailored to the current week topic.
"""

from typing import Any, Dict, List

from agents.base import BaseAgent
from rag.retriever import RAGRetriever
from state import AgentState
from tools.tools import search_learning_resources


SYSTEM_PROMPT = """You are the Content Recommendation Agent for the DEPI AI Mentor platform.
Your role is to curate the best learning resources for each student's current topic.

You have access to:
1. The DEPI internal knowledge base (retrieved via RAG)
2. External resource recommendations

Guidelines:
- Summarize the key concepts the student should learn this week.
- Present the internal knowledge base content clearly.
- Suggest external resources (videos, articles, exercises).
- Adapt the explanation depth to the student's level.
- Use encouraging and clear language.
"""


class ContentAgent(BaseAgent):
    """
    Retrieves and presents learning content for the current week topic using RAG.
    """

    def __init__(self):
        super().__init__("ContentAgent")
        self._retriever = None  # Lazy initialization

    @property
    def retriever(self) -> RAGRetriever:
        if self._retriever is None:
            self._retriever = RAGRetriever()
        return self._retriever

    def get_content_for_topic(
        self, topic: str, level: str, track: str
    ) -> Dict[str, Any]:
        """
        Retrieve relevant content chunks and format them for presentation.

        Returns:
            {
                "retrieved_chunks": [...],
                "formatted_content": "...",
                "external_resources": [...]
            }
        """
        # Retrieve relevant knowledge base chunks
        query = f"{topic} {track} {level} tutorial guide"
        chunks = self.retriever.retrieve_as_list(query)

        # Get external resource suggestions
        resource_result = search_learning_resources.invoke(
            {"query": topic, "resource_type": "course"}
        )
        import json
        external_resources = json.loads(resource_result)

        # Ask LLM to synthesize the retrieved content into a learning guide
        context_text = "\n\n".join(
            f"[Source: {c['source']}]\n{c['text']}" for c in chunks
        )
        prompt = f"""
Student Level: {level}
Track: {track}
Current Topic: {topic}

Relevant knowledge base content:
{context_text}

Create a concise learning guide for this week's topic. Include:
1. A brief introduction to the topic (2-3 sentences)
2. The 3-5 most important concepts to understand
3. A practical example or use case
4. What the student should be able to do after studying this topic

Keep the tone educational, clear, and motivating.
"""
        formatted_content = self._call_llm(
            system_prompt=SYSTEM_PROMPT,
            user_message=prompt,
            temperature=0.3,
        )

        return {
            "retrieved_chunks": chunks,
            "formatted_content": formatted_content,
            "external_resources": external_resources,
        }

    def run(self, state: AgentState) -> AgentState:
        """Main agent entry point called by the supervisor."""
        learning_path = state.get("learning_path", [])
        current_week = state.get("current_week", 1)
        level = state.get("student_level", "Beginner")
        track = state.get("track", "Data Analysis")

        # Find the current week data
        week_data = next(
            (w for w in learning_path if w.get("week") == current_week), None
        )
        if not week_data:
            state["next_agent"] = "assignment"
            return state

        topic = week_data.get("topic", "")
        state["current_topic"] = topic

        content = self.get_content_for_topic(topic, level, track)
        state["retrieved_content"] = content["retrieved_chunks"]

        # Build the reply message
        resource_lines = "\n".join(
            f"  - {r['title']}: {r['url']}" for r in content["external_resources"][:3]
        )

        reply = (
            f"Week {current_week} — {topic}\n\n"
            f"{content['formatted_content']}\n\n"
            f"Recommended External Resources:\n{resource_lines}\n\n"
            "Ready to test your knowledge? I'll now create an assignment for this week."
        )

        messages = state.get("messages", [])
        messages.append({"role": "assistant", "content": reply})
        state["messages"] = messages
        state["next_agent"] = "assignment"
        return state
