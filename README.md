# DEPI AI Mentor

## Overview

DEPI AI Mentor is a multi-agent AI-powered platform designed to act as a personalized learning mentor for students. The system goes beyond traditional chatbot interactions by managing the full learning lifecycle — from skill assessment to career guidance.

It leverages modern LLM capabilities, Retrieval-Augmented Generation (RAG), and agent orchestration to deliver adaptive, context-aware, and goal-driven learning experiences.

---

## Project Architecture

The system is built using a modular, multi-agent architecture coordinated by a central supervisor agent.

```
depi_mentor/
│
├── api/                  # FastAPI application entrypoint
├── agents/               # All AI agents (core system logic)
├── rag/                  # Retrieval-Augmented Generation pipeline
├── memory/               # Long-term memory management
├── tools/                # External tools (execution, utilities)
├── ui/                   # Streamlit frontend
├── data/                 # Knowledge base
├── scripts/              # Setup and preprocessing scripts
│
├── config.py             # Configuration management
├── state.py              # Shared state across agents
├── database.py           # Database integration (PostgreSQL)
└── requirements.txt
```

---

## System Workflow

### 1. User Onboarding

The system initializes a user profile including:

* Academic background
* Learning goals
* Target track

This information is stored and used across all agents.

---

### 2. Supervisor Agent

Located in:

```
agents/supervisor.py
```

Responsible for:

* Routing tasks between agents
* Managing execution flow
* Coordinating multi-agent interactions

Implements the core orchestration logic of the system.

---

### 3. Assessment Agent

Located in:

```
agents/assessment.py
```

Evaluates the user's current level across:

* Python
* SQL
* Data analysis fundamentals

Outputs a structured skill profile used by downstream agents.

---

### 4. Learning Path Agent

Located in:

```
agents/learning_path.py
```

Generates a personalized learning roadmap based on:

* Assessment results
* Target track
* User progress

The roadmap is structured in progressive stages.

---

### 5. Content Agent (RAG)

Located in:

```
agents/content.py
rag/retriever.py
```

Implements Retrieval-Augmented Generation:

* Retrieves relevant knowledge from `data/knowledge_base.txt`
* Uses embeddings and semantic search
* Provides contextual learning materials

Pipeline:

```
Knowledge Base → Chunking → Embeddings → Retrieval → LLM Response
```

---

### 6. Assignment Agent

Located in:

```
agents/assignment.py
```

Generates personalized assignments aligned with:

* Current learning stage
* Skill level

Assignments are dynamically adapted per user.

---

### 7. Evaluation Agent

Located in:

```
agents/evaluation.py
```

Evaluates user submissions:

* Code correctness
* Quality and structure
* Best practices

Returns structured feedback and scoring.

---

### 8. Adaptive Learning Agent

Located in:

```
agents/adaptive.py
```

Continuously adjusts the learning path based on:

* Performance
* Weak areas
* Previous evaluations

Ensures mastery before progression.

---

### 9. Interview Agent

Located in:

```
agents/interview.py
```

Simulates technical and behavioral interviews:

* Technical questions
* Scenario-based questions

Helps prepare users for real-world hiring processes.

---

### 10. Career Agent

Located in:

```
agents/career.py
```

Provides post-learning guidance:

* Career recommendations
* Skill gap analysis
* Resume improvement suggestions

---

## Memory System

Located in:

```
memory/long_term.py
```

Stores persistent user data:

* Skill levels
* Weaknesses
* Previous scores
* Learning progress

Enables long-term personalization across sessions.

---

## Tools Layer

Located in:

```
tools/tools.py
```

Includes:

* Code execution utilities
* Helper functions for agent reasoning
* External integrations

Enhances LLM capabilities via tool usage.

---

## RAG System

Located in:

```
rag/retriever.py
data/knowledge_base.txt
```

Features:

* Lightweight knowledge base
* Semantic retrieval
* Context injection into prompts

Supports dynamic content generation.

---

## API Layer

Located in:

```
api/main.py
```

Built with FastAPI.

Responsibilities:

* Exposing endpoints
* Handling user requests
* Connecting frontend with backend agents

---

## Frontend

Located in:

```
ui/app.py
```

Built with Streamlit.

Provides:

* Interactive user interface
* Chat-based interaction
* Learning progress visualization

---

## Database

Managed via:

```
database.py
scripts/init_db.py
```

Uses PostgreSQL for:

* User data
* Progress tracking
* Persistent storage

---

## Setup Instructions

### 1. Clone Repository

```
git clone <your-repo-url>
cd depi_mentor
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file based on:

```
.env.example
```

Add:

* OpenAI API Key
* Database credentials

---

### 4. Initialize Database

```
python scripts/init_db.py
```

---

### 5. Build RAG Index

```
python scripts/build_rag.py
```

---

### 6. Run Backend

```
uvicorn api.main:app --reload
```

---

### 7. Run Frontend

```
streamlit run ui/app.py
```

---

## Key Features

* Multi-agent AI system with supervisor pattern
* Personalized learning experience
* Adaptive learning paths
* Integrated RAG pipeline
* Long-term memory for user tracking
* Automated evaluation and feedback
* Interview simulation and career guidance

---

## Technologies Used

* Python
* FastAPI
* Streamlit
* LangChain / Agent-based architecture
* OpenAI GPT models
* PostgreSQL

---

## Concepts Demonstrated

### LLM Concepts

* Prompt Engineering
* Context Engineering
* Structured Outputs
* Tool Usage
* Memory Integration
* Retrieval-Augmented Generation

### Agentic AI Concepts

* Multi-Agent Systems
* Task Routing
* Planning and Orchestration
* Reflection and Adaptation
* Supervisor Pattern

---

## Conclusion

DEPI AI Mentor is a production-style AI system that demonstrates how modern LLMs can be orchestrated into a complete, intelligent application.

It showcases real-world application of:

* Agent-based architectures
* RAG pipelines
* Adaptive learning systems

This project is highly suitable as a portfolio project for roles such as:

* AI Engineer
* Generative AI Engineer
* Machine Learning Engineer

---
