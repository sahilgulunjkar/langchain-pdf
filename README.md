<div align="center">

# LangChain Multi-PDF AI Chat Assistant
[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/Orchestration-LangChain-1C3C3C.svg?logo=chainlink&logoColor=white)](https://www.langchain.com/)
[![Google Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-4285F4.svg?logo=google&logoColor=white)](https://ai.google.dev/)

</div>

## Overview

An AI-powered PDF question-answering application that allows users to upload multiple PDF documents 
and interact with their contents through natural language conversations.

The app uses RAG to retrieve relevant document context and generate meaningful responses.

---

## System Architecture

<p align="center">
  <img src="diagram/system-architecture.png" alt="System Architecture Diagram" width="850"/>
</p>

---

## Application Interface

| Document Upload & Processing | Interactive Q&A Session |
| :---: | :---: |
| <img src="Screenshots/home.png" alt="Home Screen" width="450"/> | <img src="Screenshots/working.png" alt="Chat Session" width="450"/> |

---

## Features

- **Multi-PDF Support**: Upload and analyze multiple documents in one session.
- **Google Gemini 3.5 Flash**: Fast, grounded conversational answers.
- **FAISS Vector Search**: Semantic retrieval with Google `gemini-embedding-001`.
- **Context-Aware Memory**: Retains conversation history for continuous multi-turn dialogue.

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

> Set `GOOGLE_API_KEY=your_api_key_here` in `.env` before running.


Developed using Python, LangChain & Google Gemini

