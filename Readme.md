# Minarik: AI Research Paper Assistant

Welcome to **Minarik**, an AI-powered Streamlit app that helps you explore and summarize research papers from [arXiv](https://arxiv.org/). This project was built in my free time to learn **LangChain**—a handy toolkit for building chat-like applications with large language models (LLMs).

---

## Overview

- **Fetch Papers**: Searches recent arXiv papers by topic.  
- **Summaries**: Creates easy-to-understand summaries using an LLM (OpenAI via LangChain).  
- **Chatbot**: Lets you ask follow-up questions about the papers.

---

## Why This Project?

I started this project to:
1. Practice **LangChain**, which simplifies working with large language models.
2. Explore **Streamlit** for rapid prototyping of AI-driven web apps.
3. Understand how to search, retrieve, and process large text (like scientific abstracts) in real-time.

---

## Core Libraries & Tools

- **[Streamlit](https://streamlit.io/)**: Framework for building interactive data apps.
- **[LangChain](https://github.com/hwchase17/langchain)**: An open-source library to create applications with LLMs easily.
- **[OpenAI’s Chat Models](https://platform.openai.com/docs/introduction)**: Powers the text summarization and the Q&A chatbot.
- **[arXiv](https://arxiv.org/)**: Source for academic papers covering many research areas.

---

## Setup & Usage

1. **Clone the Repository**
   ```bash
   git clone https://github.com/MikeDuran-git/Generative-AI-WebApps-langchain.git
   cd Minarik-AI-Assistant
    ```
2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
2. **create a `.streamlit/secrets.toml` file**
    ```bash
    OPENAI_API_KEY=your_openai_key
    model_name=gpt-3.5-turbo
    ```
2. **Install Dependencies**
    ```bash
    streamlit run app.py
    ```

5. **Interact with the App**
- Enter a research topic in the text area.
- Let the app fetch and display relevant arXiv papers.
- Read the AI-generated summaries.
- Ask the chatbot follow-up questions about the papers.


# What I Learned
1. **LangChain Basics**

- Creating PromptTemplates to structure LLM responses.
- Composing multiple prompts/LLM calls into LLMChains.

2. **Streamlit UI**

- Building interactive components like text areas, expanders, and chat interfaces.
- Using custom CSS to enhance the user interface.

3. **NLP Fundamentals**

- Implementing a simple frequency-based text summarization approach.
- Leveraging NLTK for tokenization and stopwords filtering.