import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize
import arxiv
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import warnings
import time

# Download necessary NLP resources
nltk.download('punkt')
nltk.download('stopwords')

# Ignore irrelevant warnings
warnings.filterwarnings("ignore")

# Load environment variables (e.g., API keys)
load_dotenv()

# Summary generator function
def summarymaker(text):
    """
    Generates a simple text summary based on word frequency.
    
    Parameters:
    - text (str): Input text to summarize.

    Returns:
    - summary (str): A concise summary of the input text.
    """
    stop_words = set(stopwords.words('english'))  # List of unimportant words
    words = word_tokenize(text)  # Split text into individual words
    freq_table = {}  # Dictionary to store word frequencies
    
    for word in words:
        word = word.lower()
        if word not in stop_words:
            freq_table[word] = freq_table.get(word, 0) + 1  # Count word frequency

    sentence_value = {}  # Dictionary to score sentences
    for sentence in sent_tokenize(text):
        for word, freq in freq_table.items():
            if word in sentence.lower():
                sentence_value[sentence] = sentence_value.get(sentence, 0) + freq

    average = sum(sentence_value.values()) / len(sentence_value)  # Calculate average sentence score
    summary = ' '.join(
        sentence for sentence in sent_tokenize(text)
        if sentence in sentence_value and sentence_value[sentence] > 1.2 * average
    )
    return summary

# Streamlit app: title and user input
st.title("Minarik: AI Research Paper Assistant")
st.write("Explore and summarize research papers with ease.")

# User enters a research topic
topic = st.text_area("Enter a research topic you are interested in:")

if topic:
    # Dictionary to store retrieved papers
    papers = {}

    # Arxiv search with retry mechanism
    for _ in range(3):  # Retry up to 3 times
        try:
            search = arxiv.Search(
                query=topic,
                max_results=5,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            for result in search.results():
                papers[result.title] = [result.entry_id, result.summary, result.pdf_url]
            if papers:
                break  # Exit retry loop if papers are found
        except Exception as e:
            st.warning(f"Error fetching papers: {e}. Retrying...")
            time.sleep(2)  # Wait before retrying

    # Handle case where no papers are found
    if not papers:
        st.error("No papers found. Try a different topic.")
    else:
        # Initialize LangChain LLM for summaries
        llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-4o-mini"
        )
        template = """
        You give a concise and easy-to-understand summary of a research paper.
        Question: Summarize: {text}
        Answer: Keep it short and easy to understand.
        """
        prompt_template = PromptTemplate(template=template, input_variables=["text"])
        answer_chain = LLMChain(llm=llm, prompt=prompt_template)

        # Display papers and generate summaries
        for paper, details in papers.items():
            st.subheader(paper)
            st.caption(f"Paper URL: {details[2]}")
            ai_summary = answer_chain.run(details[1])
            st.write(f"Bot's Summary: {ai_summary}")
            st.write(f"Original Summary: {details[1]}")
            st.divider()

        # Chatbot Section
        st.header("Chatbot: Ask About the Papers")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # User input for chatbot
        user_input = st.text_input("Ask a question about the papers:")
        if user_input:
            st.chat_message("user").markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            chatbot_template = """
            You are a chatbot that helps users understand research papers.
            Question: {text}
            Answer: Keep it concise and based on the papers provided.
            """
            chatbot_prompt = PromptTemplate(template=chatbot_template, input_variables=["text"])
            chatbot_chain = LLMChain(llm=llm, prompt=chatbot_prompt)

            # Precompute summaries for chatbot context
            papers_summary = ' '.join(summarymaker(details[1]) for details in papers.values())
            response = chatbot_chain.run({"text": user_input, "papers": papers_summary})

            # Display chatbot response
            st.chat_message("bot").markdown(response)
            st.session_state.messages.append({"role": "bot", "content": response})
