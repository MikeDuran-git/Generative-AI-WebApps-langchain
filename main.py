"""
Minarik: AI Research Paper Assistant

This Streamlit app demonstrates how to:
1. Fetch recent research papers from Arxiv based on a user-defined topic.
2. Summarize those papers using an LLM (LangChain + OpenAI).
3. Provide an interactive chatbot that answers questions about the retrieved papers.

Author's Note:
"Hello guys, I trained to learn LangChain!"
Enjoy exploring this code and customizing it to your own projects.
"""

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

# Step 1: Download NLTK resources (tokenizers, stopwords) if not already available.
nltk.download('punkt')
nltk.download('stopwords')

# Step 2: Ignore irrelevant warnings (e.g., from deprecated libraries).
warnings.filterwarnings("ignore")

# Step 3: Load environment variables (API keys, model names, etc.)
# Make sure to have a .env file with your OpenAI key, e.g.:
# OPENAI_API_KEY="your-key"
# model_name="gpt-3.5-turbo"
load_dotenv()

# Step 4: Configure the Streamlit page (title, icon, layout).
st.set_page_config(
    page_title="Minarik: AI Research Paper Assistant",
    page_icon="🧠",
    layout="centered",
)

# Step 5: Define custom CSS to improve page visuals.
# This is optional, but it helps create a cleaner, more professional UI.
st.markdown(
    """
    <style>
    /* Center the main heading and adjust margin */
    .main > div {
        max-width: 700px;
        margin: 0 auto;
        padding: 1rem;
    }
    
    /* Style the text area a bit */
    .stTextArea textarea {
        background-color: #fefefe;
        border: 1px solid #ddd;
        border-radius: 0.25rem;
        font-size: 1rem;
    }

    /* Add spacing around subheaders */
    h2 {
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }

    /* Style the chat container */
    .stChatMessage {
        border-radius: 0.5rem;
        margin: 1rem 0;
    }

    /* Divider spacing */
    .stDivider {
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def summarymaker(text):
    """
    A simple text summarization function based on word frequency.

    1. Convert the text into individual tokens (words).
    2. Build a frequency table, ignoring stopwords.
    3. Score sentences by summing the frequencies of words that appear in each sentence.
    4. Compare each sentence's score to the average; keep sentences that exceed a threshold.

    Parameters:
        text (str): The text (abstract or full content) to be summarized.

    Returns:
        summary (str): A concise summary of the input text.
    """
    stop_words = set(stopwords.words('english'))  # standard list of stopwords
    words = word_tokenize(text)  # tokenize text into individual words
    freq_table = {}  # dictionary to count word frequencies

    # Count how often each non-stopword occurs
    for word in words:
        word = word.lower()
        if word not in stop_words:
            freq_table[word] = freq_table.get(word, 0) + 1

    # Score each sentence based on the frequencies of its words
    sentence_value = {}
    for sentence in sent_tokenize(text):
        for word, freq in freq_table.items():
            if word in sentence.lower():
                sentence_value[sentence] = sentence_value.get(sentence, 0) + freq

    # Handle edge cases where the text might be too short or malformed
    if not sentence_value:
        return "No valid summary could be generated."

    # Compute an average score, then filter out sentences that don’t pass the threshold
    average = sum(sentence_value.values()) / len(sentence_value)
    summary = ' '.join(
        sentence for sentence in sent_tokenize(text)
        if sentence_value.get(sentence, 0) > 1.2 * average
    )
    return summary

# Step 6: Set up the main UI of the app.

# 6.1: Display a title and short introduction for the user.
st.title("Minarik: AI Research Paper Assistant 🧠")
st.write(
    """
    Welcome! This app helps you explore and summarize research papers from Arxiv. 
    Enter a topic below to get the latest papers, read quick summaries, 
    and ask any follow-up questions about them.
    """
)

# 6.2: Create a sidebar for instructions.
st.sidebar.header("Instructions")
st.sidebar.write(
    """
    1. Enter a **research topic** in the text area.
    2. Click outside the text area or press **Enter** to run the search.
    3. Read the short summaries generated by the app.
    4. Ask questions in the **Chatbot** section at the bottom of the page (copy paste the name of the file you want infos of).
    """
)

# 6.3: Text area for user to enter their topic.
topic = st.text_area("Enter a research topic you are interested in:", "")

# Step 7: If the user has entered a topic, proceed to search Arxiv.
if topic.strip():
    st.write("Looking for papers related to:", f"**{topic}**")
    papers = {}  # dictionary to store retrieved papers

    # 7.1: Use a spinner to indicate that the app is working.
    with st.spinner("Searching Arxiv for relevant papers..."):
        # 7.2: Attempt searching up to 3 times if there's a temporary error.
        for _ in range(3):
            try:
                search = arxiv.Search(
                    query=topic,
                    max_results=5,
                    sort_by=arxiv.SortCriterion.SubmittedDate
                )
                # 7.3: Store each paper in a dictionary with title as key and [entry_id, summary, pdf_url] as value.
                for result in search.results():
                    papers[result.title] = [result.entry_id, result.summary, result.pdf_url]
                if papers:
                    break  # stop retry loop if papers found
            except Exception as e:
                st.warning(f"Error fetching papers: {e}. Retrying...")
                time.sleep(2)  # wait 2 seconds before trying again

    # 7.4: If still no papers, display an error message.
    if not papers:
        st.error("No papers found. Try a different topic.")
    else:
        st.success("Papers successfully retrieved!")
        
        # Step 8: Summarize each paper using LangChain and OpenAI.
        
        # 8.1: Create an instance of ChatOpenAI using keys from secrets.
        llm = ChatOpenAI(
            openai_api_key=st.secrets["OPENAI_API_KEY"],
            model_name=st.secrets["model_name"]
        )
        
        # 8.2: Define a prompt template that instructs the LLM to generate concise summaries.
        template = """
        You give a concise and easy-to-understand summary of a research paper.
        Question: Summarize: {text}
        Answer: Keep it short and easy to understand.
        """
        prompt_template = PromptTemplate(template=template, input_variables=["text"])

        # 8.3: Build a chain that uses the prompt template and the ChatOpenAI model.
        answer_chain = LLMChain(llm=llm, prompt=prompt_template)

        # 8.4: Insert a divider for visual separation.
        st.divider()

        # 8.5: For each retrieved paper, display the title, link to PDF, and expansions for the summary/abstract.
        for paper, details in papers.items():
            st.subheader(paper)  # paper title
            st.caption(f"[**Paper PDF**]({details[2]})")  # clickable link to PDF

            with st.expander("Click to see AI-Generated Summary"):
                # Summarize the abstract using the LLM
                with st.spinner("Generating AI summary..."):
                    ai_summary = answer_chain.run(details[1])
                st.markdown(f"**Bot's Summary:** {ai_summary}")

            # Show the original abstract if the user wants to see it
            with st.expander("Click to see Original Abstract"):
                st.write(details[1])

            st.divider()

        # Step 9: Build a chatbot that answers questions about the retrieved papers.

        # 9.1: Section heading
        st.header("Chatbot: Ask About the Papers")
        st.write(
            "Got questions about these papers? Ask the chatbot below! It uses the paper summaries to respond."
        )

        # 9.2: Maintain a conversation state (memory) in session_state.
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 9.3: Display all past messages (user or bot).
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 9.4: Text input for user queries about the papers.
        user_input = st.text_input("Type your question here:")
        if user_input:
            # Show the user’s query in the chat UI
            st.chat_message("user").markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # 9.5: Prompt template for the chatbot to use the summary as context
            chatbot_template = """
            You are a chatbot that helps users understand research papers.
            Papers Summary: {papers}
            Question: {text}
            Answer: Keep it concise and based on the papers provided.
            """
            chatbot_prompt = PromptTemplate(template=chatbot_template, input_variables=["papers", "text"])
            
            # 9.6: Build a chain for the chatbot
            chatbot_chain = LLMChain(llm=llm, prompt=chatbot_prompt)

            # 9.7: Summarize all the papers together to provide context for the chatbot
            papers_summary = ' '.join(summarymaker(details[1]) for details in papers.values())

            # 9.8: Generate an answer to the user’s question
            with st.spinner("Thinking..."):
                response = chatbot_chain.run({
                    "text": user_input,
                    "papers": papers_summary
                })

            # 9.9: Display the bot’s response in the chat UI
            st.chat_message("bot").markdown(response)
            st.session_state.messages.append({"role": "bot", "content": response})

# Step 10: If no topic was provided, prompt the user to enter one.
else:
    st.info("Please enter a topic above to begin your search.")
