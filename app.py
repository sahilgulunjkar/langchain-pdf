from PyPDF2 import PdfReader
import streamlit as st
import os

from dotenv import load_dotenv
load_dotenv()

# Use Streamlit Secrets when deployed
if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain

from html_template import css, bot_template, user_template


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(raw_text)


def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )
    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0.5
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return conversation_chain


def handle_user_input(user_question):
    if st.session_state.conversation is None:
        st.warning("Please process your PDFs first.")
    else:
        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })

        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.conversation.invoke({
                    "question": user_question
                })

                answer = response["answer"]
                st.write(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })


def main():
    st.set_page_config(
        page_title="LangChain PDFs",
        page_icon=":books:"
    )

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.header("Chat with multiple PDFs :books:")

    with st.sidebar:
        st.subheader("Your document")

        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on Process",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)

                    if not raw_text.strip():
                        st.error("No readable text found in PDFs.")
                    else:
                        text_chunks = get_text_chunks(raw_text)

                        vectorstore = get_vectorstore(text_chunks)

                        st.session_state.conversation = (
                            get_conversation_chain(vectorstore)
                        )

                        st.session_state.messages = []
                        st.success("Processing complete!")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_question = st.chat_input(
        "Ask about your documents..."
    )

    if user_question:
        handle_user_input(user_question)


if __name__ == "__main__":
    main()
