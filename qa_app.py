
import os
import PyPDF2
import random
import itertools
import streamlit as st
from io import StringIO
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import SVMRetriever
from langchain.chains import QAGenerationChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import EnsembleRetriever


st.set_page_config(page_title="PDF Analyzer",page_icon=':shark:')



prompt_str ="""Vous êtes fourni avec un contexte ci-dessous
et ensuite il y a une question.
Vous devez répondre à la question en utilisant ce contexte.
Si vous ne connaissez pas la réponse ou ne pouvez pas la trouver, dites DÉSOLÉ, JE NE SAIS PAS :

Contexte : {contexte}

Question : {question}

Réponse : """

from langchain.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template(prompt_str)

from langchain.schema.output_parser import StrOutputParser
output_parser = StrOutputParser()

from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough
)

@st.cache_data
def load_docs(files):
    st.info("`Reading doc ...`")
    all_text = ""
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1]
        if file_extension == ".pdf":
            pdf_reader = PyPDF2.PdfReader(file_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            all_text += text
        elif file_extension == ".txt":
            stringio = StringIO(file_path.getvalue().decode("utf-8"))
            text = stringio.read()
            all_text += text
        else:
            st.warning('Please provide txt or pdf.', icon="⚠️")
    return all_text




@st.cache_resource
def create_retriever(_embeddings, splits, retriever_type):
    try:
        vectorstore = FAISS.from_texts(splits, _embeddings)
    except (IndexError, ValueError) as e:
        st.error(f"Error creating vectorstore: {e}")
        return
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})


    return retriever, vectorstore

@st.cache_resource
def split_texts(text, chunk_size, overlap, split_method):

    # Split texts
    # IN: text, chunk size, overlap, split_method
    # OUT: list of str splits

    st.info("`Splitting doc ...`")

    split_method = "RecursiveTextSplitter"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap)

    splits = text_splitter.split_text(text)
    if not splits:
        st.error("Failed to split document")
        st.stop()

    return splits

@st.cache_data
def generate_eval(text, N, chunk):

    # Generate N questions from context of chunk chars
    # IN: text, N questions, chunk size to draw question from in the doc
    # OUT: eval set as JSON list

    st.info("`Generating sample questions ...`")
    n = len(text)
    starting_indices = [random.randint(0, n-chunk) for _ in range(N)]
    sub_sequences = [text[i:i+chunk] for i in starting_indices]
    chain = QAGenerationChain.from_llm(ChatOpenAI(temperature=0))
    eval_set = []
    for i, b in enumerate(sub_sequences):
        try:
            qa = chain.run(b)
            eval_set.append(qa)
            st.write("Creating Question:",i+1)
        except:
            st.warning('Error generating question %s.' % str(i+1), icon="⚠️")
    eval_set_full = list(itertools.chain.from_iterable(eval_set))
    return eval_set_full


# ...

def main():
    
    # Add custom CSS
    st.markdown(
        """
        <style>
        
        #MainMenu {visibility: hidden;
        # }
            footer {visibility: hidden;
            }
            .css-card {
                border-radius: 0px;
                padding: 30px 10px 10px 10px;
                background-color: black;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 10px;
                font-family: "IBM Plex Sans", sans-serif;
            }
            
            .card-tag {
                border-radius: 0px;
                padding: 1px 5px 1px 5px;
                margin-bottom: 10px;
                position: absolute;
                left: 0px;
                top: 0px;
                font-size: 0.6rem;
                font-family: "IBM Plex Sans", sans-serif;
                color: white;
                background-color: green;
                }
                
            .css-zt5igj {left:0;
            }
            
            span.css-10trblm {margin-left:0;
            }
            
            div.css-1kyxreq {margin-top: -40px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.image("img/logo1.png")

    st.write(
    f"""
    <div style="display: flex; align-items: center; margin-left: 0;">
        <h1 style="display: inline-block;">PDF Analyzer</h1>
        <sup style="margin-left:5px;font-size:small; color: green;">beta</sup>
    </div>
    """,
    unsafe_allow_html=True,
        )
    
    st.sidebar.title("Some example questions and answers")
    
    embedding_option = "OpenAI Embeddings"
    splitter_type = "RecursiveCharacterTextSplitter"
    retriever_type = "SIMILARITY SEARCH"

    if 'openai_api_key' not in st.session_state:
        openai_api_key = st.text_input(
            'Please enter your OpenAI API key or [get one here](https://platform.openai.com/account/api-keys)', value="", placeholder="Enter the OpenAI API key which begins with sk-")
        if openai_api_key:
            st.session_state.openai_api_key = openai_api_key
            os.environ["OPENAI_API_KEY"] = openai_api_key
        else:
            #warning_text = 'Please enter your OpenAI API key. Get yours from here: [link](https://platform.openai.com/account/api-keys)'
            #warning_html = f'<span>{warning_text}</span>'
            #st.markdown(warning_html, unsafe_allow_html=True)
            return
    else:
        os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key

    uploaded_files = st.file_uploader("Upload a PDF or TXT Document", type=[
                                      "pdf", "txt"], accept_multiple_files=True)

    if uploaded_files:
        # Check if last_uploaded_files is not in session_state or if uploaded_files are different from last_uploaded_files
        if 'last_uploaded_files' not in st.session_state or st.session_state.last_uploaded_files != uploaded_files:
            st.session_state.last_uploaded_files = uploaded_files
            if 'eval_set' in st.session_state:
                del st.session_state['eval_set']

        # Load and process the uploaded PDF or TXT files.
        loaded_text = load_docs(uploaded_files)
        st.write("Documents uploaded and processed.")

        # Split the document into chunks
        splits = split_texts(loaded_text, chunk_size=1000,
                             overlap=0, split_method=splitter_type)

        # Display the number of text chunks
        num_chunks = len(splits)
        st.write(f"Number of text chunks: {num_chunks}")

        # Embed using OpenAI embeddings
        if embedding_option == "OpenAI Embeddings":
            embeddings = OpenAIEmbeddings()
        retriever, vectorstore = create_retriever(embeddings, splits, retriever_type)

        

        # Initialize the RetrievalQA chain with streaming output
        callback_handler = StreamingStdOutCallbackHandler()
        callback_manager = AsyncCallbackManager([callback_handler])

        chat_openai = ChatOpenAI(streaming=True, callback_manager=callback_manager, verbose=True, temperature=0)

        retriever_mquery = MultiQueryRetriever.from_llm(
            retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}), llm=chat_openai)

        # ensemble_retriever = EnsembleRetriever(
        #     retrievers=[retriever_mquery, retriever], weights=[0.8, 0.2]
        # )

        retrieval = RunnableParallel(
            {"contexte": retriever_mquery, "question": RunnablePassthrough()}
        )
        chain = retrieval | prompt | chat_openai | output_parser




        # Check if there are no generated question-answer pairs in the session state
        if 'eval_set' not in st.session_state:
            # Use the generate_eval function to generate question-answer pairs
            num_eval_questions = 6
            st.session_state.eval_set = generate_eval(
                loaded_text, num_eval_questions, 3000)

       # Display the question-answer pairs in the sidebar with smaller text
        for i, qa_pair in enumerate(st.session_state.eval_set):
            st.sidebar.markdown(
                f"""
                <div class="css-card">
                <span class="card-tag">Question {i + 1}</span>
                    <p style="font-size: 12px;">{qa_pair['question']}</p>
                    <p style="font-size: 12px;">{qa_pair['answer']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # <h4 style="font-size: 14px;">Question {i + 1}:</h4>
            # <h4 style="font-size: 14px;">Answer {i + 1}:</h4>
        st.write("Ready to answer questions.")


        
        # Question and answering
        user_question = st.text_input("Enter your question in french or english (french preferred for best results): ")
        if user_question:
            answer = chain.invoke(user_question)
            st.write("Answer:", answer)


if __name__ == "__main__":
    main()