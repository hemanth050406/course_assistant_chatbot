import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# --- Load environment variables ---
load_dotenv()
groq_api = os.getenv("GROQ_API_KEY")
if not groq_api:
    st.error("GROQ_API_KEY not found in .env file. Please add it.")
    st.stop()

# --- Initialize LLM & Embeddings ---
llm = ChatGroq(api_key=groq_api, model="llama-3.3-70b-versatile", temperature=0.7, max_tokens=512)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- Tools ---
@tool
def get_course_content(url: str) -> str:
    """
    Fetches and returns raw text content from the given course URL.
    """
    loader = WebBaseLoader(url)
    docs = loader.load()
    return " ".join([doc.page_content for doc in docs])

@tool
def retrieve_answers(query: str) -> str:
    """
    Retrieves relevant answers from the loaded course content based on the user's query.
    """
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "I could not find relevant information in the course content."
    return "\n\n".join([doc.page_content for doc in docs])

# --- Initialize vector store ---
vectordb = Chroma(persist_directory="./vectordb", embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k":5})

# --- Agent ---
system_prompt = """
You are a helpful course assistant. Use the tools to fetch course content or answer questions from it.
Be clear, concise, and do not make up information.
"""

main_agent = create_tool_calling_agent(
    llm,
    [get_course_content, retrieve_answers],
    ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
)

agent_executor = AgentExecutor(agent=main_agent, tools=[get_course_content, retrieve_answers], verbose=True)

# --- Streamlit UI ---
st.title("Course Assistant Chatbot")
st.write("Select a course and ask questions about it!")

# Drop-down for courses
course_options = {
    "Full Stack Python Online Training": "https://nareshit.com/courses/full-stack-python-online-training",
    "Full Stack Data Science & AI": "https://nareshit.com/courses/full-stack-data-science-ai-online-training"
}

selected_course = st.selectbox("Select a course:", options=list(course_options.keys()))

# Load content for selected course
if selected_course:
    st.info(f"Fetching and processing course content for **{selected_course}**...")
    url = course_options[selected_course]
    docs = WebBaseLoader(url).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    splits = splitter.split_documents(docs)
    
    vectordb = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./vectordb")
    retriever = vectordb.as_retriever(search_kwargs={"k":5})
    vectordb.persist()
    st.success("Course content loaded successfully!")

# User question input
user_question = st.text_input("Ask your question about the course:")
if user_question:
    response = agent_executor.invoke({"input": user_question})
    final_response = response if isinstance(response, str) else response.get('output', "I couldn't find an answer.")
    st.write("**Chatbot:**", final_response)
