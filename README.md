🤖 Course Assistant Chatbot

An AI-powered course assistant built with Streamlit, LangChain, Groq LLM (Llama-3), HuggingFace Embeddings, and ChromaDB.
This chatbot fetches course details directly from NareshIT course web pages, creates embeddings, and allows students to ask questions in real time.

🚀 Features
🌐 Web-based Course Loader – Pulls content from course URLs.
📚 Vector Database with Chroma – Stores and retrieves embeddings efficiently.
🔎 Semantic Search – Retrieves the most relevant answers for user queries.
🛠️ LangChain Agent Tools – Two custom tools:
get_course_content(url) → loads raw text from a course page
retrieve_answers(query) → fetches best-matching content from the vector store
🤖 Groq LLM (Llama-3.3-70B) – Generates natural, helpful responses.
🎛️ Streamlit UI – Simple interface with course selection and Q&A.
