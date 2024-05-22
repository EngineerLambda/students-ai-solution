import streamlit as st
import os
from pymongo import MongoClient
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import ServiceContext
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex

# web app config
st.set_page_config(page_title="Chat with ELE 562 notes", layout="wide", page_icon="📙")
st.title("Chat with ELE 562 notes")

# load credentials
secrets = st.secrets

os.environ["GOOGLE_API_KEY"] = secrets["GOOGLE_API_KEY"]
ATLAS_URI = secrets["ATLAS_URI"]
DB_NAME = secrets["DB_NAME"]
COLLECTION_NAME = secrets["COLLECTION_NAME"]
INDEX_NAME = secrets["INDEX_NAME"]

# set up mongo client
mongodb_client = MongoClient(ATLAS_URI)

# load embedding model
embed_model = GeminiEmbedding(model_name="models/embedding-001")

# using free google gemini-model API as llm
llm = Gemini(model_name="models/gemini-pro")

# llama_index service context
service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)

# vector store access
vector_store = MongoDBAtlasVectorSearch(
    mongodb_client=mongodb_client,
    db_name=DB_NAME,
    collection_name=COLLECTION_NAME,
    index_name=INDEX_NAME,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store, service_context=service_context
)

query_llm = index.as_query_engine()

# chat interface for consistent queries
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display for all the messages
for message, kind in st.session_state.messages:
    with st.chat_message(kind):
        st.markdown(message)

prompt = st.chat_input("Ask your questions ...")

if prompt:
    # Handling prompts and rendering to the chat interface
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append([prompt, "user"])  # updating the list of prompts

    with st.spinner("Generating response"):
        answer = query_llm.query(prompt)
        if answer:
            st.chat_message("ai").markdown(answer)
            st.session_state.messages.append([answer, "ai"])
