import os
import streamlit as st
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.vectorstores import FAISS
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory


# set google API key to environment
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# app configs
st.set_page_config(page_title="Youtube Summary and chat", layout="wide", page_icon="📹z")
st.title("Generate Summary and Chat With Youtube Video")


class YTask:
    def __init__(self, url):
        self.url = url
        self.embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.llm = GoogleGenerativeAI(model="gemini-pro", temperature=0)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        prompt_template = """
        You are a question and answer agent for Youtube video transcript documents, you can only reply in ENGLISH LANGUAGE
        You also consider that user might make typographical error in their prompts but you have to be strict on that a little bit, only looking to correct each word and not add words to sentences
        Use the following provided context to answer user questions, if the provided context does not provide answer to the prompt say that you cannot answer
        When users ask you to explain concept that are present in the context, provide more info but keep all answers short and concise, you cannot use more than three sentences
        Question: {question}
        Context:
        {context}
        """
        self.prompt_template = PromptTemplate.from_template(prompt_template)

    def load_yt_transcript(self):
        loader = YoutubeLoader.from_youtube_url(
            youtube_url=self.url, add_video_info=True
        )
        data = loader.load()

        return data

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def split_and_embed(self):
        data = self.load_yt_transcript()
        vector_db = FAISS.from_documents(data, embedding=self.embedding)

        return vector_db.as_retriever()

    def process_query(self, query):
        retriever = self.split_and_embed()

        context = retriever.get_relevant_documents(query=query, k=5)
        context_text = self.format_docs(context)
        formatted_prompt = self.prompt_template.format(
            question=query, context=context_text
        )

        response = self.llm.invoke(formatted_prompt)

        return response


with st.sidebar:
    if "response" not in st.session_state:
        st.session_state.response = ""
    url = st.text_input("Provide youtube video link")
    if url:
        if "llm" not in st.session_state:
            try:
                st.session_state.llm = YTask(url)
                with st.spinner("Loading Video Data ..."):
                    st.session_state.llm.split_and_embed()
                    st.success("Video data loaded, proceed to chats")
            except Exception as e:
                st.error(f"Error {e} encountered while processing the video")

        st.markdown("#### Video summary to guide your questions")
        st.write("You can also read the summary")
        st.write("Scroll down to see summary download button")

        template = """
        You are an AI summarization agent, you will be provided with texts, and can only generate enlish summaries and reply to enlish texts as below:
        {text}
        And you are to provide bullet points kind of summary, dont miss any important information, when there are definitions use an entire bullet point for that
        And make sure nothing worth noting is missed out but make sure to still not provide irrelevant messages that would not benefit the reader.
        """
        if "summarized" not in st.session_state:
            st.session_state.summarized = False
        if not st.session_state.summarized:
            with st.spinner("Generating Summary ..."):
                prompt = PromptTemplate.from_template(template)
                llm = GoogleGenerativeAI(model="gemini-pro")

                llm_chain = LLMChain(llm=llm, prompt=prompt)
                stuff_chain = StuffDocumentsChain(
                    llm_chain=llm_chain, document_variable_name="text"
                )

                response = stuff_chain.run(st.session_state.llm.load_yt_transcript())

                st.session_state.response = response
                st.success("Summary Generated")
                st.session_state.summarized = True

    else:
        st.error("Youtube video link not added yet")
    if st.session_state.response:
        st.write(st.session_state.response)
        download = st.download_button(
            label="Download Summary",
            data=st.session_state.response,
            file_name="summary_video.txt",
            key="download_summary",
        )

if "vid_messages" not in st.session_state:
    st.session_state.vid_messages = []

# Display for all the messages
for message, kind in st.session_state.vid_messages:
    with st.chat_message(kind):
        st.markdown(message)

prompt = st.chat_input("Ask your questions ...")

if prompt:
    # Handling prompts and rendering to the chat interface
    st.chat_message("user").markdown(prompt)
    st.session_state.vid_messages.append(
        [prompt, "user"]
    )  # updating the list of prompts

    with st.spinner("Generating response"):
        answer = st.session_state.llm.process_query(prompt)
        if answer:
            st.chat_message("ai").markdown(answer)
            st.session_state.vid_messages.append([answer, "ai"])
