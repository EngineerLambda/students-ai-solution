import os
import streamlit as st
from langchain_community.document_loaders import YoutubeLoader
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory


# set google API key to environment
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

class YTask:
    def __init__(self, url):
        self.url = url
        self.embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.llm = GoogleGenerativeAI(model="gemini-pro", temperature=0)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    def load_yt_transcript(self):
        loader = YoutubeLoader.from_youtube_url(youtube_url=self.url, add_video_info=True)
        data = loader.load()
        
        return data
    
    def split_and_embed(self):
        data = self.load_yt_transcript()
        vector_db = Chroma.from_documents(data, embedding=self.embedding)
        
        return vector_db.as_retriever()
    
    def process_query(self, query):
        retriever = self.split_and_embed()
        chat_bot = ConversationalRetrievalChain.from_llm(self.llm, retriever, memory=self.memory, verbose=False)
        response = chat_bot.invoke({"question" : query})
        
        return response["answer"]
        

if __name__ == "__main__":
    st.title("Chat With Youtube Video")
    with st.sidebar:
        if "response" not in st.session_state:
                    st.session_state.response = ""
        url = st.text_input("Provide youtube video link")
        if url:
            if "llm" not in st.session_state:
                st.session_state.llm = YTask(url)
                with st.spinner("Loading Video Data ..."):
                    st.session_state.llm.split_and_embed()
                    st.success("Video data loaded, proceed to chats")
                    
            st.write("If you want to load summary instead")
            if st.button("Summarize"):
                
                template = """
                You are an AI summarization agent, you will be provided with texts, and can only generate enlish summaries and reply to enlish texts as below:
                {text}
                And you are to provide bullet points kind of summary, dont miss any important information, when there are definitions use an entire bullet point for that
                And make sure nothing worth noting is missed out but make sure to still not provide irrelevant messages that would not benefit the reader.
                """
                with st.spinner("Generating Summary ..."):
                    prompt = PromptTemplate.from_template(template)
                    llm = GoogleGenerativeAI(model="gemini-pro")
                    
                    llm_chain = LLMChain(llm=llm, prompt=prompt)
                    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
                    
                    response = stuff_chain.run(st.session_state.llm.load_yt_transcript())
                    
                    st.session_state.response = response
                    st.success("Summary Generated")
        
        else:
            st.error("Youtube video link not added yet")
        if st.session_state.response:
            st.write(st.session_state.response)
            download = st.download_button(
                            label="Download Summary",
                            data=st.session_state.response,
                            file_name="summary_video.txt",
                            key="download_summary")   
       
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
        st.session_state.messages.append([prompt, "user"]) # updating the list of prompts 


        with st.spinner("Generating response"):
            answer = st.session_state.llm.process_query(prompt)
            if answer:
                st.chat_message("ai").markdown(answer)
                st.session_state.messages.append([answer, "ai"])
                