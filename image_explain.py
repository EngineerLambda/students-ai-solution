import os
from PIL import Image
import streamlit as st
import google.generativeai as genai


st.header("Chart Explanation from AI")
template = """
Explan this image with the given instruction here
**Instruction : {}
"""
google_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=google_key)

model = genai.GenerativeModel(model_name="gemini-1.5-flash")

image = st.file_uploader("Upload image here")

if "image" not in st.session_state:
    st.session_state.image = None
    
if image:
    st.session_state.image = image
    
    pil_image = Image.open(st.session_state.image)
    st.image(pil_image)
    

if "img_messages" not in st.session_state:
    st.session_state.img_messages = []

# Display for all the messages
for message, kind in st.session_state.img_messages:
    st.markdown(f"**{kind.upper()}**\n\n{message}")
        
# chat history function
def format_history(messages):
    history_template = ""
    for message, kind in messages:
        history_template += f"{kind}: {message}\n"
        
    return history_template
        

prompt = st.chat_input("Provide explanation instructions here")

if prompt and st.session_state.image:
    # Handling prompts and rendering to the chat interface
    st.session_state.img_messages.append([prompt, "user"])  # updating the list of prompts

    with st.spinner("Generating response"):
        try:
            full_prompt = format_history(st.session_state.img_messages)
            response = model.generate_content([full_prompt, pil_image])
            if response:
                output_mkd = f"**USER**\n\n{prompt}\n\n**AI**\n\n{response.text}"
                st.write(output_mkd)
                
            
                st.session_state.img_messages.append([response.text, "ai"])
                
        except Exception as e:
            st.error(f"Error reading Image: {e}")
       

       
   