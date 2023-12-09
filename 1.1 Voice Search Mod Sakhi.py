import streamlit as st
import speech_recognition as sr
import openai
import os

# Function to perform voice search using the microphone
def perform_voice_search():
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        st.info("Listening... Speak now!")
        try:
            audio = recognizer.listen(source, timeout=5)
            st.info("Processing...")
            recognized_text = recognizer.recognize_google(audio)
            st.success("Recognition successful! You said: " + recognized_text)
            return recognized_text
        except sr.WaitTimeoutError:
            st.warning("Listening timed out. Try again.")
            return None
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand the audio.")
            return None
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")
            return None

# Set OpenAI API key
openai.api_key = "YOUR-OPENAI-API-KEY"
st.title("🦜🔗 With Sakhi")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize model
if "model" not in st.session_state:
    st.session_state.model = "gpt-3.5-turbo"

# User input
user_prompt = st.text_input("Your prompt")
voice_search_button = st.button("🎤")

if voice_search_button:
    user_prompt = perform_voice_search()

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Generate responses
    with st.chat_message("sakhi"):
        message_placeholder = st.empty()
        full_response = ""

        for response in openai.ChatCompletion.create(
            model=st.session_state.model,
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "sakhi", "content": full_response})
