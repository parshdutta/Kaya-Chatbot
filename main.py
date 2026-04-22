import os
import uuid
from datetime import datetime
import streamlit as st 
from gemini import GeminiChatbot
from dotenv import load_dotenv
import random
import html

load_dotenv()

# --- Your Custom CSS ---
def inject_custom_css():
    st.markdown("""
    <style>
                
    .main {
    max-width: 700px;
    margin: auto;
    }
                
    /* 🌿 Clean Background */
    .stApp {
        background: #F7F9FC;
        font-family: 'Inter', sans-serif;
        color: #1F2937;
    }

    /* 🧠 Header spacing */
    .main > div {
        padding-top: 20px;
    }

    /* 💬 Chat Container */
    .stChatMessage {
        border-radius: 16px !important;
        padding: 12px 16px !important;
        margin: 10px 0 !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        font-size: 15px;
    }

    /* 👤 User */
    [data-testid="stChatMessage"][aria-label="user"] {
        background: #2563EB;
        color: white !important;
    }

    /* 🤖 Assistant */
    [data-testid="stChatMessage"][aria-label="assistant"] {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        color: #1F2937 !important;
    }

    /* 📥 Input */
    .stChatInput input {
        border-radius: 12px !important;
        border: 1px solid #D1D5DB;
        padding: 10px;
    }

    /* 🔘 Buttons */
    .stButton>button {
        background: #2563EB;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 500;
    }

    .stButton>button:hover {
        background: #1D4ED8;
    }

    /* 📦 Sections */
    .intro-section {
        background: #FFFFFF;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #E5E7EB;
        margin-bottom: 20px;
    }

    /* 📊 Sidebar */
    .stSidebar {
        background: #FFFFFF !important;
        border-right: 1px solid #E5E7EB;
    }

    .sidebar-section {
        background: #F9FAFB;
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 15px;
        border: 1px solid #E5E7EB;
    }

    .disclaimer-micro {
        margin-top: 30px;
        padding: 8px;
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        font-size: 0.75rem; /* Smaller text */
        color: #6b7280;
        line-height: 1.3;
    }

    .helpline-row {
        display: block;
        margin-top: 4px;
        font-weight: 600;
        color: #374151;
    }

    /* ✨ Subtle animation */
    .stChatMessage {
        animation: fadeIn 0.3s ease-in;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(5px); }
        to { opacity: 1; transform: translateY(0); }
    }

    </style>
    """, unsafe_allow_html=True)

class MentalHealthChatbotApp:
    def __init__(self):
        if 'user_id' not in st.session_state:
            st.session_state.user_id = str(uuid.uuid4())
        
        # Initialize the backend class
        if 'logic' not in st.session_state:
            st.session_state.logic = GeminiChatbot(api_key=os.getenv("GEMINI_API_KEY"))

    def process_user_message(self, message):
        # Crisis detection
        crisis_detected = st.session_state.logic.detect_crisis(message)
        
        # Get actual AI response
        response_text = st.session_state.logic.generate_response(st.session_state.user_id, message)

        if crisis_detected:
            crisis_msg = """
            **🚨 Important:** It sounds like you're going through immense pain. Help is available. 
            Call **112** (India) or **988** (US) immediately. 
            Visit [Befrienders Worldwide](https://www.befrienders.org).
            """
            response_text = crisis_msg + "\n\n" + response_text

        return response_text

def display_disclaimer():
    st.markdown("""
    <div class="disclaimer-micro">
        <strong>⚠️ Disclaimer:</strong> Kaya is an AI, not a medical service. 
        For crisis help in India:
        <span class="helpline-row">
            Emergency: 112 | Vandrevala: 9999 666 555 | iCall: 9152987821 | 
        </span>
    </div>
    """, unsafe_allow_html=True)

def main():
    
    st.set_page_config(page_title="Kaya", page_icon="🧠", layout="centered")

    inject_custom_css()
    
    app = MentalHealthChatbotApp()

    st.markdown("""
    <div class="header">
        <h1>🧠 Kaya</h1>
        <p>Your AI Mental Wellness Companion</p>
    </div>
    """, unsafe_allow_html=True)
        
    # Introduction Section
    st.markdown("""
    <div class="intro-section">
        <h2>Supportive Conversations When You Need Them</h2>
        <p>Our AI companion, Kaya, is here to listen and offer a gentle presence. This is your safe space 🌱</p>
    </div>
    """, unsafe_allow_html=True)

    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm Kaya. How are you feeling today?", "is_html": False}]

    # Display Chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=message.get("is_html", False))

    # --- Quick Response Starters ---
    if st.session_state.messages[-1]["role"] == "assistant":
        st.write("Quick Chat Starters:")
        starters = ["I'm feeling a bit stressed lately.", "Can we talk about anxiety?", "Suggest a mindfulness exercise."]
        cols = st.columns(3)
        for i, text in enumerate(starters):
            if cols[i].button(text):
                user_msg = text
                st.session_state.messages.append({"role": "user", "content": user_msg})
                with st.spinner("Kaya is reflecting..."):
                    res = app.process_user_message(user_msg)
                    st.session_state.messages.append({"role": "assistant", "content": res})
                st.rerun()

    # User Input
    if user_input := st.chat_input("Share your thoughts..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Kaya is thinking... 🤔"):
            response = app.process_user_message(user_input)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

    display_disclaimer()

if __name__ == "__main__":
    main()