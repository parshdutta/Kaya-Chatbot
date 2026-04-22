import os
import re
import google.generativeai as genai

class GeminiChatbot:
    def __init__(self, api_key=None):
        # Use the key provided or look in environment
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        
        if not self.api_key or self.api_key.strip() == "":
            raise ValueError("API key is missing.")

        # Correct initialization for google-generativeai
        genai.configure(api_key=self.api_key)
        
        # Using the current stable model name
        self.model = genai.GenerativeModel('gemini-3-flash-preview')
        self.chat_history = {}
        self.setup_system_prompt()

    def setup_system_prompt(self):
        self.system_prompt = """
        You are Kaya, a supportive mental health AI. 
        Focus on empathy and listening. If asked about non-mental health topics, 
        politely redirect. Do not provide medical prescriptions.
        """

    def generate_response(self, user_id, message):
        """Standard method to match your frontend call."""
        try:
            # We pass the system prompt as context
            chat = self.model.start_chat(history=[])
            full_prompt = f"System Instruction: {self.system_prompt}\n\nUser: {message}"
            
            response = chat.send_message(full_prompt)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"

    def detect_crisis(self, message):
        """Helper for crisis detection."""
        keywords = ["suicide", "kill myself", "end my life", "harm myself", "hurt myself"]
        return any(k in message.lower() for k in keywords)