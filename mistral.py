from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MistralChatbot:
    def __init__(self, api_key=None):
        """
        Initialize the Mistral chatbot
        """
        # Set API key
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Mistral API key is required")

        # Initialize the chat model
        self.chat_model = ChatMistralAI(
            model="mistral-large",  # Can be changed to "mistral-medium" or "mistral-large"
            mistral_api_key=self.api_key,
            temperature=0.7,
            max_tokens=1024
        )

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Respond concisely and accurately."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # Create the chain
        self.chain = self.prompt | self.chat_model | StrOutputParser()
        
        # Initialize chat history
        self.chat_history = []

    def get_response(self, user_input):
        """
        Get response for user input
        """
        try:
            # Prepare the messages with history
            messages = {
                "chat_history": self.chat_history,
                "input": user_input
            }

            # Get response
            response = self.chain.invoke(messages)

            # Update chat history
            self.chat_history.extend([
                HumanMessage(content=user_input),
                AIMessage(content=response)
            ])

            return response

        except Exception as e:
            return f"Error: {str(e)}"

    def clear_history(self):
        """
        Clear chat history
        """
        self.chat_history = []
        return "Chat history cleared."

def main():
    """
    Main function to run the chatbot
    """
    try:
        # Initialize chatbot
        chatbot = MistralChatbot()
        print("Chatbot initialized! Type 'quit' to exit or 'clear' to clear chat history.")
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'clear':
                print(chatbot.clear_history())
                continue
            elif not user_input:
                continue
                
            response = chatbot.get_response(user_input)
            print("\nAI:", response)

    except Exception as e:
        print(f"Error initializing chatbot: {str(e)}")
        print("\nPlease ensure you have:")
        print("1. Installed required packages: pip install langchain-mistralai python-dotenv")
        print("2. Set your Mistral API key in environment variables or .env file")

if __name__ == "__main__":
    main()