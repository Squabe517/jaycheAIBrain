import getpass
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import openai
import tiktoken

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
conversation_history = [{"role": "system", "content": "You are an expert at utilizing chatGPT."}]
model = "gpt-4o"


def breached_context(context):
    encoding = tiktoken.encoding_for_model('gpt-4o')
    tokens = 0
    for message in context:
        for key, value in message.items():
            tokens += len(encoding.encode(value))
    
    return tokens > 128000

def add_episodic_memory(conversation):
    


while True:
    
    if breached_context(conversation_history):
        
        
    
    # Get user input
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    

    # Add user input to the conversation
    conversation_history.append({"role": "user", "content": user_input})

    # Make API call
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation_history,
    )

    # Get and print the assistant's response
    assistant_response = response.choices[0].message.content
    print(f"ChatGPT: {assistant_response}")

    # Add assistant's response to the conversation
    conversation_history.append({"role": "assistant", "content": assistant_response})
    



    
