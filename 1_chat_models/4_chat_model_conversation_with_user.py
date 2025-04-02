from langchain_ollama import OllamaLLM
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Load environment variables from .env

# Create a ChatOpenAI model
model = OllamaLLM(model="mistral")

chat_history = []  # Use a list to store messages

# Set an initial system message (optional)
system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history.append(system_message)  # Add system message to chat history

# Chat loop
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))  # Add user message

    # Get AI response using history
    result = model.invoke(chat_history)
    
    chat_history.append(AIMessage(content=result))  # Add AI message

    print(f"AI: {result}")


print("---- Message History ----")
print(chat_history)
