import asyncio
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

# Initialize the chat model with specific configurations
chat_model = ChatOllama(
    model="qwen2.5:0.5b",
    temperature=0.3,  # Lower temperature for more deterministic outputs
    base_url="http://localhost:11434",
)

# Define a prompt for generating a basic function in a data science project
messages = [
    SystemMessage(
        content="You are a data scientist who writes efficient Python code."
    ),
    HumanMessage(
        content=(
            "Given a DataFrame with columns 'product', 'year', and 'sales', calculates the total sales for each product over the specified years. ")
    ),
]

# Invoke the model and print the generated function
response = chat_model.invoke(messages)
print(response.content)

print("\n\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\n")

async def generate_async():
    response = await chat_model.ainvoke(messages)
    return response.content

# In async context
# Exécution dans un contexte asynchrone
async def main():
    result = await generate_async()
    print(result)

# Lancement du programme asynchrone
asyncio.run(main())

