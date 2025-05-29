from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="qwen2.5:0.5b")
text = """
Write a function that takes a DataFrame with columns 'product', 'year', and 'sales' and calculate the total sales for each product over the specified years.

```python
def calculate_total_sales(df):
"""
# completion_response = llm.invoke(text)
# print(completion_response)


'''
    La différence entre les classes et :ChatOllama vs OllamaLLM

    - OllamaLLM Utilise le point de terminaison pour la complétion de texte/api/generate
    - ChatOllama Utilise le point de terminaison pour les interactions de type chat/api/chat
    - La complétion est meilleure pour la poursuite du code, l’écriture créative et les invites à tour unique
    - Le chat est meilleur pour les conversations à plusieurs tours et lors de l’utilisation des invites système
'''

# Pour les réponses en streaming (montrant les jetons au fur et à mesure qu’ils sont générés), utilisez :llm.stream

for chunk in llm.stream(text):
    print(chunk, end="", flush=True)

#Cela permet d’afficher les résultats en temps réel,
# ce qui rend les applications interactives telles que
# les chatbots plus rapides et plus réactives.