from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

'''
Le code ci-dessus :

    - Utilise une invite flexible qui accepte des variables telles que , , et . Il est ainsi facile de réutiliser le même modèle pour différentes entrées sans réécrire le texte de l’invite.PromptTemplatedate_colgroup_colvalue_colwindow
    - Initialise le code llama :7b model à l’aide de , qui exécute le modèle localement et gère l’inférence avec une faible latence.OllamaLLM
    - Permet d’extraire et de nettoyer la sortie de chaîne brute du modèle, en s’assurant qu’elle ne renvoie que le code généré.StrOutputParser
    - Utilise l’opérateur pour envoyer la sortie de l’invite dans le modèle (), puis la sortie du modèle dans l’analyseur ().|function_prompt | model| StrOutputParser()
    - Utilise la méthode pour exécuter la chaîne entière avec des entrées spécifiques, en déclenchant chaque composant dans l’ordre et en renvoyant le résultat final.invoke()

'''

model = OllamaLLM(model="codellama:7b")
function_prompt = PromptTemplate.from_template(
    """
    Write a Python function using pandas that takes a DataFrame with columns '{date_col}', '{group_col}', and '{value_col}'.
    The function should return a new DataFrame that includes a {window}-day rolling average of {value_col} for each {group_col}.
    """
)

# Build the chain
code_chain = function_prompt | model | StrOutputParser()

# Run the chain with specific variable values
chain_response = code_chain.invoke({
    "date_col": "date",
    "group_col": "store_id",
    "value_col": "sales",
    "window": 7
})

print(chain_response)


print("\n\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\n")

test_model = OllamaLLM(model="codellama:7b", temperature=0.3)

test_prompt = PromptTemplate.from_template(
    """
    Given the following Python function:
    ```python
    {code}
    ```
    Write 1–2 simple unit tests for this function using pytest.
    """
)

test_chain = (
    {"code": code_chain}
    | test_prompt
    | test_model
    | StrOutputParser()
)

# Invoke the test chain
test_response = test_chain.invoke({
    "date_col": "date",
    "group_col": "store_id",
    "value_col": "sales",
    "window": 7
})

print(test_response)