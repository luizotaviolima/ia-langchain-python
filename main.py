# from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

numero_dias = 7
numero_criancas = 2
atividades = "praia"
cidade = "Balneário Camboriú"

prompt_template = PromptTemplate(
    template="""
    Crie um roteiro de viagem de {numero_dias} dias, 
    para uma família com {numero_criancas} crianças, 
    que buscam atividades relacionadas a {atividades} 
    em {cidade}.
""")

prompt = prompt_template.format(
    numero_dias=numero_dias,
    numero_criancas=numero_criancas,
    atividades=atividades,
    cidade=cidade
)

print("Prompt:\n", prompt)

model =ChatOpenAI(
    model="gpt-3.5-turbo", 
    temperature=0.5, 
    api_key=api_key)

response = model.invoke(prompt)
print(response.content)