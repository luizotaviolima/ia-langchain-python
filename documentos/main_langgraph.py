from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
import asyncio
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")



model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    api_key=api_key
    )

prompt_consultor_praia = ChatPromptTemplate.from_messages(
    [
        ("system", "Apresente-se como Senhora Praia. Você é uma especialista em viagens com destinos para praias."),
        ("human", "{query}")
    ]
)


prompt_consultor_montanha = ChatPromptTemplate.from_messages(
    [
        ("system", "Apresente-se como Senhora Montanha. Você é uma especialista em viagens com destinos para montanhas e atividades radicais."),
        ("human", "{query}")
    ]
)

chain_praia = prompt_consultor_praia | model | StrOutputParser()
chain_montanha = prompt_consultor_montanha | model | StrOutputParser()

class Rota(TypedDict):
    destino: Literal["praia", "montanha"]

prompt_roteador = ChatPromptTemplate.from_messages(
    [
        ("system", "Responda apenas com 'praia' ou 'montanha' dependendo do interesse do usuário."),
        ("human", "{query}")
    ]
)

roteador = prompt_roteador | model.with_structured_output(Rota)

class Estado(TypedDict):
    qyery: str
    destino: Rota
    resposta:str

async def no_roteador(estado: Estado, config=RunnableConfig):
    return {"destino": await roteador.ainvoke({"query": estado["qyery"]}, config)}


async def no_praia(estado: Estado, config=RunnableConfig):
    return {"resposta": await chain_praia.ainvoke({"query": estado["qyery"]}, config)}


async def no_montanha(estado: Estado, config=RunnableConfig):
    return {"resposta": await chain_montanha.ainvoke({"query": estado["qyery"]}, config)}

def escolher_no(estado:Estado)->Literal["praia", "montanha"]:
    return "praia" if estado["destino"]["destino"] == "praia" else "montanha"

grafo = StateGraph(Estado)
grafo.add_node("roteador", no_roteador)
grafo.add_node("praia", no_praia)
grafo.add_node("montanha", no_montanha)

grafo.add_edge(START, "roteador")
grafo.add_conditional_edges("roteador", escolher_no)
grafo.add_edge("praia", END)
grafo.add_edge("montanha", END)

app = grafo.compile()

async def main():
    resposta = await app.ainvoke({"qyery": "Quero escalar montanhas radicais no sul do Brasil."})
    print(resposta["resposta"])


asyncio.run(main())