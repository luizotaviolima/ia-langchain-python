from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import Field, BaseModel
from dotenv import load_dotenv
from langchain.globals import set_debug
import os

set_debug(True)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


class Destiny (BaseModel):
    cidade: str = Field(description="A cidade recomendada para visitar")
    motivo: str = Field(
        description="Motivo pelo qual é interessante visitar essa cidade")


class Restaurants (BaseModel):
    cidade: str = Field(description="A cidade recomendada para visitar")
    restaurantes: str = Field(
        description="Restaurantes recomendados na cidade")


destiny_parser = JsonOutputParser(pydantic_object=Destiny)
restaurants_parser = JsonOutputParser(pydantic_object=Restaurants)


prompt_city = PromptTemplate(
    template="""
    Sugira uma cidade dado o meu interesse por {interesse}.
    {output_format}
    """,
    input_variables=["interesse"],
    partial_variables={"output_format": destiny_parser.get_format_instructions()})

prompt_restaurants = PromptTemplate(
    template="""
    Sugira restaurantes populares entre locais em {cidade}.
    {output_format}
    """,
    partial_variables={"output_format": restaurants_parser.get_format_instructions()})

prompt_culture = PromptTemplate(
    template="""Sugira atividades e locais culturais para em {cidade}.""")


model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=api_key)

chain_1 = prompt_city | model | destiny_parser
chain_2 = prompt_restaurants | model | restaurants_parser
chain_3 = prompt_culture | model | StrOutputParser()

chain = (chain_1 | chain_2 | chain_3)

response = chain.invoke(
    {"interesse": "praias"}
)
print(response)
