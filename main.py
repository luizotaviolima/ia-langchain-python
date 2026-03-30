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


parser = JsonOutputParser(pydantic_object=Destiny)

prompt_city = PromptTemplate(
    template="""
    Sugira uma cidade dado o meu interesse por {interesse}.
    {output_format}
    """,
    input_variables=["interesse"],
    partial_variables={"output_format": parser.get_format_instructions()}

)

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=api_key)

chain = prompt_city | model | parser

response = chain.invoke(
    {"interesse": "praias"}
)
print(response)
