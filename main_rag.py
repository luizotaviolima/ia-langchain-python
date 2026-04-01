from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    api_key=api_key
    )

embeddings = OpenAIEmbeddings()

files = [
    "documentos\GTB_standard_Nov23.pdf", 
    "documentos\GTB_gold_Nov23.pdf",
    "documentos\GTB_platinum_Nov23.pdf"
    ]

documents = sum(
    [
        PyPDFLoader(file).load() for file in files
        ], []
)

parts = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100
    ).split_documents(documents)

dados_recuperados = FAISS.from_documents(
    parts, embeddings
    ).as_retriever(search_kwargs={"k": 2})

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Responda usando exclusivamente as informações fornecidas."),
        ("human", "{query}\n\nContexto: \n{context}\n\nResposta:")
    ]
)

chain = prompt | model | StrOutputParser()

def anwer_query(question:str):
    trechos = dados_recuperados.invoke(question)
    context = "\n".join(um_trecho.page_content for um_trecho in trechos)
    return chain.invoke({
        "query": question, "context": context})

print(anwer_query("Como devo proceder caso tenha um item comprado roubado e caso eu tenha o cartão Platinum?"))