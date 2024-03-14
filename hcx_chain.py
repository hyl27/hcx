import os
import tiktoken
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import chroma
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from hcx import CompletionExecutor
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# os.environ["OPENAI_API_KEY"] =

# ------db 저장되어 있으면 여기서 부터
tokenizer = tiktoken.get_encoding("cl100k_base")


def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)


loader = PyPDFLoader("./data/멜버른 자유여행.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50, length_function=tiktoken_len
)
docs = text_splitter.split_documents(pages)

# text embedding
model_name = "jhgan/ko-sbert-nli"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
# --------------여기까지 주석처리

# db = chroma.Chroma.from_documents(docs, hf, persist_directory="./Mel_tour")
# db = chroma.Chroma(persist_directory="./Mel_tour", embedding_function=hf)
db = Milvus(
    hf,
    collection_name="melbourne",
    connection_args={"host": "milvus.claion.io", "port": "19530"},
)
retriever = db.as_retriever()
# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


hcx = CompletionExecutor()

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)


def load_memory(input):
    print(input)
    return memory.load_memory_variables({})["chat_history"]


template = """You are a chatbot having a conversation with a human.
Answer the question based only on the following context:
{context} 

Previous conversation history:
{chat_history}

Response:"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        ("user", "{content}"),
        MessagesPlaceholder(variable_name="chat_history"),
    ]
)

# chain
chain = (
    {
        "context": retriever,
        "content": RunnablePassthrough(),
        "chat_history": load_memory,
    }
    | prompt
    | hcx
    | StrOutputParser()
)


def invoke_chain(question):
    result = chain.invoke(question)  # 입력 형태
    memory.save_context(
        {"input": question},
        {"output": result},
    )
    print(result)  # 답변 출력


while True:
    question = input("질문: ")
    if question != "quit":
        invoke_chain(question)
        # print(memory)
    else:
        break
