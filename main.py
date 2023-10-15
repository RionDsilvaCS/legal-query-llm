from fastapi import FastAPI
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain import HuggingFaceHub

load_dotenv()

db_faiss_path = 'vectorstores/db_faiss'

custom_prompt_template = """
    Use the following pieces of information to answer the user's question.
    
    If the user is asking content for filling out the form then take the necessary information from the prompt and fill in the blanks in document.

    Context:{context}
    Question:{question}

    Guide the user and give results which helps the user to solve their problem.
"""

def set_custom_prompt():
    """
        Prompt template for QA retrieval for each vector stores
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", 'question'])
    return prompt

def load_llm():
    
    llm = HuggingFaceHub(repo_id="google/flan-t5-base")

    return llm

def retrieval_qa_chain(llm, prompt, db):
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever = db.as_retriever(search_kwargs={'k':2}),
        return_source_documents=True, 
        chain_type_kwargs={'prompt':prompt}
    )
    return chain

def bot():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device':'cpu'})
    db = FAISS.load_local(db_faiss_path, embeddings=embedding)
    llm = load_llm()
    prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, prompt=prompt, db=db)
    return qa

def final_result(query):
    result=bot()
    response = result({'query':query})
    return response


app = FastAPI()

class Msg(BaseModel):
    msg: str


@app.get("/")
async def root():
    return {"message": "Hello World. Welcome to FastAPI!"}


@app.post("/predict")
async def LLM_Pred(inp: Msg):
    res = final_result(inp.msg)
    return {"message": str(res['result'])}


