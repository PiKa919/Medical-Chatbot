# Importing necessary libraries and modules
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from typing import Dict, Optional
import chainlit as cl

# Defining the path to the FAISS database
DB_FAISS_PATH = "vectorstores/db_faiss"

custom_prompt_template = """To provide you with accurate medical information, please use the following context and guidelines:

Medical Context:
{context}

User's Medical Query:
{question}

Guidelines:
- Please provide evidence-based medical information.
- If you're uncertain or don't know the answer, state that clearly.
- Avoid making medical diagnoses; instead, offer general advice.
- If the term is not medical related, don't answer the question.
Your Helpful Response:"""

# Defining a function to set the custom prompt template
def set_custom_prompt_template():
    """
    Prompt Template for QARetrival for each vector stores
    """
    
    prompt = PromptTemplate(template = custom_prompt_template, input_variables=['context', 'question'])
    
    return prompt

# Defining a function to load the LLM model
def load_llm():
    llm = CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type = "llama",
        max_new_tokens = 512, # preffered max = 8192
        temperature = 0.3, #controls the randomness of the output
    )
    return llm


# Defining a function to create a retrieval QA chain
# def retrieval_qa_chain(llm, prompt, db):
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriver=db.as_retriever(search_kwargs = {"k": 2}),
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": prompt},
#     )
    
#     return qa_chain

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

# Defining a function to create a QA bot
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs = {"device": 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt_template()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    
    return qa

# Defining a function to get the final result of the QA bot
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response


##Chainlit  
#https://docs.chainlit.io/get-started/overview
from typing import Dict, Optional
import chainlit as cl


# Defining a function to start the chatbot
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Hello, I am a QA bot. Please ask me a question.")
    await msg.send()
    msg.content = "Hi, Welcome to Medical Bot. Please ask me a question."
    await msg.update()
    cl.user_session.set("chain", chain)

    
# Defining a function to handle messages sent to the chatbot
@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]
    
    if sources:
        answer += f"\n\n\nSources: {str(sources)}"
    else:
        answer += "\nNo sources found."
        
    await cl.Message(content=answer).send()