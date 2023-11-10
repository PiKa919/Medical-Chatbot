from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
# import pypdf
import time

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstores/db_faiss"

#create Vector Databases
def create_vector_db():
    start_time = time.time()  # Record the start time
    
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs = {"device": 'cpu'})
    
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    
    end_time = time.time()  # Record the end time
    runtime = end_time - start_time  # Calculate the runtime in seconds

    print(f"Code execution completed in {runtime:.2f} seconds.")
    
if __name__ == "__main__":
    create_vector_db()