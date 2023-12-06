from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, CSVLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_PATH_PDF = 'data/'
DATA_PATH_CSV = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Create vector database
def create_vector_db():
    # Load PDF documents
    pdf_loader = DirectoryLoader(DATA_PATH_PDF, glob='*.pdf', loader_cls=PyPDFLoader)
    pdf_documents = pdf_loader.load()

    # Load CSV documents
    csv_loader = DirectoryLoader(DATA_PATH_CSV, glob='*.csv', loader_cls=CSVLoader)
    csv_documents = csv_loader.load()

    # Combine all documents
    all_documents = pdf_documents + csv_documents

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(all_documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()

