from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("knowledge_base.txt", encoding="utf-8")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      
    chunk_overlap=50,
)
docs = splitter.split_documents(documents)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

vector_store = FAISS.from_documents(docs, embeddings)
retriever = vector_store.as_retriever(
    search_kwargs={"k":5}
)