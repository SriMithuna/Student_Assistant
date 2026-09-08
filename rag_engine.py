"""
Core engine for the Study Assistant: RAG + Memory + Quiz generation.
Used by both app.py (CLI) and server.py (web app).
"""

import os
import glob
import hashlib
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
try:
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
except ImportError:
    from langchain_classic.chains import ConversationalRetrievalChain
    from langchain_classic.memory import ConversationBufferMemory

DATA_DIR = "data"          # reads every .txt and .pdf file in this folder now
MODEL_NAME = os.environ.get("STUDY_MODEL", "llama3")  # set STUDY_MODEL=llama3.2 for a faster, lighter model
PERSIST_DIR = "chroma_db"
HASH_FILE = os.path.join(PERSIST_DIR, "_source_hash.txt")


class StudyAssistant:
    def __init__(self):
        self.vectorstore = self._build_vectorstore()
        # num_predict caps how many words/tokens the model can generate,
        # which keeps answers concise and reduces response time
        self.llm = ChatOllama(model=MODEL_NAME, temperature=0.3, num_predict=300)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        # Forces the model to always reply in English, regardless of how
        # short or ambiguous the question is (small models can otherwise
        # randomly switch languages on inputs like "hi" or "hello").
        qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a study assistant. Always answer in English, "
                "clearly and concisely, using only the context below.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n"
                "Answer (in English):"
            ),
        )

        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            # mmr = pulls diverse-but-relevant chunks instead of near-duplicates,
            # fetch_k widens the initial candidate pool before picking the best k
            retriever=self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 15}
            ),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
        )

    def _find_source_files(self):
        files = sorted(
            glob.glob(os.path.join(DATA_DIR, "*.txt"))
            + glob.glob(os.path.join(DATA_DIR, "*.pdf"))
        )
        if not files:
            raise FileNotFoundError(f"No .txt or .pdf files found in '{DATA_DIR}/'")
        return files

    def _source_hash(self, files):
        """Fingerprint of file names + sizes + modified times, to detect changes."""
        h = hashlib.sha256()
        for f in files:
            stat = os.stat(f)
            h.update(f"{f}-{stat.st_size}-{stat.st_mtime}".encode())
        return h.hexdigest()

    def _read_text_file(self, path):
        """Try common encodings in order until one works, instead of relying
        on LangChain's built-in auto-detection (which has a version bug)."""
        for enc in ("utf-8", "cp1252", "latin-1"):
            try:
                with open(path, encoding=enc) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
        # last resort: read raw bytes and drop anything that can't be decoded
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _load_all_documents(self, files):
        from langchain_core.documents import Document
        documents = []
        for f in files:
            if f.lower().endswith(".pdf"):
                documents.extend(PyPDFLoader(f).load())
            else:
                text = self._read_text_file(f)
                documents.append(Document(page_content=text, metadata={"source": f}))
        return documents

    def _build_vectorstore(self):
        files = self._find_source_files()
        current_hash = self._source_hash(files)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Reuse the existing index if the notes haven't changed since last run.
        # This is what makes startup fast after the first time.
        if os.path.exists(PERSIST_DIR) and os.path.exists(HASH_FILE):
            with open(HASH_FILE) as fh:
                saved_hash = fh.read().strip()
            if saved_hash == current_hash:
                print(f"Notes unchanged — reusing existing index for {len(files)} file(s).")
                return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

        print(f"Indexing {len(files)} file(s): {', '.join(os.path.basename(f) for f in files)}")
        documents = self._load_all_documents(files)
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
        os.makedirs(PERSIST_DIR, exist_ok=True)
        with open(HASH_FILE, "w") as fh:
            fh.write(current_hash)
        return vectorstore

    def ask(self, question: str) -> str:
        result = self.chain.invoke({"question": question})
        return result["answer"]

    def quiz(self, topic: str, num_questions: int = 5) -> str:
        docs = self.vectorstore.similarity_search(topic, k=6)
        context = "\n\n".join(d.page_content for d in docs)
        prompt = f"""You are a study assistant. Always respond in English.
Based ONLY on the following course material, create {num_questions}
multiple-choice questions about "{topic}". Each question should have 4 options (A-D)
and clearly indicate the correct answer at the end.

Course material:
{context}
"""
        response = self.llm.invoke(prompt)
        return response.content

    def reset_memory(self):
        self.memory.clear()