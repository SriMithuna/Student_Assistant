"""
AI Learning & Study Assistant
------------------------------
A simple local chatbot that:
1. Reads your course notes (RAG - searches them for relevant info)
2. Remembers the conversation (Memory)
3. Can generate quizzes from the material

Requires: Ollama running locally with a model pulled (e.g. `ollama pull llama3`)
"""

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# ---------- CONFIG ----------
DATA_FILE = "data/sample_notes.txt"   # swap this for your own notes file later
MODEL_NAME = "llama3"                 # change to whatever model you've pulled in Ollama
PERSIST_DIR = "chroma_db"             # where the searchable index is stored


def build_vectorstore():
    """Load the notes, split them into chunks, and build a searchable index."""
    print("Loading and indexing your notes...")
    loader = TextLoader(DATA_FILE)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
    return vectorstore


def build_chatbot(vectorstore):
    """Create the conversational RAG chain with memory."""
    llm = ChatOllama(model=MODEL_NAME, temperature=0.3)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )
    return chain


def generate_quiz(llm, vectorstore, topic, num_questions=5):
    """Generate a quiz based on the notes about a given topic."""
    docs = vectorstore.similarity_search(topic, k=4)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""Based ONLY on the following course material, create {num_questions}
multiple-choice questions about "{topic}". Each question should have 4 options (A-D)
and clearly indicate the correct answer at the end.

Course material:
{context}
"""
    response = llm.invoke(prompt)
    return response.content


def main():
    if not os.path.exists(DATA_FILE):
        print(f"Could not find {DATA_FILE}. Add your notes there first.")
        return

    vectorstore = build_vectorstore()
    chain = build_chatbot(vectorstore)
    llm = ChatOllama(model=MODEL_NAME, temperature=0.3)

    print("\nStudy Assistant ready! Ask a question about your notes.")
    print("Commands: 'quiz <topic>' to generate a quiz, 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye! Happy studying.")
            break

        if user_input.lower().startswith("quiz"):
            topic = user_input[4:].strip() or "the material"
            print("\nGenerating quiz...\n")
            quiz = generate_quiz(llm, vectorstore, topic)
            print(quiz, "\n")
            continue

        result = chain.invoke({"question": user_input})
        print("\nAssistant:", result["answer"], "\n")


if __name__ == "__main__":
    main()
