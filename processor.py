from pathlib import Path
from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
import os


def is_playbook(pdf_path: Path):
    name = pdf_path.name.lower()
    return any(keyword in name for keyword in ["playbook", "circuit", "agent", "manual"])


def extract_agents_from_text(text):
    agent_list = ["HYPE", "STRIKE", "CARE", "VISION", "FLOW", "ASSET", "TEAM", "CODE"]
    agent_blocks = {}
    positions = [(agent, text.find(agent)) for agent in agent_list if agent in text]
    positions = [p for p in positions if p[1] != -1]
    positions.sort(key=lambda x: x[1])

    for i in range(len(positions)):
        agent, start = positions[i]
        end = positions[i + 1][1] if i + 1 < len(positions) else len(text)
        agent_blocks[agent] = text[start:end].strip()

    return agent_blocks


def load_all_documents(pdf_paths):
    all_sections = []

    for pdf_path in pdf_paths:
        print(f" Processing PDF: {pdf_path}")
        elements = partition_pdf(
            filename=str(pdf_path),
            strategy="fast",
            extract_images_in_pdf=False
        )

        full_text = "\n".join([el.text.strip() for el in elements if el.text and el.text.strip()])

        if is_playbook(pdf_path):
            agent_blocks = extract_agents_from_text(full_text)
            for agent, content in agent_blocks.items():
                all_sections.append({
                    "title": f"{agent}",
                    "text": content,
                    "source": pdf_path.name,
                    "type": "agent"
                })
        else:
            current_title = "Untitled Section"
            for el in elements:
                if el.text and el.text.strip():
                    text = el.text.strip()
                    if el.category == "Title":
                        current_title = text
                    else:
                        all_sections.append({
                            "title": current_title,
                            "text": text,
                            "source": pdf_path.name,
                            "type": "transcript"
                        })

    return all_sections


def split_into_chunks(sections, chunk_size=700, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "?", "!", " "]
    )
    documents = []
    for sec in sections:
        text = sec["text"].replace("\n", " ")
        chunks = splitter.split_text(text)
        for chunk in chunks:
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "title": sec["title"],
                        "source": sec["source"],
                        "type": sec.get("type", "unknown")
                    }
                )
            )
    return documents

def build_faiss_index(documents):
    if not documents:
        raise ValueError(" No documents provided for indexing.")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

def rag_chain(vectorstore, query):
    print(" Running similarity search...")
    all_docs = vectorstore.similarity_search(query, k=80)
    transcript_docs = [doc for doc in all_docs if doc.metadata.get("type") == "transcript"]
    agent_docs = [doc for doc in all_docs if doc.metadata.get("type") == "agent"]

    relevant_docs = transcript_docs[:30] + agent_docs[:10]

    llm = ChatOpenAI(temperature=0.8, model_name="gpt-4", max_tokens=3000)
    chain = load_qa_with_sources_chain(llm, chain_type="stuff")
    result = chain({"input_documents": relevant_docs, "question": query}, return_only_outputs=True)

    return result["output_text"]
