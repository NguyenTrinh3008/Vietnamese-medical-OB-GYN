#!/usr/bin/env python3
"""
Hierarchical Ingest Script for Medical Articles

Ingests articles from all_articles.json with hierarchical structure:
- Each section/subsection content becomes a separate chunk
- Only content is embedded (100-300 tokens each)
- Rich metadata preserved for traceability

Creates new ChromaDB collection: medical_docs_v2
"""

import os
import json
from typing import List, Dict, Any
from tqdm import tqdm

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Plus
import pickle

# === Configuration ===
JSON_PATH = "all_articles.json"
PERSIST_DIR = "chroma_db_v2"
COLLECTION = "medical_docs_v2"
BM25_INDEX_PATH = "bm25_index_v2.pkl"
EMBED_MODEL = "dangvantuan/vietnamese-document-embedding"


def load_articles(json_path: str) -> List[Dict]:
    """Load articles from JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    print(f"📚 Loaded {len(articles)} articles from {json_path}")
    return articles


def flatten_article(article: Dict, article_index: int) -> List[Dict]:
    """
    Flatten hierarchical article structure into chunks
    
    Each chunk contains:
    - text: Only the content (for embedding)
    - metadata: Full context (article title, section info, etc.)
    """
    chunks = []
    article_title = article.get("title", "Unknown")
    article_link = article.get("link", "")
    
    # Use article index for unique IDs (handles duplicate titles)
    article_id = f"art{article_index:04d}"
    
    for section in article.get("chunks", []):
        section_number = section.get("section", "")
        section_title = section.get("title", "")
        section_content = section.get("content", "")
        
        # Skip empty content
        if not section_content.strip():
            continue
        
        # Main section chunk - use index for uniqueness
        chunk_id = f"{article_id}::{section_number}"
        chunks.append({
            "id": chunk_id,
            "text": section_content,
            "metadata": {
                "article_title": article_title,
                "article_link": article_link,
                "section_number": section_number,
                "section_title": section_title,
                "parent_section": None,
                "chunk_type": "section",
                # Combined title for display
                "title": f"{article_title} - {section_title}"
            }
        })
        
        # Subsection chunks
        for subsection in section.get("subsections", []):
            sub_number = subsection.get("section", "")
            sub_title = subsection.get("title", "")
            sub_content = subsection.get("content", "")
            
            if not sub_content.strip():
                continue
            
            sub_chunk_id = f"{article_id}::{sub_number}"
            chunks.append({
                "id": sub_chunk_id,
                "text": sub_content,
                "metadata": {
                    "article_title": article_title,
                    "article_link": article_link,
                    "section_number": sub_number,
                    "section_title": sub_title,
                    "parent_section": section_title,
                    "chunk_type": "subsection",
                    "title": f"{article_title} - {sub_title}"
                }
            })
    
    return chunks


def count_tokens_approx(text: str) -> int:
    """Approximate token count (Vietnamese ~1.5 char/token)"""
    return len(text) // 3  # Conservative estimate


def analyze_chunks(all_chunks: List[Dict]) -> None:
    """Analyze chunk statistics"""
    token_counts = [count_tokens_approx(c["text"]) for c in all_chunks]
    
    print(f"\n📊 Chunk Statistics:")
    print(f"   Total chunks: {len(all_chunks)}")
    print(f"   Average tokens: {sum(token_counts) / len(token_counts):.0f}")
    print(f"   Min tokens: {min(token_counts)}")
    print(f"   Max tokens: {max(token_counts)}")
    print(f"   Chunks > 512 tokens: {sum(1 for t in token_counts if t > 512)}")
    print(f"   Chunks < 100 tokens: {sum(1 for t in token_counts if t < 100)}")
    
    # Section vs Subsection
    sections = sum(1 for c in all_chunks if c["metadata"]["chunk_type"] == "section")
    subsections = len(all_chunks) - sections
    print(f"   Sections: {sections}, Subsections: {subsections}")


def create_vectordb(all_chunks: List[Dict], embeddings) -> Chroma:
    """Create ChromaDB vector store from chunks"""
    print(f"\n🔧 Creating ChromaDB collection: {COLLECTION}")
    
    # Clear existing if present
    if os.path.exists(PERSIST_DIR):
        import shutil
        shutil.rmtree(PERSIST_DIR)
        print(f"   Removed existing {PERSIST_DIR}")
    
    # Prepare texts and metadatas
    texts = [c["text"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]
    ids = [c["id"] for c in all_chunks]
    
    # Create Chroma
    vectordb = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids,
        collection_name=COLLECTION,
        persist_directory=PERSIST_DIR
    )
    
    print(f"   ✅ Created {len(texts)} vectors in {PERSIST_DIR}")
    return vectordb


def create_bm25_index(all_chunks: List[Dict]) -> None:
    """Create BM25 index for hybrid search"""
    print(f"\n📝 Creating BM25 index: {BM25_INDEX_PATH}")
    
    doc_texts = []
    for c in all_chunks:
        doc_texts.append({
            'text': c["text"],
            'metadata': {**c["metadata"], 'id': c["id"]}
        })
    
    # Tokenize for BM25
    corpus = [doc['text'].lower().split() for doc in doc_texts]
    bm25_index = BM25Plus(corpus)
    
    # Save
    with open(BM25_INDEX_PATH, 'wb') as f:
        pickle.dump({'index': bm25_index, 'doc_texts': doc_texts}, f)
    
    print(f"   ✅ Created BM25 index with {len(doc_texts)} documents")


def main():
    print("="*60)
    print("🚀 Hierarchical Ingest for Medical RAG V2")
    print("="*60)
    
    # Load articles
    articles = load_articles(JSON_PATH)
    
    # Flatten all articles
    print(f"\n🔄 Flattening hierarchical structure...")
    all_chunks = []
    for idx, article in enumerate(tqdm(articles, desc="Processing articles")):
        chunks = flatten_article(article, idx)
        all_chunks.extend(chunks)
    
    # Analyze
    analyze_chunks(all_chunks)
    
    # Load embeddings
    print(f"\n🔧 Loading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cuda", "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # Create vector DB
    vectordb = create_vectordb(all_chunks, embeddings)
    
    # Create BM25 index
    create_bm25_index(all_chunks)
    
    print("\n" + "="*60)
    print("✅ Hierarchical ingest completed!")
    print(f"   ChromaDB: {PERSIST_DIR}/{COLLECTION}")
    print(f"   BM25: {BM25_INDEX_PATH}")
    print("="*60)
    
    # Quick test
    print("\n🧪 Quick retrieval test...")
    query = "Liều acid folic trong thai kỳ"
    results = vectordb.similarity_search(query, k=3)
    
    print(f"   Query: '{query}'")
    for i, doc in enumerate(results, 1):
        print(f"\n   [{i}] {doc.metadata.get('title', 'N/A')}")
        print(f"       Section: {doc.metadata.get('section_number', 'N/A')}")
        print(f"       Content: {doc.page_content[:100]}...")


if __name__ == "__main__":
    main()
