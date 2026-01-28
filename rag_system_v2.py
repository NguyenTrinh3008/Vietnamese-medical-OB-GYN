#!/usr/bin/env python3
"""
RAG System V2 - Hierarchical Chunking Version

Key Differences from V1:
- Uses hierarchical chunks (~100-300 tokens each)
- Uses medical_docs_v2 collection
- NLI grader checks per-chunk instead of concatenated
- Better suited for NLI 512 token limit

Can run in parallel with rag_system.py (V1)
"""

import os
import sys
import asyncio
from typing import List, Dict, Any
from dotenv import load_dotenv
import tiktoken

# Fix import paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import AsyncOpenAI, OpenAI

# Import existing infrastructure
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Plus
from sentence_transformers import CrossEncoder
import pickle

# Import async framework
from async_agents.base import AsyncAgentOrchestrator, AgentState
from async_agents.router import AsyncRouterAgent, AsyncQueryExpansionAgent
from async_agents.retriever_v2 import AsyncRetrieverAgentV2  # V2 retriever
from async_agents.retrieval_grader import AsyncRetrievalGraderAgent
from async_agents.query_decomposition import AsyncQueryDecompositionAgent
from async_agents.query_rewriter import AsyncQueryRewriterAgent

# Import sync agents
from agents import GeneratorAgent, CriticAgent
from agents.reranker_v2 import RerankerAgentV2  # Ensemble reranker
from agents.nli_hallucination_grader_v2 import NLIHallucinationGraderV2

# Load environment variables
load_dotenv()

# === Configuration V2 ===
PERSIST_DIR = "chroma_db_v2"           # NEW: V2 collection
COLLECTION = "medical_docs_v2"          # NEW: V2 collection
BM25_INDEX_PATH = "bm25_index_v2.pkl"   # NEW: V2 BM25
EMBED_MODEL = "dangvantuan/vietnamese-document-embedding"
RERANK_MODEL = "namdp-ptit/ViRanker"
NLI_MODEL_PATH = "NguyenTrinh/mdeberta-v3-medical-nli-vietnamese"  # HuggingFace model - auto-download
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Hyperparameters V2 - More chunks since they're smaller
# V1: 10 chunks x ~1500 tokens = ~15000 tokens context
# V2: 20 chunks x ~166 tokens = ~3300 tokens context (still less!)
TOP_K_TO_LLM = 20

# Advanced RAG Config
ENABLE_RETRIEVAL_GRADER = True
ENABLE_HALLUCINATION_GRADER = True
ENABLE_RAG_FUSION = True
ENABLE_QUERY_DECOMPOSITION = True
NUM_QUERY_VARIANTS = 2

# === Utilities ===
enc = tiktoken.get_encoding("cl100k_base")

def truncate_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to max_tokens"""
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])

# === Initialize Models ===
print("🔧 Initializing RAG System V2 (Hierarchical Chunking)...")
print("🔧 Loading embedding model...")

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cuda", "trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True}
)

# Check if V2 collection exists
if not os.path.exists(PERSIST_DIR):
    print(f"❌ V2 collection not found at {PERSIST_DIR}")
    print("   Please run: python ingest_hierarchical.py")
    sys.exit(1)

# Vector store V2
vectordb = Chroma(
    collection_name=COLLECTION,
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR
)
print(f"   ✅ Loaded V2 collection: {COLLECTION} ({vectordb._collection.count()} chunks)")

# Reranker
print(f"🔧 Loading ViRanker model: {RERANK_MODEL}")
reranker = CrossEncoder(RERANK_MODEL, max_length=512, device="cuda")

# NLI model
print(f"🔧 Loading NLI model: {NLI_MODEL_PATH}")
nli_model = CrossEncoder(NLI_MODEL_PATH, device="cuda")
print("   ✅ NLI model loaded (mDeBERTa-v3 fine-tuned)")

# BM25 index V2
bm25_index = None
doc_texts = []

def load_bm25_index():
    """Load BM25 index V2"""
    global bm25_index, doc_texts
    try:
        if os.path.exists(BM25_INDEX_PATH):
            print(f"📚 Loading BM25 index V2...")
            with open(BM25_INDEX_PATH, 'rb') as f:
                data = pickle.load(f)
            bm25_index = data['index']
            doc_texts = data['doc_texts']
            print(f"   ✅ BM25 V2 loaded with {len(doc_texts)} documents")
        else:
            print(f"❌ BM25 V2 not found at {BM25_INDEX_PATH}")
            print("   Please run: python ingest_hierarchical.py")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading BM25 V2: {e}")
        sys.exit(1)

load_bm25_index()

# OpenAI clients
async_openai_client = AsyncOpenAI()
sync_openai_client = OpenAI()

print("✅ RAG System V2 initialized!")
print(f"   - Collection: {COLLECTION} (hierarchical chunks)")
print(f"   - NLI: Per-chunk checking (512 token compatible)")
print(f"   - RAG-Fusion: ✅ Enabled")

# === Build Async Orchestrator V2 ===
def build_async_orchestrator_v2(
    enable_retrieval_grader=ENABLE_RETRIEVAL_GRADER,
    enable_hallucination_grader=ENABLE_HALLUCINATION_GRADER,
    enable_rag_fusion=ENABLE_RAG_FUSION,
    enable_query_decomposition=ENABLE_QUERY_DECOMPOSITION
) -> AsyncAgentOrchestrator:
    """Build async orchestrator V2 with hierarchical chunks"""
    
    orch = AsyncAgentOrchestrator(
        name="AsyncMedicalRAG_V2",
        openai_client=async_openai_client,
        model=OPENAI_MODEL,
        embedding_model=embeddings
    )
    
    # Phase 1: Async Query Understanding
    orch.add_agent(AsyncRouterAgent(async_openai_client, OPENAI_MODEL))
    
    # Phase 1.2: Query Decomposition
    if enable_query_decomposition:
        orch.add_agent(AsyncQueryDecompositionAgent(
            async_openai_client,
            OPENAI_MODEL,
            max_sub_queries=5,
            min_query_length=15,
            enable_decomposition=True
        ))
    
    # Phase 1.5: Query Expansion (RAG-Fusion)
    if enable_rag_fusion:
        orch.add_agent(AsyncQueryExpansionAgent(async_openai_client, OPENAI_MODEL, NUM_QUERY_VARIANTS))
    
    # Phase 2: Retrieval V2 (higher k for small chunks)
    orch.add_agent(AsyncRetrieverAgentV2(
        vectordb, bm25_index, doc_texts,
        semantic_distance_threshold=0.8,
        bm25_score_threshold=12,
        # V2 config - higher counts for smaller chunks
        semantic_k=30,
        bm25_k=30,
        single_query_k=30,
        aspect_k_per_query=15,
        total_aggregation_limit=40
    ))
    
    # Phase 3: Retrieval Grader (CRAG) - LLM-based relevance grading
    if enable_retrieval_grader:
        orch.add_agent(AsyncRetrievalGraderAgent(
            async_openai_client,
            OPENAI_MODEL,
            confidence_threshold=0.6,  # 60% confident chunks required
            batch_size=10  # Parallel batch processing
        ))
    
    # Phase 4: Reranker V2 - Ensemble scoring
    # Combines RRF (retrieval) + Reranker + Keyword for balanced ranking
    sync_reranker = RerankerAgentV2(
        reranker,
        top_k=20,
        # Ensemble weights
        rrf_weight=0.35,      # Trust retrieval consensus
        reranker_weight=0.45, # Cross-encoder matching
        keyword_weight=0.20,  # Exact term boost
        crag_quality_threshold=0.5  # Trigger rewrite if top_score < 0.5
    )
    orch.add_agent(_wrap_sync_agent(sync_reranker))
    
    # Phase 5: Generator
    sync_generator = GeneratorAgent(sync_openai_client, OPENAI_MODEL, truncate_tokens, TOP_K_TO_LLM)
    orch.add_agent(_wrap_sync_agent(sync_generator))
    
    # Phase 6: NLI Hallucination Detection V2 (per-chunk)
    if enable_hallucination_grader:
        sync_hallucination_grader = NLIHallucinationGraderV2(
            nli_model=nli_model,
            truncate_func=truncate_tokens,
            openai_client=sync_openai_client,
            model=OPENAI_MODEL,
            use_llm_extraction=False
        )
        orch.add_agent(_wrap_sync_agent(sync_hallucination_grader))
    
    # Phase 7: Critic
    sync_critic = CriticAgent(sync_openai_client, OPENAI_MODEL, truncate_tokens, TOP_K_TO_LLM)
    orch.add_agent(_wrap_sync_agent(sync_critic))
    
    return orch


def _wrap_sync_agent(sync_agent) -> 'AsyncBaseAgent':
    """Wrap sync agent for async orchestrator"""
    from async_agents.base import AsyncBaseAgent
    
    class SyncAgentWrapper(AsyncBaseAgent):
        def __init__(self, sync_agent):
            super().__init__(name=sync_agent.name)
            self.sync_agent = sync_agent
        
        async def execute(self, state: AgentState) -> AgentState:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.sync_agent.execute, state)
        
        def validate_input(self, state: AgentState) -> bool:
            return self.sync_agent.validate_input(state)
    
    return SyncAgentWrapper(sync_agent)


# === Main Async API ===
async def ask_async_v2(
    query: str,
    verbose: bool = False,
    enable_retrieval_grader: bool = True,
    enable_hallucination_grader: bool = True,
    enable_rag_fusion: bool = True,
    enable_query_decomposition: bool = True
):
    """Main async entry point for RAG V2"""
    orch = build_async_orchestrator_v2(
        enable_retrieval_grader=enable_retrieval_grader,
        enable_hallucination_grader=enable_hallucination_grader,
        enable_rag_fusion=enable_rag_fusion,
        enable_query_decomposition=enable_query_decomposition
    )
    
    result = await orch.run_with_self_rag(
        query,
        max_iterations=3,
        verbose=verbose
    )
    
    return result.answer


def ask_v2(question: str, verbose: bool = True, **kwargs) -> str:
    """Sync wrapper for RAG V2"""
    return asyncio.run(ask_async_v2(question, verbose, **kwargs))


# === Test Entry Point ===
if __name__ == "__main__":
    import time
    
    print("\n" + "="*60)
    print("Testing RAG System V2 (Hierarchical Chunking)")
    print("="*60)
    
    test_questions = [
        "Rong kinh là gì?",
    ]
    
    for question in test_questions:
        print(f"\n📝 Question: {question}")
        start = time.time()
        
        answer = ask_v2(question, verbose=True)
        
        elapsed = time.time() - start
        print(f"\n{'='*60}")
        print(f"✅ V2 pipeline completed in {elapsed:.2f}s")
        print(f"📄 Answer: {answer[:300]}...")
        print(f"{'='*60}\n")
