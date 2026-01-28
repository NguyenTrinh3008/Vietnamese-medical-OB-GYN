#!/usr/bin/env python3
"""
Reranker Agent - Using ViRanker cross-encoder

OPTIMIZED: Handles RERANKING, FILTERING, and CRAG QUALITY GATE
- Reranks ALL retrieved chunks (typically 20-30)
- ADAPTIVE filtering by score threshold based on top chunk quality
  * High quality (>=0.6): lenient threshold (0.1)
  * Borderline (0.4-0.6): moderate threshold (0.25)  
  * Low (<0.4): strict threshold (0.3) + triggers CRAG rewrite
- Returns top-k high-quality chunks

Performance: ~0.8s for 30 chunks (vs 7.5s for LLM grader)
Model: namdp-ptit/ViRanker
"""

import sys
sys.path.append('..')

from async_agents.base import BaseAgent, AgentState


class RerankerAgent(BaseAgent):
    """
    Reranking + Filtering Agent with ViRanker cross-encoder
    
    Optimization Strategy:
    - Replace AsyncRetrievalGrader (LLM-based, 7.5s) 
    - Use ViRanker for both ranking AND relevance filtering
    - Cross-encoders are as accurate as LLMs for relevance scoring
    - But 10x faster (local inference vs API calls)
    
    Input: All retrieved chunks (20-30 from retriever)
    Process: 
      1. Rerank ALL chunks with ViRanker scores
      2. Filter low-relevance chunks (score < threshold)
      3. Return top-k high-quality chunks
    
    Model: namdp-ptit/ViRanker (Vietnamese cross-encoder)
    """
    
    def __init__(self, reranker, score_threshold: float = None, top_k: int = 10, 
                 crag_quality_threshold: float = 0.4, adaptive_filtering: bool = True):
        """
        Args:
            reranker: ViRanker cross-encoder model
            score_threshold: Minimum score to keep chunk (default None = adaptive)
                           If set explicitly, disables adaptive filtering
            top_k: Maximum chunks to return (default 10)
            crag_quality_threshold: CRAG quality gate - if best chunk < this, trigger rewrite
                                   (0.4 = moderate quality requirement)
            adaptive_filtering: If True, adjust threshold based on top chunk quality
                              - High quality (>=0.6): threshold = 0.1 (lenient)
                              - Borderline (0.4-0.6): threshold = 0.25 (moderate)
                              - Low (<0.4): triggers CRAG rewrite anyway
        """
        super().__init__(name="Reranker")
        self.reranker = reranker  # ViRanker cross-encoder
        self.base_score_threshold = score_threshold  # User-set threshold (None = adaptive)
        self.adaptive_filtering = adaptive_filtering and score_threshold is None
        self.top_k = top_k  # Max chunks to return
        self.crag_quality_threshold = crag_quality_threshold  # CRAG gate
    
    def execute(self, state: AgentState) -> AgentState:
        """Execute reranking + filtering + CRAG quality check"""
        cands = state.candidate_chunks
        
        if not cands:
            new_state = state.copy()
            new_state.reranked_chunks = []
            new_state.metadata["reranker_stats"] = {
                "total_chunks": 0,
                "filtered_chunks": 0,
                "kept_chunks": 0,
                "top_score": 0.0,
                "crag_quality_check": "no_chunks"
            }
            new_state.metadata["trigger_crag_rewrite"] = True  # No chunks → rewrite
            return new_state
        
        verbose = state.metadata.get("verbose", False)
        
        if verbose:
            print(f"🔄 Reranking {len(cands)} chunks with ViRanker...")
        
        # Rerank with ViRanker cross-encoder
        query = state.query
        pairs = [(query, c["text"]) for c in cands]
        scores = self.reranker.predict(pairs)
        
        # Update scores
        for i, c in enumerate(cands):
            c["rerank_score"] = float(scores[i])
        
        # Sort by rerank score (descending)
        reranked = sorted(cands, key=lambda x: x["rerank_score"], reverse=True)
        
        # CRAG QUALITY GATE: Check if best chunk has sufficient quality
        top_score = reranked[0]["rerank_score"] if reranked else 0.0
        trigger_crag_rewrite = False
        quality_status = "good"
        
        if top_score < self.crag_quality_threshold:
            # Best chunk quality insufficient → Trigger CRAG corrective action
            trigger_crag_rewrite = True
            quality_status = "insufficient"
            
            if verbose:
                print(f"⚠️ CRAG Quality Gate: Top chunk score ({top_score:.3f}) < {self.crag_quality_threshold}")
                print(f"   Quality insufficient for reliable answer")
                print(f"   → Will trigger query rewriting (CRAG corrective action)")
        
        
        # ADAPTIVE FILTERING: Adjust threshold based on top chunk quality
        if self.adaptive_filtering:
            # Determine threshold based on top score
            if top_score >= 0.6:
                # High quality retrieval → Lenient filtering
                score_threshold = 0.1
                quality_tier = "high"
            elif top_score >= self.crag_quality_threshold:
                # Borderline quality → Moderate filtering to reduce noise
                score_threshold = 0.25
                quality_tier = "borderline"
            else:
                # Low quality → Will trigger CRAG anyway, but use strict filtering
                score_threshold = 0.3
                quality_tier = "low"
            
            if verbose:
                print(f"   🎯 Adaptive filtering: tier={quality_tier}, threshold={score_threshold:.2f}")
                print(f"      (top={top_score:.3f}, high>=0.6, borderline>=0.4, low<0.4)")
        else:
            # Fixed threshold mode
            score_threshold = self.base_score_threshold or 0.1
            if verbose:
                print(f"   🔧 Fixed filtering: threshold={score_threshold:.2f}")
        
        # Apply threshold filtering
        filtered = [c for c in reranked if c["rerank_score"] >= score_threshold]
        
        # Take top-k
        top_chunks = filtered[:self.top_k]
        
        # Stats
        total = len(cands)
        kept = len(top_chunks)
        filtered_count = total - len(filtered)
        
        if verbose:
            print(f"📊 Reranking results:")
            print(f"   Total chunks: {total}")
            print(f"   Top score: {top_score:.3f}")
            print(f"   CRAG quality: {'✅ PASS' if not trigger_crag_rewrite else '⚠️ INSUFFICIENT'}")
            print(f"   Filtered (score < {score_threshold:.2f}): {filtered_count}")
            print(f"   Kept: {kept}")
            
            if top_chunks:
                scores_str = [f'{r["rerank_score"]:.3f}' for r in top_chunks[:5]]
                print(f"   🏆 Top 5 scores: {scores_str}")
                
                # Print detailed results
                print(f"\n📊 CHI TIẾT TOP {min(10, len(top_chunks))} CHUNKS SAU RERANKING:")
                print("=" * 80)
                for i, chunk in enumerate(top_chunks[:10], 1):
                    print(f"\n🏆 RANK {i} (Score: {chunk['rerank_score']:.3f}):")
                    print(f"   Title: {chunk.get('title', 'N/A')}")
                    print(f"   Chunk ID: {chunk.get('chunk_id', 'N/A')}")
                    print(f"   Source: {chunk.get('source', 'N/A')}")
                    print(f"   Text preview: {chunk.get('text', '')[:150]}...")
                    print("-" * 60)
                print("=" * 80)
        
        new_state = state.copy()
        new_state.reranked_chunks = top_chunks
        new_state.metadata["reranker_stats"] = {
            "total_chunks": total,
            "filtered_chunks": filtered_count,
            "kept_chunks": kept,
            "top_score": top_score,
            "crag_quality_check": quality_status,
            "adaptive_threshold_used": score_threshold if self.adaptive_filtering else None,
            "filtering_mode": "adaptive" if self.adaptive_filtering else "fixed"
        }
        
        # Signal to orchestrator for CRAG behavior
        new_state.metadata["trigger_crag_rewrite"] = trigger_crag_rewrite
        
        return new_state
    
    def validate_input(self, state: AgentState) -> bool:
        """Validate input state"""
        return isinstance(state.candidate_chunks, list)
