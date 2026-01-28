#!/usr/bin/env python3
"""
Reranker Agent V2.1 - Ensemble Scoring with Soft Matching

KEY INNOVATIONS:
1. RRF Score (từ retriever) - consensus của semantic + BM25
2. Reranker Score (ViRanker) - cross-encoder matching
3. Soft Keyword Match - fuzzy matching với synonyms
4. Soft Penalty - penalty tỷ lệ với mức độ thiếu keyword

IMPROVEMENTS OVER V2:
- Fuzzy Matching với rapidfuzz (chấp nhận typo, biến thể)
- Synonym Expansion (rong kinh ↔ menorrhagia ↔ ra máu kéo dài)
- Soft Penalty thay vì hard -0.3
- Multi-entity query support (so sánh A và B)
"""

import sys
import re
import json
from pathlib import Path
sys.path.append('..')

from async_agents.base import BaseAgent, AgentState
from typing import List, Dict, Set, Optional

# Fuzzy matching
try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    print("⚠️ rapidfuzz not installed, using exact matching")


# ============================================================
# LOAD MEDICAL SYNONYMS FROM JSON FILE
# ============================================================
def load_medical_synonyms() -> Dict[str, List[str]]:
    """Load medical synonyms from JSON file"""
    synonyms_file = Path(__file__).parent.parent / "medical_synonyms.json"
    
    if not synonyms_file.exists():
        print(f"⚠️ medical_synonyms.json not found at {synonyms_file}")
        return {}
    
    try:
        with open(synonyms_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Flatten all categories into one dict
        flat_synonyms = {}
        for category, terms in data.get("categories", {}).items():
            for term, syns in terms.items():
                flat_synonyms[term.lower()] = [s.lower() for s in syns]
        
        return flat_synonyms
    except Exception as e:
        print(f"❌ Error loading synonyms: {e}")
        return {}

# Load synonyms at module level
MEDICAL_SYNONYMS = load_medical_synonyms()

# Build reverse lookup for efficiency
SYNONYM_LOOKUP = {}
for primary, synonyms in MEDICAL_SYNONYMS.items():
    SYNONYM_LOOKUP[primary] = primary  # map to itself
    for syn in synonyms:
        SYNONYM_LOOKUP[syn.lower()] = primary


class RerankerAgentV2(BaseAgent):
    """
    Ensemble Reranker with Soft Matching
    
    V2.1 Improvements:
    - Fuzzy Matching: Handles typos and word variations
    - Synonym Expansion: Recognizes medical term equivalents
    - Soft Penalty: Proportional to keyword match quality
    - Multi-entity Support: Doesn't penalize comparison queries
    """
    
    def __init__(self, reranker, 
                 top_k: int = 20,
                 # Ensemble weights (must sum to 1.0)
                 rrf_weight: float = 0.35,
                 reranker_weight: float = 0.45,
                 keyword_weight: float = 0.20,
                 # Thresholds
                 crag_quality_threshold: float = 0.5,
                 min_score_threshold: float = 0.05,
                 # Soft penalty
                 max_penalty: float = 0.25,  # Max penalty (when keyword_score=0)
                 fuzzy_threshold: int = 85   # Min fuzzy score to count as match
                 ):
        """
        Args:
            reranker: ViRanker cross-encoder
            top_k: Max chunks to return
            rrf_weight: Weight for RRF score (retrieval consensus)
            reranker_weight: Weight for reranker score
            keyword_weight: Weight for soft keyword match score
            crag_quality_threshold: Trigger rewrite if below this
            min_score_threshold: Filter chunks below this final score
            max_penalty: Maximum penalty when no keywords match
            fuzzy_threshold: Minimum fuzzy match score (0-100) to count as match
        """
        super().__init__(name="RerankerV2")
        self.reranker = reranker
        self.top_k = top_k
        
        # Ensemble weights
        self.rrf_weight = rrf_weight
        self.reranker_weight = reranker_weight
        self.keyword_weight = keyword_weight
        
        self.crag_quality_threshold = crag_quality_threshold
        self.min_score_threshold = min_score_threshold
        self.max_penalty = max_penalty
        self.fuzzy_threshold = fuzzy_threshold
    
    def execute(self, state: AgentState) -> AgentState:
        """Execute ensemble reranking with soft matching"""
        cands = state.candidate_chunks
        query = state.query
        verbose = state.metadata.get("verbose", False)
        
        if not cands:
            new_state = state.copy()
            new_state.reranked_chunks = []
            new_state.metadata["trigger_crag_rewrite"] = True
            return new_state
        
        if verbose:
            print(f"🔄 [V2.1] Ensemble reranking with Soft Matching...")
            print(f"   Weights: RRF={self.rrf_weight:.0%}, VR={self.reranker_weight:.0%}, KW={self.keyword_weight:.0%}")
        
        # Step 1: Get ViRanker scores
        pairs = [(query, c["text"]) for c in cands]
        reranker_scores = self.reranker.predict(pairs)
        
        # Step 2: Normalize reranker scores to 0-1
        reranker_min = min(reranker_scores)
        reranker_max = max(reranker_scores)
        reranker_range = reranker_max - reranker_min if reranker_max > reranker_min else 1
        
        # Step 3: Extract and expand query keywords (with synonyms)
        query_terms = self._extract_and_expand_terms(query)
        is_comparison_query = self._is_comparison_query(query)
        
        if verbose:
            print(f"   🔑 Expanded keywords: {query_terms}")
            if is_comparison_query:
                print(f"   📊 Comparison query detected - using lenient penalty")
        
        # Step 4: Compute ensemble scores for each chunk
        scored_chunks = []
        for i, chunk in enumerate(cands):
            # RRF score (from retriever) - already normalized
            rrf_score = chunk.get("rrf_score", 0.0)
            rrf_normalized = min(rrf_score * 30, 1.0)  # Scale 0-0.03 to 0-1
            
            # Reranker score (normalized to 0-1)
            reranker_raw = float(reranker_scores[i])
            reranker_normalized = (reranker_raw - reranker_min) / reranker_range
            
            # Soft Keyword Match Score (0.0 - 1.0) with fuzzy + synonyms
            keyword_score = self._soft_keyword_score(chunk, query_terms)
            
            # Soft Penalty (proportional to missing keywords)
            # If keyword_score = 1.0 → penalty = 0
            # If keyword_score = 0.0 → penalty = max_penalty
            if is_comparison_query:
                # Lenient for comparison queries
                penalty = (1.0 - keyword_score) * (self.max_penalty * 0.3)
            else:
                penalty = (1.0 - keyword_score) * self.max_penalty
            
            # Ensemble final score
            final_score = (
                self.rrf_weight * rrf_normalized +
                self.reranker_weight * reranker_normalized +
                self.keyword_weight * keyword_score
            ) - penalty
            
            # Ensure non-negative
            final_score = max(final_score, 0.0)
            
            # Store all scores for debugging
            chunk["rrf_score_norm"] = rrf_normalized
            chunk["reranker_score"] = reranker_raw
            chunk["reranker_score_norm"] = reranker_normalized
            chunk["keyword_score"] = keyword_score
            chunk["penalty"] = penalty
            chunk["ensemble_score"] = final_score
            
            scored_chunks.append(chunk)
        
        # Step 5: Sort by ensemble score
        reranked = sorted(scored_chunks, key=lambda x: x["ensemble_score"], reverse=True)
        
        # Step 6: Filter by threshold
        filtered = [c for c in reranked if c["ensemble_score"] >= self.min_score_threshold]
        
        # Step 7: Take top-k
        top_chunks = filtered[:self.top_k]
        
        # CRAG quality check
        top_score = top_chunks[0]["ensemble_score"] if top_chunks else 0
        trigger_crag = top_score < self.crag_quality_threshold
        
        if verbose:
            print(f"\n📊 Ensemble Reranking Results:")
            print(f"   Top ensemble score: {top_score:.3f}")
            print(f"   CRAG quality: {'✅ PASS' if not trigger_crag else '⚠️ INSUFFICIENT'}")
            print(f"   Kept: {len(top_chunks)}/{len(cands)}")
            
            if top_chunks:
                print(f"\n🏆 Top 5 chunks:")
                for i, c in enumerate(top_chunks[:5], 1):
                    title = c.get("title", "N/A")[:40]
                    kw_icon = "🔑" if c["keyword_score"] > 0.5 else "  "
                    pen_str = f", pen={c['penalty']:.2f}" if c['penalty'] > 0.01 else ""
                    print(f"   {i}. {kw_icon} [{c['ensemble_score']:.3f}] "
                          f"(rrf={c['rrf_score_norm']:.2f}, vr={c['reranker_score']:.2f}, "
                          f"kw={c['keyword_score']:.2f}{pen_str})")
                    print(f"      {title}...")
        
        new_state = state.copy()
        new_state.reranked_chunks = top_chunks
        new_state.metadata["trigger_crag_rewrite"] = trigger_crag
        new_state.metadata["reranker_stats"] = {
            "mode": "ensemble_v2.1_soft",
            "total_chunks": len(cands),
            "kept_chunks": len(top_chunks),
            "top_ensemble_score": top_score,
            "weights": {
                "rrf": self.rrf_weight,
                "reranker": self.reranker_weight,
                "keyword": self.keyword_weight
            },
            "fuzzy_matching": RAPIDFUZZ_AVAILABLE
        }
        
        return new_state
    
    def _extract_and_expand_terms(self, query: str) -> Set[str]:
        """
        Extract keywords from query and expand with synonyms
        
        Example:
        "Rong kinh là gì?" → {"rong kinh", "menorrhagia", "ra máu kinh nhiều", ...}
        """
        query_lower = query.lower()
        
        # Remove common question words
        stop_patterns = [
            r'\blà gì\b', r'\bnhư thế nào\b', r'\btại sao\b',
            r'\bcó thể\b', r'\bđược không\b', r'\bcần\b',
            r'\bso sánh\b', r'\bkhác nhau\b', r'\bgiống nhau\b'
        ]
        for pattern in stop_patterns:
            query_lower = re.sub(pattern, '', query_lower)
        
        terms = set()
        
        # Define stopwords FIRST (including medical generic terms)
        stopwords = {'là', 'gì', 'có', 'được', 'như', 'thế', 'nào', 'tại', 'sao',
                     'và', 'với', 'cho', 'để', 'từ', 'về', 'trong', 'khi', 'của',
                     'này', 'đó', 'những', 'các', 'một', 'hai', 'ba', 'bốn',
                     # Medical generic terms that appear in almost every document
                     'bệnh', 'bệnh lý', 'triệu chứng', 'điều trị', 'chẩn đoán',
                     'thuốc', 'phương pháp', 'nguyên nhân', 'yếu tố', 'dấu hiệu',
                     'tình trạng', 'biểu hiện', 'xét nghiệm', 'chăm sóc',
                     'pathology', 'tình trạng bệnh', 'disease', 'symptoms',
                     'rối loạn', 'disorder', 'treatment', 'diagnosis'}
        
        # Check for compound medical terms first (but skip generic terms)
        for primary_term, synonyms in MEDICAL_SYNONYMS.items():
            # Skip if primary term is a stopword
            if primary_term in stopwords:
                continue
            if primary_term in query_lower:
                terms.add(primary_term)
                terms.update(synonyms)
            # Also check if any synonym is in query
            for syn in synonyms:
                if syn.lower() in query_lower and syn.lower() not in stopwords:
                    terms.add(primary_term)
                    terms.update(synonyms)
                    break
        
        # Add significant individual words (>3 chars, not stopwords)
        words = query_lower.split()
        for word in words:
            word = re.sub(r'[^\w]', '', word)
            if len(word) > 3 and word not in stopwords:
                terms.add(word)
                # Expand if it's a known synonym
                if word in SYNONYM_LOOKUP:
                    primary = SYNONYM_LOOKUP[word]
                    if primary not in stopwords:  # Skip if primary is stopword
                        terms.add(primary)
                        if primary in MEDICAL_SYNONYMS:
                            terms.update(MEDICAL_SYNONYMS[primary])
        
        # FINAL FILTER: Remove any stopwords that slipped through
        terms = {t for t in terms if t not in stopwords}
        
        return terms
    
    def _is_comparison_query(self, query: str) -> bool:
        """
        Detect if query is comparing multiple entities
        
        Example:
        "So sánh thuốc A và thuốc B" → True
        "Khác nhau giữa X và Y" → True
        "Rong kinh là gì?" → False
        """
        query_lower = query.lower()
        comparison_patterns = [
            r'\bso sánh\b', r'\bkhác nhau\b', r'\bgiống nhau\b',
            r'\bvà\b.*\bkhác\b', r'\bhay\b.*\btốt hơn\b',
            r'\bnên.*hay\b', r'\bchọn.*hay\b'
        ]
        
        for pattern in comparison_patterns:
            if re.search(pattern, query_lower):
                return True
        
        # Check for "X và Y" pattern
        if ' và ' in query_lower or ' hay ' in query_lower:
            # Simple heuristic: if there are 2+ entities mentioned
            entities = re.findall(r'\b\w+\b', query_lower)
            if len(entities) > 6:  # Complex query likely comparing
                return True
        
        return False
    
    def _soft_keyword_score(self, chunk: Dict, query_terms: Set[str]) -> float:
        """
        Calculate soft keyword match score using fuzzy matching
        
        Returns 0.0 - 1.0 based on:
        - Exact matches get 1.0
        - Fuzzy matches (>85% similarity) get proportional score
        - No match gets 0.0
        """
        if not query_terms:
            return 0.0
        
        chunk_text = chunk.get("text", "").lower()
        chunk_title = chunk.get("title", "").lower()
        combined = chunk_text + " " + chunk_title
        
        matched_terms = 0
        total_match_quality = 0.0
        
        for term in query_terms:
            term_lower = term.lower()
            
            # First try exact match (fastest)
            if term_lower in combined:
                matched_terms += 1
                total_match_quality += 1.0
                continue
            
            # Then try fuzzy match if rapidfuzz available
            if RAPIDFUZZ_AVAILABLE and len(term_lower) >= 4:
                # Use partial_ratio for substring matching
                fuzzy_score = fuzz.partial_ratio(term_lower, combined)
                
                if fuzzy_score >= self.fuzzy_threshold:
                    matched_terms += 1
                    total_match_quality += fuzzy_score / 100.0
        
        if matched_terms == 0:
            return 0.0
        
        # Score = average match quality, weighted by coverage
        coverage = matched_terms / len(query_terms)
        avg_quality = total_match_quality / matched_terms
        
        # Combined score: both coverage and quality matter
        # At least 50% if any match, scaling up based on coverage and quality
        if matched_terms > 0:
            score = 0.5 + 0.5 * (coverage * avg_quality)
            return min(score, 1.0)
        
        return 0.0
    
    def validate_input(self, state: AgentState) -> bool:
        return isinstance(state.candidate_chunks, list)
