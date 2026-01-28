#!/usr/bin/env python3
"""
Async Retriever V2 - Higher chunk counts for hierarchical chunking

Key differences from V1:
- semantic_k=30 instead of 20
- bm25_k=30 instead of 20
- Single query retrieval k=30 instead of 10
- Aspect-based retrieval k=15 instead of 10
- Total aggregation limit=40 instead of 30

Optimized for smaller hierarchical chunks (~166 tokens avg)
"""

import asyncio
import time
from typing import List, Dict, Any

from async_agents.base import AsyncBaseAgent, AgentState
from async_agents.utils import async_chromadb_search, async_bm25_search


class AsyncRetrieverAgentV2(AsyncBaseAgent):
    """
    Async Retriever V2 - Higher retrieval counts for small chunks
    
    V2 chunks are ~166 tokens avg vs V1 ~1500 tokens
    → Need ~9x more chunks for same context
    → Conservative: 2-3x more chunks
    """
    
    def __init__(self, vectordb, bm25_index, doc_texts, 
                 semantic_distance_threshold: float = 0.5,
                 bm25_score_threshold: float = 5.0,
                 # V2 config - higher counts
                 semantic_k: int = 30,
                 bm25_k: int = 30,
                 single_query_k: int = 30,
                 aspect_k_per_query: int = 15,
                 total_aggregation_limit: int = 40):
        super().__init__(name="AsyncRetrieverV2")
        self.vectordb = vectordb
        self.bm25_index = bm25_index
        self.doc_texts = doc_texts
        
        # Score thresholds
        self.semantic_distance_threshold = semantic_distance_threshold
        self.bm25_score_threshold = bm25_score_threshold
        
        # V2 retrieval counts (higher for small chunks)
        self.semantic_k = semantic_k
        self.bm25_k = bm25_k
        self.single_query_k = single_query_k
        self.aspect_k = aspect_k_per_query
        self.total_limit = total_aggregation_limit
        
        # Vietnamese stop words
        self.stop_words = {
            'là', 'gì', 'có', 'được', 'như', 'thế', 'nào', 'tại', 'sao', 'khi',
            'các', 'và', 'trong', 'của', 'với', 'cho', 'để', 'từ', 'về', 'trên', 'dưới',
            'này', 'đó', 'những', 'một', 'hai', 'ba', 'bốn', 'năm'
        }
    
    async def execute(self, state: AgentState) -> AgentState:
        """Execute with V2 retrieval config (Metadata Filtering DISABLED for now)"""
        if not state.plan.get("need_retrieval", True):
            if state.metadata.get("verbose"):
                print("⏭️ Skip retrieval")
            new_state = state.copy()
            new_state.candidate_chunks = []
            return new_state
        
        verbose = state.metadata.get("verbose", False)
        
        # Check for decomposition
        use_decomposition = state.metadata.get("use_decomposition", False)
        decomposition_result = state.metadata.get("decomposition_result", {})
        
        if use_decomposition and decomposition_result.get("sub_queries"):
            # Multi-aspect retrieval 
            chunks = await self._async_decomposed_query_retrieval(decomposition_result, verbose)
            new_state = state.copy()
            new_state.candidate_chunks = chunks
            new_state.metadata["retrieval_mode"] = "v2_decomposed"
            return new_state
        
        # Check for multi-query (RAG-Fusion)
        queries = state.metadata.get("query_variants", [state.query])
        
        if len(queries) > 1:
            # Multi-query retrieval (RAG-Fusion)
            chunks = await self._async_multi_query_retrieval(queries, verbose)
            new_state = state.copy()
            new_state.candidate_chunks = chunks
            new_state.metadata["retrieval_mode"] = "v2_multi_query"
            return new_state
        
        # Single query retrieval
        query = state.query
        
        # DISABLED: Metadata Filtering Dynamic (will be expanded later)
        # ─────────────────────────────────────────────────────────────
        # metadata_filters = state.plan.get("metadata_filters", [])
        # intent_category = state.plan.get("intent_category", "tổng_quan")
        # 
        # if metadata_filters:
        #     # Union Retrieval: Filtered (k=15) + General (k=15) streams
        #     if verbose:
        #         print(f"   🎯 Union Retrieval: intent={intent_category}")
        #         print(f"      Stream 1: Filtered by section_title keywords (k=15)")
        #         print(f"      Stream 2: General retrieval (k=15)")
        #     
        #     chunks = await self._union_retrieval(query, metadata_filters, verbose)
        #     
        #     new_state = state.copy()
        #     new_state.candidate_chunks = chunks
        #     new_state.metadata["retrieval_mode"] = "v2_union_filtered"
        #     new_state.metadata["intent_category"] = intent_category
        #     return new_state
        # ─────────────────────────────────────────────────────────────
        
        # Standard single query retrieval (hybrid: semantic + BM25)
        if verbose:
            print(f"   🔍 V2 Single query retrieval (semantic_k={self.semantic_k}, bm25_k={self.bm25_k})")
        
        semantic_task = self._async_semantic_search(query, k=self.single_query_k, verbose=verbose)
        bm25_task = self._async_bm25_search(query, k=self.single_query_k, verbose=verbose)
        
        semantic_results, bm25_results = await asyncio.gather(semantic_task, bm25_task)
        
        fused_chunks = self._rrf_fusion(semantic_results, bm25_results, verbose)
        
        new_state = state.copy()
        new_state.candidate_chunks = fused_chunks
        new_state.metadata["retrieval_mode"] = "v2_single_query"
        return new_state
    
    async def _union_retrieval(self, query: str, metadata_filters: List[str], verbose: bool) -> List[Dict]:
        """
        Union Retrieval: Soft Boost via parallel filtered + general retrieval
        
        Strategy:
        1. Stream 1 (Filtered): Retrieve k=15 with section_title keyword filtering
        2. Stream 2 (General): Retrieve k=15 without filters (ensures recall)
        3. Merge: Deduplicate by chunk_id, prioritize filtered stream
        
        This prevents Router misclassification from causing empty/wrong results
        """
        union_k = 15  # k per stream
        
        # Execute both streams in parallel
        filtered_task = self._async_semantic_search_filtered(query, k=union_k, 
                                                              filter_keywords=metadata_filters, verbose=verbose)
        general_task = self._async_semantic_search(query, k=union_k, verbose=verbose)
        bm25_task = self._async_bm25_search(query, k=union_k, verbose=verbose)
        
        filtered_results, general_results, bm25_results = await asyncio.gather(
            filtered_task, general_task, bm25_task
        )
        
        if verbose:
            print(f"      📋 Filtered: {len(filtered_results)} | General: {len(general_results)} | BM25: {len(bm25_results)}")
        
        # Merge: Filtered first (higher priority), then general, then BM25
        merged = {}
        
        # Filtered stream: boost score by 1.2x
        for chunk in filtered_results:
            chunk_id = chunk.get("chunk_id") or chunk.get("doc_id")
            if chunk_id:
                chunk["filtered_boost"] = True
                merged[chunk_id] = chunk
        
        # General stream: only add if not already present
        for chunk in general_results:
            chunk_id = chunk.get("chunk_id") or chunk.get("doc_id")
            if chunk_id and chunk_id not in merged:
                chunk["filtered_boost"] = False
                merged[chunk_id] = chunk
        
        # BM25 stream: merge with RRF-style scoring
        for chunk in bm25_results:
            chunk_id = chunk.get("chunk_id") or chunk.get("doc_id")
            if chunk_id and chunk_id not in merged:
                chunk["filtered_boost"] = False
                merged[chunk_id] = chunk
        
        # Sort: filtered first, then by semantic score
        result_chunks = list(merged.values())
        result_chunks.sort(key=lambda x: (
            0 if x.get("filtered_boost") else 1,  # Filtered first
            x.get("semantic_score", 1.0)          # Lower distance = better
        ))
        
        if verbose:
            filtered_count = sum(1 for c in result_chunks if c.get("filtered_boost"))
            print(f"      ✅ Union merged: {len(result_chunks)} chunks ({filtered_count} filtered-boosted)")
        
        return result_chunks[:self.total_limit]
    
    async def _async_semantic_search_filtered(self, query: str, k: int, 
                                               filter_keywords: List[str], verbose: bool) -> List[Dict]:
        """
        Semantic search WITH metadata filtering on section_title
        
        Uses ChromaDB where clause to filter by section_title containing keywords
        """
        try:
            # Build ChromaDB where filter for section_title
            # Match if section_title contains any of the filter keywords
            where_conditions = []
            for kw in filter_keywords[:5]:  # Limit to 5 keywords
                where_conditions.append({"section_title": {"$contains": kw}})
            
            if len(where_conditions) == 1:
                where_filter = where_conditions[0]
            else:
                where_filter = {"$or": where_conditions}
            
            # Query ChromaDB with filter
            results = self.vectordb.similarity_search_with_score(
                query, k=k, where=where_filter
            )
            
            chunks = []
            for rank, (doc, distance) in enumerate(results):
                if distance > self.semantic_distance_threshold:
                    continue
                
                chunks.append({
                    "doc_id": doc.metadata.get("doc_id", doc.metadata.get("id", "")),
                    "title": doc.metadata.get("title", ""),
                    "section_title": doc.metadata.get("section_title", ""),
                    "source": doc.metadata.get("source", ""),
                    "text": doc.page_content,
                    "chunk_id": doc.metadata.get("doc_id", doc.metadata.get("id", "")),
                    "semantic_score": float(distance),
                    "semantic_rank": rank + 1
                })
            
            if verbose:
                print(f"      🔎 Filtered search: {len(chunks)} chunks match section_title keywords")
            
            return chunks
            
        except Exception as e:
            # Fallback if filter query fails (e.g., no matching section_titles)
            if verbose:
                print(f"      ⚠️ Filtered search failed ({e}), using general search")
            return []
    
    async def _async_semantic_search(self, query: str, k: int, verbose: bool) -> List[Dict]:
        """V2 semantic search with configurable k"""
        results = await async_chromadb_search(self.vectordb, query, k)
        chunks = []
        filtered_count = 0
        
        for rank, (doc, distance) in enumerate(results):
            if distance > self.semantic_distance_threshold:
                filtered_count += 1
                continue
            
            # V2 metadata uses article_link instead of source
            source = doc.metadata.get("source", "") or doc.metadata.get("article_link", "")
            
            chunks.append({
                "doc_id": doc.metadata.get("doc_id", doc.metadata.get("id", "")),
                "title": doc.metadata.get("title", ""),
                "section_title": doc.metadata.get("section_title", ""),
                "source": source,
                "text": doc.page_content,
                "chunk_id": doc.metadata.get("doc_id", doc.metadata.get("id", "")),
                "semantic_score": float(distance),
                "semantic_rank": rank + 1
            })
        
        if verbose and filtered_count > 0:
            print(f"      🗑️ Semantic: Filtered {filtered_count}/{k} chunks (distance > {self.semantic_distance_threshold})")
        
        return chunks
    
    async def _async_bm25_search(self, query: str, k: int, verbose: bool) -> List[Dict]:
        """V2 BM25 search with configurable k"""
        if self.bm25_index is None or not self.doc_texts:
            return []
        
        query_tokens = [t for t in query.lower().split() 
                       if t not in self.stop_words and len(t) > 2]
        
        if not query_tokens:
            return []
        
        results = await async_bm25_search(self.bm25_index, query_tokens, k, self.doc_texts)
        
        chunks = []
        filtered_count = 0
        
        for rank, (idx, score) in enumerate(results):
            if score < self.bm25_score_threshold:
                filtered_count += 1
                continue
            
            doc_info = self.doc_texts[idx]
            meta = doc_info['metadata']
            
            # V2 metadata uses article_link instead of source
            source = meta.get("source", "") or meta.get("article_link", "")
            
            chunks.append({
                "doc_id": meta.get("doc_id", meta.get("id", "")),
                "title": meta.get("title", ""),
                "section_title": meta.get("section_title", ""),
                "source": source,
                "text": doc_info['text'],
                "chunk_id": meta.get("doc_id", meta.get("id", "")),
                "bm25_score": float(score),
                "bm25_rank": rank + 1
            })
        
        if verbose and filtered_count > 0:
            print(f"      🗑️ BM25: Filtered {filtered_count}/{k} chunks (score < {self.bm25_score_threshold})")
        
        return chunks
    
    def _rrf_fusion(self, semantic: List[Dict], bm25: List[Dict], verbose: bool, k: int = 60) -> List[Dict]:
        """RRF fusion of semantic and BM25 results"""
        scores = {}
        chunks = {}
        
        for rank, chunk in enumerate(semantic, 1):
            chunk_id = chunk.get("chunk_id", chunk.get("title", str(rank)))
            rrf_score = 1.0 / (k + rank)
            scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score
            if chunk_id not in chunks:
                chunks[chunk_id] = chunk.copy()
        
        for rank, chunk in enumerate(bm25, 1):
            chunk_id = chunk.get("chunk_id", chunk.get("title", str(rank)))
            rrf_score = 1.0 / (k + rank)
            scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score
            if chunk_id not in chunks:
                chunks[chunk_id] = chunk.copy()
        
        # Add RRF score
        for chunk_id, score in scores.items():
            if chunk_id in chunks:
                chunks[chunk_id]["rrf_score"] = score
        
        # Sort by RRF score
        sorted_chunks = sorted(chunks.values(), key=lambda x: x.get("rrf_score", 0), reverse=True)
        
        # V2: Return more chunks (limit by total_limit)
        result = sorted_chunks[:self.total_limit]
        
        if verbose:
            print(f"   📊 RRF Fusion: {len(semantic)} semantic + {len(bm25)} BM25 → {len(result)} fused")
        
        return result
    
    async def _async_multi_query_retrieval(self, queries: List[str], verbose: bool) -> List[Dict]:
        """Multi-query retrieval with V2 config"""
        start_time = time.time()
        
        if verbose:
            print(f"   🧵 V2: Creating {len(queries)} async retrieval tasks (k={self.semantic_k})...")
        
        tasks = [self._async_retrieve_single(idx, q, verbose) for idx, q in enumerate(queries)]
        all_results = await asyncio.gather(*tasks)
        all_results.sort(key=lambda x: x[1])
        
        elapsed = time.time() - start_time
        if verbose:
            print(f"   ✅ V2 Parallel retrieval completed in {elapsed:.2f}s")
        
        fused = self._multi_query_rrf_fusion(all_results, verbose)
        return fused
    
    async def _async_retrieve_single(self, query_idx: int, query: str, verbose: bool) -> tuple:
        """Retrieve for single query with V2 k values"""
        is_original = query_idx == 0
        k = self.semantic_k if is_original else self.aspect_k
        
        semantic_task = self._async_semantic_search(query, k=k, verbose=False)
        bm25_task = self._async_bm25_search(query, k=k, verbose=False)
        
        semantic, bm25 = await asyncio.gather(semantic_task, bm25_task)
        return (query, query_idx, is_original, semantic, bm25)
    
    def _multi_query_rrf_fusion(self, all_results: List[tuple], verbose: bool) -> List[Dict]:
        """Multi-query RRF fusion"""
        scores = {}
        chunks = {}
        k = 60
        
        for query, query_idx, is_original, semantic, bm25 in all_results:
            weight = 1.2 if is_original else 1.0
            
            for rank, chunk in enumerate(semantic, 1):
                chunk_id = chunk.get("chunk_id", chunk.get("title"))
                rrf_score = weight / (k + rank)
                scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score
                if chunk_id not in chunks:
                    chunks[chunk_id] = chunk.copy()
            
            for rank, chunk in enumerate(bm25, 1):
                chunk_id = chunk.get("chunk_id", chunk.get("title"))
                rrf_score = weight / (k + rank)
                scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score
                if chunk_id not in chunks:
                    chunks[chunk_id] = chunk.copy()
        
        for chunk_id, score in scores.items():
            if chunk_id in chunks:
                chunks[chunk_id]["rrf_score"] = score
        
        sorted_chunks = sorted(chunks.values(), key=lambda x: x.get("rrf_score", 0), reverse=True)
        result = sorted_chunks[:self.total_limit]
        
        if verbose:
            print(f"   📊 Multi-query RRF: → {len(result)} fused chunks")
        
        return result
    
    async def _async_decomposed_query_retrieval(self, decomposition_result: Dict, verbose: bool) -> List[Dict]:
        """Aspect-based retrieval with V2 config"""
        start_time = time.time()
        
        sub_queries = decomposition_result.get("sub_queries", [])
        if not sub_queries:
            return []
        
        if verbose:
            print(f"   🧵 V2: Retrieving for {len(sub_queries)} aspects (k={self.aspect_k} each)...")
        
        tasks = []
        for i, sq in enumerate(sub_queries):
            task = self._async_retrieve_single_aspect(
                sq.get("text", ""),  # FIX: was "sub_query", decomposition uses "text"
                sq.get("aspect", f"aspect_{i}"),
                i,
                False
            )
            tasks.append(task)
        
        aspect_results = await asyncio.gather(*tasks)
        
        elapsed = time.time() - start_time
        if verbose:
            print(f"   ✅ V2 Aspect-based retrieval completed in {elapsed:.2f}s")
        
        aggregated = self._aggregate_aspect_results(aspect_results, verbose)
        return aggregated
    
    async def _async_retrieve_single_aspect(self, sub_query: str, aspect: str, 
                                            order: int, verbose: bool) -> tuple:
        """Retrieve for single aspect with V2 k"""
        semantic_task = self._async_semantic_search(sub_query, k=self.aspect_k, verbose=False)
        bm25_task = self._async_bm25_search(sub_query, k=self.aspect_k, verbose=False)
        
        semantic, bm25 = await asyncio.gather(semantic_task, bm25_task)
        fused = self._rrf_fusion(semantic, bm25, verbose=False)
        
        for chunk in fused:
            chunk["aspect"] = aspect
            chunk["sub_query"] = sub_query
            chunk["aspect_order"] = order
        
        return (aspect, fused)
    
    def _aggregate_aspect_results(self, aspect_results: List[tuple], verbose: bool) -> List[Dict]:
        """Aggregate aspect results with V2 limits"""
        top_k_per_aspect = 10  # V2: Increased from 7
        
        all_chunks = []
        seen_ids = set()
        
        for aspect_name, chunks in aspect_results:
            aspect_chunks = []
            for i, chunk in enumerate(chunks[:top_k_per_aspect]):
                chunk_id = chunk.get("chunk_id", chunk.get("title"))
                if i < 5 or chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    aspect_chunks.append(chunk)
            all_chunks.extend(aspect_chunks)
        
        all_chunks.sort(key=lambda x: (x.get("aspect_order", 999), -x.get("rrf_score", 0)))
        aggregated = all_chunks[:self.total_limit]
        
        if verbose:
            print(f"   📦 V2 Aggregated {len(aggregated)} chunks from {len(aspect_results)} aspects")
        
        return aggregated
    
    def validate_input(self, state: AgentState) -> bool:
        return bool(state.query)
