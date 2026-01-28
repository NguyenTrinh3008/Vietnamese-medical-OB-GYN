#!/usr/bin/env python3
"""
Async Agents - Pure async/await versions of RAG agents
All agents converted to use async/await for optimal performance
"""

from .base import AsyncBaseAgent, AgentState
from .utils import async_llm_json, async_llm_text, async_chromadb_search, async_bm25_search
from typing import List, Dict, Any
import asyncio
import time


class AsyncRouterAgent(AsyncBaseAgent):
    """Async Router/Planner Agent - Query classification and safety checking"""
    
    def __init__(self, openai_client, model: str, top_docs_coarse: int = 20):
        super().__init__(name="AsyncRouter")
        self.openai_client = openai_client  # Must be AsyncOpenAI
        self.model = model
        self.top_docs_coarse = top_docs_coarse
    
    async def execute(self, state: AgentState) -> AgentState:
        """Async execute - classify query and create plan"""
        query = state.query
        verbose = state.metadata.get("verbose", False)
        
        system_prompt = (
            "Bạn là chuyên gia AN TOÀN Y TẾ, phân loại câu hỏi về sức khỏe thai sản.\n\n"
            "NHIỆM VỤ: Phân loại câu hỏi thành PERSONAL (từ chối) hoặc GENERAL (chấp nhận).\n\n"
            "QUY TRÌNH SUY LUẬN:\n\n"
            "Bước 1 - PHÂN TÍCH CẤU TRÚC:\n"
            "- Câu có chứa đại từ nhân xưng không? (tôi, con tôi, em, mình, bạn...)\n"
            "- Câu có mô tả triệu chứng CỤ THỂ của một NGƯỜI CỤ THỂ không?\n"
            "- Câu có yêu cầu hành động y tế CỤ THỂ không? (khám, chẩn đoán, kê đơn...)\n\n"
            "Bước 2 - PHÂN LOẠI:\n"
            "PERSONAL (Cá nhân - PHẢI TỪ CHỐI):\n"
            "  ✗ Mô tả triệu chứng của bản thân/người thân: 'Tôi bị...', 'Con tôi...', 'Em bị...'\n"
            "  ✗ Yêu cầu chẩn đoán: '...có phải [bệnh] không?', 'Tôi mắc bệnh gì?'\n"
            "  ✗ Yêu cầu kê đơn: 'Cho tôi liều...', 'Tôi nên uống thuốc gì?'\n"
            "  ✗ Yêu cầu tư vấn điều trị cá nhân: 'Tôi nên làm gì?', 'Em phải khám gì?'\n\n"
            "GENERAL (Tổng quát - CHẤP NHẬN):\n"
            "  ✓ Định nghĩa bệnh/thuật ngữ: 'Tiền sản giật là gì?'\n"
            "  ✓ Kiến thức y khoa: 'Triệu chứng của [bệnh] là gì?'\n"
            "  ✓ Thông tin sức khỏe: 'Acid folic có tác dụng gì?'\n"
            "  ✓ So sánh/phân tích: 'So sánh A và B'\n\n"
            "Bước 3 - XÁC ĐỊNH COMPLEXITY:\n"
            "Nếu PERSONAL → action='NO_RETRIEVAL', rejection_reason rõ ràng\n"
            "Nếu GENERAL:\n"
            "  - Câu đơn giản (<15 từ, 1 khía cạnh) → 'SIMPLE_RETRIEVAL'\n"
            "  - Câu phức tạp (>15 từ, nhiều khía cạnh, so sánh) → 'COMPLEX_MULTI_STEP'\n\n"
            "VÍ DỤ:\n\n"
            "Query: 'Tôi bị chảy máu nhiều, có phải ung thư không?'\n"
            "→ PERSONAL (mô tả triệu chứng bản thân + yêu cầu chẩn đoán)\n"
            "Output: {\"action\": \"NO_RETRIEVAL\", \"complexity\": \"n/a\", \"rejection_reason\": \"Câu hỏi chẩn đoán cá nhân\"}\n\n"
            "Query: 'Triệu chứng của ung thư cổ tử cung là gì?'\n"
            "→ GENERAL (hỏi kiến thức y khoa tổng quát)\n"
            "Output: {\"action\": \"SIMPLE_RETRIEVAL\", \"complexity\": \"simple\"}\n\n"
            "Query: 'So sánh triệu chứng tiền sản giật và sản giật'\n"
            "→ GENERAL (so sánh, phức tạp)\n"
            "Output: {\"action\": \"COMPLEX_MULTI_STEP\", \"complexity\": \"complex\"}\n\n"
            "QUAN TRỌNG:\n"
            "- AN TOÀN là ưu tiên TUYỆT ĐỐI\n"
            "- KHI NGHI NGỜ → TỪ CHỐI (NO_RETRIEVAL)\n"
            "- Mọi câu có 'Tôi/Em/Con tôi/Bạn + [triệu chứng/bệnh]' → PHẢI TỪ CHỐI\n\n"
            "OUTPUT FORMAT:\n"
            "Trả về JSON với keys:\n"
            "- action: 'NO_RETRIEVAL' | 'SIMPLE_RETRIEVAL' | 'COMPLEX_MULTI_STEP'\n"
            "- complexity: 'simple' | 'complex' | 'n/a'\n"
            "- rejection_reason: string (CHỈ khi action='NO_RETRIEVAL')"
        )
        
        user_prompt = f"Câu hỏi: {query}"
        
        # Async LLM call
        result = await async_llm_json(system_prompt, user_prompt, self.openai_client, self.model)
        
        action = result.get("action", "SIMPLE_RETRIEVAL")
        need_retrieval = action != "NO_RETRIEVAL"
        
        plan = {
            "need_retrieval": need_retrieval,
            "complexity": result.get("complexity", "simple"),
            "action": action,
            "rejection_reason": result.get("rejection_reason", "")
        }
        
        # Verbose output
        if verbose:
            if not need_retrieval:
                print(f"🚫 Router decision: {action}")
                if plan["rejection_reason"]:
                    print(f"   Reason: {plan['rejection_reason']}")
            else:
                print(f"✅ Router decision: {action} (complexity: {plan['complexity']})")
        
        new_state = state.copy()
        new_state.plan = plan
        return new_state
    
    def validate_input(self, state: AgentState) -> bool:
        return bool(state.query)



class AsyncQueryExpansionAgent(AsyncBaseAgent):
    """Async Query Expansion Agent - RAG-Fusion multi-query generation"""
    
    def __init__(self, openai_client, model: str, num_variants: int = 2):
        super().__init__(name="AsyncQueryExpansion")
        self.openai_client = openai_client
        self.model = model
        self.num_variants = num_variants
    
    async def execute(self, state: AgentState) -> AgentState:
        """Async execute - generate query variants"""
        if not state.plan.get("need_retrieval", True):
            new_state = state.copy()
            new_state.metadata["query_variants"] = [state.query]
            return new_state
        
        original_query = state.query
        verbose = state.metadata.get("verbose", False)
        
        # Skip if query was decomposed - decomposition already handles multi-aspect queries
        use_decomposition = state.metadata.get("use_decomposition", False)
        if use_decomposition:
            if verbose:
                print("⏭️ QueryExpansion skipped (query already decomposed)")
            new_state = state.copy()
            new_state.metadata["query_variants"] = [original_query]
            return new_state
        
        enable_rag_fusion = state.metadata.get("enable_rag_fusion", True)
        if not enable_rag_fusion:
            if verbose:
                print("⏭️ RAG-Fusion disabled, using original query only")
            new_state = state.copy()
            new_state.metadata["query_variants"] = [original_query]
            return new_state
        
        if verbose:
            print(f"🔄 RAG-Fusion: Generating {self.num_variants} query variants...")
        
        # Async variant generation
        variants = await self._generate_variants(original_query, self.num_variants, verbose)
        all_queries = [original_query] + variants
        
        if verbose:
            print(f"✅ Generated {len(all_queries)} total queries:")
            for i, q in enumerate(all_queries):
                label = "📌 Original" if i == 0 else f"🔀 Variant {i}"
                print(f"   {label}: {q}")
        
        new_state = state.copy()
        new_state.metadata["query_variants"] = all_queries
        new_state.metadata["original_query"] = original_query
        return new_state
    
    async def _generate_variants(self, query: str, num_variants: int, verbose: bool) -> List[str]:
        """Async generate query variants using LLM"""
        system_prompt = (
            "Bạn là chuyên gia sức khỏe thai sản, nhiệm vụ sinh các biến thể câu hỏi.\n\n"
            "CHIẾN LƯỢC:\n"
            "1. Thuật ngữ Y khoa: Thay ngôn ngữ thường bằng thuật ngữ chuyên môn\n"
            "2. Mở rộng Triệu chứng: Thêm chi tiết lâm sàng\n"
            "3. Góc nhìn Lâm sàng: Diễn đạt từ góc độ bác sĩ/chẩn đoán\n\n"
            f"Trả JSON với key 'variants': array of {num_variants} strings."
        )
        
        user_prompt = f"Câu hỏi gốc: {query}\n\nSinh {num_variants} biến thể khác nhau."
        
        # Use temperature=0 for deterministic expansion (avoid stochastic retrieval failures)
        result = await async_llm_json(system_prompt, user_prompt, self.openai_client, 
                                      self.model, temperature=0.0, max_tokens=800)
        
        variants = result.get("variants", [])
        variants = variants[:num_variants]
        variants = [v for v in variants if v.lower() != query.lower()]
        
        return variants
    
    def validate_input(self, state: AgentState) -> bool:
        return bool(state.query) and isinstance(state.plan, dict)


class AsyncRetrieverAgent(AsyncBaseAgent):
    """
    Async Retriever Agent - Native async multi-query retrieval
    
    Biggest performance gain: 400ms → 250ms (-150ms improvement)
    Uses asyncio.gather() for truly parallel async retrieval
    """
    
    def __init__(self, vectordb, bm25_index, doc_texts, 
                 semantic_distance_threshold: float = 0.5,
                 bm25_score_threshold: float = 5.0):
        super().__init__(name="AsyncRetriever")
        self.vectordb = vectordb
        self.bm25_index = bm25_index
        self.doc_texts = doc_texts
        
        # Score thresholds for filtering low-quality chunks
        self.semantic_distance_threshold = semantic_distance_threshold
        self.bm25_score_threshold = bm25_score_threshold
        
        # Vietnamese stop words for BM25
        self.stop_words = {
            'là', 'gì', 'có', 'được', 'như', 'thế', 'nào', 'tại', 'sao', 'khi', 'nào',
            'các', 'và', 'trong', 'của', 'với', 'cho', 'để', 'từ', 'về', 'trên', 'dưới',
            'này', 'đó', 'những', 'một', 'hai', 'ba', 'bốn', 'năm', 'sáu', 'bảy', 'tám', 'chín', 'mười'
        }
    
    async def execute(self, state: AgentState) -> AgentState:
        """Async execute with native async multi-query retrieval"""
        if not state.plan.get("need_retrieval", True):
            if state.metadata.get("verbose"):
                print("⏭️ Skip retrieval")
            new_state = state.copy()
            new_state.candidate_chunks = []
            return new_state
        
        verbose = state.metadata.get("verbose", False)
        
        # Check if query was decomposed
        use_decomposition = state.metadata.get("use_decomposition", False)
        decomposition_result = state.metadata.get("decomposition_result", {})
        
        if use_decomposition and decomposition_result.get("should_decompose"):
            # Aspect-based retrieval for decomposed queries
            if verbose:
                print(f"🎯 Aspect-based retrieval for decomposed query")
            
            fused_chunks = await self._async_decomposed_query_retrieval(
                decomposition_result,
                verbose
            )
            
            new_state = state.copy()
            new_state.candidate_chunks = fused_chunks
            new_state.metadata["retrieval_mode"] = "aspect_based_decomposition"  
            new_state.metadata["num_aspects"] = len(decomposition_result.get("sub_queries", []))
            return new_state
        
        # Fall back to standard query expansion (RAG-Fusion) flow
        query_variants = state.metadata.get("query_variants", [state.query])
        is_multi_query = len(query_variants) > 1
        
        if is_multi_query:
            # Native async multi-query retrieval
            if verbose:
                print(f"🚀 Async RAG-Fusion: Native async retrieval for {len(query_variants)} queries")
            
            fused_chunks = await self._async_multi_query_retrieval(query_variants, verbose)
            
            new_state = state.copy()
            new_state.candidate_chunks = fused_chunks
            new_state.metadata["retrieval_mode"] = "async_rag_fusion"
            new_state.metadata["num_queries"] = len(query_variants)
            return new_state
        
        else:
            # Single async query retrieval
            query = state.query
            intent, alpha = self._classify_query_intent(query, verbose)
            
            if verbose:
                print(f"🔍 Async hybrid retrieval with {intent} intent (α={alpha:.2f})")
            
            # Async retrieval with gather
            semantic_task = self._async_semantic_search(query, k=20, verbose=verbose)
            bm25_task = self._async_bm25_search(query, k=20, verbose=verbose)
            
            semantic_results, bm25_results = await asyncio.gather(semantic_task, bm25_task)
            
            # RRF fusion
            fused_chunks = self._rrf_fusion(semantic_results, bm25_results, verbose)
            
            new_state = state.copy()
            new_state.candidate_chunks = fused_chunks
            new_state.metadata["retrieval_mode"] = "async_single_query"
            return new_state
    
    async def _async_multi_query_retrieval(self, queries: List[str], verbose: bool) -> List[Dict]:
        """Native async multi-query retrieval with asyncio.gather()"""
        start_time = time.time()
        
        if verbose:
            print(f"   🧵 Creating {len(queries)} async retrieval tasks...")
        
        # Create all retrieval tasks
        tasks = [self._async_retrieve_single(idx, q, verbose) for idx, q in enumerate(queries)]
        
        # Execute all in parallel with asyncio.gather
        all_results = await asyncio.gather(*tasks)
        
        # Sort by query index
        all_results.sort(key=lambda x: x[1])
        
        elapsed = time.time() - start_time
        if verbose:
            print(f"   ✅ Async parallel retrieval completed in {elapsed:.2f}s")
            for query, idx, is_original, semantic, bm25 in all_results:
                label = "📌" if is_original else f"🔀 {idx}"
                print(f"      {label} Retrieved {len(semantic)} semantic + {len(bm25)} BM25")
        
        # Multi-query RRF fusion
        fused_chunks = self._multi_query_rrf_fusion(all_results, verbose)
        
        return fused_chunks
    
    async def _async_decomposed_query_retrieval(self, decomposition_result: Dict, verbose: bool) -> List[Dict]:
        """
        Aspect-based parallel retrieval for decomposed queries
        
        Each sub-query is retrieved independently, then results are aggregated
        with aspect metadata preserved for structured generation
        """
        start_time = time.time()
        
        sub_queries = decomposition_result.get("sub_queries", [])
        
        if not sub_queries:
            return []
        
        if verbose:
            print(f"   🧵 Retrieving for {len(sub_queries)} aspects in parallel...")
        
        # Create retrieval tasks for each sub-query/aspect
        tasks = []
        for sq in sub_queries:
            task = self._async_retrieve_single_aspect(
                sq["text"],
                sq["aspect"],
                sq["order"],
                verbose=False
            )
            tasks.append(task)
        
        # Execute all aspect retrievals in parallel
        aspect_results = await asyncio.gather(*tasks)
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"   ✅ Aspect-based retrieval completed in {elapsed:.2f}s")
            for aspect_name, chunks in aspect_results:
                print(f"      📍 [{aspect_name}]: {len(chunks)} chunks")
        
        # Aggregate results with aspect metadata preservation
        aggregated_chunks = self._aggregate_aspect_results(aspect_results, verbose)
        
        return aggregated_chunks
    
    async def _async_retrieve_single_aspect(
        self,
        sub_query: str,
        aspect: str,
        order: int,
        verbose: bool
    ) -> tuple:
        """
        Retrieve for a single aspect/sub-query
        
        Returns: (aspect_name, chunks_with_metadata)
        """
        # Parallel semantic + BM25 for this aspect
        semantic_task = self._async_semantic_search(sub_query, k=10, verbose=False)
        bm25_task = self._async_bm25_search(sub_query, k=10, verbose=False)
        
        semantic, bm25 = await asyncio.gather(semantic_task, bm25_task)
        
        # RRF fusion for this aspect
        fused = self._rrf_fusion(semantic, bm25, verbose=False)
        
        # Add aspect metadata to each chunk
        for chunk in fused:
            chunk["aspect"] = aspect
            chunk["sub_query"] = sub_query
            chunk["aspect_order"] = order
        
        return (aspect, fused)
    
    def _aggregate_aspect_results(self, aspect_results: List[tuple], verbose: bool) -> List[Dict]:
        """
        Aggregate results from multiple aspects
        
        Strategy:
        1. Take top-k from each aspect
        2. Less aggressive deduplication - allow some overlap for context
        3. Sort by aspect_order to maintain structure
        4. Keep top-N overall
        """
        top_k_per_aspect = 7  # Top 7 from each aspect (increased from 5)
        total_limit = 30      # Max 30 total chunks (increased from 20)
        
        all_chunks = []
        seen_ids = set()
        
        # First pass: collect top-k from each aspect
        # Allow first 3 chunks per aspect without dedup to preserve aspect-specific context
        for aspect_name, chunks in aspect_results:
            aspect_chunks = []
            for i, chunk in enumerate(chunks[:top_k_per_aspect]):
                chunk_id = chunk.get("chunk_id") or chunk.get("id")
                
                # First 3 chunks per aspect: always keep (even if duplicate)
                # Rest: deduplicate
                if i < 3 or chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    aspect_chunks.append(chunk)
            
            all_chunks.extend(aspect_chunks)
        
        # Sort by aspect_order to preserve structure
        all_chunks.sort(key=lambda x: (x.get("aspect_order", 999), -x.get("rrf_score", 0)))
        
        # Limit total
        aggregated = all_chunks[:total_limit]
        
        if verbose:
            print(f"   📦 Aggregated {len(aggregated)} chunks from {len(aspect_results)} aspects")
        
        return aggregated
    
    async def _async_retrieve_single(self, query_idx: int, query: str, verbose: bool) -> tuple:
        """Async retrieve for a single query"""
        is_original = query_idx == 0
        
        # Parallel semantic + BM25 retrieval for this query (reduced to 10 for speed)
        semantic_task = self._async_semantic_search(query, k=10, verbose=False)
        bm25_task = self._async_bm25_search(query, k=10, verbose=False)
        
        semantic, bm25 = await asyncio.gather(semantic_task, bm25_task)
        
        return (query, query_idx, is_original, semantic, bm25)
    
    async def _async_semantic_search(self, query: str, k: int, verbose: bool) -> List[Dict]:
        """Async semantic search using ChromaDB with distance threshold filtering"""
        results = await async_chromadb_search(self.vectordb, query, k)
        chunks = []
        filtered_count = 0
        
        for rank, (doc, distance) in enumerate(results):
            # Filter by distance threshold (lower distance = more similar)
            if distance > self.semantic_distance_threshold:
                filtered_count += 1
                continue
            
            chunks.append({
                "doc_id": doc.metadata.get("doc_id", doc.metadata.get("id", "")),
                "title": doc.metadata.get("title", ""),
                "source": doc.metadata.get("source", ""),
                "text": doc.page_content,
                "chunk_id": doc.metadata.get("doc_id", doc.metadata.get("id", "")),
                "semantic_score": float(distance),
                "semantic_rank": rank + 1
            })
        
        if verbose and filtered_count > 0:
            print(f"      🗑️  Semantic: Filtered {filtered_count}/{k} chunks (distance > {self.semantic_distance_threshold})")
        
        return chunks
    
    async def _async_bm25_search(self, query: str, k: int, verbose: bool) -> List[Dict]:
        """Async BM25 sparse search with score threshold filtering"""
        if self.bm25_index is None or not self.doc_texts:
            return []
        
        # Preprocess query
        query_tokens = [token for token in query.lower().split() 
                       if token not in self.stop_words and len(token) > 2]
        
        if not query_tokens:
            return []
        
        # Async BM25 search
        results = await async_bm25_search(self.bm25_index, query_tokens, k, self.doc_texts)
        
        chunks = []
        filtered_count = 0
        
        for rank, (idx, score) in enumerate(results):
            # Filter by BM25 score threshold (higher score = more relevant)
            if score < self.bm25_score_threshold:
                filtered_count += 1
                continue
            
            doc_info = self.doc_texts[idx]
            chunks.append({
                "doc_id": doc_info['metadata'].get("doc_id", doc_info['metadata'].get("id", "")),
                "title": doc_info['metadata'].get("title", ""),
                "source": doc_info['metadata'].get("source", ""),
                "text": doc_info['text'],
                "chunk_id": doc_info['metadata'].get("doc_id", doc_info['metadata'].get("id", "")),
                "bm25_score": float(score),
                "bm25_rank": rank + 1
            })
        
        if verbose and filtered_count > 0:
            print(f"      🗑️  BM25: Filtered {filtered_count}/{k} chunks (score < {self.bm25_score_threshold})")
        
        return chunks
    
    # Intent classification and fusion methods (unchanged from sync version)
    def _classify_query_intent(self, query: str, verbose: bool) -> tuple:
        """Classify query intent for maternal health domain"""
        query_lower = query.lower()
        
        medication_patterns = ['liều', 'thuốc', 'aspirin', 'vitamin', 'mg', 'mcg']
        prenatal_patterns = ['xét nghiệm', 'siêu âm', 'sàng lọc', 'khám thai']
        condition_patterns = ['triệu chứng', 'buồn nôn', 'đau', 'tiền sản giật']
        
        for pattern in medication_patterns:
            if pattern in query_lower:
                return ("medication_supplement", 0.30)
        
        for pattern in prenatal_patterns:
            if pattern in query_lower:
                return ("prenatal_care", 0.55)
        
        for pattern in condition_patterns:
            if pattern in query_lower:
                return ("pregnancy_condition", 0.70)
        
        return ("general", 0.50)
    
    def _rrf_fusion(self, semantic_results: List[Dict], bm25_results: List[Dict], 
                   verbose: bool, k: int = 60) -> List[Dict]:
        """RRF fusion (reused from sync implementation)"""
        fusion_scores = {}
        
        for chunk in semantic_results:
            doc_id = chunk["doc_id"]
            rrf_score = 1.0 / (k + chunk["semantic_rank"])
            
            if doc_id not in fusion_scores:
                fusion_scores[doc_id] = {"chunk": chunk.copy(), "rrf_score": 0.0}
            
            fusion_scores[doc_id]["rrf_score"] += rrf_score
        
        for chunk in bm25_results:
            doc_id = chunk["doc_id"]
            rrf_score = 1.0 / (k + chunk["bm25_rank"])
            
            if doc_id not in fusion_scores:
                fusion_scores[doc_id] = {"chunk": chunk.copy(), "rrf_score": 0.0}
            
            fusion_scores[doc_id]["rrf_score"] += rrf_score
        
        sorted_docs = sorted(fusion_scores.items(), key=lambda x: x[1]["rrf_score"], reverse=True)
        fused_chunks = [{"fusion_score": data["rrf_score"], **data["chunk"]} 
                        for doc_id, data in sorted_docs]
        
        return fused_chunks
    
    def _multi_query_rrf_fusion(self, all_results: List[tuple], verbose: bool, k: int = 60) -> List[Dict]:
        """Multi-query RRF fusion (reused from sync implementation)"""
        fusion_scores = {}
        
        for query, query_idx, is_original, semantic_results, bm25_results in all_results:
            for chunk in semantic_results:
                doc_id = chunk["doc_id"]
                rrf_score = 1.0 / (k + chunk["semantic_rank"])
                if doc_id not in fusion_scores:
                    fusion_scores[doc_id] = {"chunk": chunk.copy(), "rrf_score": 0.0}
                fusion_scores[doc_id]["rrf_score"] += rrf_score
            
            for chunk in bm25_results:
                doc_id = chunk["doc_id"]
                rrf_score = 1.0 / (k + chunk["bm25_rank"])
                if doc_id not in fusion_scores:
                    fusion_scores[doc_id] = {"chunk": chunk.copy(), "rrf_score": 0.0}
                fusion_scores[doc_id]["rrf_score"] += rrf_score
        
        sorted_docs = sorted(fusion_scores.items(), key=lambda x: x[1]["rrf_score"], reverse=True)
        fused_chunks = [{"fusion_score": data["rrf_score"], **data["chunk"]} 
                        for doc_id, data in sorted_docs]
        
        if verbose:
            print(f"   ✅ Multi-query RRF: {len(fused_chunks)} unique chunks")
        
        return fused_chunks
    
    def validate_input(self, state: AgentState) -> bool:
        return bool(state.query) and isinstance(state.plan, dict)
