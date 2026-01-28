#!/usr/bin/env python3
"""
Async Query Rewriter Agent - For Self-RAG iterative refinement

Rewrites queries when initial retrieval fails to find relevant documents.
Used by Self-RAG loop in orchestrator when chunks_count == 0.

Strategies:
1. expanded_terminology - Expand colloquial to medical terminology
2. added_context - Add medical context
3. simplified - Simplify complex queries
4. synonyms - Use medical synonyms
"""

import json
from .base import AsyncBaseAgent, AgentState
from .utils import async_llm_json
from typing import Dict, Any, List, Optional


class AsyncQueryRewriterAgent(AsyncBaseAgent):
    """
    Query Rewriter for Self-RAG iterative refinement
    
    Triggers when:
    - No chunks found after Reranker
    - Need alternative phrasing for better retrieval
    
    Strategies:
    1. Expand medical terminology (rong kinh → chảy máu kinh nguyệt bất thường)
    2. Add clinical context
    3. Simplify complex queries
    4. Rephrase for clarity
    """
    
    REWRITE_PROMPT = """Bạn là chuyên gia tối ưu hóa câu hỏi y khoa.

NHIỆM VỤ: Viết lại câu hỏi để tìm kiếm tài liệu y khoa tốt hơn.

QUERY GỐC: {original_query}

⚠️ CRITICAL RULES:
1. PRESERVE STRUCTURE - Giữ nguyên cấu trúc câu (hỏi → hỏi, ngắn → ngắn)
2. MINIMAL CHANGES - Chỉ thay đổi tối thiểu cần thiết
3. SAME INTENT - Phải cùng chủ đề y khoa
4. ADD MEDICAL TERMS - Thêm thuật ngữ y khoa trong ngoặc

═══════════════════════════════════════════════════════════
🧠 CHAIN OF THOUGHT - HÃY SUY NGHĨ TỪNG BƯỚC:
═══════════════════════════════════════════════════════════

STEP 1: ANALYZE STRUCTURE (Phân tích cấu trúc)
- Query type: [Câu hỏi / Cụm từ / Câu trần thuật]
- Length: [Ngắn / Vừa / Dài]
- Tone: [Thông thường / Y khoa]

STEP 2: IDENTIFY TERMS (Xác định thuật ngữ)
- Colloquial terms found: [Liệt kê các thuật ngữ thường]
- Medical equivalent: [Thuật ngữ y khoa tương ứng]
- Should add English term? [Yes/No + lý do]

STEP 3: CHECK CHANGES NEEDED (Kiểm tra thay đổi cần thiết)
- Structure change needed? [Yes/No - SHOULD BE NO!]
- Add context needed? [Yes/No - Only if extremely vague]
- Simplify needed? [Yes/No]
- Replace terms? [Yes/No + which ones]

STEP 4: VALIDATE INTENT (Kiểm tra ý định)
- Original intent: [Mô tả ngắn gọn]
- Rewritten intent: [Mô tả ngắn gọn]
- Same topic? [✅ Yes / ❌ No]
- If No, STOP - reject rewrite!

STEP 5: ESTIMATE SIMILARITY (Ước tính độ tương đồng)
- How much did we change? [Small / Medium / Large]
- Estimated similarity: [High > 0.7 / Medium 0.5-0.7 / Low < 0.5]
- If Low, RECONSIDER - make smaller changes!

═══════════════════════════════════════════════════════════
📋 EXAMPLES (Học từ ví dụ):
═══════════════════════════════════════════════════════════

EXAMPLE 1 - GOOD ✅:
Original: "Rong kinh là gì?"

Step 1: Câu hỏi, ngắn, thông thường
Step 2: "rong kinh" (colloquial) → "menorrhagia" (medical)
Step 3: No structure change, just add term
Step 4: Same intent - both ask about menorrhagia ✅
Step 5: Small change, High similarity (0.85+)

Rewritten: "Rong kinh (menorrhagia) là gì?"

EXAMPLE 2 - BAD ❌:
Original: "Rong kinh là gì?"

Step 1: Câu hỏi, ngắn, thông thường
Step 2: "rong kinh" → "chảy máu kinh nguyệt kéo dài bất thường"
Step 3: ❌ Changed structure (question → statement), added "tổng quan"
Step 4: Different structure - fails validation ❌
Step 5: Large change, Low similarity (0.3)

Rewritten: "Tổng quan về chảy máu kinh nguyệt kéo dài bất thường"
→ REJECTED - Too different!

EXAMPLE 3 - GOOD ✅:
Original: "Triệu chứng tiền sản giật"

Step 1: Cụm từ, ngắn, y khoa partial
Step 2: "tiền sản giật" → "preeclampsia"
Step 3: No structure change, just add English term
Step 4: Same intent ✅
Step 5: Small change, High similarity (0.9+)

Rewritten: "Triệu chứng tiền sản giật (preeclampsia)"

═══════════════════════════════════════════════════════════
🎯 YOUR TURN - APPLY CHAIN OF THOUGHT:
═══════════════════════════════════════════════════════════

Now analyze: "{original_query}"

Think through all 5 steps carefully, then return JSON:

{{
  "reasoning": {{
    "step1_structure": "...",
    "step2_terms": "...",
    "step3_changes": "...",
    "step4_intent_check": "...",
    "step5_similarity_estimate": "..."
  }},
  "rewritten_query": "...",
  "strategy": "expanded_terminology",
  "explanation": "Giải thích ngắn gọn",
  "changes": ["Change 1", "Change 2"]
}}

CHỈ trả về JSON, KHÔNG thêm text."""
    
    def __init__(
        self,
        openai_client,
        model: str,
        embedding_model=None,  # Intent Guardrail (better than ViRanker for query-query)
        max_rewrites: int = 2,
        intent_similarity_threshold: float = 0.65  # Query drift detection (lowered for medical terms)
    ):
        super().__init__(name="AsyncQueryRewriter")
        self.client = openai_client
        self.model = model
        self.max_rewrites = max_rewrites
        
        # Intent Guardrail (using Embedding for semantic similarity)
        self.embedding_model = embedding_model
        self.intent_threshold = intent_similarity_threshold
    
    async def execute(self, state: AgentState) -> AgentState:
        """Execute query rewriting"""
        
        original_query = state.metadata.get("original_query", state.query)
        current_attempt = state.metadata.get("self_rag_iteration", 1)
        
        verbose = state.metadata.get("verbose", False)
        
        # Check rewrite history to avoid loops
        rewrite_history = state.metadata.get("rewrite_history", [])
        
        if verbose:
            print(f"🔄 Rewriting query (attempt {current_attempt})...")
        
        # Rewrite query
        rewrite_result = await self._rewrite_query(
            original_query,
            current_attempt,
            rewrite_history
        )
        
        if not rewrite_result:
            # Fallback: return original
            new_state = state.copy()
            return new_state
        
        # Update state
        new_state = state.copy()
        new_query = rewrite_result["rewritten_query"]
        
        # Check if we've seen this before (loop detection)
        if new_query in rewrite_history or new_query == state.query:
            if verbose:
                print("⚠️ Query rewrite created a loop, using alternative strategy")
            # Try alternative strategy or give up
            new_state.metadata["rewrite_failed"] = True
            return new_state
        
        # Update query and metadata
        new_state.query = new_query
        new_state.metadata["rewritten_query"] = new_query
        new_state.metadata["rewrite_strategy"] = rewrite_result["strategy"]
        new_state.metadata["rewrite_explanation"] = rewrite_result["explanation"]
        new_state.metadata["rewrite_changes"] = rewrite_result.get("changes", [])
        new_state.metadata["intent_similarity_score"] = rewrite_result.get("intent_similarity_score", 0)
        
        # Track history
        rewrite_history.append(state.query)
        new_state.metadata["rewrite_history"] = rewrite_history
        
        if verbose:
            print(f"🔀 New query: {new_query}")
            print(f"💡 Strategy: {rewrite_result['strategy']}")
            print(f"📝 Explanation: {rewrite_result['explanation']}")
            if rewrite_result.get("changes"):
                print(f"   Changes:")
                for change in rewrite_result["changes"]:
                    print(f"      - {change}")
        
        return new_state
    
    async def _rewrite_query(
        self,
        original_query: str,
        attempt: int,
        rewrite_history: List[str]
    ) -> Optional[Dict]:
        """
        Use LLM to rewrite query
        
        Returns:
            Dict with rewritten_query, strategy, explanation, changes
        """
        try:
            prompt = self.REWRITE_PROMPT.format(
                original_query=original_query,
                attempt=attempt
            )
            
            # Add context about previous attempts
            if rewrite_history:
                prompt += f"\n\nCác query đã thử (không hiệu quả):\n"
                for i, prev_q in enumerate(rewrite_history, 1):
                    prompt += f"{i}. {prev_q}\n"
                prompt += "\nHãy thử chiến lược KHÁC để tránh lặp lại.\n"
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical query optimization expert. Always respond with valid JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Some creativity but controlled
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            
            # INTENT GUARDRAIL: Verify rewritten query preserves original intent
            if self.embedding_model and "rewritten_query" in result:
                is_safe, similarity_score = self._validate_intent(
                    original_query,
                    result["rewritten_query"]
                )
                
                if not is_safe:
                    print(f"⚠️ INTENT GUARDRAIL: Query drift detected!")
                    print(f"   Similarity score: {similarity_score:.3f} < {self.intent_threshold}")
                    print(f"   Original: {original_query}")
                    print(f"   Rewritten: {result['rewritten_query']}")
                    print(f"   → Rejecting rewrite, using original query")
                    return None  # Signal to use original
                else:
                    print(f"✅ Intent preserved (similarity: {similarity_score:.3f})")
                    # Store for transparency
                    result["intent_similarity_score"] = similarity_score
            
            return result
            
        except Exception as e:
            print(f"❌ Query rewrite failed: {e}")
            return None
    
    def _validate_intent(self, original_query: str, rewritten_query: str) -> tuple[bool, float]:
        """
        Intent Guardrail: Validate that rewritten query preserves original intent
        
        Uses embedding model with cosine similarity to check semantic similarity
        between original and rewritten queries. Prevents query drift in medical domain.
        
        Args:
            original_query: User's original query
            rewritten_query: LLM-rewritten query
            
        Returns:
            (is_safe, similarity_score): 
                - is_safe: True if similarity >= threshold
                - similarity_score: 0-1 cosine similarity score
        
        Threshold Reasoning:
            - 0.80-1.00: Very similar (safe paraphrasing)
            - 0.70-0.80: Similar (acceptable variation)
            - 0.50-0.70: Different phrasing, same topic (warning zone)
            - < 0.50: Different topic (dangerous)
        """
        if not self.embedding_model:
            # No guardrail available, assume safe
            return True, 1.0
        
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            # LangChain HuggingFaceEmbeddings uses different API
            # Access underlying SentenceTransformer model
            if hasattr(self.embedding_model, 'client'):
                # LangChain HuggingFaceEmbeddings
                model = self.embedding_model.client
                emb1 = model.encode([original_query])[0]
                emb2 = model.encode([rewritten_query])[0]
            elif hasattr(self.embedding_model, 'encode'):
                # Direct SentenceTransformer
                emb1 = self.embedding_model.encode([original_query])[0]
                emb2 = self.embedding_model.encode([rewritten_query])[0]
            else:
                # Fallback: use LangChain embed_query
                emb1 = np.array(self.embedding_model.embed_query(original_query))
                emb2 = np.array(self.embedding_model.embed_query(rewritten_query))
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                emb1.reshape(1, -1),
                emb2.reshape(1, -1)
            )[0][0]
            
            similarity_score = float(similarity)
            
            # Check against threshold
            is_safe = similarity_score >= self.intent_threshold
            
            return is_safe, similarity_score
            
        except Exception as e:
            print(f"⚠️ Intent validation failed: {e}")
            # On error, be conservative: assume safe to not block valid rewrites
            return True, 0.0
    
    def validate_input(self, state: AgentState) -> bool:
        """Validate input state"""
        return bool(state.query)
