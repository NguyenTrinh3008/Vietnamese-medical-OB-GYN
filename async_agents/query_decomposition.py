#!/usr/bin/env python3
"""
Async Query Decomposition Agent

Breaks complex multi-part queries into focused sub-queries for:
1. Better targeted retrieval
2. Structured generation (can answer each aspect systematically)
3. Higher accuracy for complex medical questions

Example:
  Input: "Nguyên nhân, triệu chứng và cách điều trị tiền sản giật?"
  Output: 3 sub-queries for each aspect
"""

import asyncio
import json
from typing import List, Dict, Any
from dataclasses import dataclass

from .base import AsyncBaseAgent, AgentState
from .utils import async_llm_json


class SubQuery:
    """Represents a decomposed sub-query"""
    def __init__(self, text: str, aspect: str, order: int):
        self.text = text
        self.aspect = aspect  # e.g., "nguyên nhân", "biểu hiện", "điều trị"
        self.order = order

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "aspect": self.aspect,
            "order": self.order
        }


@dataclass
class DecompositionResult:
    """Result of query decomposition"""
    should_decompose: bool
    sub_queries: List[SubQuery]
    original_query: str
    
    def to_dict(self) -> Dict:
        return {
            "should_decompose": self.should_decompose,
            "sub_queries": [sq.to_dict() for sq in self.sub_queries],
            "original_query": self.original_query
        }


class AsyncQueryDecompositionAgent(AsyncBaseAgent):
    """
    Query Decomposition Agent for complex multi-part queries
    
    Workflow:
    1. Analyze query complexity
    2. If complex: decompose into focused sub-queries
    3. Store sub-queries for downstream parallel retrieval
    4. If simple: pass through unchanged
    
    Example:
        Complex: "Các nguyên nhân, biểu hiện và cách điều trị tiểu đường thai kỳ"
        → Sub-queries:
          - "Nguyên nhân gây tiểu đường thai kỳ?"
          - "Biểu hiện của tiểu đường thai kỳ?"
          - "Cách điều trị tiểu đường thai kỳ?"
    """
    
    DECOMPOSITION_PROMPT = """Bạn là chuyên gia y khoa. Hãy phân tích câu hỏi sau và quyết định có nên chia thành các câu hỏi con tập trung không.

Câu hỏi gốc: {query}

Hướng dẫn:
1. CHỈ chia nếu câu hỏi có NHIỀU khía cạnh rõ ràng (nguyên nhân, biểu hiện, chẩn đoán, điều trị, phòng ngừa...)
2. Mỗi câu hỏi con phải:
   - ĐỘC LẬP (có thể trả lời riêng)
   - Giữ ngữ cảnh y khoa (bệnh/tình trạng) 
   - Tập trung vào 1 khía cạnh duy nhất
3. TỐI ĐA 5 câu hỏi con
4. Câu hỏi đơn giản (1 khía cạnh) → KHÔNG chia

Trả về JSON format:

{{
  "should_decompose": true/false,
  "sub_queries": [
    {{
      "text": "Câu hỏi con đầy đủ, rõ ràng?",
      "aspect": "nguyên nhân",
      "order": 1
    }},
    {{
      "text": "Câu hỏi con 2?",
      "aspect": "biểu hiện", 
      "order": 2
    }}
  ]
}}

Ví dụ:

Input: "Tiểu đường thai kỳ là gì?"
Output: {{"should_decompose": false, "sub_queries": []}}

Input: "Các nguyên nhân, triệu chứng và cách điều trị tiểu đường thai kỳ"
Output: {{
  "should_decompose": true,
  "sub_queries": [
    {{"text": "Nguyên nhân gây tiểu đường thai kỳ là gì?", "aspect": "nguyên nhân", "order": 1}},
    {{"text": "Triệu chứng của tiểu đường thai kỳ là gì?", "aspect": "triệu chứng", "order": 2}},
    {{"text": "Cách điều trị tiểu đường thai kỳ như thế nào?", "aspect": "điều trị", "order": 3}}
  ]
}}

CHỈ trả về JSON, KHÔNG giải thích thêm.
"""
    
    def __init__(
        self,
        openai_client,
        model: str,
        max_sub_queries: int = 5,
        min_query_length: int = 15,
        enable_decomposition: bool = True
    ):
        super().__init__(name="AsyncQueryDecomposition")
        self.client = openai_client
        self.model = model
        self.max_sub_queries = max_sub_queries
        self.min_query_length = min_query_length
        self.enable_decomposition = enable_decomposition
    
    async def execute(self, state: AgentState) -> AgentState:
        """Execute query decomposition"""
        query = state.query
        
        # Skip if disabled globally
        if not self.enable_decomposition:
            new_state = state.copy()
            new_state.metadata["decomposition_result"] = DecompositionResult(
                should_decompose=False,
                sub_queries=[],
                original_query=query
            ).to_dict()
            return new_state
        
        # Always call LLM to decide - let it evaluate complexity
        decomposition = await self._decompose_query(query)
        
        # Store result in state
        new_state = state.copy()
        new_state.metadata["decomposition_result"] = decomposition.to_dict()
        
        # If decomposed, update queries list for downstream processing
        if decomposition.should_decompose and decomposition.sub_queries:
            new_state.metadata["sub_queries"] = [sq.text for sq in decomposition.sub_queries]
            new_state.metadata["use_decomposition"] = True
            
            verbose = state.metadata.get("verbose", False)
            if verbose:
                print(f"🔀 Query decomposed into {len(decomposition.sub_queries)} sub-queries:")
                for sq in decomposition.sub_queries:
                    print(f"   {sq.order}. [{sq.aspect}] {sq.text}")
        else:
            new_state.metadata["use_decomposition"] = False
            
            verbose = state.metadata.get("verbose", False)
            if verbose:
                print("✅ Query is simple, no decomposition needed")
        
        return new_state
    
    async def _decompose_query(self, query: str) -> DecompositionResult:
        """
        Use LLM to decompose query into sub-queries
        
        Returns:
            DecompositionResult with should_decompose flag and sub_queries list
        """
        try:
            prompt = self.DECOMPOSITION_PROMPT.format(query=query)
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a medical query analysis expert. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # Deterministic
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content
            result_json = json.loads(result_text)
            
            # Parse result
            should_decompose = result_json.get("should_decompose", False)
            sub_queries_data = result_json.get("sub_queries", [])
            
            # Limit to max_sub_queries
            if len(sub_queries_data) > self.max_sub_queries:
                sub_queries_data = sub_queries_data[:self.max_sub_queries]
            
            # Convert to SubQuery objects
            sub_queries = [
                SubQuery(
                    text=sq["text"],
                    aspect=sq["aspect"],
                    order=sq["order"]
                )
                for sq in sub_queries_data
            ]
            
            return DecompositionResult(
                should_decompose=should_decompose,
                sub_queries=sub_queries,
                original_query=query
            )
            
        except Exception as e:
            print(f"❌ Decomposition failed: {e}")
            # Fallback: no decomposition
            return DecompositionResult(
                should_decompose=False,
                sub_queries=[],
                original_query=query
            )
    
    def validate_input(self, state: AgentState) -> bool:
        """Validate input state"""
        return bool(state.query)
