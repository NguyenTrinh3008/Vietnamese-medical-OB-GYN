#!/usr/bin/env python3
"""
NLI Hallucination Grader V2 - Per-Chunk Checking

Key Differences from V1:
- Checks each claim against EACH chunk separately (not concatenated)
- Each chunk ~100-300 tokens → fits NLI 512 token limit
- More accurate entailment/contradiction detection

Compatible with hierarchical chunking in RAG V2.
"""

import sys
sys.path.append('..')

from async_agents.base import BaseAgent, AgentState
from typing import List, Dict, Any, Tuple, Optional
import re
import numpy as np


class NLIHallucinationGraderV2(BaseAgent):
    """
    NLI Hallucination Grader V2 with Per-Chunk Checking
    
    For each claim:
    1. Check against EACH retrieved chunk separately
    2. If ANY chunk shows ENTAILMENT → claim is supported
    3. If ANY chunk shows CONTRADICTION → claim conflicts
    4. If ALL chunks show NEUTRAL → claim not addressed
    
    This approach works with hierarchical chunks (~100-300 tokens)
    that fit within NLI model's 512 token limit.
    """
    
    NLI_LABELS = ["entailment", "neutral", "contradiction"]
    
    def __init__(self, nli_model,
                 truncate_func=None,
                 openai_client=None,
                 model: str = "gpt-4.1-mini",
                 use_llm_extraction: bool = True,
                 max_chunks_per_claim: int = 10,  
                 verbose: bool = False):
        """
        Args:
            nli_model: CrossEncoder NLI model (mDeBERTa-v3)
            truncate_func: Function to truncate text
            openai_client: OpenAI client for claim extraction
            model: Model name for LLM
            use_llm_extraction: Use LLM for claim extraction
            max_chunks_per_claim: Max chunks to check per claim (top ranked)
            verbose: Enable verbose logging
        """
        super().__init__(name="NLIHallucinationGraderV2")
        self.nli_model = nli_model
        self.truncate_tokens = truncate_func or (lambda text, max_len: text[:max_len*4])
        self.openai_client = openai_client
        self.model = model
        self.use_llm_extraction = use_llm_extraction and openai_client is not None
        self.max_chunks_per_claim = max_chunks_per_claim
        self._verbose = verbose
    
    def execute(self, state: AgentState) -> AgentState:
        """Execute hallucination detection using per-chunk NLI"""
        answer = state.answer
        chunks = state.reranked_chunks[:self.max_chunks_per_claim]
        
        if not answer or not chunks:
            new_state = state.copy()
            new_state.metadata["hallucination_check"] = "skipped"
            return new_state
        
        verbose = state.metadata.get("verbose", self._verbose)
        
        if verbose:
            print(f"🔍 [V2] Checking hallucinations with per-chunk NLI...")
            print(f"   Using {len(chunks)} chunks (hierarchical, ~100-300 tokens each)")
        
        # Extract claims
        claims = self._extract_claims(answer)
        
        if verbose:
            print(f"   Extracted {len(claims)} claims to verify")
        
        if not claims:
            new_state = state.copy()
            new_state.metadata["hallucination_check"] = "no_claims"
            new_state.metadata["grounded"] = True
            return new_state
        
        # ============================================================
        # BATCH NLI PROCESSING - All pairs in one model.predict() call
        # ============================================================
        # Build all (chunk, claim) pairs for batch inference
        pairs = []
        pair_mapping = []  # Track which pair belongs to which (claim_idx, chunk_idx)
        
        for claim_idx, claim in enumerate(claims):
            for chunk_idx, chunk in enumerate(chunks):
                chunk_text = chunk.get("text", "")
                # Truncate chunk to ensure < 512 tokens
                chunk_text = self.truncate_tokens(chunk_text, 400)
                pairs.append((chunk_text, claim))
                pair_mapping.append((claim_idx, chunk_idx))
        
        if verbose:
            print(f"   🚀 Batch NLI: {len(pairs)} pairs ({len(claims)} claims × {len(chunks)} chunks)")
        
        # Single batch inference - MUCH FASTER than sequential
        all_scores = self.nli_model.predict(pairs)
        
        # ============================================================
        # Process batch results - aggregate per claim
        # ============================================================
        # Initialize per-claim tracking
        claim_results = {i: {
            "best_entail_score": -999,
            "best_entail_chunk": None,
            "best_contra_score": -999,
            "best_contra_chunk": None,
            "entail_count": 0,
            "contra_count": 0
        } for i in range(len(claims))}
        
        for pair_idx, (claim_idx, chunk_idx) in enumerate(pair_mapping):
            scores = all_scores[pair_idx]
            pred_idx = np.argmax(scores)
            label = self.NLI_LABELS[pred_idx]
            
            # Track best scores per claim
            if label == "entailment" and scores[0] > claim_results[claim_idx]["best_entail_score"]:
                claim_results[claim_idx]["best_entail_score"] = scores[0]
                claim_results[claim_idx]["best_entail_chunk"] = chunks[chunk_idx]
            
            if label == "contradiction" and scores[2] > claim_results[claim_idx]["best_contra_score"]:
                claim_results[claim_idx]["best_contra_score"] = scores[2]
                claim_results[claim_idx]["best_contra_chunk"] = chunks[chunk_idx]
            
            # Count confident predictions
            if label == "entailment" and scores[0] > 2.0:
                claim_results[claim_idx]["entail_count"] += 1
            elif label == "contradiction" and scores[2] > 2.0:
                claim_results[claim_idx]["contra_count"] += 1
        
        # ============================================================
        # Determine verdict for each claim using Support-Dominant Logic
        # ============================================================
        results = []
        contradictions = []
        neutrals = []
        entailments = []
        
        for claim_idx, claim in enumerate(claims):
            cr = claim_results[claim_idx]
            
            # Support-Dominant: ANY entailment > contradiction > neutral
            if cr["best_entail_score"] > 0:
                verdict = "entailment"
                best_chunk = cr["best_entail_chunk"]
            elif cr["best_contra_score"] > 4.5:
                verdict = "contradiction"
                best_chunk = cr["best_contra_chunk"]
            else:
                verdict = "neutral"
                best_chunk = None
            
            if verbose:
                print(f"   [{claim_idx+1}] {claim[:60]}... → {verdict.upper()}")
            
            results.append({
                "claim": claim,
                "verdict": verdict,
                "best_chunk": best_chunk,
                "scores": {
                    "entailment": cr["best_entail_score"],
                    "contradiction": cr["best_contra_score"],
                    "entailment_count": cr["entail_count"],
                    "contradiction_count": cr["contra_count"]
                }
            })
            
            if verdict == "contradiction":
                contradictions.append(claim)
            elif verdict == "neutral":
                neutrals.append(claim)
            else:
                entailments.append(claim)
        
        # Final verdict
        has_contradictions = len(contradictions) > 0
        has_only_neutrals = len(neutrals) > 0 and len(entailments) == 0 and not has_contradictions
        
        if has_contradictions:
            grounded = False
            verdict = "REJECTED"
        elif has_only_neutrals:
            grounded = False
            verdict = "WARNING"
        else:
            grounded = True
            verdict = "APPROVED"
        
        # Update metadata
        new_state = state.copy()
        new_state.metadata["hallucination_check"] = verdict
        new_state.metadata["grounded"] = grounded
        new_state.metadata["nli_results"] = results
        new_state.metadata["contradictions"] = contradictions
        new_state.metadata["neutrals"] = neutrals
        new_state.metadata["entailments"] = entailments
        
        if verbose:
            print(f"\n   📊 Summary: {len(entailments)} entailed, {len(neutrals)} neutral, {len(contradictions)} contradictions")
        
        if grounded:
            if verbose:
                print(f"   ✅ All claims grounded ({len(entailments)} entailments)")
        elif has_contradictions:
            if verbose:
                print(f"   ❌ Contradiction detected! ({len(contradictions)} claims)")
                print(f"      → Sending to Critic for removal...")
            
            # Mark for Critic to handle - don't modify answer here
            new_state.metadata["contradicted_claims"] = contradictions
            new_state.metadata["need_critic_removal"] = True
            # Don't set grounded=True so Critic will run
        else:
            # NEW: Remove neutral claims from answer
            if neutrals and self.openai_client:
                if verbose:
                    print(f"   ⚠️ Removing {len(neutrals)} unsupported claims from answer...")
                
                cleaned_answer = self._remove_neutral_claims(answer, neutrals, verbose)
                new_state.answer = cleaned_answer
                new_state.metadata["removed_claims"] = neutrals
                new_state.metadata["original_answer"] = answer
                
                if verbose:
                    print(f"   ✅ Answer cleaned - removed unsupported information")
            else:
                if verbose:
                    print(f"   ⚠️ Some claims not directly supported ({len(neutrals)} neutral)")
                
                warning = (
                    "\n\n⚠️ **LƯU Ý**: Một số thông tin không được hỗ trợ trực tiếp bởi tài liệu."
                )
                new_state.answer = answer + warning
        
        return new_state
    
    def _remove_neutral_claims(self, answer: str, neutral_claims: List[str], verbose: bool = False) -> str:
        """
        Remove neutral (unsupported) claims from the answer using LLM
        
        Args:
            answer: Original answer text
            neutral_claims: List of claims not supported by any chunk
            verbose: Enable verbose logging
            
        Returns:
            Cleaned answer without unsupported claims
        """
        if not neutral_claims or not self.openai_client:
            return answer
        
        # Preserve source references section
        source_section = ""
        main_answer = answer
        if "## 📚 Nguồn tham khảo:" in answer:
            parts = answer.split("## 📚 Nguồn tham khảo:")
            main_answer = parts[0]
            source_section = "\n\n## 📚 Nguồn tham khảo:" + parts[1]
        
        # Build prompt
        claims_list = "\n".join([f"- {claim}" for claim in neutral_claims])
        
        system_prompt = """Bạn là trợ lý y tế. Nhiệm vụ: Viết lại câu trả lời, LOẠI BỎ các thông tin không được hỗ trợ.

QUY TẮC:
1. LOẠI BỎ hoàn toàn các câu/đoạn chứa thông tin không được hỗ trợ
2. GIỮ NGUYÊN định dạng (markdown, bullet points, citations)
3. GIỮ NGUYÊN các thông tin đã được xác minh
4. KHÔNG thêm thông tin mới
5. ĐẢM BẢO câu trả lời vẫn mạch lạc sau khi xóa
6. Nếu sau khi xóa, một section trống → xóa luôn section header

Trả về câu trả lời đã được chỉnh sửa, KHÔNG giải thích."""

        user_prompt = f"""CÂU TRẢ LỜI GỐC:
{main_answer}

CÁC THÔNG TIN KHÔNG ĐƯỢC HỖ TRỢ (CẦN XÓA):
{claims_list}

Viết lại câu trả lời, loại bỏ các thông tin không được hỗ trợ ở trên:"""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=2000
            )
            cleaned = response.choices[0].message.content.strip()
            
            # Re-append source section
            if source_section:
                cleaned = cleaned + source_section
            
            return cleaned
            
        except Exception as e:
            if verbose:
                print(f"   ⚠️ Failed to remove claims: {e}")
            # Fallback: return original with warning
            return answer + "\n\n⚠️ **LƯU Ý**: Một số thông tin không được hỗ trợ trực tiếp bởi tài liệu."

    def _extract_claims(self, answer: str) -> List[str]:
        """Extract claims from answer using LLM with optimized prompt"""
        if self.use_llm_extraction:
            return self._extract_claims_llm(answer)
        else:
            return self._extract_claims_regex(answer)
    
    def _extract_claims_llm(self, answer: str) -> List[str]:
        """
        LLM-based claim extraction with optimized prompt for medical domain
        
        Each claim is limited to ~100 tokens to ensure:
        - claim + chunk < 512 tokens (NLI model limit)
        - More atomic claims = more accurate verification
        """
        # Pre-clean answer: remove sources, disclaimers, citations
        clean_answer = answer
        
        # Remove source reference section
        if "## 📚 Nguồn tham khảo:" in clean_answer:
            clean_answer = clean_answer.split("## 📚 Nguồn tham khảo:")[0]
        if "Thông tin chỉ nhằm tham khảo" in clean_answer:
            clean_answer = clean_answer.split("Thông tin chỉ nhằm tham khảo")[0]
        
        # Remove inline citations [1], [2], etc.
        clean_answer = re.sub(r'\[\d+\]', '', clean_answer)
        
        # Remove markdown headers for cleaner extraction
        clean_answer = re.sub(r'^#+\s+.*$', '', clean_answer, flags=re.MULTILINE)
        
        # Remove emoji prefixes
        clean_answer = re.sub(r'[✅❌⚠️🔹🔸•]\s*', '', clean_answer)
        
        system_prompt = """Bạn là chuyên gia trích xuất câu từ văn bản. Nhiệm vụ: Tách các CÂU KHẲNG ĐỊNH Y KHOA từ câu trả lời.

QUY TẮC TUYỆT ĐỐI:
1. TRÍCH XUẤT NGUYÊN VĂN - KHÔNG ĐƯỢC viết lại hay paraphrase
2. Mỗi claim là một câu hoặc mệnh đề độc lập từ văn bản gốc
3. LOẠI BỎ: câu mở đầu, kết luận, disclaimer, lời khuyên đi khám
4. GIỮ LẠI: Các câu chứa thông tin y khoa cụ thể (liều, triệu chứng, nguyên nhân)
5. Tối đa 10 claims

VÍ DỤ:
Input: "Acid folic rất quan trọng trong thai kỳ vì giúp ngăn ngừa dị tật ống thần kinh. Liều khuyến nghị là 400mcg/ngày."
Output: {"claims": ["Acid folic rất quan trọng trong thai kỳ vì giúp ngăn ngừa dị tật ống thần kinh", "Liều khuyến nghị là 400mcg/ngày"]}

Return JSON: {"claims": [...]}"""

        try:
            import json
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Trích xuất claims từ câu trả lời sau:\n\n{clean_answer}"}
                ],
                temperature=0.0,
                max_tokens=800,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            claims = result.get("claims", [])
            
            # Post-process: ensure each claim is within token limit
            processed_claims = []
            for claim in claims:
                claim = claim.strip()
                # Skip too short or too long
                if len(claim) < 15:
                    continue
                # Truncate to ~100 tokens (~400 chars for Vietnamese)
                if len(claim) > 400:
                    claim = claim[:400] + "..."
                processed_claims.append(claim)
            
            # Limit to top 10 claims for speed
            return processed_claims[:10]
            
        except Exception as e:
            if self._verbose:
                print(f"   ⚠️ LLM extraction failed: {e}, using regex fallback")
            return self._extract_claims_regex(answer)
    
    def _extract_claims_regex(self, answer: str) -> List[str]:
        """Regex-based claim extraction (fallback)"""
        # Pre-clean
        if "## 📚 Nguồn tham khảo:" in answer:
            answer = answer.split("## 📚 Nguồn tham khảo:")[0]
        if "Thông tin chỉ nhằm tham khảo" in answer:
            answer = answer.split("Thông tin chỉ nhằm tham khảo")[0]
        
        # Remove citations and markdown
        answer = re.sub(r'\[\d+\]', '', answer)
        answer = re.sub(r'^#+\s+.*$', '', answer, flags=re.MULTILINE)
        answer = re.sub(r'[✅❌⚠️🔹🔸•]\s*', '', answer)
        
        # Split by sentence
        sentences = re.split(r'[.!?]\s+|\n+', answer)
        claims = []
        
        for sent in sentences:
            sent = sent.strip()
            # Skip too short/long
            if len(sent) < 20 or len(sent) > 400:
                continue
            # Skip headers, bullets that slipped through
            if sent.startswith(('#', '-', '*')):
                continue
            if sent.startswith('[') and sent.endswith(']'):
                continue
            claims.append(sent)
        
        return claims[:10]  # Limit for speed
    
    def validate_input(self, state: AgentState) -> bool:
        return bool(state.answer) and isinstance(state.reranked_chunks, list)


# === Quick Test ===
if __name__ == "__main__":
    from sentence_transformers import CrossEncoder
    
    print("Loading mDeBERTa-v3 NLI model...")
    nli_model = CrossEncoder("./mdeberta_v3_medical_nli_v2", device="cuda")
    
    grader = NLIHallucinationGraderV2(nli_model, verbose=True)
    
    # Simulate chunks
    chunks = [
        {"text": "Acid folic quan trọng trong thai kỳ vì giúp ngăn dị tật ống thần kinh.", 
         "title": "Acid Folic - Tại sao quan trọng"},
        {"text": "Liều khuyến nghị là 400 mcg mỗi ngày, bắt đầu ít nhất 1 tháng trước khi mang thai.",
         "title": "Acid Folic - Liều dùng"},
    ]
    
    # Test answer
    test_answer = "Acid folic giúp ngăn ngừa dị tật ống thần kinh. Liều khuyến nghị là 400mcg mỗi ngày."
    
    print("\n" + "="*60)
    print("NLI HALLUCINATION GRADER V2 - Batch Processing Test")
    print("="*60)
    
    # Create test state
    state = AgentState(query="test", answer=test_answer)
    state.reranked_chunks = chunks
    state.metadata = {"verbose": True}
    
    result = grader.execute(state)
    print(f"\nFinal: {result.metadata.get('hallucination_check')}")

