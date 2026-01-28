#!/usr/bin/env python3
"""
Critic Agent - Kiểm tra faithfulness và medical safety
Migrate từ critic_node() trong agentic_rag.py
"""

import sys
sys.path.append('..')

from async_agents.base import BaseAgent, AgentState
from typing import List, Dict, Any
import json


def llm_json(system: str, user: str, openai_client, model: str, max_tokens=1000) -> Dict[str, Any]:
    """Call OpenAI API với JSON response format"""
    try:
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"⚠️ LLM JSON call failed: {e}")
        return {"error": str(e)}


def llm_text(system: str, user: str, openai_client, model: str, temperature=0.2) -> str:
    """Call OpenAI API với text response"""
    try:
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=temperature,
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"⚠️ LLM text call failed: {e}")
        return f"Lỗi: {e}"


class CriticAgent(BaseAgent):
    """
    Critic/Safety Agent - kiểm tra faithfulness và medical safety
    
    Features:
    - Faithfulness check: Answer có trung thực với sources không?
    - Medical safety check: Không đưa lời khuyên cá nhân, liều lượng thuốc
    - 1-pass revision nếu vi phạm
    - Đảm bảo disclaimer y tế present
    """
    
    def __init__(self, openai_client, model: str, truncate_func, top_k: int = 8):
        super().__init__(name="Critic")
        self.openai_client = openai_client
        self.model = model
        self.truncate_tokens = truncate_func
        self.top_k = top_k
    
    def execute(self, state: AgentState) -> AgentState:
        """Execute critic review"""
        verbose = state.metadata.get("verbose", False)
        
        # ============================================================
        # CASE 1: Handle NLI-detected contradictions
        # Remove contradicted claims from answer
        # ============================================================
        need_removal = state.metadata.get("need_critic_removal", False)
        contradicted_claims = state.metadata.get("contradicted_claims", [])
        
        if need_removal and contradicted_claims:
            if verbose:
                print(f"🔧 Critic removing {len(contradicted_claims)} contradicted claims...")
            
            cleaned_answer = self._remove_contradicted_claims(
                state.answer, 
                contradicted_claims, 
                verbose
            )
            
            new_state = state.copy()
            new_state.answer = cleaned_answer
            new_state.metadata["critic_action"] = "REMOVED_CONTRADICTIONS"
            
            if verbose:
                print("✅ Contradicted claims removed from answer")
            
            return new_state
        
        # ============================================================
        # CASE 2: Skip Critic if NLI already passed (APPROVED)
        # ============================================================
        nli_grounded = state.metadata.get("grounded", False)
        hallucination_check = state.metadata.get("hallucination_check", "")
        
        if nli_grounded and hallucination_check == "APPROVED":
            if verbose:
                print("⏭️ Critic skipped (NLI already verified: APPROVED)")
            return state
        
        # ============================================================
        # CASE 3: Full Critic check if NLI was skipped or has warnings
        # ============================================================
        chunks = state.reranked_chunks[:self.top_k]
        refs = "\n\n".join(f'[{c.get("title","Doc")} §{c["chunk_id"]}]\n{self.truncate_tokens(c["text"],400)}'
                          for c in chunks)
        
        if verbose:
            print("🔍 Critic reviewing answer...")
        
        system_prompt = (
            "Bạn là Critic & Safety Evaluator. Nhiệm vụ QUAN TRỌNG: Kiểm tra nghiêm ngặt tính trung thực của câu trả lời.\\n\\n"
            "KIỂM TRA FAITHFULNESS (Độ trung thực):\\n"
            "- Từng câu trong answer có được hỗ trợ bởi sources không?\\n"
            "- Có thông tin nào được thêm vào mà không có trong sources?\\n"
            "- Có suy diễn hoặc giả định nào vượt quá thông tin trong sources?\\n"
            "- Trích dẫn [title §chunk_id] có chính xác không?\\n"
            "- Có phần 'Nguồn tham khảo' với links bài viết không?\\n\\n"
            "KIỂM TRA MEDICAL SAFETY:\\n"
            "- Không đưa lời khuyên chẩn đoán/điều trị cá nhân\\n"
            "- Không đưa liều lượng thuốc cụ thể\\n"
            "- Có disclaimer y tế không?\\n\\n"
            "NẾU vi phạm bất kỳ điều nào → action='REVISE'\\n"
            "NẾU tất cả đều đúng → action='APPROVE'\\n\\n"
            "Trả JSON với keys: action (APPROVE/REVISE), notes (string), suggestion (string nếu cần sửa)"
        )
        
        user_prompt = f"Answer:\n{state.answer}\n\nSources:\n{refs}"
        verdict = llm_json(system_prompt, user_prompt, self.openai_client, self.model)
        
        if verdict.get("action") == "REVISE":
            if verbose:
                print("⚠️ Critic requests revision")
            
            # 1-pass revision
            revise_prompt = (
                f"NHIỆM VỤ: Sửa câu trả lời để tuân thủ ghi chú của critic, CHỈ sử dụng thông tin từ nguồn tham chiếu.\\n\\n"
                f"GHI CHÚ TỪ CRITIC: {verdict.get('notes', '')}\\n"
                f"GỢI Ý SỬA CHỮA: {verdict.get('suggestion', '')}\\n\\n"
                f"CÂU TRẢ LỜI CŨ (CẦN SỬA):\\n{state.answer}\\n\\n"
                f"NGUỒN THAM CHIẾU (CHỈ ĐƯỢC SỬ DỤNG THÔNG TIN TRONG ĐÂY):\\n"
                f"{'='*60}\\n{refs}\\n{'='*60}\\n\\n"
                f"YÊU CẦU: Sửa câu trả lời dựa trên ghi chú của critic, đảm bảo:\\n"
                f"- CHỈ sử dụng thông tin có trong nguồn tham chiếu\\n"
                f"- Có trích dẫn [title §chunk_id] cho mọi thông tin\\n"
                f"- Không thêm kiến thức bên ngoài\\n"
                f"- Có disclaimer y tế"
            )
            
            revision_system = (
                "Bạn là Medical RAG Answerer. NHIỆM VỤ: Sửa câu trả lời theo yêu cầu của critic.\\n\\n"
                "QUY TẮC NGHIÊM NGẶT KHI SỬA:\\n"
                "- CHỈ sử dụng thông tin từ nguồn tham chiếu được cung cấp\\n"
                "- KHÔNG thêm kiến thức từ bên ngoài\\n"
                "- PHẢI có trích dẫn [title §chunk_id] cho mọi thông tin\\n"
                "- PHẢI có disclaimer y tế\\n"
                "- KHÔNG cần thêm phần 'Nguồn tham khảo' - hệ thống sẽ tự động thêm\\n"
                "- Nếu thiếu thông tin trong nguồn, nói rõ 'Không đủ thông tin trong tài liệu'\\n"
                "- KHÔNG SỬ DỤNG emoji hoặc icon trong câu trả lời. Giữ văn phong chuyên nghiệp.\\n\\n"
                "Trả lời bằng tiếng Việt."
            )
            
            revised_answer = llm_text(
                revision_system,
                revise_prompt,
                self.openai_client,
                self.model,
                temperature=0.1
            )
            
            # Re-append source references for revised answer
            source_refs = self._build_source_references(chunks)
            if source_refs:
                final_revised_answer = f"{revised_answer}\n\n{source_refs}"
            else:
                final_revised_answer = revised_answer
            
            new_state = state.copy()
            new_state.answer = final_revised_answer
            return new_state
        else:
            if verbose:
                print("✅ Critic approves answer")
        
        return state
    
    def _build_source_references(self, chunks: List[Dict[str, Any]]) -> str:
        """Xây dựng danh sách nguồn tham khảo với links"""
        if not chunks:
            return ""
        
        # Group chunks by source to avoid duplicate links
        sources = {}
        for c in chunks:
            title = c.get("title", "Tài liệu")
            source = c.get("source", "")
            if source and source not in sources:
                sources[source] = title
        
        if not sources:
            return ""
        
        ref_lines = ["## 📚 Nguồn tham khảo:"]
        for i, (source, title) in enumerate(sources.items(), 1):
            ref_lines.append(f"{i}. **{title}** - {source}")
        
        return "\n".join(ref_lines)
    
    def _remove_contradicted_claims(self, answer: str, claims: List[str], verbose: bool = False) -> str:
        """
        Remove contradicted claims from the answer using LLM
        
        Args:
            answer: Original answer text
            claims: List of claims that contradict the sources
            verbose: Enable verbose logging
        """
        if not claims:
            return answer
        
        # Preserve source references section
        source_section = ""
        main_answer = answer
        if "## 📚 Nguồn tham khảo:" in answer:
            parts = answer.split("## 📚 Nguồn tham khảo:")
            main_answer = parts[0]
            source_section = "\n\n## 📚 Nguồn tham khảo:" + parts[1]
        
        claims_list = "\n".join([f"- {claim}" for claim in claims])
        
        system_prompt = """Bạn là trợ lý y tế. Nhiệm vụ: Loại bỏ các thông tin SAI khỏi câu trả lời.

QUY TẮC:
1. XÓA hoàn toàn các câu chứa thông tin sai (được liệt kê bên dưới)
2. GIỮ NGUYÊN format: markdown, bullet points, citations
3. GIỮ NGUYÊN các thông tin đúng
4. KHÔNG thêm thông tin mới
5. ĐẢM BẢO câu trả lời vẫn mạch lạc

Trả về câu trả lời đã chỉnh sửa, KHÔNG giải thích."""

        user_prompt = f"""CÂU TRẢ LỜI GỐC:
{main_answer}

CÁC THÔNG TIN SAI (CẦN XÓA):
{claims_list}

Viết lại câu trả lời, loại bỏ các thông tin sai ở trên:"""

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
            
            # Add warning about removed content
            warning = "\n\n⚠️ **Lưu ý**: Một số thông tin đã được loại bỏ do không phù hợp với nguồn tài liệu."
            
            # Re-append source section
            if source_section:
                return cleaned + warning + source_section
            return cleaned + warning
            
        except Exception as e:
            if verbose:
                print(f"   ⚠️ Failed to remove claims: {e}")
            # Fallback: add warning to original
            return answer + "\n\n❌ **CẢNH BÁO**: Một số thông tin trong câu trả lời có thể không chính xác."
    
    def validate_input(self, state: AgentState) -> bool:
        """Validate input state"""
        return bool(state.answer) and isinstance(state.reranked_chunks, list)

