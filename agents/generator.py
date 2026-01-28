#!/usr/bin/env python3
"""
Generator Agent - Sinh câu trả lời từ top chunks
Migrate từ generator_node(), build_context(), build_source_references() trong agentic_rag.py
"""

import sys
sys.path.append('..')

from async_agents.base import BaseAgent, AgentState
from typing import List, Dict, Any, Tuple


def llm_text(system: str, user: str, openai_client, model: str, temperature=0.0) -> str:
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


def llm_text_stream(system: str, user: str, openai_client, model: str, temperature=0.0):
    """
    Call OpenAI API với STREAMING response
    
    Yields chunks of text as they arrive from the API
    For progressive reveal in UI (better UX)
    """
    try:
        stream = openai_client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=temperature,
            stream=True  # Enable streaming
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        print(f"⚠️ LLM streaming failed: {e}")
        yield f"Lỗi: {e}"


class GeneratorAgent(BaseAgent):
    """
    Generator Agent - sinh câu trả lời từ top chunks
    
    Features:
    - Build context từ top-K chunks
    - Strict rules: CHỈ dùng thông tin từ sources
    - Auto-append citations [title §chunk_id]
    - Auto-append source references với links
    - Rejection handling cho câu hỏi không phù hợp
    """
    
    def __init__(self, openai_client, model: str, truncate_func, top_k: int = 20):
        super().__init__(name="Generator")
        self.openai_client = openai_client
        self.model = model
        self.truncate_tokens = truncate_func
        self.top_k = top_k
    
    def execute(self, state: AgentState) -> AgentState:
        """Execute generation"""
        chunks = state.reranked_chunks
        verbose = state.metadata.get("verbose", False)
        
        # Handle rejection case
        if not state.plan.get("need_retrieval", True):
            rejection_reason = state.plan.get("rejection_reason", "")
            if rejection_reason:
                answer = (
                    f"Xin lỗi, tôi không thể trả lời câu hỏi này vì: {rejection_reason}\n\n"
                    "Hệ thống chỉ cung cấp thông tin y khoa tổng quát từ tài liệu tham khảo, "
                    "không đưa ra lời khuyên chẩn đoán hoặc điều trị cá nhân.\n\n"
                    "Vui lòng tham khảo ý kiến bác sĩ chuyên khoa cho các vấn đề sức khỏe cụ thể.\n\n"
                    "Thông tin chỉ nhằm tham khảo, không thay thế tư vấn y khoa cá nhân."
                )
                new_state = state.copy()
                new_state.answer = answer
                return new_state
        
        ctx, chosen = self._build_context(chunks, self.top_k, verbose)
        
        if verbose:
            print(f"✍️ Generating answer from {len(chosen)} chunks...")
        
        system_prompt = (
            "Bạn là Medical RAG Answerer chuyên nghiệp.\\n\\n"
            "QUY TẮC:\\n"
            "- Trả lời DỰA CHÍNH XÁC trên thông tin trong các đoạn tham chiếu\\n"
            "- CÓ THỂ diễn giải lại (paraphrase) để dễ hiểu hơn, NHƯNG PHẢI GIỮ NGHĨA CHÍNH XÁC từ nguồn\\n"
            "- KHÔNG được thêm thông tin từ kiến thức riêng của bạn\\n"
            "- Khi tổng hợp nhiều điểm thành danh sách, CHỈ liệt kê những gì CÓ TRONG nguồn\\n"
            "- NẾU không tìm thấy thông tin liên quan, hãy nói rõ 'Không tìm thấy thông tin về vấn đề này trong tài liệu được cung cấp'\\n"
            "- Mỗi câu/đoạn PHẢI có trích dẫn đầy đủ dạng [Tên bài viết - Tên mục - Nguồn X]\\n"
            "- Ví dụ: 'Acid folic giúp ngăn ngừa dị tật ống thần kinh [Acid Folic - Tại sao acid folic quan trọng - Nguồn 1]'\\n"
            "- KHÔNG SỬ DỤNG emoji hoặc icon (🔹❌✅📌...) trong câu trả lời. Giữ văn phong chuyên nghiệp, học thuật.\\n\\n"
            "CẤU TRÚC:\\n"
            "1. Tóm tắt ngắn gọn\\n"
            "2. Các điểm chính với trích dẫn cụ thể dạng [Tên bài - Tên mục - Nguồn X]\\n"
            "3. Kết luận từ thông tin trong nguồn\\n"
            "4. Disclaimer: 'Thông tin chỉ nhằm tham khảo, không thay thế tư vấn y khoa cá nhân.'\\n"
            "5. KHÔNG cần thêm phần 'Nguồn tham khảo' - hệ thống sẽ tự động thêm\\n\\n"
            "Trả lời bằng tiếng Việt."
        )
        
        user_prompt = (
            f"Câu hỏi: {state.query}\\n\\n"
            f"ĐOẠN THAM CHIẾU (CHỈ ĐƯỢC SỬ DỤNG THÔNG TIN TRONG ĐÂY):\\n"
            f"{'='*60}\\n"
            f"{ctx}\\n"
            f"{'='*60}\\n\\n"
            "LƯU Ý: Bạn CHỈ được trả lời dựa trên thông tin có trong các đoạn tham chiếu ở trên. "
            "Không được sử dụng kiến thức bên ngoài. Nếu không tìm thấy thông tin liên quan, "
            "hãy nói rõ 'Không tìm thấy thông tin về vấn đề này trong tài liệu được cung cấp'.\\n\\n"
            "Hãy trả lời câu hỏi:"
        )
        
        answer = llm_text(system_prompt, user_prompt, self.openai_client, self.model, temperature=0.0)
        
        # Automatically append source references
        source_refs = self._build_source_references(chosen)
        if source_refs:
            final_answer = f"{answer}\n\n{source_refs}"
        else:
            final_answer = answer
        
        # Add query rewrite transparency if applicable
        rewritten_query = state.metadata.get("rewritten_query")
        original_query = state.metadata.get("original_query", state.query)
        
        if rewritten_query and rewritten_query != original_query:
            # Query was rewritten - add transparency note
            strategy = state.metadata.get("rewrite_strategy", "unknown")
            explanation = state.metadata.get("rewrite_explanation", "")
            intent_score = state.metadata.get("intent_similarity_score", 0)
            
            transparency_note = f"""

---

📝 **Lưu ý về xử lý câu hỏi:**

Câu hỏi gốc của bạn không tìm thấy tài liệu phù hợp, vì vậy hệ thống đã tự động viết lại để tìm kiếm tốt hơn:

- **Câu hỏi gốc:** "{original_query}"
- **Câu hỏi đã tối ưu:** "{rewritten_query}"
- **Phương pháp:** {strategy}
- **Giải thích:** {explanation}
- **Intent Guardrail:** ✅ Verified (similarity: {intent_score:.2f}/1.00)

Câu trả lời trên được tạo dựa trên câu hỏi đã tối ưu, nhưng vẫn giữ đúng ý định ban đầu của bạn.
"""
            final_answer = final_answer + transparency_note
        
        new_state = state.copy()
        new_state.answer = final_answer
        return new_state
    
    def _build_context(self, chunks: List[Dict[str, Any]], k: int, verbose: bool = False) -> Tuple[str, List[Dict[str, Any]]]:
        """Xây dựng context từ top-k chunks với numbered citations và chunk IDs"""
        chosen = chunks[:k]  # Use configurable k (default: 20 for V2)
        blocks = []
        
        # Build source mapping for numbered citations
        source_map = {}
        source_counter = 1
        for c in chosen:
            source = c.get('source', 'N/A')
            if source and source not in source_map:
                source_map[source] = source_counter
                source_counter += 1
        
        if verbose:
            print(f"\n📄 CHI TIẾT CÁC CHUNKS ĐƯỢC SỬ DỤNG:")
            print("=" * 80)
        
        for i, c in enumerate(chosen, 1):
            # Extract section info for citation linking
            chunk_id = c.get('chunk_id', c.get('doc_id', 'N/A'))
            section_title = self._extract_section_title(c)
            article_title = c.get('title', 'Doc').split(' - ')[0]  # Get article title only
            
            if verbose:
                print(f"\n🔍 CHUNK {i}:")
                print(f"   Article: {article_title}")
                print(f"   Section: {section_title}")
                print(f"   Chunk ID: {chunk_id}")
                print(f"   Source: {c.get('source', 'N/A')}")
                print(f"   Text length: {len(c.get('text', ''))} chars")
                print(f"   Text preview: {c.get('text', '')[:200]}...")
                print("-" * 60)
            
            # NEW: Citation format with section title for user-friendly display
            source_num = source_map.get(c.get('source', 'N/A'), i)
            tag = f'[{article_title} - {section_title} - Nguồn {source_num}]'
            source_info = f"URL: {c.get('source', 'N/A')}"
            blocks.append(f"{tag}\n{source_info}\n{self.truncate_tokens(c['text'], 800)}")
        
        if verbose:
            print("=" * 80)
            print(f"✅ Đã chọn {len(chosen)} chunks để tạo context\n")
        
        return "\n\n---\n\n".join(blocks), chosen
    
    def _extract_section_title(self, chunk: Dict[str, Any]) -> str:
        """
        Extract section title from chunk metadata for user-friendly citation
        
        Priority:
        1. section_title from V2 metadata
        2. Parse from full title ("Article - Section")
        3. Use chunk_id section number
        """
        # V2 hierarchical chunks have section_title in metadata
        section_title = chunk.get('section_title', '')
        if section_title:
            return section_title
        
        # Parse from full title format "Article Title - Section Title"
        full_title = chunk.get('title', '')
        if ' - ' in full_title:
            parts = full_title.split(' - ')
            if len(parts) >= 2:
                return parts[-1]  # Return last part as section
        
        # Fallback: use section number from chunk_id
        chunk_id = chunk.get('chunk_id', chunk.get('doc_id', ''))
        if '::' in str(chunk_id):
            section_num = chunk_id.split('::')[-1]
            return f"Mục {section_num}"
        
        return "Nội dung"
    
    def _extract_section_number(self, chunk: Dict[str, Any]) -> str:
        """
        Extract section number from chunk metadata
        
        Priority:
        1. section_number from V2 metadata (e.g., "1", "3.2")
        2. Parse from chunk_id (e.g., "art0001::3.2" -> "3.2")
        """
        # V2 hierarchical chunks have section_number in metadata
        section_number = chunk.get('section_number', '')
        if section_number:
            return section_number
        
        # Fallback: extract from chunk_id
        chunk_id = chunk.get('chunk_id', chunk.get('doc_id', ''))
        if '::' in str(chunk_id):
            return chunk_id.split('::')[-1]
        
        return ""
    
    def _build_source_references(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Xây dựng danh sách nguồn tham khảo với section numbers theo cấu trúc hierarchical
        
        Format: [X] Article Title  Section Y: Section Title
        """
        if not chunks:
            return ""
        
        # Group chunks by source, keeping track of sections with their numbers
        sources = {}  # {source: {"title": article_title, "sections": [(section_num, section_title, html_id)]}}
        for c in chunks:
            source = c.get("source", "")
            if not source:
                continue
            
            # Get article title (first part before " - ")
            full_title = c.get("title", "Tài liệu")
            article_title = full_title.split(' - ')[0] if ' - ' in full_title else full_title
            
            # Get section info
            section_number = self._extract_section_number(c)
            section_title = self._extract_section_title(c)
            chunk_id = c.get('chunk_id', c.get('doc_id', ''))
            
            # Create HTML-safe ID for frontend deep linking
            html_id = str(chunk_id).replace('::', '_').replace('.', '_')
            
            if source not in sources:
                sources[source] = {
                    "title": article_title,
                    "sections": []
                }
            
            # Add section if not already present (check by section_number to avoid duplicates)
            section_entry = (section_number, section_title, html_id)
            existing_nums = [s[0] for s in sources[source]["sections"]]
            if section_number not in existing_nums:
                sources[source]["sections"].append(section_entry)
        
        if not sources:
            return ""
        
        # Build reference list with hierarchical section format
        ref_lines = ["\n---\n## 📚 Nguồn tham khảo:\n"]
        for i, (source, info) in enumerate(sources.items(), 1):
            article_title = info["title"]
            sections = info["sections"]
            
            # Sort sections by section number
            sections.sort(key=lambda x: self._section_sort_key(x[0]))
            
            # Build section references (limit to 3)
            valid_sections = [(num, title, hid) for num, title, hid in sections if num and title][:3]
            
            if valid_sections:
                # Format: Section X: Title • Section Y: Title
                section_parts = []
                for section_num, section_title, html_id in valid_sections:
                    section_parts.append(f"Section {section_num}: {section_title}")
                
                sections_str = " • ".join(section_parts)
                # Main article reference with sections on same line
                ref_lines.append(f"**[{i}] [{article_title}]({source})**  {sections_str}")
            else:
                # No sections, just article title
                ref_lines.append(f"**[{i}] [{article_title}]({source})**")
            
            ref_lines.append("")  # Empty line between sources
        
        return "\n".join(ref_lines)
    
    def _section_sort_key(self, section_num: str) -> tuple:
        """
        Convert section number to sortable tuple
        e.g., "3.2" -> (3, 2), "1" -> (1, 0)
        """
        if not section_num:
            return (999, 999)
        try:
            parts = section_num.split('.')
            return tuple(int(p) for p in parts) + (0,) * (2 - len(parts))
        except ValueError:
            return (999, 999)
    
    def validate_input(self, state: AgentState) -> bool:
        """Validate input state"""
        return isinstance(state.reranked_chunks, list) and isinstance(state.plan, dict)
