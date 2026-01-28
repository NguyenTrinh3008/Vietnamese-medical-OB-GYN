#!/usr/bin/env python3
"""
Streamlit UI for Advanced RAG Medical System
"""

import streamlit as st
import time
import os
from datetime import datetime

# === Cached Model Loading ===
@st.cache_resource(show_spinner=False)
def load_rag_system():
    """Load and cache RAG system V2 components - only runs once!"""
    with st.spinner("🔧 Initializing RAG system V2 (first time only)..."):
        import rag_system_v2 as rag_system
        return rag_system

# Load cached RAG system
rag = load_rag_system()
ask = rag.ask_v2  # V2 uses ask_v2 instead of ask

# === Page Config ===
st.set_page_config(
    page_title="Advanced RAG - Medical QA",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Custom CSS ===
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .answer-container {
        background-color: transparent !important;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #28a745;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(40, 167, 69, 0.2);
        color: #ffffff !important;
    }
    .metric-card {
        background-color: #fff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# === Sidebar ===
with st.sidebar:
    st.markdown("## ⚙️ Cấu hình")
    
    # Advanced Features
    st.markdown("### 🚀 Advanced RAG Features")
    
    enable_retrieval_grader = st.checkbox(
        "🔍 Retrieval Grader (CRAG)",
        value=True,
        help="Đánh giá và filter chunks không liên quan"
    )
    
    enable_hallucination_grader = st.checkbox(
        "🛡️ Hallucination Grader (Self-RAG)",
        value=True,
        help="Phát hiện thông tin không có nguồn"
    )
    
    st.markdown("---")
    
    # System info
    st.markdown("## 📊 Thông tin hệ thống")
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        st.success("✅ OpenAI API Key: OK")
    else:
        st.error("❌ OpenAI API Key: Chưa set")
    

    st.markdown("---")
    
    # Quick stats
    st.markdown("## 📈 Thống kê")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Documents", "443", "✅")
    with col2:
        st.metric("Model", "ViRanker", "🔄")

# === Main Content ===
st.markdown('<div class="main-header"> Advanced RAG System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Hệ thống trả lời câu hỏi y tế với CRAG + Self-RAG</div>', unsafe_allow_html=True)

# Feature indicators
col1, col2, col3 = st.columns(3)
with col1:
    if enable_retrieval_grader:
        st.markdown('<div class="success-box">✅ CRAG: Enabled</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">⏸️ CRAG: Disabled</div>', unsafe_allow_html=True)

with col2:
    if enable_hallucination_grader:
        st.markdown('<div class="success-box">✅ Self-RAG: Enabled</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">⏸️ Self-RAG: Disabled</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="success-box">✅ ViRanker: Enabled</div>', unsafe_allow_html=True)

st.markdown("---")

# === Tabs ===
tab1, tab2, tab3 = st.tabs(["💬 Hỏi đáp", "🧪 Test mẫu", "ℹ️ Hướng dẫn"])

with tab1:
    st.markdown("### 💬 Đặt câu hỏi y tế")
    
    # Question input
    question = st.text_area(
        "Câu hỏi của bạn:",
        placeholder="Ví dụ: Tại sao acid folic quan trọng trong thai kỳ?",
        height=120,
        help="Nhập câu hỏi y tế bằng tiếng Việt"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        ask_button = st.button("🔍 Tìm câu trả lời", type="primary", use_container_width=True)
    
    with col2:
        clear_button = st.button("🗑️ Xóa", use_container_width=True)
    
    with col3:
        show_details = st.checkbox("📋 Hiển thị chi tiết", value=False, help="Hiển thị quá trình xử lý")
    
    if clear_button:
        st.rerun()
    
    # Process question
    if ask_button and question:
        if not api_key:
            st.error("❌ Vui lòng cấu hình OpenAI API key trong file .env")
        else:
            with st.spinner("🤔 Đang suy nghĩ..."):
                start_time = time.time()
                
                try:
                    # Call the RAG system
                    answer = ask(
                        question, 
                        verbose=show_details,
                        enable_retrieval_grader=enable_retrieval_grader,
                        enable_hallucination_grader=enable_hallucination_grader
                    )
                    
                    processing_time = time.time() - start_time
                    
                    # Display answer
                    st.markdown("### 💡 Câu trả lời")
                    
                    # Use expander for better visibility
                    with st.expander("💡 Xem câu trả lời chi tiết", expanded=True):
                        st.markdown("**🤖 Hệ thống Advanced RAG trả lời:**")
                        st.markdown("---")
                        st.write(answer)
                        st.markdown("---")
                        st.warning("⚠️ Thông tin chỉ nhằm tham khảo, không thay thế tư vấn y khoa cá nhân.")
                    
                    # Show metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("⏱️ Thời gian", f"{processing_time:.1f}s")
                    
                    with col2:
                        if enable_retrieval_grader:
                            st.metric("🔍 CRAG", "Enabled")
                        else:
                            st.metric("🔍 CRAG", "Disabled")
                    
                    with col3:
                        if enable_hallucination_grader:
                            st.metric("🛡️ Self-RAG", "Enabled")
                        else:
                            st.metric("🛡️ Self-RAG", "Disabled")
                    
                    with col4:
                        st.metric("📅 Thời điểm", datetime.now().strftime("%H:%M:%S"))
                    
                    # Performance indicator
                    if processing_time < 30:
                        st.success(f"✅ Phản hồi nhanh: {processing_time:.1f}s")
                    elif processing_time < 60:
                        st.info(f"ℹ️ Phản hồi trung bình: {processing_time:.1f}s")
                    else:
                        st.warning(f"⚠️ Phản hồi chậm: {processing_time:.1f}s")
                    
                except Exception as e:
                    st.error(f"❌ Lỗi: {str(e)}")
                    st.markdown("**Gợi ý khắc phục:**")
                    st.markdown("- Kiểm tra OpenAI API key")
                    st.markdown("- Kiểm tra kết nối internet")
                    st.markdown("- Thử tắt một số advanced features")

with tab2:
    st.markdown("### 🧪 Test với câu hỏi mẫu")
    
    # Sample questions
    sample_questions = [
        "Tại sao acid folic quan trọng trong thai kỳ?",
        "Triệu chứng của teo âm đạo là gì?",
        "Khi nào nên thực hiện chọc ối?",
        "Lợi ích của việc tập thể dục sau sinh là gì?",
        "Vitamin B có tác dụng gì trong thai kỳ?",
        "Choline và Omega-3 quan trọng như thế nào trong thai kỳ?",
        "Sàng lọc quý I thai kỳ là gì?",
        "Dinh dưỡng trong thai kỳ cần chú ý gì?",
        "Chăm sóc phụ nữ cho con bú như thế nào?",
        "Các nhóm thực phẩm thiết yếu trong thai kỳ là gì?"
    ]
    
    # Question selection
    selected_question = st.selectbox(
        "Chọn câu hỏi mẫu:",
        options=sample_questions,
        index=0
    )
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        test_button = st.button("🧪 Test câu hỏi này", type="secondary", use_container_width=True)
    
    with col2:
        st.markdown(f"**Câu hỏi đã chọn:** {selected_question}")
    
    if test_button:
        if not api_key:
            st.error("❌ Vui lòng cấu hình OpenAI API key")
        else:
            with st.spinner("🧪 Đang test..."):
                start_time = time.time()
                
                try:
                    answer = ask(
                        selected_question, 
                        verbose=False,
                        enable_retrieval_grader=enable_retrieval_grader,
                        enable_hallucination_grader=enable_hallucination_grader
                    )
                    processing_time = time.time() - start_time
                    
                    # Display results
                    st.markdown("### 📋 Kết quả test")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("**💡 Câu trả lời:**")
                        with st.expander("📋 Xem kết quả test", expanded=True):
                            st.write(answer)
                            st.info("ℹ️ Thông tin tham khảo y tế")
                    
                    with col2:
                        st.markdown("**📊 Metrics:**")
                        st.metric("Thời gian", f"{processing_time:.1f}s")
                        st.metric("CRAG", "✅" if enable_retrieval_grader else "❌")
                        st.metric("Self-RAG", "✅" if enable_hallucination_grader else "❌")
                        st.metric("Status", "✅ Thành công")
                
                except Exception as e:
                    st.error(f"❌ Test thất bại: {str(e)}")

with tab3:
    st.markdown("### ℹ️ Hướng dẫn sử dụng")
    
    st.markdown("""
    #### 🚀 Cách sử dụng
    
    1. **Bật/tắt features**: Sử dụng sidebar để enable/disable CRAG và Self-RAG
    2. **Đặt câu hỏi**: Nhập câu hỏi y tế bằng tiếng Việt
    3. **Nhấn "Tìm câu trả lời"**: Hệ thống sẽ xử lý và trả về kết quả
    4. **Xem chi tiết**: Bật "Hiển thị chi tiết" để xem quá trình xử lý
    
    #### 🔬 Advanced Features
    
    **🔍 CRAG (Corrective RAG):**
    - Đánh giá chất lượng chunks được retrieve
    - Filter out chunks không liên quan (incorrect)
    - Cải thiện precision ~28%
    - Trade-off: +10-15s latency
    
    **🛡️ Self-RAG (Hallucination Detection):**
    - Phát hiện thông tin không có nguồn
    - Đảm bảo answer grounded vào sources
    - Critical cho medical AI safety
    - Trade-off: +5-8s latency
    
    #### ⚙️ Hiệu năng
    
    - **Fast mode** (tắt cả 2): ~26s
    - **Balanced** (chỉ CRAG): ~35-40s
    - **Quality** (bật cả 2): ~45-50s
    
    #### 🔧 Troubleshooting
    
    - **Lỗi API**: Kiểm tra OpenAI API key trong `.env`
    - **Chậm**: Thử tắt một số advanced features
    - **Lỗi model**: Kiểm tra kết nối internet
    """)
    
    # Technical details
    st.markdown("### 🔧 Chi tiết kỹ thuật")
    
    tech_details = {
        "Framework": "Custom Agent Framework (No LangGraph)",
        "Vector DB": "ChromaDB",
        "Embedding": "Vietnamese Document Embedding",
        "Reranker": "ViRanker (namdp-ptit/ViRanker)",
        "LLM": "OpenAI GPT-4o-mini",
        "Advanced Features": "CRAG + Self-RAG",
        "Language": "Python 3.10+"
    }
    
    for key, value in tech_details.items():
        st.markdown(f"**{key}**: {value}")

# === Footer ===
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666; font-size: 0.9rem;">'
    '🩺 Advanced RAG System with CRAG + Self-RAG | '
    'Custom Agent Framework + ChromaDB + ViRanker + OpenAI | '
    f'© {datetime.now().year}'
    '</div>',
    unsafe_allow_html=True
)

# === Disclaimer ===
st.markdown("""
<div class="warning-box">
<strong>⚠️ Lưu ý quan trọng:</strong><br>
Hệ thống này chỉ cung cấp thông tin tham khảo từ các tài liệu y khoa. 
Không thay thế tư vấn, chẩn đoán hay điều trị y tế chuyên nghiệp. 
Luôn tham khảo ý kiến bác sĩ cho các vấn đề sức khỏe cụ thể.
</div>
""", unsafe_allow_html=True)
