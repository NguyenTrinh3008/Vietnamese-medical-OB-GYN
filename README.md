# 🏥 Vietnamese Medical OB-GYN RAG System

Hệ thống RAG (Retrieval-Augmented Generation) chuyên về lĩnh vực **Sản Phụ Khoa** (Obstetrics & Gynecology) cho tiếng Việt.

## ✨ Tính năng

- **Agentic RAG Pipeline**: Sử dụng nhiều agent phối hợp để trả lời câu hỏi y khoa
- **Hierarchical Chunking**: Chia nhỏ tài liệu theo cấu trúc phân cấp (~100-300 tokens/chunk)
- **NLI Hallucination Detection**: Phát hiện thông tin không chính xác sử dụng model NLI fine-tuned
- **Hybrid Search**: Kết hợp semantic search (ChromaDB) và BM25 để tìm kiếm tốt hơn
- **RAG-Fusion**: Mở rộng query để tăng recall
- **Query Decomposition**: Phân tách câu hỏi phức tạp thành các câu hỏi con

## 📁 Cấu trúc dự án

```
nguyentrinhmedical/
├── all_articles.json          # Dữ liệu bài viết y khoa đã được chunked
├── ingest_hierarchical.py     # Script tạo database
├── rag_system_v2.py           # Hệ thống RAG chính
├── streamlit_ui.py            # Giao diện người dùng
├── agents/                    # Các agent xử lý
│   ├── generator.py           # Agent sinh câu trả lời
│   ├── critic.py              # Agent phê bình và cải thiện
│   ├── nli_hallucination_grader_v2.py  # Agent phát hiện hallucination
│   └── reranker_v2.py         # Agent xếp hạng kết quả
├── async_agents/              # Các agent bất đồng bộ
├── evaluation/                # Scripts đánh giá
└── trainNLImodels/            # Scripts huấn luyện model NLI
```

## 🚀 Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/NguyenTrinh3008/Vietnamese-medical-OB-GYN.git
cd Vietnamese-medical-OB-GYN
```

### 2. Tạo virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements_agentic_rag.txt
```

### 4. Cấu hình biến môi trường

Tạo file `.env` trong thư mục gốc:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

### 5. Tạo Database (Bắt buộc!)

⚠️ **QUAN TRỌNG**: Bạn phải chạy script này trước khi sử dụng hệ thống!

```bash
python ingest_hierarchical.py
```

Script này sẽ:
- Đọc dữ liệu từ `all_articles.json`
- Tạo ChromaDB vector database trong `chroma_db_v2/`
- Tạo BM25 index trong `bm25_index_v2.pkl`
- Tự động download embedding model từ HuggingFace

Thời gian: ~2-5 phút tùy thuộc vào cấu hình máy.

### 6. Chạy ứng dụng

```bash
streamlit run streamlit_ui.py
```

Mở trình duyệt và truy cập: `http://localhost:8501`

## 🧠 Models sử dụng

| Model | Nguồn | Mục đích |
|-------|-------|----------|
| `dangvantuan/vietnamese-document-embedding` | HuggingFace | Embedding tiếng Việt |
| `namdp-ptit/ViRanker` | HuggingFace | Reranking documents |
| `NguyenTrinh/mdeberta-v3-medical-nli-vietnamese` | HuggingFace | NLI Hallucination Detection |
| `gpt-4o-mini` | OpenAI | Sinh câu trả lời |

**Lưu ý**: Các models từ HuggingFace sẽ được tự động download khi chạy lần đầu.

## 💻 Yêu cầu hệ thống

- **Python**: 3.10+
- **GPU**: NVIDIA GPU với CUDA (khuyến nghị ≥8GB VRAM)
- **RAM**: ≥16GB
- **Disk**: ≥10GB trống (cho models và database)

## 📊 Workflow

```
User Query
    ↓
┌─────────────────────────────────────┐
│ 1. Query Understanding              │
│    - Router: Phân loại câu hỏi      │
│    - Query Decomposition            │
│    - Query Expansion (RAG-Fusion)   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 2. Retrieval                        │
│    - Semantic Search (ChromaDB)     │
│    - Keyword Search (BM25)          │
│    - Hybrid Fusion                  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 3. Reranking                        │
│    - ViRanker Cross-Encoder         │
│    - Ensemble Scoring               │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 4. Generation                       │
│    - GPT-4o-mini                    │
│    - Contextual Answer              │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 5. Hallucination Detection          │
│    - NLI Per-Chunk Checking         │
│    - Claim Verification             │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 6. Critic & Refinement              │
│    - Quality Assessment             │
│    - Answer Improvement             │
└─────────────────────────────────────┘
    ↓
Final Answer
```

## 📝 Ví dụ sử dụng

### Python API

```python
from rag_system_v2 import ask_v2

# Hỏi một câu hỏi
answer = ask_v2("Acid folic có vai trò gì trong thai kỳ?", verbose=True)
print(answer)
```

### Streamlit UI

1. Chạy `streamlit run streamlit_ui.py`
2. Nhập câu hỏi vào ô chat
3. Nhận câu trả lời kèm nguồn tham khảo

## 🔧 Cấu hình nâng cao

Chỉnh sửa các tham số trong `rag_system_v2.py`:

```python
# Số lượng documents gửi cho LLM
TOP_K_TO_LLM = 20

# Bật/tắt các tính năng
ENABLE_RETRIEVAL_GRADER = True      # CRAG grading
ENABLE_HALLUCINATION_GRADER = True  # NLI checking
ENABLE_RAG_FUSION = True            # Query expansion
ENABLE_QUERY_DECOMPOSITION = True   # Multi-aspect queries
```

## 📚 Dữ liệu

Dữ liệu được crawl từ [yhoccongdong.com](https://yhoccongdong.com) - chuyên trang y khoa sản phụ khoa tiếng Việt.

- **Tổng số bài viết**: 100+ bài
- **Chủ đề**: Thai kỳ, sinh sản, phụ khoa, sức khỏe phụ nữ
- **Format**: Hierarchical chunks với metadata

## 🏆 Model NLI

Model NLI được fine-tuned từ `microsoft/mdeberta-v3-base` trên dữ liệu y khoa tiếng Việt:

- **Base model**: mDeBERTa-v3-base
- **Fine-tuned on**: Vietnamese medical NLI pairs
- **Task**: Entailment/Neutral/Contradiction classification
- **Purpose**: Phát hiện hallucination trong câu trả lời

Model được host tại: [HuggingFace - NguyenTrinh/mdeberta-v3-medical-nli-vietnamese](https://huggingface.co/NguyenTrinh/mdeberta-v3-medical-nli-vietnamese)

## 📄 License

MIT License

## 👤 Tác giả

**Nguyen Trinh**
- GitHub: [@NguyenTrinh3008](https://github.com/NguyenTrinh3008)
- HuggingFace: [NguyenTrinh](https://huggingface.co/NguyenTrinh)

## 🙏 Acknowledgments

- [yhoccongdong.com](https://yhoccongdong.com) - Nguồn dữ liệu y khoa
- [LangChain](https://langchain.com) - Framework RAG
- [ChromaDB](https://www.trychroma.com) - Vector database
- [Sentence Transformers](https://www.sbert.net) - Embedding và CrossEncoder
