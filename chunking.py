import json, os
import re
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chunk_article_with_openai(article: dict) -> list:
    """Sử dụng OpenAI để tách bài viết thành chunks theo các mục trong 'Nội dung chính'"""
    
    system_prompt = """Bạn là một AI chuyên phân tích và tách nội dung bài viết y tế. 

Nhiệm vụ của bạn:
1. Đọc bài viết và tìm phần "Nội dung chính" 
2. Tách bài viết thành các chunks dựa trên các mục được liệt kê trong "Nội dung chính"
3. Tạo cấu trúc JSON lồng nhau: các mục con (3.1, 3.2...) sẽ nằm trong mảng "subsections" của mục chính (3, 4...)
4. Mỗi chunk phải chứa đầy đủ nội dung liên quan đến mục đó

Trả về JSON với format lồng nhau:
{
  "chunks": [
    {
      "section": "1",
      "title": "Tiêu đề mục 1", 
      "content": "Nội dung chi tiết của mục 1",
      "subsections": []
    },
    {
      "section": "2", 
      "title": "Tiêu đề mục 2",
      "content": "Nội dung chi tiết của mục 2",
      "subsections": []
    },
    {
      "section": "3", 
      "title": "Tiêu đề mục 3",
      "content": "Nội dung tổng quan của mục 3",
      "subsections": [
        {
          "section": "3.1",
          "title": "Tiêu đề mục con 3.1",
          "content": "Nội dung chi tiết của mục con 3.1"
        },
        {
          "section": "3.2",
          "title": "Tiêu đề mục con 3.2", 
          "content": "Nội dung chi tiết của mục con 3.2"
        }
      ]
    }
  ]
}

Lưu ý:
- Chỉ trả về JSON, không có text khác
- Các mục con được lồng trong mảng "subsections" của mục chính
- Nếu mục không có con thì "subsections" là mảng rỗng []
- Đảm bảo nội dung của mỗi chunk đầy đủ và chính xác, không tự ý thêm, bớt hoặc sửa đổi nội dung
- Nếu không tìm thấy "Nội dung chính", trả về chunks rỗng"""

    user_prompt = f"""Phân tích bài viết sau và tách thành chunks theo các mục trong "Nội dung chính":

Tiêu đề: {article['title']}

Nội dung:
{article['content']}"""

    try:
        response = client.chat.completions.create(
        model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=12000
        )
        
        result = json.loads(response.choices[0].message.content)
        chunks = []

        for chunk_data in result.get("chunks", []):
            # Tạo mục chính với subsections lồng nhau
            main_chunk = {
                "section": chunk_data.get("section", ""),
                "title": chunk_data.get("title", ""),
                "content": chunk_data.get("content", ""),
                "subsections": []
            }
            
            # Thêm các mục con vào subsections
            for sub_chunk_data in chunk_data.get("subsections", []):
                sub_chunk = {
                    "section": sub_chunk_data.get("section", ""),
                    "title": sub_chunk_data.get("title", ""),
                    "content": sub_chunk_data.get("content", "")
                }
                main_chunk["subsections"].append(sub_chunk)
            
            chunks.append(main_chunk)

        return chunks
        
    except Exception as e:
        print(f"❌ Lỗi khi xử lý bài '{article['title']}': {e}")
        return []

def load_articles_from_json(file_path: str) -> list:
    """Load articles từ file JSON"""
    with open(file_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    return articles

if __name__ == "__main__":
    # Load articles từ file JSON
    articles_file = "/home/ltnga/nguyentrinhmedical/yhoccongdong/yhoccongdong/articles-complete.json"
    
    print("📚 Loading articles from JSON file...")
    articles = load_articles_from_json(articles_file)
    print(f"✅ Loaded {len(articles)} articles")
    
    # Test với 2 bài đầu tiên
    print("\n🧪 Testing with first 2 articles...")
    test_articles = articles[:2]
    all_test_articles = []
    
    for idx, test_article in enumerate(test_articles, 1):
        print(f"\n📖 Testing article {idx}: {test_article['title']}")
        
        chunks = chunk_article_with_openai(test_article)
        print(f"✂️ Generated {len(chunks)} chunks")
        
        # Tạo cấu trúc mới cho bài viết
        article_data = {
            "title": test_article['title'],
            "link": test_article['link'],
            "chunks": chunks
        }
        all_test_articles.append(article_data)
        
        # Hiển thị các chunks được tạo
        for i, chunk in enumerate(chunks, 1):
            print(f"\n📄 Chunk {i}:")
            print(f"   Section: {chunk['section']} - {chunk['title']}")
            print(f"   Content length: {len(chunk['content'])} chars")
            print(f"   Content preview: {chunk['content'][:100]}...")
            if chunk['subsections']:
                print(f"   Subsections: {len(chunk['subsections'])}")
    
    # Lưu chunks vào file JSON với format đẹp
    with open("test_chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_test_articles, f, ensure_ascii=False, indent=2)
    
    total_chunks = sum(len(article['chunks']) for article in all_test_articles)
    print(f"\n✅ Saved {len(all_test_articles)} articles with {total_chunks} total chunks to test_chunks.json")
    
    # Hỏi user có muốn xử lý tất cả articles không
    print(f"\n❓ Do you want to process all {len(articles)} articles? (y/n): ", end="")
    choice = input().strip().lower()
    
    if choice == 'y':
        print(f"\n🔄 Processing all {len(articles)} articles...")
        all_articles = []
        
        for i, article in enumerate(articles, 1):
            print(f"Processing {i}/{len(articles)}: {article['title']}")
            chunks = chunk_article_with_openai(article)
            
            # Tạo cấu trúc cho mỗi bài viết
            article_data = {
                "title": article['title'],
                "link": article['link'],
                "chunks": chunks
            }
            all_articles.append(article_data)
            print(f"   → Generated {len(chunks)} chunks")
        
        # Lưu tất cả articles với format đẹp
        with open("all_articles.json", "w", encoding="utf-8") as f:
            json.dump(all_articles, f, ensure_ascii=False, indent=2)
        
        total_chunks = sum(len(article['chunks']) for article in all_articles)
        print(f"\n✅ Processed all articles!")
        print(f"📊 Total articles: {len(all_articles)}")
        print(f"📊 Total chunks generated: {total_chunks}")
        print(f"💾 Saved to all_articles.json")
    else:
        print("⏹️ Stopped after test. Only processed 1 article.")
