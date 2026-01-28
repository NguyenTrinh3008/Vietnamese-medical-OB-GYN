#!/usr/bin/env python3
"""
Medical Synonyms Generator using OpenAI API

Tự động sinh từ đồng nghĩa cho các thuật ngữ y khoa Việt Nam.
Giúp mở rộng từ điển MEDICAL_SYNONYMS cho RerankerV2.

Usage:
    python generate_medical_synonyms.py --terms "rong kinh,thống kinh"
    python generate_medical_synonyms.py --category obstetrics
    python generate_medical_synonyms.py --expand-all
"""

import json
import os
import argparse
from typing import List, Dict, Set
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Paths
SYNONYMS_FILE = Path(__file__).parent / "medical_synonyms.json"

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def load_synonyms() -> Dict:
    """Load existing synonyms from JSON file"""
    if SYNONYMS_FILE.exists():
        with open(SYNONYMS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"version": "1.0", "categories": {}}


def save_synonyms(data: Dict):
    """Save synonyms to JSON file"""
    data["last_updated"] = __import__('datetime').datetime.now().isoformat()
    with open(SYNONYMS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved to {SYNONYMS_FILE}")


def generate_synonyms_for_term(term: str, category: str = "general") -> List[str]:
    """
    Generate medical synonyms for a term using OpenAI API
    
    Args:
        term: Vietnamese medical term (e.g., "rong kinh")
        category: Medical category for context
        
    Returns:
        List of synonyms
    """
    prompt = f"""Bạn là chuyên gia y khoa Việt Nam. Cho thuật ngữ y khoa sau:

Thuật ngữ: {term}
Chuyên khoa: {category}

Hãy liệt kê TẤT CẢ các từ đồng nghĩa, cách gọi khác, thuật ngữ tiếng Anh tương đương.
Bao gồm:
1. Tên tiếng Việt thông dụng
2. Tên tiếng Việt chuyên ngành
3. Tên tiếng Anh (viết thường)
4. Cách bệnh nhân thường mô tả (ngôn ngữ thông thường)

Trả về dạng JSON array, chỉ trả về array, không giải thích thêm.
Ví dụ: ["menorrhagia", "ra máu kinh nhiều", "kinh kéo dài"]"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Bạn là chuyên gia y khoa, trả lời bằng JSON array."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON
        if content.startswith('['):
            synonyms = json.loads(content)
        else:
            # Try to extract array from response
            import re
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                synonyms = json.loads(match.group())
            else:
                print(f"⚠️ Could not parse response for '{term}': {content[:100]}")
                return []
        
        # Clean and dedupe
        synonyms = [s.lower().strip() for s in synonyms if s and s.lower() != term.lower()]
        synonyms = list(set(synonyms))
        
        return synonyms
        
    except Exception as e:
        print(f"❌ Error generating synonyms for '{term}': {e}")
        return []


def generate_new_terms(category: str, count: int = 10) -> Dict[str, List[str]]:
    """
    Generate entirely new medical terms for a category
    
    Args:
        category: Medical specialty (e.g., "obstetrics", "gynecology")
        count: Number of terms to generate
        
    Returns:
        Dict of term -> synonyms
    """
    prompt = f"""Bạn là chuyên gia y khoa Việt Nam về {category}.

Hãy liệt kê {count} thuật ngữ y khoa quan trọng nhất trong chuyên ngành này mà bệnh nhân thường hỏi.
Với mỗi thuật ngữ, cung cấp các từ đồng nghĩa.

Trả về dạng JSON object:
{{
  "thuật ngữ chính 1": ["đồng nghĩa 1", "đồng nghĩa 2"],
  "thuật ngữ chính 2": ["đồng nghĩa 1", "đồng nghĩa 2"]
}}

Chỉ trả về JSON, không giải thích thêm."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Bạn là chuyên gia y khoa, trả lời bằng JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=2000
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON
        if content.startswith('{'):
            terms = json.loads(content)
        else:
            import re
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                terms = json.loads(match.group())
            else:
                print(f"⚠️ Could not parse response: {content[:200]}")
                return {}
        
        # Clean
        cleaned = {}
        for term, syns in terms.items():
            term_clean = term.lower().strip()
            syns_clean = [s.lower().strip() for s in syns if s]
            cleaned[term_clean] = list(set(syns_clean))
        
        return cleaned
        
    except Exception as e:
        print(f"❌ Error generating terms for '{category}': {e}")
        return {}


def expand_existing_terms(data: Dict) -> Dict:
    """
    Expand all existing terms with OpenAI-generated synonyms
    """
    print("🔄 Expanding existing terms with OpenAI...")
    
    for category_name, terms in data.get("categories", {}).items():
        print(f"\n📂 Category: {category_name}")
        
        for term, existing_synonyms in terms.items():
            print(f"   🔍 {term}...", end=" ")
            
            new_synonyms = generate_synonyms_for_term(term, category_name)
            
            # Merge, keeping unique values
            all_synonyms = set(existing_synonyms)
            all_synonyms.update(new_synonyms)
            
            # Remove term itself from synonyms
            all_synonyms.discard(term.lower())
            
            added = len(all_synonyms) - len(existing_synonyms)
            if added > 0:
                print(f"+{added} synonyms")
                terms[term] = sorted(list(all_synonyms))
            else:
                print("no new")
    
    return data


def add_new_category(data: Dict, category: str, count: int = 15) -> Dict:
    """
    Add a new category with OpenAI-generated terms
    """
    print(f"\n🆕 Generating new category: {category}")
    
    new_terms = generate_new_terms(category, count)
    
    if new_terms:
        if category not in data["categories"]:
            data["categories"][category] = {}
        
        data["categories"][category].update(new_terms)
        print(f"   ✅ Added {len(new_terms)} terms to '{category}'")
    
    return data


def add_single_term(data: Dict, term: str, category: str = "general") -> Dict:
    """
    Add a single term with generated synonyms
    """
    print(f"🔍 Generating synonyms for: {term}")
    
    synonyms = generate_synonyms_for_term(term, category)
    
    if synonyms:
        if category not in data["categories"]:
            data["categories"][category] = {}
        
        data["categories"][category][term.lower()] = synonyms
        print(f"   ✅ Added {len(synonyms)} synonyms")
    
    return data


def extract_synonyms_from_corpus(articles_path: str = None, batch_size: int = 20, 
                                  limit: int = None, dry_run: bool = False) -> Dict:
    """
    Extract medical synonyms from all_articles.json corpus using LLM
    
    Strategy:
    1. Load all articles from corpus
    2. Batch articles (batch_size per LLM call)
    3. Use LLM to extract (Term, Synonym) pairs from article content
    4. Merge extracted synonyms with existing dictionary
    
    Args:
        articles_path: Path to all_articles.json (default: same directory)
        batch_size: Number of articles per LLM call
        limit: Max articles to process (None = all)
        dry_run: If True, print extracted pairs but don't save
        
    Returns:
        Updated synonyms dictionary
    """
    import math
    
    # Load articles
    if articles_path is None:
        articles_path = Path(__file__).parent / "all_articles.json"
    
    print(f"\n📚 Loading corpus from {articles_path}...")
    
    try:
        with open(articles_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load articles: {e}")
        return {}
    
    if limit:
        articles = articles[:limit]
    
    total_articles = len(articles)
    num_batches = math.ceil(total_articles / batch_size)
    
    print(f"   📊 Total articles: {total_articles}")
    print(f"   📦 Batch size: {batch_size}")
    print(f"   🔄 LLM calls needed: {num_batches}")
    
    if dry_run:
        print("   ⚠️ DRY RUN mode - will not save results")
    
    # Load existing synonyms
    data = load_synonyms()
    if "corpus_extracted" not in data["categories"]:
        data["categories"]["corpus_extracted"] = {}
    
    total_extracted = 0
    
    # Process in batches
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_articles)
        batch_articles = articles[start_idx:end_idx]
        
        print(f"\n🔄 Processing batch {batch_idx + 1}/{num_batches} (articles {start_idx + 1}-{end_idx})...")
        
        # Extract article titles and key content
        batch_content = []
        for art in batch_articles:
            title = art.get("title", "")
            # Get content from first few chunks
            chunks = art.get("chunks", [])[:3]
            content_preview = " ".join([c.get("content", "")[:200] for c in chunks])
            batch_content.append(f"=== {title} ===\n{content_preview[:500]}")
        
        combined_content = "\n\n".join(batch_content)
        
        # LLM extraction
        extracted = _extract_synonyms_batch(combined_content)
        
        if extracted:
            for term, synonyms in extracted.items():
                term_lower = term.lower().strip()
                if term_lower and len(term_lower) > 2:
                    # Merge with existing
                    if term_lower in data["categories"]["corpus_extracted"]:
                        existing = set(data["categories"]["corpus_extracted"][term_lower])
                        existing.update(synonyms)
                        data["categories"]["corpus_extracted"][term_lower] = list(existing)
                    else:
                        data["categories"]["corpus_extracted"][term_lower] = synonyms
                    
                    total_extracted += 1
            
            print(f"   ✅ Extracted {len(extracted)} term-synonym pairs")
    
    print(f"\n📊 Extraction complete!")
    print(f"   Total terms extracted: {total_extracted}")
    print(f"   Category: corpus_extracted")
    
    if not dry_run:
        return data
    else:
        # Print sample for dry run
        print("\n📋 Sample extracted pairs:")
        corpus_terms = data["categories"].get("corpus_extracted", {})
        for i, (term, syns) in enumerate(list(corpus_terms.items())[:10]):
            print(f"   {term}: {syns[:3]}...")
        return data


def _extract_synonyms_batch(content: str) -> Dict[str, List[str]]:
    """
    Extract synonym pairs from a batch of article content using LLM
    """
    prompt = f"""Phân tích các bài báo y khoa sau và trích xuất các thuật ngữ y khoa quan trọng cùng với từ đồng nghĩa.

BÀI VIẾT:
{content[:3000]}

NHIỆM VỤ:
1. Xác định các thuật ngữ y khoa chính trong bài viết
2. Với mỗi thuật ngữ, liệt kê các từ đồng nghĩa, tên tiếng Anh, cách gọi thông thường

Trả về JSON object:
{{
  "thuật ngữ 1": ["đồng nghĩa 1", "đồng nghĩa 2", "english term"],
  "thuật ngữ 2": ["đồng nghĩa 1", "đồng nghĩa 2"]
}}

CHỈ trả về JSON, không giải thích. Tối đa 15 thuật ngữ quan trọng nhất."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Bạn là chuyên gia y khoa, trích xuất thuật ngữ từ văn bản y khoa Việt Nam. Trả về JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1500
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON
        if content.startswith('{'):
            result = json.loads(content)
        else:
            # Try to extract JSON object
            import re
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                result = json.loads(match.group())
            else:
                return {}
        
        # Clean up
        cleaned = {}
        for term, synonyms in result.items():
            if isinstance(synonyms, list):
                cleaned_syns = [s.lower().strip() for s in synonyms if s and len(s) > 1]
                if cleaned_syns:
                    cleaned[term.lower().strip()] = cleaned_syns
        
        return cleaned
        
    except Exception as e:
        print(f"   ⚠️ Batch extraction error: {e}")
        return {}


def show_stats(data: Dict):
    """Show dictionary statistics"""
    total_terms = 0
    total_synonyms = 0
    
    print("\n📊 Dictionary Statistics")
    print("=" * 50)
    
    for category, terms in data.get("categories", {}).items():
        term_count = len(terms)
        syn_count = sum(len(syns) for syns in terms.values())
        total_terms += term_count
        total_synonyms += syn_count
        print(f"   {category}: {term_count} terms, {syn_count} synonyms")
    
    print("-" * 50)
    print(f"   TOTAL: {total_terms} terms, {total_synonyms} synonyms")
    print(f"   File: {SYNONYMS_FILE}")


def main():
    parser = argparse.ArgumentParser(description="Generate medical synonyms using OpenAI")
    parser.add_argument("--terms", type=str, help="Comma-separated terms to add")
    parser.add_argument("--category", type=str, help="Category for new terms")
    parser.add_argument("--add-category", type=str, help="Generate new category with terms")
    parser.add_argument("--expand-all", action="store_true", help="Expand all existing terms")
    parser.add_argument("--stats", action="store_true", help="Show dictionary statistics")
    parser.add_argument("--count", type=int, default=15, help="Number of terms to generate for new category")
    
    # NEW: Corpus extraction options
    parser.add_argument("--extract-from-corpus", action="store_true", 
                        help="Extract synonyms from all_articles.json using LLM")
    parser.add_argument("--batch-size", type=int, default=20, 
                        help="Articles per LLM call (default: 20)")
    parser.add_argument("--limit", type=int, default=None, 
                        help="Max articles to process (default: all)")
    parser.add_argument("--dry-run", action="store_true", 
                        help="Preview extraction without saving")
    
    args = parser.parse_args()
    
    # Load existing data
    data = load_synonyms()
    
    if args.stats:
        show_stats(data)
        return
    
    # NEW: Corpus extraction
    if args.extract_from_corpus:
        data = extract_synonyms_from_corpus(
            batch_size=args.batch_size,
            limit=args.limit,
            dry_run=args.dry_run
        )
        if data and not args.dry_run:
            save_synonyms(data)
            show_stats(data)
        return
    
    if args.terms:
        terms = [t.strip() for t in args.terms.split(",")]
        category = args.category or "general"
        
        for term in terms:
            data = add_single_term(data, term, category)
        
        save_synonyms(data)
        show_stats(data)
        return
    
    if args.add_category:
        data = add_new_category(data, args.add_category, args.count)
        save_synonyms(data)
        show_stats(data)
        return
    
    if args.expand_all:
        data = expand_existing_terms(data)
        save_synonyms(data)
        show_stats(data)
        return
    
    # Default: show help
    parser.print_help()
    print("\n📌 Examples:")
    print("   python generate_medical_synonyms.py --stats")
    print("   python generate_medical_synonyms.py --terms 'viêm âm đạo,viêm cổ tử cung' --category gynecology")
    print("   python generate_medical_synonyms.py --add-category pediatrics --count 20")
    print("   python generate_medical_synonyms.py --expand-all")
    print("   python generate_medical_synonyms.py --extract-from-corpus --batch-size 20")
    print("   python generate_medical_synonyms.py --extract-from-corpus --limit 10 --dry-run")


if __name__ == "__main__":
    main()
