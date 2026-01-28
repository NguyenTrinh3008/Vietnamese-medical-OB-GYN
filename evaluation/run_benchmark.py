#!/usr/bin/env python3
"""
V-OBGYN-50 RAG System Benchmark Runner

Evaluates the full RAG pipeline on the Vietnamese OB/GYN benchmark dataset.
Measures: Accuracy, Latency, Retrieval Precision, and Hallucination Detection

Usage:
    python run_benchmark.py --limit 10  # Run first 10 questions
    python run_benchmark.py --level L1  # Run only Level 1
    python run_benchmark.py             # Run all
"""

import os
import sys
import json
import time
import asyncio
import argparse
from datetime import datetime
from typing import Dict, List, Any, Tuple
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def load_benchmark(filepath: str) -> Dict:
    """Load benchmark questions from JSON"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# Initialize OpenAI client for evaluation
from openai import OpenAI
eval_client = OpenAI()

async def evaluate_with_llm(
    question: str,
    expected_answer: str,
    generated_answer: str,
    level: str,
    analysis: str = None,
    adversarial_type: str = None
) -> Dict[str, Any]:
    """
    Evaluate generated answer using GPT-4o-mini based on:
    - Expected answer (đáp án chuẩn)
    - Analysis criteria (tiêu chí phân tích từ benchmark.md)
    
    Returns:
        Dict with verdict, score, reasoning
    """
    # Build evaluation prompt with analysis criteria
    system_prompt = """Bạn là chuyên gia đánh giá hệ thống RAG y khoa Sản Phụ khoa.

NHIỆM VỤ: So sánh câu trả lời của AI với ĐÁP ÁN CHUẨN và TIÊU CHÍ PHÂN TÍCH.

QUY TẮC ĐÁNH GIÁ:
1. Câu trả lời phải ĐÚNG với đáp án chuẩn về mặt NỘI DUNG và SỐ LIỆU
2. Phải tuân thủ TIÊU CHÍ PHÂN TÍCH (nếu có)
3. KHÔNG ĐƯỢC bịa đặt thông tin ngoài nguồn tài liệu
4. Với câu hỏi L3 (adversarial): AI phải biết TỪ CHỐI nếu thông tin không có

VERDICT:
- CORRECT: Đúng hoàn toàn với đáp án chuẩn và tiêu chí phân tích
- PARTIAL: Đúng một phần nhưng thiếu thông tin quan trọng hoặc có lỗi nhỏ
- INCORRECT: Sai số liệu, sai nội dung, hoặc vi phạm tiêu chí phân tích

Return JSON: {"verdict": "CORRECT|PARTIAL|INCORRECT", "score": 0.0-1.0, "reasoning": "..."}"""

    # Build user prompt with analysis criteria
    user_parts = [
        f"CÂU HỎI ({level}): {question}",
        f"\nĐÁP ÁN CHUẨN: {expected_answer}"
    ]
    
    if analysis:
        user_parts.append(f"\nTIÊU CHÍ PHÂN TÍCH: {analysis}")
    
    if adversarial_type:
        user_parts.append(f"\nLOẠI CÂU HỎI: {adversarial_type} (adversarial - AI phải cẩn thận)")
    
    user_parts.append(f"\nCÂU TRẢ LỜI CỦA AI:\n{generated_answer[:1500]}")
    user_parts.append("\nĐánh giá câu trả lời. Return JSON.")
    
    user_prompt = "".join(user_parts)

    try:
        response = eval_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        verdict = result.get("verdict", "INCORRECT")
        score = result.get("score", 0.0)
        reasoning = result.get("reasoning", "")
        
        return {
            "verdict": verdict,
            "score": score,
            "reasoning": reasoning,
            "correct": verdict == "CORRECT",
            "partial": verdict == "PARTIAL"
        }
        
    except Exception as e:
        return {
            "verdict": "ERROR",
            "score": 0.0,
            "reasoning": str(e),
            "correct": False,
            "partial": False
        }



async def run_single_question(
    orchestrator,
    question: Dict,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run a single benchmark question through the RAG system
    """
    q_id = question["id"]
    q_text = question["question"]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"[{q_id}] {q_text[:80]}...")
    
    start_time = time.time()
    
    try:
        # Run through RAG orchestrator
        result_state = await orchestrator.run_with_self_rag(
            query=q_text,
            max_iterations=2,
            verbose=False
        )
        
        latency = time.time() - start_time
        
        # Extract answer from AgentState
        generated_answer = result_state.answer or ""
        metadata = result_state.metadata or {}
        
        # Evaluate using GPT-4o-mini with analysis criteria
        eval_result = await evaluate_with_llm(
            question=q_text,
            expected_answer=question["expected_answer"],
            generated_answer=generated_answer,
            level=question["level"],
            analysis=question.get("analysis"),
            adversarial_type=question.get("adversarial_type")
        )
        
        return {
            "id": q_id,
            "level": question["level"],
            "category": question.get("category", "unknown"),
            "question": q_text,
            "expected_answer": question["expected_answer"],
            "generated_answer": generated_answer[:800],  # Truncate for report
            "latency": latency,
            "router_action": metadata.get("router_action", "unknown"),
            "chunks_retrieved": len(result_state.reranked_chunks) if result_state.reranked_chunks else 0,
            "hallucination_check": metadata.get("hallucination_check", "skipped"),
            "grounded": metadata.get("grounded", None),
            **eval_result
        }
        
    except Exception as e:
        import traceback
        if verbose:
            traceback.print_exc()
        return {
            "id": q_id,
            "level": question["level"],
            "error": str(e),
            "correct": False,
            "latency": time.time() - start_time
        }


def generate_report(results: List[Dict], output_path: str) -> Dict:
    """
    Generate comprehensive benchmark report with LLM evaluation results
    """
    # Overall stats
    total = len(results)
    correct = sum(1 for r in results if r.get("correct", False))
    partial = sum(1 for r in results if r.get("partial", False))
    incorrect = total - correct - partial
    errors = sum(1 for r in results if "error" in r)
    
    # Average LLM score
    scores = [r.get("score", 0) for r in results if "score" in r]
    avg_score = sum(scores) / len(scores) if scores else 0
    
    # Per-level stats
    level_stats = defaultdict(lambda: {"total": 0, "correct": 0, "partial": 0, "latencies": [], "scores": []})
    for r in results:
        level = r.get("level", "unknown")
        level_stats[level]["total"] += 1
        if r.get("correct"):
            level_stats[level]["correct"] += 1
        if r.get("partial"):
            level_stats[level]["partial"] += 1
        if "latency" in r:
            level_stats[level]["latencies"].append(r["latency"])
        if "score" in r:
            level_stats[level]["scores"].append(r["score"])
    
    # Calculate level metrics
    for level, stats in level_stats.items():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        stats["avg_latency"] = sum(stats["latencies"]) / len(stats["latencies"]) if stats["latencies"] else 0
        stats["avg_score"] = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0
    
    # Hallucination stats
    hall_results = [r for r in results if r.get("hallucination_check") != "skipped"]
    grounded_count = sum(1 for r in hall_results if r.get("grounded"))
    
    # Build report
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_questions": total,
            "benchmark_file": "v_obgyn_benchmark.json",
            "evaluator": "gpt-4o-mini"
        },
        "overall": {
            "accuracy": correct / total if total > 0 else 0,
            "correct": correct,
            "partial": partial,
            "incorrect": incorrect,
            "total": total,
            "errors": errors,
            "avg_score": round(avg_score, 3),
            "avg_latency": sum(r.get("latency", 0) for r in results) / total if total > 0 else 0
        },
        "by_level": {
            level: {
                "accuracy": stats["accuracy"],
                "correct": stats["correct"],
                "partial": stats["partial"],
                "total": stats["total"],
                "avg_score": round(stats["avg_score"], 3),
                "avg_latency": round(stats["avg_latency"], 2)
            }
            for level, stats in level_stats.items()
        },
        "hallucination": {
            "checked": len(hall_results),
            "grounded": grounded_count,
            "grounding_rate": grounded_count / len(hall_results) if hall_results else 0
        },
        "detailed_results": results
    }
    
    # Save report
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    return report



def print_summary(report: Dict):
    """Print formatted summary to console"""
    print("\n" + "="*70)
    print("📊 V-OBGYN-50 BENCHMARK REPORT")
    print("="*70)
    
    overall = report["overall"]
    print(f"\n📈 OVERALL METRICS:")
    print(f"   Accuracy: {overall['accuracy']*100:.1f}% ({overall['correct']}/{overall['total']})")
    print(f"   Avg Latency: {overall['avg_latency']:.2f}s")
    print(f"   Errors: {overall['errors']}")
    
    print(f"\n📊 BY LEVEL:")
    for level, stats in sorted(report["by_level"].items()):
        print(f"   {level}: {stats['accuracy']*100:.1f}% ({stats['correct']}/{stats['total']}) - {stats['avg_latency']:.2f}s avg")
    
    hall = report["hallucination"]
    print(f"\n🔍 HALLUCINATION CHECK:")
    print(f"   Grounding Rate: {hall['grounding_rate']*100:.1f}% ({hall['grounded']}/{hall['checked']})")
    
    print("\n" + "="*70)


async def main():
    parser = argparse.ArgumentParser(description="Run V-OBGYN-50 Benchmark")
    parser.add_argument("--limit", type=int, help="Limit number of questions")
    parser.add_argument("--level", choices=["L1", "L2", "L3"], help="Run only specific level")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--output", default="evaluation/benchmark_report.json", help="Output file")
    args = parser.parse_args()
    
    # Load benchmark
    benchmark_path = os.path.join(os.path.dirname(__file__), "v_obgyn_benchmark.json")
    benchmark = load_benchmark(benchmark_path)
    questions = benchmark["questions"]
    
    # Filter by level if specified
    if args.level:
        questions = [q for q in questions if q["level"] == args.level]
    
    # Limit if specified
    if args.limit:
        questions = questions[:args.limit]
    
    print(f"📋 Loading V-OBGYN-50 Benchmark...")
    print(f"   Total questions: {len(questions)}")
    
    # Initialize RAG system
    print(f"\n🔧 Initializing RAG System V2...")
    from rag_system_v2 import build_async_orchestrator_v2
    orchestrator = build_async_orchestrator_v2()
    print(f"   ✅ RAG System ready")
    
    # Run benchmark
    print(f"\n🚀 Running benchmark...")
    results = []
    
    for i, q in enumerate(questions):
        print(f"   [{i+1}/{len(questions)}] {q['id']}: {q['question'][:50]}...")
        result = await run_single_question(orchestrator, q, verbose=args.verbose)
        results.append(result)
        
        # Progress indicator
        if result.get("correct"):
            print(f"      ✅ Correct ({result.get('latency', 0):.1f}s)")
        else:
            print(f"      ❌ Incorrect ({result.get('latency', 0):.1f}s)")
    
    # Generate report
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.output)
    report = generate_report(results, output_path)
    
    # Print summary
    print_summary(report)
    
    print(f"\n📄 Full report saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
