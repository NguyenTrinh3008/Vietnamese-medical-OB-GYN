#!/usr/bin/env python3
"""
Async utilities for the RAG system
Provides async wrappers for LLM calls, ChromaDB, and BM25 search
"""

import asyncio
import json
from typing import Dict, Any, List, Tuple


async def async_llm_json(system: str, user: str, openai_client, model: str, 
                         temperature: float = 0.0, max_tokens: int = 3000) -> Dict[str, Any]:
    """
    Async call to OpenAI API with JSON response format
    
    Improved error handling:
    - Increased default max_tokens to 2000 to avoid truncation
    - Retry on JSON parse errors
    - Better fallback handling
    """
    max_retries = 2
    
    for attempt in range(max_retries):
        try:
            resp = await openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            content = resp.choices[0].message.content
            
            # Try to parse JSON
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                # If JSON parse fails but we have retries left, try again
                if attempt < max_retries - 1:
                    print(f"⚠️ LLM JSON parse failed (attempt {attempt+1}/{max_retries}): {e}")
                    await asyncio.sleep(0.5)  # Brief delay before retry
                    continue
                else:
                    # Last attempt failed, return error dict
                    print(f"❌ LLM JSON call failed after {max_retries} attempts: {e}")
                    print(f"   Response preview: {content[:200]}...")
                    return {"error": f"JSON parse error: {str(e)}"}
                    
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️ LLM call failed (attempt {attempt+1}/{max_retries}): {e}")
                await asyncio.sleep(0.5)
                continue
            else:
                print(f"❌ LLM API call failed after {max_retries} attempts: {e}")
                return {"error": str(e)}
    
    # Should not reach here, but just in case
    return {"error": "Unknown error in async_llm_json"}


async def async_llm_text(system: str, user: str, openai_client, model: str,
                         temperature: float = 0.0, max_tokens: int = 1000) -> str:
    """
    Async OpenAI API call with text response
    
    Args:
        system: System prompt
        user: User prompt
        openai_client: AsyncOpenAI client instance
        model: Model name
        temperature: Sampling temperature
        max_tokens: Max tokens in response
        
    Returns:
        Text response
    """
    try:
        response = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"⚠️ Async LLM call failed: {e}")
        return f"Error: {str(e)}"


async def async_chromadb_search(vectordb, query: str, k: int) -> List[Tuple]:
    """
    Async wrapper for ChromaDB similarity search
    
    ChromaDB doesn't have native async support, so we use run_in_executor
    to avoid blocking the event loop.
    
    Args:
        vectordb: ChromaDB vector store instance
        query: Query string
        k: Number of results
        
    Returns:
        List of (document, score) tuples
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,  # Use default ThreadPoolExecutor
        lambda: vectordb.similarity_search_with_score(query, k=k)
    )


async def async_bm25_search(bm25_index, query_tokens: List[str], k: int, 
                           doc_texts: List, verbose: bool = False) -> List[Tuple[int, float]]:
    """
    Async wrapper for BM25 search
    
    BM25 scoring is CPU-bound, so we run it in executor to avoid blocking.
    
    Args:
        bm25_index: BM25 index instance
        query_tokens: Preprocessed query tokens
        k: Number of results
        doc_texts: List of document texts
        verbose: Enable verbose logging
        
    Returns:
        List of (index, score) tuples for top-k results
    """
    if not query_tokens:
        return []
    
    import numpy as np
    loop = asyncio.get_event_loop()
    
    def _search():
        """Synchronous search function to run in executor"""
        scores = bm25_index.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
    
    return await loop.run_in_executor(None, _search)


async def async_batch_process(items: List[Any], process_fn, batch_size: int = 10, 
                              max_concurrent: int = 5) -> List[Any]:
    """
    Process items in batches with concurrency control
    
    Useful for processing multiple items with async functions while
    controlling concurrency to avoid rate limits.
    
    Args:
        items: List of items to process
        process_fn: Async function to process each item
        batch_size: Batch size for processing
        max_concurrent: Maximum concurrent tasks
        
    Returns:
        List of processed results
    """
    results = []
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(item):
        async with semaphore:
            return await process_fn(item)
    
    # Process in batches
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = await asyncio.gather(*[process_with_semaphore(item) for item in batch])
        results.extend(batch_results)
    
    return results
