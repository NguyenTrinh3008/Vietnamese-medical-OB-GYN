#!/usr/bin/env python3
"""
Async Agent Framework - Pure async/await version (Self-contained)
Provides async base classes for optimal performance
NO dependency on agent_base.py
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
import logging
import time
import asyncio
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class AgentStatus(Enum):
    """Status of agent execution"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class AgentState:
    """
    State container for entire RAG pipeline
    Replaces GraphState TypedDict from LangGraph
    """
    # Core data
    query: str = ""
    plan: Dict[str, Any] = field(default_factory=dict)
    coarse_docs: List[Dict[str, Any]] = field(default_factory=list)
    candidate_chunks: List[Dict[str, Any]] = field(default_factory=list)
    reranked_chunks: List[Dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    
    # Metadata for logging and debugging
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Execution tracking
    start_time: float = field(default_factory=time.time)
    agent_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize metadata if not provided"""
        if not self.metadata:
            self.metadata = {
                "verbose": False,
                "start_time": self.start_time,
                "errors": []
            }
    
    def copy(self) -> 'AgentState':
        """Create a copy of current state (immutable pattern)"""
        return AgentState(
            query=self.query,
            plan=self.plan.copy(),
            coarse_docs=self.coarse_docs.copy(),
            candidate_chunks=self.candidate_chunks.copy(),
            reranked_chunks=self.reranked_chunks.copy(),
            answer=self.answer,
            metadata=self.metadata.copy(),
            start_time=self.start_time,
            agent_history=self.agent_history.copy()
        )
    
    def add_agent_execution(self, agent_name: str, status: AgentStatus, 
                           duration: float, error: Optional[str] = None):
        """Track agent execution in history"""
        self.agent_history.append({
            "agent": agent_name,
            "status": status.value,
            "duration": duration,
            "timestamp": time.time(),
            "error": error
        })





class BaseAgent(ABC):
    """
    Sync base class for all synchronous agents
    (Reranker, Generator, Critic, Hallucination Grader)
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"Agent.{name}")
    
    @abstractmethod
    def execute(self, state: AgentState) -> AgentState:
        """
        Execute agent logic và return updated state
        
        Args:
            state: Current AgentState
            
        Returns:
            Updated AgentState
        """
        pass
    
    def validate_input(self, state: AgentState) -> bool:
        """
        Validate input state trước khi execute
        
        Args:
            state: Current AgentState
            
        Returns:
            True nếu valid, False nếu không
        """
        # Default: luôn valid, subclass có thể override
        return True
    
    def run(self, state: AgentState, verbose: bool = False) -> AgentState:
        """
        Wrapper cho execute với error handling và logging
        
        Args:
            state: Current AgentState
            verbose: Enable verbose logging
            
        Returns:
            Updated AgentState
        """
        start_time = time.time()
        
        try:
            # Update verbose in metadata
            state.metadata["verbose"] = verbose
            
            # Validate input
            if not self.validate_input(state):
                raise ValueError(f"Invalid input state for {self.name}")
            
            if verbose:
                self.logger.info(f"🚀 Starting {self.name}...")
            
            # Execute agent logic (sync)
            new_state = self.execute(state)
            
            duration = time.time() - start_time
            
            # Track execution
            new_state.add_agent_execution(
                agent_name=self.name,
                status=AgentStatus.SUCCESS,
                duration=duration
            )
            
            if verbose:
                self.logger.info(f"✅ {self.name} completed in {duration:.2f}s")
            
            return new_state
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"{self.name} failed: {str(e)}"
            
            self.logger.error(error_msg, exc_info=True)
            
            # Track failed execution
            state.add_agent_execution(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                duration=duration,
                error=error_msg
            )
            
            # Add error to metadata
            state.metadata["errors"].append({
                "agent": self.name,
                "error": error_msg,
                "timestamp": time.time()
            })
            
            # Re-raise for orchestrator to handle
            raise


class AsyncBaseAgent(ABC):
    """
    Async abstract base class for all agents
    
    All agents should inherit from this and implement async execute()
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"AsyncAgent.{name}")
    
    @abstractmethod
    async def execute(self, state: AgentState) -> AgentState:
        """
        Async execute agent logic and return updated state
        
        Args:
            state: Current AgentState
            
        Returns:
            Updated AgentState
        """
        pass
    
    def validate_input(self, state: AgentState) -> bool:
        """
        Validate input state before execute
        
        Args:
            state: Current AgentState
            
        Returns:
            True if valid, False otherwise
        """
        # Default: always valid, subclass can override
        return True
    
    async def run(self, state: AgentState, verbose: bool = False) -> AgentState:
        """
        Async wrapper for execute with error handling and logging
        
        Args:
            state: Current AgentState
            verbose: Enable verbose logging
            
        Returns:
            Updated AgentState
        """
        start_time = time.time()
        
        try:
            # Update verbose in metadata
            state.metadata["verbose"] = verbose
            
            # Validate input
            if not self.validate_input(state):
                raise ValueError(f"Invalid input state for {self.name}")
            
            if verbose:
                self.logger.info(f"🚀 Starting {self.name}...")
            
            # Execute agent logic (async!)
            new_state = await self.execute(state)
            
            duration = time.time() - start_time
            
            # Track execution
            new_state.add_agent_execution(
                agent_name=self.name,
                status=AgentStatus.SUCCESS,
                duration=duration
            )
            
            if verbose:
                self.logger.info(f"✅ {self.name} completed in {duration:.2f}s")
            
            return new_state
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"{self.name} failed: {str(e)}"
            
            self.logger.error(error_msg, exc_info=True)
            
            # Track failed execution
            state.add_agent_execution(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                duration=duration,
                error=error_msg
            )
            
            # Add error to metadata
            state.metadata["errors"].append({
                "agent": self.name,
                "error": error_msg,
                "timestamp": time.time()
            })
            
            # Re-raise for orchestrator to handle
            raise




class AgentOrchestrator:
    """
    Sync Orchestrator for agent pipeline
    Used for pure synchronous execution (legacy/testing)
    """
    
    def __init__(self, name: str = "RAGOrchestrator"):
        self.name = name
        self.agents: List[BaseAgent] = []
        self.logger = logging.getLogger(f"Orchestrator.{name}")
        self.progress_callback: Optional[Callable] = None
    
    def add_agent(self, agent: BaseAgent) -> 'AgentOrchestrator':
        """Add agent to pipeline"""
        self.agents.append(agent)
        self.logger.info(f"Added agent: {agent.name}")
        return self
    
    def set_progress_callback(self, callback: Callable):
        """Set callback function for UI progress updates"""
        self.progress_callback = callback
    
    def run(self, query: str, verbose: bool = False) -> AgentState:
        """Execute full pipeline"""
        state = AgentState(query=query)
        state.metadata["verbose"] = verbose
        
        if verbose:
            self.logger.info(f"🚀 Starting orchestration for query: {query[:50]}...")
            self.logger.info(f"📋 Pipeline: {' → '.join(a.name for a in self.agents)}")
        
        for i, agent in enumerate(self.agents, 1):
            try:
                if verbose:
                    self.logger.info(f"\n{'='*60}")
                    self.logger.info(f"Step {i}/{len(self.agents)}: {agent.name}")
                    self.logger.info(f"{'='*60}")
                
                if self.progress_callback:
                    self.progress_callback(agent.name, AgentStatus.RUNNING)
                
                state = agent.run(state, verbose=verbose)
                
                if self.progress_callback:
                    self.progress_callback(agent.name, AgentStatus.SUCCESS)
                    
            except Exception as e:
                if self.progress_callback:
                    self.progress_callback(agent.name, AgentStatus.FAILED)
                
                self.logger.error(f"❌ Pipeline failed at {agent.name}: {e}")
                state.answer = f"Xin lỗi, đã xảy ra lỗi trong quá trình xử lý: {str(e)}"
                return state
        
        total_time = time.time() - state.start_time
        
        if verbose:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"✅ Pipeline completed in {total_time:.2f}s")
            self.logger.info(f"📊 Execution history:")
            for entry in state.agent_history:
                status_emoji = "✅" if entry["status"] == "success" else "❌"
                self.logger.info(f"   {status_emoji} {entry['agent']}: {entry['duration']:.2f}s")
            self.logger.info(f"{'='*60}\n")
        
        return state
    
    def run_until(self, query: str, stop_at: str, verbose: bool = False) -> AgentState:
        """Execute pipeline until a specific agent (debugging/testing)"""
        state = AgentState(query=query)
        state.metadata["verbose"] = verbose
        
        if verbose:
            self.logger.info(f"🚀 Partial execution until: {stop_at}")
        
        for agent in self.agents:
            if verbose:
                self.logger.info(f"Executing: {agent.name}")
            
            state = agent.run(state, verbose=verbose)
            
            if agent.name == stop_at:
                if verbose:
                    self.logger.info(f"⏹️ Stopped at {stop_at}")
                break
        
        return state
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get agent by name"""
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None


class AsyncAgentOrchestrator:
    """
    Async orchestrator for agent pipeline
    
    Executes agents sequentially with async/await for optimal performance
    """
    
    def __init__(self, name: str = "AsyncRAGOrchestrator", 
                 openai_client=None, model: str = "gpt-4o-mini",
                 embedding_model=None):  # For Intent Guardrail
        """
        Initialize async orchestrator
        
        Args:
            name: Orchestrator name
            openai_client: OpenAI async client (for on-demand QueryRewriter)
            model: LLM model name (for on-demand QueryRewriter)
            embedding_model: Embedding model (for Intent Guardrail in QueryRewriter)
        """
        self.name = name
        self.agents: List[AsyncBaseAgent] = []
        self.logger = logging.getLogger(f"AsyncOrchestrator.{name}")
        self.progress_callback: Optional[Callable] = None
        
        # For on-demand agent creation (e.g., QueryRewriter in Self-RAG loop)
        self.openai_client = openai_client
        self.model = model
        
        # For Intent Guardrail in QueryRewriter
        self._embedding_model = embedding_model
    
    def add_agent(self, agent: AsyncBaseAgent) -> 'AsyncAgentOrchestrator':
        """
        Add agent to pipeline
        
        Args:
            agent: AsyncBaseAgent instance
            
        Returns:
            Self for chaining
        """
        self.agents.append(agent)
        self.logger.info(f"Added async agent: {agent.name}")
        return self
    
    def set_progress_callback(self, callback: Callable):
        """
        Set callback function to report progress (for UI)
        
        Args:
            callback: Function(agent_name, status) -> None
        """
        self.progress_callback = callback
    
    async def run(self, query: str, verbose: bool = False) -> AgentState:
        """
        Async execute full pipeline from start to finish
        
        Args:
            query: User query
            verbose: Enable verbose logging
            
        Returns:
            Final AgentState with answer
        """
        # Initialize state
        state = AgentState(query=query)
        state.metadata["verbose"] = verbose
        
        if verbose:
            self.logger.info(f"🚀 Starting async orchestration for query: {query[:50]}...")
            self.logger.info(f"📋 Pipeline: {' → '.join(a.name for a in self.agents)}")
        
        # Execute agents sequentially (but each agent can be async internally)
        for i, agent in enumerate(self.agents, 1):
            try:
                if verbose:
                    self.logger.info(f"\n{'='*60}")
                    self.logger.info(f"Step {i}/{len(self.agents)}: {agent.name}")
                    self.logger.info(f"{'='*60}")
                
                # Progress callback for UI
                if self.progress_callback:
                    self.progress_callback(agent.name, AgentStatus.RUNNING)
                
                # Execute agent (async!)
                state = await agent.run(state, verbose=verbose)
                
                # EARLY RETURN: Check if Router rejected the query
                if agent.name == "AsyncRouter" and not state.plan.get("need_retrieval", True):
                    rejection_reason = state.plan.get("rejection_reason", "Câu hỏi không phù hợp")
                    
                    if verbose:
                        self.logger.info(f"\n🛑 Pipeline stopped: Router rejected query")
                        self.logger.info(f"   Reason: {rejection_reason}")
                    
                    # Build rejection message
                    rejection_message = (
                        f"Xin lỗi, tôi không thể trả lời câu hỏi này vì: {rejection_reason}\n\n"
                        "Hệ thống chỉ cung cấp thông tin y khoa tổng quát từ tài liệu tham khảo, "
                        "không đưa ra lời khuyên chẩn đoán hoặc điều trị cá nhân.\n\n"
                        "Vui lòng tham khảo ý kiến bác sĩ chuyên khoa cho các vấn đề sức khỏe cụ thể.\n\n"
                        "⚕️ Thông tin chỉ nhằm tham khảo, không thay thế tư vấn y khoa cá nhân."
                    )
                    
                    state.answer = rejection_message
                    state.metadata["router_rejected"] = True
                    
                    # Log completion
                    total_time = time.time() - state.start_time
                    if verbose:
                        self.logger.info(f"\n{'='*60}")
                        self.logger.info(f"✅ Pipeline stopped after Router in {total_time:.2f}s")
                        self.logger.info(f"{'='*60}\n")
                    
                    return state
                
                # Progress callback
                if self.progress_callback:
                    self.progress_callback(agent.name, AgentStatus.SUCCESS)
                
            except Exception as e:
                # Progress callback
                if self.progress_callback:
                    self.progress_callback(agent.name, AgentStatus.FAILED)
                
                self.logger.error(f"❌ Pipeline failed at {agent.name}: {e}")
                
                # Return state with error
                state.answer = f"Xin lỗi, đã xảy ra lỗi trong quá trình xử lý: {str(e)}"
                return state
        
        # Pipeline completed successfully
        total_time = time.time() - state.start_time
        
        if verbose:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"✅ Pipeline completed in {total_time:.2f}s")
            self.logger.info(f"📊 Execution history:")
            for entry in state.agent_history:
                status_emoji = "✅" if entry["status"] == "success" else "❌"
                self.logger.info(f"   {status_emoji} {entry['agent']}: {entry['duration']:.2f}s")
            self.logger.info(f"{'='*60}\n")
        
        return state
    
    async def run_until(self, query: str, stop_at: str, verbose: bool = False) -> AgentState:
        """
        Async execute pipeline until a specific agent (for debugging/testing)
        
        Args:
            query: User query
            stop_at: Agent name to stop at
            verbose: Enable verbose logging
            
        Returns:
            AgentState at stop point
        """
        state = AgentState(query=query)
        state.metadata["verbose"] = verbose
        
        if verbose:
            self.logger.info(f"🚀 Partial async execution until: {stop_at}")
        
        for agent in self.agents:
            if verbose:
                self.logger.info(f"Executing: {agent.name}")
            
            state = await agent.run(state, verbose=verbose)
            
            if agent.name == stop_at:
                if verbose:
                    self.logger.info(f"⏹️ Stopped at {stop_at}")
                break
        
        return state
    
    async def run_with_self_rag(
        self,
        query: str,
        max_iterations: int = 3,
        verbose: bool = False,
        notify_callback: Optional[Callable] = None
    ) -> AgentState:
        """
        Run pipeline with Self-RAG iterative refinement
        
        Features:
        - Query rewriting when no chunks found
        - Iterative refinement when Critic rejects
        - User notifications for transparency
        - Max iteration limit
        
        Args:
            query: User query
            max_iterations: Max retry attempts (default 3)
            verbose: Enable verbose logging
            notify_callback: Function(message: str) for UI updates
            
        Returns:
            Final AgentState with answer or fallback message
        """
        iteration = 0
        current_query = query
        original_query = query
        best_state = None  # Track best attempt
        
        def notify(message: str):
            """Helper to send notifications"""
            if notify_callback:
                notify_callback(message)
            if verbose:
                self.logger.info(message)
        
        while iteration < max_iterations:
            iteration += 1
            
            # Notify iteration start
            if iteration > 1:
                notify(f"\n{'='*60}")
                notify(f"🔄 Self-RAG Iteration {iteration}/{max_iterations}")
                notify(f"{'='*60}")
            
            # Initialize state for this iteration
            state = AgentState(query=current_query)
            state.metadata["verbose"] = verbose
            state.metadata["original_query"] = original_query
            state.metadata["self_rag_iteration"] = iteration
            
            # Pass ViRanker for Intent Guardrail (if available)
            # This will be used by QueryRewriter to validate query rewrites
            if hasattr(self, '_viranker_instance'):
                state.metadata["viranker_instance"] = self._viranker_instance
            
            # Run full pipeline
            state = await self.run(current_query, verbose=verbose)
            
            # Track best state
            if best_state is None or len(state.reranked_chunks) > len(best_state.reranked_chunks):
                best_state = state
            
            # CRAG QUALITY GATE CHECK
            # Reranker signals if:
            # 1. No chunks found (chunks_count == 0)
            # 2. Top chunk quality insufficient (top_score < threshold)
            trigger_rewrite = state.metadata.get("trigger_crag_rewrite", False)
            chunks_count = len(state.reranked_chunks)
            
            # Check 1: CRAG detected low quality retrieval?
            if trigger_rewrite:
                top_score = state.metadata.get("reranker_stats", {}).get("top_score", 0)
                quality_status = state.metadata.get("reranker_stats", {}).get("crag_quality_check", "unknown")
                
                if iteration >= max_iterations:
                    if chunks_count == 0:
                        notify("❌ Đã thử tối đa 3 lần nhưng không tìm thấy tài liệu phù hợp")
                        return self._create_fallback_state(
                            original_query,
                            "Rất tiếc, tôi không tìm thấy thông tin phù hợp để trả lời câu hỏi này.",
                            best_state
                        )
                    else:
                        notify(f"⚠️ Chất lượng tài liệu chưa đạt yêu cầu sau {max_iterations} lần thử")
                        notify(f"💡 Sử dụng kết quả tốt nhất có (top score: {top_score:.3f})")
                        # Use best chunks we have with disclaimer
                        best_state.metadata["quality_warning"] = True
                        best_state.metadata["crag_warning"] = f"Top score: {top_score:.3f}"
                        return best_state
                
                # Query rewriting with AsyncQueryRewriterAgent (CRAG corrective action)
                if chunks_count == 0:
                    notify("⚠️ Không tìm thấy tài liệu phù hợp")
                else:
                    notify(f"⚠️ CRAG: Chất lượng tài liệu không đủ tốt (top score: {top_score:.3f})")
                
                notify(f"🔀 Đang viết lại câu hỏi với LLM (lần thử {iteration + 1})...")
                
                # Import and create QueryRewriter on-demand
                try:
                    from async_agents.query_rewriter import AsyncQueryRewriterAgent
                    
                    # Get embedding model for Intent Guardrail from orchestrator
                    embedding_model = self._embedding_model if hasattr(self, '_embedding_model') else None
                    
                    # Create temporary rewriter agent (not in pipeline)
                    rewriter = AsyncQueryRewriterAgent(
                        self.openai_client,
                        self.model,
                        embedding_model=embedding_model,  # Intent Guardrail
                        max_rewrites=2,
                        intent_similarity_threshold=0.65  # Prevent query drift (lowered for medical terminology)
                    )
                    
                    # Execute rewriter agent
                    rewrite_state = state.copy()
                    rewrite_state = await rewriter.execute(rewrite_state)
                    
                    # Check if rewrite was successful
                    if rewrite_state.metadata.get("rewrite_failed", False):
                        notify("⚠️ Không thể tạo query mới khác biệt")
                        if chunks_count > 0:
                            # We have some chunks, use them with warning
                            notify("💡 Sử dụng kết quả hiện có với cảnh báo chất lượng")
                            best_state.metadata["quality_warning"] = True
                            return best_state
                        else:
                            return self._create_fallback_state(
                                original_query,
                                "Không tìm thấy thông tin phù hợp sau khi thử nhiều cách viết lại.",
                                best_state
                            )
                    
                    # Get new query
                    new_query = rewrite_state.metadata.get("rewritten_query", current_query)
                    
                    if new_query and new_query != current_query:
                        current_query = new_query
                        strategy = rewrite_state.metadata.get("rewrite_strategy", "unknown")
                        explanation = rewrite_state.metadata.get("rewrite_explanation", "")
                        changes = rewrite_state.metadata.get("rewrite_changes", [])
                        
                        notify(f"💡 Câu hỏi mới: {current_query}")
                        notify(f"📝 Chiến lược: {strategy}")
                        if verbose and explanation:
                            notify(f"   Giải thích: {explanation}")
                        if verbose and changes:
                            for change in changes[:2]:  # Show first 2 changes
                                notify(f"   - {change}")
                        continue
                    else:
                        notify("⚠️ Không thể tạo query mới khác biệt")
                        if chunks_count > 0:
                            notify("💡 Sử dụng kết quả hiện có")
                            best_state.metadata["quality_warning"] = True
                            return best_state
                        return self._create_fallback_state(
                            original_query,
                            "Không tìm thấy thông tin phù hợp.",
                            best_state
                        )
                        
                except Exception as e:
                    notify(f"❌ Lỗi khi viết lại query: {str(e)}")
                    if chunks_count > 0:
                        notify("💡 Sử dụng kết quả hiện có")
                        best_state.metadata["quality_warning"] = True
                        return best_state
                    return self._create_fallback_state(
                        original_query,
                        f"Lỗi trong quá trình viết lại câu hỏi: {str(e)}",
                        best_state
                    )
            
            # Check 2: Answer quality (if Critic exists and rejected)
            critic_approved = state.metadata.get("critic_approved", True)
            
            if not critic_approved:
                if iteration >= max_iterations:
                    notify(f"⚠️ Chất lượng câu trả lời có thể chưa đầy đủ (sau {iteration} lần thử)")
                    # Return best attempt with disclaimer
                    best_state.metadata["quality_warning"] = True
                    return best_state
                
                notify("🔍 Chất lượng câu trả lời chưa đạt yêu cầu")
                notify("💡 Đang cải thiện câu trả lời...")
                
                # Try to get more context or regenerate
                # For now, continue to next iteration with original query
                continue
            
            # Success!
            if iteration > 1:
                notify(f"✅ Hoàn thành sau {iteration} lần thử!")
            
            return state
        
        # Max iterations reached
        notify(f"⚠️ Đã đạt giới hạn {max_iterations} lần thử")
        
        if best_state and best_state.answer:
            notify("💡 Trả về câu trả lời tốt nhất")
            best_state.metadata["quality_warning"] = True
            return best_state
        
        return self._create_fallback_state(
            original_query,
            "Rất tiếc, tôi không thể tìm được câu trả lời thỏa đáng sau nhiều lần thử.",
            best_state
        )
    
    def _create_fallback_state(
        self,
        query: str,
        message: str,
        best_state: Optional[AgentState]
    ) -> AgentState:
        """Create fallback state with helpful message"""
        if best_state:
            state = best_state.copy()
        else:
            state = AgentState(query=query)
        
        state.answer = f"""{message}

📌 Gợi ý:
- Thử đặt câu hỏi cụ thể hơn
- Sử dụng thuật ngữ y khoa nếu có
- Chia nhỏ câu hỏi phức tạp thành nhiều câu đơn giản

Ví dụ:
❌ "bệnh tiểu đường"
✅ "nguyên nhân gây tiểu đường thai kỳ"

Bạn có muốn thử lại không?"""
        
        state.metadata["fallback"] = True
        return state
    
    def get_agent(self, name: str) -> Optional[AsyncBaseAgent]:
        """Get agent by name"""
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None
