# app/memory.py

from typing import Dict, List, Any, Optional, Tuple
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, ConversationBufferWindowMemory
from langchain.memory.chat_message_histories import ChatMessageHistory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from datetime import datetime, timedelta
import json
import os
import pickle
from dataclasses import dataclass, asdict
from enum import Enum


class ConversationContext(Enum):
    """Types of conversation contexts for business intelligence."""
    SALES_ANALYSIS = "sales_analysis"
    CUSTOMER_INSIGHTS = "customer_insights"
    PRODUCT_PERFORMANCE = "product_performance"
    REGIONAL_ANALYSIS = "regional_analysis"
    STRATEGIC_PLANNING = "strategic_planning"
    PROBLEM_SOLVING = "problem_solving"
    GENERAL_BI = "general_bi"


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    timestamp: datetime
    question: str
    answer: str
    context_type: ConversationContext
    sources_used: List[str]
    key_insights: List[str]
    follow_up_suggestions: List[str]


@dataclass
class ConversationSession:
    """Represents a complete conversation session."""
    session_id: str
    start_time: datetime
    last_activity: datetime
    turns: List[ConversationTurn]
    session_summary: str
    key_topics: List[str]
    business_focus_areas: List[str]


class InsightForgeMemorySystem:
    """
    Advanced memory system for InsightForge AI that maintains conversation context,
    tracks business insights, and enables intelligent follow-up conversations.
    """
    
    def __init__(
        self,
        memory_type: str = "buffer_window",
        window_size: int = 10,
        max_token_limit: int = 2000,
        session_timeout_hours: int = 24,
        persist_memory: bool = True,
        memory_file_path: str = None
    ):
        """
        Initialize the memory system.
        
        Args:
            memory_type: Type of memory ('buffer', 'buffer_window', 'summary')
            window_size: Number of recent interactions to keep in buffer window
            max_token_limit: Maximum tokens for summary memory
            session_timeout_hours: Hours before a session is considered inactive
            persist_memory: Whether to save memory to disk
            memory_file_path: Path to save memory data
        """
        self.memory_type = memory_type
        self.window_size = window_size
        self.max_token_limit = max_token_limit
        self.session_timeout = timedelta(hours=session_timeout_hours)
        self.persist_memory = persist_memory
        self.memory_file_path = memory_file_path or self._get_default_memory_path()
        
        # Core memory components
        self.conversation_memory = None
        self.current_session = None
        self.conversation_history = []
        self.business_context_cache = {}
        self.insight_tracker = {}
        
        # Initialize memory system
        self._initialize_memory()
        self._load_persistent_memory()
    


    def _get_default_memory_path(self) -> str:
        """Get default path for memory persistence."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(current_dir), "data")
        os.makedirs(data_dir, exist_ok=True)
        return os.path.join(data_dir, "conversation_memory.pkl")
    


    def _initialize_memory(self):
        """Initialize the appropriate type of conversation memory."""
        
        if self.memory_type == "buffer":
            self.conversation_memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        elif self.memory_type == "buffer_window":
            self.conversation_memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
                k=self.window_size
            )
        elif self.memory_type == "summary":
            # Note: Summary memory requires an LLM for summarization
            self.conversation_memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        else:
            raise ValueError(f"Unsupported memory type: {self.memory_type}")
        
        print(f"✅ Initialized {self.memory_type} memory system")
    


    def start_new_session(self, session_id: str = None) -> str:
        """
        Start a new conversation session.
        
        Args:
            session_id: Optional custom session ID
            
        Returns:
            str: Session ID
        """
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session = ConversationSession(
            session_id=session_id,
            start_time=datetime.now(),
            last_activity=datetime.now(),
            turns=[],
            session_summary="",
            key_topics=[],
            business_focus_areas=[]
        )
        
        # Clear conversation memory for new session
        if self.conversation_memory:
            self.conversation_memory.clear()
        
        print(f"🆕 Started new session: {session_id}")
        return session_id
    


    def add_conversation_turn(
        self,
        question: str,
        answer: str,
        context_type: ConversationContext,
        sources_used: List[str] = None,
        key_insights: List[str] = None,
        follow_up_suggestions: List[str] = None
    ):
        """
        Add a new conversation turn to memory.
        
        Args:
            question: User's question
            answer: AI's response
            context_type: Type of business context
            sources_used: Sources referenced in the answer
            key_insights: Key business insights extracted
            follow_up_suggestions: Suggested follow-up questions
        """
        if self.current_session is None:
            self.start_new_session()
        
        # Create conversation turn
        turn = ConversationTurn(
            timestamp=datetime.now(),
            question=question,
            answer=answer,
            context_type=context_type,
            sources_used=sources_used or [],
            key_insights=key_insights or [],
            follow_up_suggestions=follow_up_suggestions or []
        )
        
        # Add to current session
        self.current_session.turns.append(turn)
        self.current_session.last_activity = datetime.now()
        
        # Add to LangChain memory
        if self.conversation_memory:
            self.conversation_memory.chat_memory.add_user_message(question)
            self.conversation_memory.chat_memory.add_ai_message(answer)
        
        # Update business context tracking
        self._update_business_context(turn)
        
        # Update insight tracking
        self._update_insight_tracker(turn)
        
        # Persist memory if enabled
        if self.persist_memory:
            self._save_persistent_memory()
        
        print(f"💭 Added conversation turn to memory ({context_type.value})")
    


    def get_conversation_context(self, include_insights: bool = True) -> Dict[str, Any]:
        """
        Get current conversation context for enhanced responses.
        
        Args:
            include_insights: Whether to include accumulated insights
            
        Returns:
            Dict containing conversation context
        """
        if not self.current_session or not self.current_session.turns:
            return {"context": "No previous conversation", "insights": []}
        
        # Get recent conversation history
        recent_turns = self.current_session.turns[-5:]  # Last 5 turns
        
        context = {
            "session_id": self.current_session.session_id,
            "session_duration": str(datetime.now() - self.current_session.start_time),
            "total_turns": len(self.current_session.turns),
            "recent_topics": [turn.context_type.value for turn in recent_turns],
            "recent_questions": [turn.question for turn in recent_turns[-3:]],  # Last 3 questions
        }
        
        if include_insights:
            context["accumulated_insights"] = self._get_accumulated_insights()
            context["suggested_follow_ups"] = self._get_relevant_follow_ups()
        
        # Add business focus summary
        context["business_focus_areas"] = list(set([
            turn.context_type.value for turn in self.current_session.turns
        ]))
        
        return context
    


    def get_memory_for_llm(self) -> str:
        """
        Get formatted memory content for LLM context.
        
        Returns:
            str: Formatted conversation history
        """
        if not self.conversation_memory:
            return "No conversation history available."
        
        # Get conversation history from LangChain memory
        memory_variables = self.conversation_memory.load_memory_variables({})
        chat_history = memory_variables.get("chat_history", [])
        
        if not chat_history:
            return "No previous conversation in this session."
        
        # Format memory for LLM
        formatted_history = []
        for message in chat_history[-6:]:  # Last 6 messages (3 exchanges)
            if isinstance(message, HumanMessage):
                formatted_history.append(f"User: {message.content}")
            elif isinstance(message, AIMessage):
                formatted_history.append(f"Assistant: {message.content[:200]}...")
        
        return "\n".join(formatted_history)
    


    def generate_contextual_prompt_enhancement(self, current_question: str) -> str:
        """
        Generate context-aware prompt enhancement based on conversation history.
        
        Args:
            current_question: The current user question
            
        Returns:
            str: Enhanced context for the prompt
        """
        if not self.current_session or not self.current_session.turns:
            return ""
        
        context = self.get_conversation_context()
        
        enhancement = f"""
**Conversation Context:**
- Session Duration: {context['session_duration']}
- Previous Topics Discussed: {', '.join(context['recent_topics'])}
- Business Focus Areas: {', '.join(context['business_focus_areas'])}

**Recent Context:**
{self.get_memory_for_llm()}

**Instructions:**
- Build upon the previous conversation context when relevant
- Reference earlier insights if they relate to the current question
- Maintain consistency with previous analysis
- If this is a follow-up question, explicitly connect it to prior discussions
"""
        
        return enhancement
    
    def suggest_follow_up_questions(self, current_answer: str, context_type: ConversationContext) -> List[str]:
        """
        Generate intelligent follow-up questions based on current answer and context.
        
        Args:
            current_answer: The AI's current response
            context_type: Type of business context
            
        Returns:
            List of suggested follow-up questions
        """
        base_suggestions = {
            ConversationContext.SALES_ANALYSIS: [
                "How do these sales trends compare to the same period last year?",
                "What factors might be driving these sales patterns?",
                "Which sales strategies could improve our performance?",
                "How do these results impact our quarterly targets?"
            ],
            ConversationContext.CUSTOMER_INSIGHTS: [
                "What customer retention strategies would work best for these segments?",
                "How can we increase the lifetime value of these customer groups?",
                "What marketing messages would resonate with our top segments?",
                "Are there seasonal patterns in customer behavior we should consider?"
            ],
            ConversationContext.PRODUCT_PERFORMANCE: [
                "What product improvements could increase sales?",
                "How do our products compare to competitors?",
                "Which products have the highest profit margins?",
                "What new products should we consider developing?"
            ],
            ConversationContext.REGIONAL_ANALYSIS: [
                "What regional expansion opportunities exist?",
                "How do local market conditions affect regional performance?",
                "What region-specific strategies should we implement?",
                "Are there supply chain optimizations for underperforming regions?"
            ]
        }
        
        suggestions = base_suggestions.get(context_type, [
            "What additional insights would be helpful?",
            "How can we apply these findings strategically?",
            "What trends should we monitor going forward?"
        ])
        
        # Customize based on conversation history
        if self.current_session and len(self.current_session.turns) > 0:
            # Add context-aware suggestions
            recent_topics = [turn.context_type for turn in self.current_session.turns[-3:]]
            if len(set(recent_topics)) > 1:
                suggestions.append("How do these insights connect to our previous analysis?")
        
        return suggestions[:4]  # Return top 4 suggestions
    


    def _update_business_context(self, turn: ConversationTurn):
        """Update business context cache based on conversation turn."""
        context_key = turn.context_type.value
        
        if context_key not in self.business_context_cache:
            self.business_context_cache[context_key] = {
                "count": 0,
                "last_discussed": None,
                "key_questions": [],
                "insights": []
            }
        
        cache = self.business_context_cache[context_key]
        cache["count"] += 1
        cache["last_discussed"] = turn.timestamp
        cache["key_questions"].append(turn.question)
        cache["insights"].extend(turn.key_insights)
        
        # Keep only recent questions (last 5)
        cache["key_questions"] = cache["key_questions"][-5:]
        cache["insights"] = cache["insights"][-10:]  # Last 10 insights
    


    def _update_insight_tracker(self, turn: ConversationTurn):
        """Track and accumulate business insights across conversations."""
        for insight in turn.key_insights:
            insight_key = insight.lower()[:50]  # Use first 50 chars as key
            
            if insight_key not in self.insight_tracker:
                self.insight_tracker[insight_key] = {
                    "insight": insight,
                    "first_mentioned": turn.timestamp,
                    "mention_count": 0,
                    "related_contexts": []
                }
            
            tracker = self.insight_tracker[insight_key]
            tracker["mention_count"] += 1
            if turn.context_type not in tracker["related_contexts"]:
                tracker["related_contexts"].append(turn.context_type)
    


    def _get_accumulated_insights(self) -> List[str]:
        """Get key insights accumulated during the session."""
        if not self.current_session:
            return []
        
        insights = []
        for turn in self.current_session.turns:
            insights.extend(turn.key_insights)
        
        # Return unique insights, most recent first
        unique_insights = list(dict.fromkeys(insights))  # Preserve order, remove duplicates
        return unique_insights[-5:]  # Last 5 unique insights
    


    def _get_relevant_follow_ups(self) -> List[str]:
        """Get relevant follow-up suggestions from recent turns."""
        if not self.current_session:
            return []
        
        follow_ups = []
        for turn in self.current_session.turns[-2:]:  # Last 2 turns
            follow_ups.extend(turn.follow_up_suggestions)
        
        return list(dict.fromkeys(follow_ups))[:3]  # Top 3 unique suggestions
    


    def _save_persistent_memory(self):
        """Save memory to persistent storage."""
        try:
            memory_data = {
                "current_session": asdict(self.current_session) if self.current_session else None,
                "business_context_cache": self.business_context_cache,
                "insight_tracker": self.insight_tracker,
                "last_saved": datetime.now().isoformat()
            }
            
            with open(self.memory_file_path, 'wb') as f:
                pickle.dump(memory_data, f)
            
        except Exception as e:
            print(f"⚠️ Warning: Could not save memory to disk: {e}")
    


    def _load_persistent_memory(self):
        """Load memory from persistent storage."""
        if not os.path.exists(self.memory_file_path):
            return
        
        try:
            with open(self.memory_file_path, 'rb') as f:
                memory_data = pickle.load(f)
            
            # Restore business context and insights
            self.business_context_cache = memory_data.get("business_context_cache", {})
            self.insight_tracker = memory_data.get("insight_tracker", {})
            
            print(f"✅ Loaded persistent memory from {self.memory_file_path}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not load memory from disk: {e}")
    


    def get_session_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the current session."""
        if not self.current_session:
            return {"status": "No active session"}
        
        session = self.current_session
        
        # Analyze conversation patterns
        context_counts = {}
        for turn in session.turns:
            context = turn.context_type.value
            context_counts[context] = context_counts.get(context, 0) + 1
        
        # Get top insights
        all_insights = []
        for turn in session.turns:
            all_insights.extend(turn.key_insights)
        
        return {
            "session_id": session.session_id,
            "duration": str(datetime.now() - session.start_time),
            "total_turns": len(session.turns),
            "business_areas_discussed": list(context_counts.keys()),
            "most_discussed_area": max(context_counts.items(), key=lambda x: x[1])[0] if context_counts else None,
            "total_insights_generated": len(all_insights),
            "unique_insights": len(set(all_insights)),
            "conversation_topics": context_counts
        }
    


    def clear_session_memory(self):
        """Clear the current session memory."""
        if self.conversation_memory:
            self.conversation_memory.clear()
        
        self.current_session = None
        print("🧹 Session memory cleared")
    


    def export_session_data(self, export_path: str = None) -> str:
        """
        Export session data to JSON for analysis or backup.
        
        Args:
            export_path: Path to save the export file
            
        Returns:
            str: Path to exported file
        """
        if not self.current_session:
            raise ValueError("No active session to export")
        
        if export_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = f"session_export_{timestamp}.json"
        
        # Convert session to JSON-serializable format
        export_data = {
            "session_summary": self.get_session_summary(),
            "conversation_turns": [],
            "business_context": self.business_context_cache,
            "export_timestamp": datetime.now().isoformat()
        }
        
        for turn in self.current_session.turns:
            turn_data = asdict(turn)
            turn_data["timestamp"] = turn.timestamp.isoformat()
            turn_data["context_type"] = turn.context_type.value
            export_data["conversation_turns"].append(turn_data)
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📤 Session data exported to {export_path}")
        return export_path



def create_memory_enhanced_prompt(
    base_prompt: str,
    memory_system: InsightForgeMemorySystem,
    current_question: str
) -> str:
    """
    Enhance a base prompt with conversation memory context.
    
    Args:
        base_prompt: The original prompt template
        memory_system: Initialized memory system
        current_question: Current user question
        
    Returns:
        str: Memory-enhanced prompt
    """
    memory_context = memory_system.generate_contextual_prompt_enhancement(current_question)
    
    if memory_context:
        enhanced_prompt = f"{base_prompt}\n\n{memory_context}"
    else:
        enhanced_prompt = base_prompt
    
    return enhanced_prompt
