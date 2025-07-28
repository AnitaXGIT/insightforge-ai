# evaluation/evaluator.py

import os
import json
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.evaluation import QAEvalChain
from langchain.schema import Document
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class EvaluationResult:
    """Results from a single evaluation."""
    timestamp: datetime
    question: str
    predicted_answer: str
    reference_answer: str
    accuracy_score: float
    relevance_score: float
    completeness_score: float
    context_quality: float
    response_time: float
    sources_used: List[str]
    evaluation_notes: str


@dataclass
class EvaluationSession:
    """Complete evaluation session results."""
    session_id: str
    start_time: datetime
    end_time: datetime
    total_questions: int
    average_accuracy: float
    average_relevance: float
    average_completeness: float
    average_response_time: float
    evaluation_results: List[EvaluationResult]
    summary_report: str


class InsightForgeEvaluator:
    """
    Comprehensive evaluation system for InsightForge AI.
    Assesses response quality, accuracy, relevance, and performance metrics.
    """
    
    def __init__(
        self,
        evaluation_data_path: str = None,
        save_results: bool = True,
        results_directory: str = None
    ):
        """
        Initialize the evaluation system.
        
        Args:
            evaluation_data_path: Path to evaluation dataset (Q&A pairs)
            save_results: Whether to save evaluation results
            results_directory: Directory to save evaluation results
        """
        self.evaluation_data_path = evaluation_data_path or self._get_default_eval_data_path()
        self.save_results = save_results
        self.results_directory = results_directory or self._get_default_results_dir()
        
        # Initialize evaluation components
        self.qa_eval_chain = None
        self.evaluation_questions = []
        self.reference_answers = []
        self.current_session = None
        
        # Performance tracking
        self.response_times = []
        self.accuracy_scores = []
        self.quality_metrics = []
        
        # Setup evaluation environment
        self._setup_evaluation_environment()
    
    def _get_default_eval_data_path(self) -> str:
        """Get default path for evaluation data."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(os.path.dirname(current_dir), "data", "evaluation_data.json")
    
    def _get_default_results_dir(self) -> str:
        """Get default directory for evaluation results."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(os.path.dirname(current_dir), "data", "evaluation_results")
        os.makedirs(results_dir, exist_ok=True)
        return results_dir
    
    def _setup_evaluation_environment(self):
        """Setup the evaluation environment and load test data."""
        print("Setting up InsightForge evaluation environment...")
        
        # Create evaluation data if it doesn't exist
        if not os.path.exists(self.evaluation_data_path):
            self._create_default_evaluation_data()
        
        # Load evaluation questions and answers
        self._load_evaluation_data()
        
        print(f"✅ Evaluation environment ready with {len(self.evaluation_questions)} test questions")
    
    def _create_default_evaluation_data(self):
        """Create default evaluation dataset for business intelligence testing."""
        
        default_eval_data = {
            "business_intelligence_qa_pairs": [
                {
                    "question": "What are our total sales figures?",
                    "reference_answer": "Based on the sales data, total sales are $125,000 with strong performance indicators.",
                    "category": "sales_analysis",
                    "difficulty": "easy",
                    "expected_sources": ["sales_data"]
                },
                {
                    "question": "Which products are performing best and why?",
                    "reference_answer": "Product C leads with $50,000 revenue (40% of total), followed by Product A at $45,000 (36%). Product C's success is attributed to strong market demand and effective positioning.",
                    "category": "product_analysis",
                    "difficulty": "medium",
                    "expected_sources": ["sales_data", "product_analysis"]
                },
                {
                    "question": "How do regional sales compare across different areas?",
                    "reference_answer": "West region dominates with $55,000 (44% of total sales), while East region shows $35,000 (28%) and needs strategic attention for growth.",
                    "category": "regional_analysis",
                    "difficulty": "medium",
                    "expected_sources": ["sales_data"]
                },
                {
                    "question": "What customer demographics drive the highest sales?",
                    "reference_answer": "The 26-35 age group generates the highest average sales, with 65% female customer base showing strong purchasing power and brand loyalty.",
                    "category": "customer_segmentation",
                    "difficulty": "medium",
                    "expected_sources": ["sales_data", "customer_analysis"]
                },
                {
                    "question": "Based on our performance data, what strategic recommendations would you make?",
                    "reference_answer": "Focus marketing efforts on West region success factors, develop East region strategies, prioritize Product C expansion, and target 26-35 demographic with tailored campaigns.",
                    "category": "strategic_planning",
                    "difficulty": "hard",
                    "expected_sources": ["sales_data", "strategic_documents"]
                },
                {
                    "question": "How can AI business model innovations improve our current performance?",
                    "reference_answer": "AI innovations can enhance predictive analytics for demand forecasting, automate customer segmentation, optimize regional resource allocation, and enable real-time performance monitoring.",
                    "category": "innovation_analysis",
                    "difficulty": "hard",
                    "expected_sources": ["AI_business_innovation.pdf", "sales_data"]
                },
                {
                    "question": "What trends do you see in our monthly sales performance?",
                    "reference_answer": "Sales show upward trajectory with Q3 being the strongest quarter, indicating seasonal demand patterns and successful product positioning strategies.",
                    "category": "trend_analysis",
                    "difficulty": "medium",
                    "expected_sources": ["sales_data"]
                },
                {
                    "question": "Compare our top products across different customer segments.",
                    "reference_answer": "Product C performs best across all segments, with particularly strong adoption in the 26-35 age group. Product A shows regional preferences in West markets.",
                    "category": "comparative_analysis",
                    "difficulty": "hard",
                    "expected_sources": ["sales_data", "customer_analysis"]
                }
            ]
        }
        
        # Save evaluation data
        os.makedirs(os.path.dirname(self.evaluation_data_path), exist_ok=True)
        with open(self.evaluation_data_path, 'w') as f:
            json.dump(default_eval_data, f, indent=2)
        
        print(f"✅ Created default evaluation dataset at {self.evaluation_data_path}")
    
    def _load_evaluation_data(self):
        """Load evaluation questions and reference answers."""
        try:
            with open(self.evaluation_data_path, 'r') as f:
                eval_data = json.load(f)
            
            qa_pairs = eval_data.get("business_intelligence_qa_pairs", [])
            
            self.evaluation_questions = []
            self.reference_answers = []
            
            for pair in qa_pairs:
                self.evaluation_questions.append({
                    "question": pair["question"],
                    "category": pair.get("category", "general"),
                    "difficulty": pair.get("difficulty", "medium"),
                    "expected_sources": pair.get("expected_sources", [])
                })
                self.reference_answers.append(pair["reference_answer"])
            
            print(f"✅ Loaded {len(self.evaluation_questions)} evaluation questions")
            
        except Exception as e:
            print(f"⚠️ Error loading evaluation data: {e}")
            self.evaluation_questions = []
            self.reference_answers = []
    
    def start_evaluation_session(self, session_id: str = None) -> str:
        """
        Start a new evaluation session.
        
        Args:
            session_id: Optional custom session ID
            
        Returns:
            str: Session ID
        """
        if session_id is None:
            session_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session = EvaluationSession(
            session_id=session_id,
            start_time=datetime.now(),
            end_time=None,
            total_questions=0,
            average_accuracy=0.0,
            average_relevance=0.0,
            average_completeness=0.0,
            average_response_time=0.0,
            evaluation_results=[],
            summary_report=""
        )
        
        print(f"🧪 Started evaluation session: {session_id}")
        return session_id
    
    def evaluate_response(
        self,
        question: str,
        predicted_answer: str,
        reference_answer: str = None,
        sources_used: List[str] = None,
        response_time: float = 0.0,
        context_documents: List[Document] = None
    ) -> EvaluationResult:
        """
        Evaluate a single response against reference answer and quality metrics.
        
        Args:
            question: The input question
            predicted_answer: AI-generated response
            reference_answer: Expected/reference answer
            sources_used: Sources referenced in the response
            response_time: Time taken to generate response
            context_documents: Context documents used
            
        Returns:
            EvaluationResult: Detailed evaluation metrics
        """
        # If no reference answer provided, try to find it in evaluation data
        if reference_answer is None:
            reference_answer = self._find_reference_answer(question)
        
        # Calculate evaluation metrics
        accuracy_score = self._calculate_accuracy_score(predicted_answer, reference_answer)
        relevance_score = self._calculate_relevance_score(question, predicted_answer)
        completeness_score = self._calculate_completeness_score(predicted_answer, reference_answer)
        context_quality = self._calculate_context_quality(context_documents, question)
        
        # Create evaluation result
        result = EvaluationResult(
            timestamp=datetime.now(),
            question=question,
            predicted_answer=predicted_answer,
            reference_answer=reference_answer or "No reference available",
            accuracy_score=accuracy_score,
            relevance_score=relevance_score,
            completeness_score=completeness_score,
            context_quality=context_quality,
            response_time=response_time,
            sources_used=sources_used or [],
            evaluation_notes=self._generate_evaluation_notes(
                accuracy_score, relevance_score, completeness_score, context_quality
            )
        )
        
        # Add to current session if active
        if self.current_session:
            self.current_session.evaluation_results.append(result)
            self.current_session.total_questions += 1
        
        return result
    
    def _find_reference_answer(self, question: str) -> str:
        """Find reference answer for a question in evaluation data."""
        for i, eval_q in enumerate(self.evaluation_questions):
            if eval_q["question"].lower() == question.lower():
                return self.reference_answers[i]
        return "Reference answer not found"
    
    def _calculate_accuracy_score(self, predicted: str, reference: str) -> float:
        """Calculate accuracy score using semantic similarity."""
        try:
            # Simple keyword-based accuracy for now
            # In production, you'd use embedding similarity
            predicted_words = set(predicted.lower().split())
            reference_words = set(reference.lower().split())
            
            if not reference_words:
                return 0.0
            
            intersection = predicted_words.intersection(reference_words)
            union = predicted_words.union(reference_words)
            
            # Jaccard similarity
            similarity = len(intersection) / len(union) if union else 0.0
            
            # Boost score for key business terms
            business_terms = {'sales', 'revenue', 'product', 'customer', 'region', 'performance'}
            business_matches = len(intersection.intersection(business_terms))
            boost = min(0.2, business_matches * 0.05)
            
            return min(1.0, similarity + boost)
            
        except Exception:
            return 0.5  # Default score if calculation fails
    
    def _calculate_relevance_score(self, question: str, answer: str) -> float:
        """Calculate how relevant the answer is to the question."""
        try:
            question_words = set(question.lower().split())
            answer_words = set(answer.lower().split())
            
            # Remove common words
            stop_words = {'the', 'is', 'are', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'what', 'how', 'which', 'our'}
            question_words = question_words - stop_words
            answer_words = answer_words - stop_words
            
            if not question_words:
                return 0.5
            
            overlap = len(question_words.intersection(answer_words))
            relevance = overlap / len(question_words)
            
            # Boost for business intelligence terms
            bi_terms = {'sales', 'analysis', 'performance', 'data', 'insights', 'trends'}
            bi_overlap = len(question_words.intersection(answer_words).intersection(bi_terms))
            boost = min(0.3, bi_overlap * 0.1)
            
            return min(1.0, relevance + boost)
            
        except Exception:
            return 0.6  # Default score
    
    def _calculate_completeness_score(self, predicted: str, reference: str) -> float:
        """Calculate how complete the answer is compared to reference."""
        try:
            # Length-based completeness with quality adjustment
            pred_length = len(predicted.split())
            ref_length = len(reference.split())
            
            if ref_length == 0:
                return 0.5
            
            # Ideal range is 80-150% of reference length
            length_ratio = pred_length / ref_length
            
            if 0.8 <= length_ratio <= 1.5:
                length_score = 1.0
            elif length_ratio < 0.8:
                length_score = length_ratio / 0.8  # Penalize too short
            else:
                length_score = max(0.3, 1.5 / length_ratio)  # Penalize too long
            
            # Adjust for information density
            info_density = self._calculate_info_density(predicted)
            
            return min(1.0, length_score * info_density)
            
        except Exception:
            return 0.6  # Default score
    
    def _calculate_info_density(self, text: str) -> float:
        """Calculate information density of text."""
        try:
            words = text.split()
            if len(words) < 5:
                return 0.5
            
            # Count information-rich words
            info_words = ['analysis', 'performance', 'data', 'insights', 'revenue', 'sales', 
                         'customer', 'product', 'region', 'trend', 'growth', 'strategy']
            
            info_count = sum(1 for word in words if any(info_word in word.lower() for info_word in info_words))
            density = info_count / len(words)
            
            # Normalize to 0.5-1.0 range
            return max(0.5, min(1.0, 0.5 + density * 2))
            
        except Exception:
            return 0.7
    
    def _calculate_context_quality(self, context_documents: List[Document], question: str) -> float:
        """Calculate quality of retrieved context for the question."""
        if not context_documents:
            return 0.0
        
        try:
            question_words = set(question.lower().split())
            
            relevance_scores = []
            for doc in context_documents:
                doc_words = set(doc.page_content.lower().split())
                overlap = len(question_words.intersection(doc_words))
                relevance = overlap / len(question_words) if question_words else 0
                relevance_scores.append(relevance)
            
            # Average relevance with bonus for multiple relevant sources
            avg_relevance = np.mean(relevance_scores)
            diversity_bonus = min(0.2, len(set(doc.metadata.get('source', '') for doc in context_documents)) * 0.1)
            
            return min(1.0, avg_relevance + diversity_bonus)
            
        except Exception:
            return 0.5
    
    def _generate_evaluation_notes(self, accuracy: float, relevance: float, completeness: float, context: float) -> str:
        """Generate human-readable evaluation notes."""
        notes = []
        
        # Accuracy assessment
        if accuracy >= 0.8:
            notes.append("High accuracy - well-aligned with expected answer")
        elif accuracy >= 0.6:
            notes.append("Good accuracy - mostly correct information")
        else:
            notes.append("Low accuracy - significant deviations from expected answer")
        
        # Relevance assessment
        if relevance >= 0.8:
            notes.append("Highly relevant to the question")
        elif relevance >= 0.6:
            notes.append("Generally relevant with some tangential content")
        else:
            notes.append("Low relevance - answer may be off-topic")
        
        # Completeness assessment
        if completeness >= 0.8:
            notes.append("Comprehensive and complete response")
        elif completeness >= 0.6:
            notes.append("Adequate completeness with room for more detail")
        else:
            notes.append("Incomplete response - missing key information")
        
        # Context quality assessment
        if context >= 0.7:
            notes.append("Excellent context retrieval")
        elif context >= 0.5:
            notes.append("Good context relevance")
        else:
            notes.append("Poor context quality - may need better retrieval")
        
        return "; ".join(notes)
    
    def run_comprehensive_evaluation(self, rag_pipeline) -> EvaluationSession:
        """
        Run comprehensive evaluation using all test questions.
        
        Args:
            rag_pipeline: The RAG pipeline to evaluate
            
        Returns:
            EvaluationSession: Complete evaluation results
        """
        session_id = self.start_evaluation_session("comprehensive_eval")
        
        print(f"\n🧪 Running comprehensive evaluation with {len(self.evaluation_questions)} questions...")
        print("=" * 60)
        
        for i, eval_item in enumerate(self.evaluation_questions, 1):
            question = eval_item["question"]
            category = eval_item["category"]
            reference_answer = self.reference_answers[i-1]
            
            print(f"\n📝 Question {i}/{len(self.evaluation_questions)} ({category})")
            print(f"❓ {question}")
            
            try:
                # Measure response time
                start_time = datetime.now()
                
                # Get response from RAG pipeline
                response = rag_pipeline.query(question, return_sources=True)
                
                end_time = datetime.now()
                response_time = (end_time - start_time).total_seconds()
                
                # Extract response details
                predicted_answer = response.get("answer", "")
                sources_used = [src.get("source_type", "") for src in response.get("sources", [])]
                
                # Get context documents
                context_docs = rag_pipeline.get_relevant_context(question, k=3)
                
                # Evaluate response
                result = self.evaluate_response(
                    question=question,
                    predicted_answer=predicted_answer,
                    reference_answer=reference_answer,
                    sources_used=sources_used,
                    response_time=response_time,
                    context_documents=context_docs
                )
                
                # Display results
                print(f"⏱️  Response time: {response_time:.2f}s")
                print(f"📊 Accuracy: {result.accuracy_score:.2f}")
                print(f"🎯 Relevance: {result.relevance_score:.2f}")
                print(f"📋 Completeness: {result.completeness_score:.2f}")
                print(f"📚 Context Quality: {result.context_quality:.2f}")
                
            except Exception as e:
                print(f"❌ Evaluation failed for question {i}: {e}")
        
        # Finalize session
        self._finalize_evaluation_session()
        
        return self.current_session
    
    def _finalize_evaluation_session(self):
        """Finalize the current evaluation session with summary statistics."""
        if not self.current_session or not self.current_session.evaluation_results:
            return
        
        results = self.current_session.evaluation_results
        
        # Calculate averages
        self.current_session.average_accuracy = np.mean([r.accuracy_score for r in results])
        self.current_session.average_relevance = np.mean([r.relevance_score for r in results])
        self.current_session.average_completeness = np.mean([r.completeness_score for r in results])
        self.current_session.average_response_time = np.mean([r.response_time for r in results])
        self.current_session.end_time = datetime.now()
        
        # Generate summary report
        self.current_session.summary_report = self._generate_summary_report()
        
        # Save results if enabled
        if self.save_results:
            self._save_evaluation_results()
        
        print(f"\n✅ Evaluation session completed!")
        print(f"📊 Overall Performance:")
        print(f"   • Average Accuracy: {self.current_session.average_accuracy:.3f}")
        print(f"   • Average Relevance: {self.current_session.average_relevance:.3f}")
        print(f"   • Average Completeness: {self.current_session.average_completeness:.3f}")
        print(f"   • Average Response Time: {self.current_session.average_response_time:.2f}s")
    
    def _generate_summary_report(self) -> str:
        """Generate a comprehensive summary report."""
        if not self.current_session:
            return "No evaluation session active"
        
        results = self.current_session.evaluation_results
        session = self.current_session
        
        # Performance categorization
        def get_performance_category(score):
            if score >= 0.8:
                return "Excellent"
            elif score >= 0.6:
                return "Good"
            elif score >= 0.4:
                return "Fair"
            else:
                return "Needs Improvement"
        
        report = f"""
InsightForge AI - Evaluation Summary Report
==========================================

Session: {session.session_id}
Duration: {session.start_time.strftime('%Y-%m-%d %H:%M')} - {session.end_time.strftime('%H:%M')}
Total Questions Evaluated: {session.total_questions}

OVERALL PERFORMANCE METRICS:
• Accuracy: {session.average_accuracy:.3f} ({get_performance_category(session.average_accuracy)})
• Relevance: {session.average_relevance:.3f} ({get_performance_category(session.average_relevance)})
• Completeness: {session.average_completeness:.3f} ({get_performance_category(session.average_completeness)})
• Response Time: {session.average_response_time:.2f}s

PERFORMANCE BREAKDOWN:
• Questions with >80% accuracy: {len([r for r in results if r.accuracy_score >= 0.8])}/{len(results)}
• Questions with >80% relevance: {len([r for r in results if r.relevance_score >= 0.8])}/{len(results)}
• Average context quality: {np.mean([r.context_quality for r in results]):.3f}

RECOMMENDATIONS:
"""
        
        # Add specific recommendations based on performance
        if session.average_accuracy < 0.6:
            report += "• Improve answer accuracy through better prompt engineering\n"
        if session.average_relevance < 0.6:
            report += "• Enhance context retrieval and query understanding\n"
        if session.average_completeness < 0.6:
            report += "• Expand response detail and information coverage\n"
        if session.average_response_time > 3.0:
            report += "• Optimize response time through better caching or model efficiency\n"
        
        if all(score >= 0.7 for score in [session.average_accuracy, session.average_relevance, session.average_completeness]):
            report += "• System performance is strong - consider advanced features or deployment\n"
        
        return report
    
    def _save_evaluation_results(self):
        """Save evaluation results to disk."""
        if not self.current_session:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        results_file = os.path.join(self.results_directory, f"evaluation_{timestamp}.json")
        
        # Convert session to JSON-serializable format
        session_data = asdict(self.current_session)
        session_data["start_time"] = self.current_session.start_time.isoformat()
        session_data["end_time"] = self.current_session.end_time.isoformat()
        
        for result in session_data["evaluation_results"]:
            result["timestamp"] = datetime.fromisoformat(result["timestamp"]).isoformat()
        
        with open(results_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        # Save summary report
        report_file = os.path.join(self.results_directory, f"summary_report_{timestamp}.txt")
        with open(report_file, 'w') as f:
            f.write(self.current_session.summary_report)
        
        print(f"💾 Evaluation results saved to {self.results_directory}")
    
    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get detailed performance analytics for the current session."""
        if not self.current_session or not self.current_session.evaluation_results:
            return {"error": "No evaluation data available"}
        
        results = self.current_session.evaluation_results
        
        analytics = {
            "total_questions": len(results),
            "performance_metrics": {
                "accuracy": {
                    "mean": np.mean([r.accuracy_score for r in results]),
                    "std": np.std([r.accuracy_score for r in results]),
                    "min": np.min([r.accuracy_score for r in results]),
                    "max": np.max([r.accuracy_score for r in results])
                },
                "relevance": {
                    "mean": np.mean([r.relevance_score for r in results]),
                    "std": np.std([r.relevance_score for r in results]),
                    "min": np.min([r.relevance_score for r in results]),
                    "max": np.max([r.relevance_score for r in results])
                },
                "completeness": {
                    "mean": np.mean([r.completeness_score for r in results]),
                    "std": np.std([r.completeness_score for r in results]),
                    "min": np.min([r.completeness_score for r in results]),
                    "max": np.max([r.completeness_score for r in results])
                },
                "response_time": {
                    "mean": np.mean([r.response_time for r in results]),
                    "std": np.std([r.response_time for r in results]),
                    "min": np.min([r.response_time for r in results]),
                    "max": np.max([r.response_time for r in results])
                }
            },
            "quality_distribution": {
                "high_quality": len([r for r in results if all([r.accuracy_score >= 0.8, r.relevance_score >= 0.8])]),
                "medium_quality": len([r for r in results if all([r.accuracy_score >= 0.6, r.relevance_score >= 0.6]) and not all([r.accuracy_score >= 0.8, r.relevance_score >= 0.8])]),
                "low_quality": len([r for r in results if r.accuracy_score < 0.6 or r.relevance_score < 0.6])
            }
        }
        
        return analytics
