# evaluation/improved_evaluator.py - Better evaluation for business intelligence responses

import os
import json
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np
import re
from langchain.schema import Document


@dataclass
class ImprovedEvaluationResult:
    """Enhanced evaluation results with multiple metrics."""
    timestamp: datetime
    question: str
    predicted_answer: str
    reference_answer: str
    
    # Core metrics
    semantic_similarity: float
    keyword_coverage: float
    business_relevance: float
    factual_accuracy: float
    
    # Overall scores
    accuracy_score: float
    relevance_score: float
    completeness_score: float
    
    # Metadata
    response_time: float
    sources_used: List[str]
    evaluation_notes: str


class ImprovedBIEvaluator:
    """
    Improved evaluator specifically designed for business intelligence responses.
    Uses multiple evaluation approaches for more accurate scoring.
    """
    
    def __init__(self):
        """Initialize the improved evaluator."""
        
        # Business intelligence keywords for relevance scoring
        self.bi_keywords = {
            'sales', 'revenue', 'profit', 'growth', 'performance', 'metrics',
            'customers', 'products', 'regions', 'analysis', 'trends', 'data',
            'insights', 'strategy', 'recommendations', 'optimization', 'ai'
        }
        
        # Factual indicators (numbers, percentages, etc.)
        self.factual_patterns = [
            r'\$[\d,]+(?:\.\d{2})?',  # Money amounts
            r'\d+(?:\.\d+)?%',        # Percentages
            r'\d{1,3}(?:,\d{3})*',    # Large numbers with commas
            r'Product [A-Z]',         # Product names
            r'(North|South|East|West)', # Regions
        ]
    
    def evaluate_response(
        self,
        question: str,
        predicted_answer: str,
        reference_answer: str = None,
        sources_used: List[str] = None,
        response_time: float = 0.0,
        context_documents: List[Document] = None
    ) -> ImprovedEvaluationResult:
        """
        Comprehensive evaluation of a business intelligence response.
        """
        
        # If no reference answer, use a more lenient evaluation
        if not reference_answer:
            reference_answer = "No reference available"
        
        # Calculate individual metrics
        semantic_similarity = self._calculate_semantic_similarity(predicted_answer, reference_answer)
        keyword_coverage = self._calculate_keyword_coverage(predicted_answer, reference_answer)
        business_relevance = self._calculate_business_relevance(question, predicted_answer)
        factual_accuracy = self._calculate_factual_accuracy(predicted_answer, reference_answer)
        
        # Calculate overall scores (weighted combination)
        accuracy_score = self._calculate_overall_accuracy(
            semantic_similarity, keyword_coverage, factual_accuracy
        )
        
        relevance_score = self._calculate_overall_relevance(
            business_relevance, keyword_coverage
        )
        
        completeness_score = self._calculate_completeness_score(
            predicted_answer, reference_answer
        )
        
        # Generate evaluation notes
        evaluation_notes = self._generate_evaluation_notes(
            semantic_similarity, keyword_coverage, business_relevance, factual_accuracy
        )
        
        return ImprovedEvaluationResult(
            timestamp=datetime.now(),
            question=question,
            predicted_answer=predicted_answer,
            reference_answer=reference_answer,
            semantic_similarity=semantic_similarity,
            keyword_coverage=keyword_coverage,
            business_relevance=business_relevance,
            factual_accuracy=factual_accuracy,
            accuracy_score=accuracy_score,
            relevance_score=relevance_score,
            completeness_score=completeness_score,
            response_time=response_time,
            sources_used=sources_used or [],
            evaluation_notes=evaluation_notes
        )
    
    def _calculate_semantic_similarity(self, predicted: str, reference: str) -> float:
        """Calculate semantic similarity using improved word overlap."""
        
        if not reference or reference == "No reference available":
            # If no reference, score based on content quality
            return self._score_content_quality(predicted)
        
        # Clean and tokenize
        pred_words = set(self._clean_text(predicted).split())
        ref_words = set(self._clean_text(reference).split())
        
        if not ref_words:
            return 0.5
        
        # Calculate Jaccard similarity with business term boost
        intersection = pred_words.intersection(ref_words)
        union = pred_words.union(ref_words)
        
        jaccard = len(intersection) / len(union) if union else 0
        
        # Boost for business terms
        business_intersection = intersection.intersection(self.bi_keywords)
        business_boost = min(0.3, len(business_intersection) * 0.1)
        
        return min(1.0, jaccard + business_boost)
    
    def _calculate_keyword_coverage(self, predicted: str, reference: str) -> float:
        """Calculate how well predicted answer covers key concepts from reference."""
        
        if not reference or reference == "No reference available":
            # Score based on presence of business keywords
            pred_words = set(self._clean_text(predicted).split())
            business_matches = len(pred_words.intersection(self.bi_keywords))
            return min(1.0, business_matches * 0.15)
        
        # Extract key terms from reference (numbers, business terms, etc.)
        ref_key_terms = self._extract_key_terms(reference)
        pred_key_terms = self._extract_key_terms(predicted)
        
        if not ref_key_terms:
            return 0.7  # Default score if no key terms in reference
        
        # Calculate coverage
        covered_terms = len(set(ref_key_terms).intersection(set(pred_key_terms)))
        coverage = covered_terms / len(ref_key_terms)
        
        return min(1.0, coverage)
    
    def _calculate_business_relevance(self, question: str, predicted: str) -> float:
        """Calculate how relevant the answer is to business intelligence."""
        
        question_words = set(self._clean_text(question).split())
        answer_words = set(self._clean_text(predicted).split())
        
        # Check for business intelligence relevance
        bi_in_question = len(question_words.intersection(self.bi_keywords))
        bi_in_answer = len(answer_words.intersection(self.bi_keywords))
        
        # Question-answer alignment
        overlap = len(question_words.intersection(answer_words))
        alignment = overlap / len(question_words) if question_words else 0
        
        # Combine metrics
        bi_relevance = min(1.0, (bi_in_question + bi_in_answer) * 0.1)
        
        return min(1.0, alignment + bi_relevance)
    
    def _calculate_factual_accuracy(self, predicted: str, reference: str) -> float:
        """Calculate factual accuracy based on numbers and facts."""
        
        if not reference or reference == "No reference available":
            # Score based on presence of factual information
            return self._score_factual_content(predicted)
        
        # Extract factual information using patterns
        pred_facts = []
        ref_facts = []
        
        for pattern in self.factual_patterns:
            pred_facts.extend(re.findall(pattern, predicted))
            ref_facts.extend(re.findall(pattern, reference))
        
        if not ref_facts:
            return 0.8  # Default score if no facts in reference
        
        # Check how many reference facts appear in prediction
        fact_matches = sum(1 for fact in ref_facts if fact in predicted)
        accuracy = fact_matches / len(ref_facts) if ref_facts else 0.5
        
        return min(1.0, accuracy)
    
    def _calculate_overall_accuracy(self, semantic_sim: float, keyword_cov: float, factual_acc: float) -> float:
        """Calculate weighted overall accuracy score."""
        
        # Weighted combination
        weights = {
            'semantic': 0.4,
            'keyword': 0.3,
            'factual': 0.3
        }
        
        overall = (
            semantic_sim * weights['semantic'] +
            keyword_cov * weights['keyword'] +
            factual_acc * weights['factual']
        )
        
        return min(1.0, overall)
    
    def _calculate_overall_relevance(self, business_rel: float, keyword_cov: float) -> float:
        """Calculate overall relevance score."""
        
        return min(1.0, (business_rel * 0.6) + (keyword_cov * 0.4))
    
    def _calculate_completeness_score(self, predicted: str, reference: str) -> float:
        """Calculate completeness based on information coverage."""
        
        pred_length = len(predicted.split())
        
        if not reference or reference == "No reference available":
            # Score based on response length and detail
            if pred_length < 10:
                return 0.3
            elif pred_length < 50:
                return 0.6
            elif pred_length < 150:
                return 0.9
            else:
                return 0.8  # Slight penalty for very long responses
        
        ref_length = len(reference.split())
        
        if ref_length == 0:
            return 0.5
        
        # Ideal range is 80-200% of reference length
        length_ratio = pred_length / ref_length
        
        if 0.8 <= length_ratio <= 2.0:
            return 1.0
        elif length_ratio < 0.8:
            return length_ratio / 0.8
        else:
            return max(0.4, 2.0 / length_ratio)
    
    def _clean_text(self, text: str) -> str:
        """Clean text for analysis."""
        
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())  # Normalize whitespace
        
        return text
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms (numbers, business terms, etc.) from text."""
        
        key_terms = []
        
        # Extract factual information
        for pattern in self.factual_patterns:
            key_terms.extend(re.findall(pattern, text))
        
        # Extract business keywords
        words = set(self._clean_text(text).split())
        key_terms.extend(list(words.intersection(self.bi_keywords)))
        
        return key_terms
    
    def _score_content_quality(self, text: str) -> float:
        """Score content quality when no reference is available."""
        
        # Check for various quality indicators
        word_count = len(text.split())
        business_terms = len(set(self._clean_text(text).split()).intersection(self.bi_keywords))
        factual_elements = sum(len(re.findall(pattern, text)) for pattern in self.factual_patterns)
        
        # Base score from length
        length_score = min(1.0, word_count / 50)  # Normalize to 50 words
        
        # Business relevance score
        business_score = min(0.5, business_terms * 0.1)
        
        # Factual content score
        factual_score = min(0.3, factual_elements * 0.1)
        
        return min(1.0, length_score + business_score + factual_score)
    
    def _score_factual_content(self, text: str) -> float:
        """Score factual content when no reference is available."""
        
        factual_elements = sum(len(re.findall(pattern, text)) for pattern in self.factual_patterns)
        
        # Score based on number of factual elements
        if factual_elements >= 3:
            return 0.9
        elif factual_elements >= 2:
            return 0.7
        elif factual_elements >= 1:
            return 0.5
        else:
            return 0.3
    
    def _generate_evaluation_notes(self, semantic_sim: float, keyword_cov: float, 
                                 business_rel: float, factual_acc: float) -> str:
        """Generate human-readable evaluation notes."""
        
        notes = []
        
        # Semantic similarity
        if semantic_sim >= 0.8:
            notes.append("High semantic similarity to reference")
        elif semantic_sim >= 0.6:
            notes.append("Good semantic alignment")
        else:
            notes.append("Limited semantic similarity")
        
        # Keyword coverage
        if keyword_cov >= 0.8:
            notes.append("Excellent keyword coverage")
        elif keyword_cov >= 0.6:
            notes.append("Good keyword coverage")
        else:
            notes.append("Limited keyword coverage")
        
        # Business relevance
        if business_rel >= 0.8:
            notes.append("Highly relevant to business context")
        elif business_rel >= 0.6:
            notes.append("Good business relevance")
        else:
            notes.append("Limited business context")
        
        # Factual accuracy
        if factual_acc >= 0.8:
            notes.append("Strong factual accuracy")
        elif factual_acc >= 0.6:
            notes.append("Adequate factual content")
        else:
            notes.append("Limited factual accuracy")
        
        return "; ".join(notes)


# Helper function to replace the original evaluator
def create_improved_evaluator() -> ImprovedBIEvaluator:
    """Create an improved evaluator instance."""
    return ImprovedBIEvaluator()