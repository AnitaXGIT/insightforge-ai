# app/prompt_engineering.py

from typing import Dict, List, Any, Optional
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.schema import Document
import re


class InsightForgePromptEngineering:
    """
    Advanced prompt engineering system for InsightForge AI.
    Creates specialized, context-aware prompts for different types of business intelligence queries.
    """
    
    def __init__(self):
        """Initialize the prompt engineering system with specialized templates."""
        self.prompt_templates = {}
        self.chain_prompts = {}
        self.few_shot_examples = {}
        
        # Initialize all prompt templates
        self._initialize_base_prompts()
        self._initialize_analysis_prompts()
        self._initialize_chain_prompts()
        self._initialize_few_shot_examples()
    

    def _initialize_base_prompts(self):
        """Initialize base prompt templates for different query types."""
        
        # General Business Intelligence Prompt
        self.prompt_templates['general_bi'] = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are InsightForge AI, an expert Business Intelligence consultant with deep expertise in data analysis and strategic insights.

Context Information:
{context}

Analysis Guidelines:
- Provide data-driven insights based solely on the provided context
- Use specific numbers, percentages, and metrics when available
- Identify trends, patterns, and key performance indicators
- Offer actionable business recommendations
- Structure your response with clear headings when appropriate
- Always cite which source your information comes from (sales data, documents, etc.)


Question: {question}

Business Intelligence Analysis:"""
        )
        
        # Sales Performance Specific Prompt
        self.prompt_templates['sales_performance'] = PromptTemplate(
            input_variables=["context", "question", "time_period"],
            template="""You are a Sales Performance Analyst for InsightForge AI. Focus on sales metrics, trends, and performance optimization.

            
Sales Data Context:
{context}

Time Period: {time_period}

Sales Analysis Framework:
1. **Key Metrics**: Total sales, average sale size, growth rates
2. **Performance Trends**: Month-over-month, seasonal patterns
3. **Top Performers**: Best products, regions, customer segments
4. **Opportunities**: Underperforming areas with growth potential
5. **Recommendations**: Specific, actionable next steps


Question: {question}

Sales Performance Insights:"""
        )
        
        # Product and Regional Analysis Prompt
        self.prompt_templates['product_regional'] = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a Product & Regional Performance Specialist for InsightForge AI. Analyze product performance across different geographic regions.

Product & Regional Data:
{context}

Analysis Focus Areas:
• Product Performance: Revenue, volume, market share by product
• Regional Analysis: Geographic distribution, regional preferences
• Cross-Analysis: Which products perform best in which regions
• Market Opportunities: Underperforming product-region combinations
• Strategic Recommendations: Product positioning and regional expansion


Question: {question}

Product & Regional Analysis:"""
        )
        
        # Customer Segmentation Prompt
        self.prompt_templates['customer_segmentation'] = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a Customer Analytics Specialist for InsightForge AI. Focus on customer behavior, demographics, and segmentation insights.

Customer Data Context:
{context}

Customer Analysis Framework:
🎯 **Demographics**: Age groups, gender, geographic distribution
📊 **Behavior Patterns**: Purchase frequency, average order value, preferences
💰 **Value Segments**: High-value vs. low-value customers
📈 **Trends**: Customer acquisition, retention, satisfaction scores
🔍 **Insights**: Customer lifetime value, segment-specific opportunities


Question: {question}

Customer Segmentation Analysis:"""
        )
        
        # Statistical Analysis Prompt
        self.prompt_templates['statistical_analysis'] = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a Data Science Analyst for InsightForge AI. Provide statistical insights and quantitative analysis.

            
Statistical Data Context:
{context}

Statistical Analysis Approach:
📐 **Descriptive Statistics**: Mean, median, standard deviation, ranges
📊 **Distribution Analysis**: Data spread, outliers, patterns
📈 **Trend Analysis**: Correlations, relationships, statistical significance
🔢 **Key Metrics**: Ratios, percentages, growth rates
📋 **Conclusions**: Data-driven insights with statistical backing


Question: {question}

Statistical Analysis Report:"""
        )
    
    def _initialize_analysis_prompts(self):
        """Initialize specialized analysis prompts for complex scenarios."""
        
        # Comparative Analysis Prompt
        self.prompt_templates['comparative_analysis'] = PromptTemplate(
            input_variables=["context", "question", "comparison_type"],
            template="""You are performing {comparison_type} analysis for InsightForge AI.
         
Comparative Data:
{context}

Comparison Framework:
🔄 **Side-by-Side Analysis**: Direct comparisons with specific metrics
📊 **Performance Gaps**: What's working vs. what needs improvement
💡 **Best Practices**: Lessons from top performers
⚠️ **Risk Areas**: Underperforming segments requiring attention
🎯 **Strategic Actions**: Specific recommendations to close gaps


Question: {question}

Comparative Analysis Results:"""
        )
        
        # Trend Forecasting Prompt
        self.prompt_templates['trend_forecasting'] = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a Business Forecasting Analyst for InsightForge AI. Focus on identifying trends and making data-driven predictions.

Historical Trend Data:
{context}

Forecasting Framework:
📈 **Historical Patterns**: What trends are evident in the data
🔮 **Trend Projection**: Where current patterns are heading
⚡ **Influencing Factors**: What's driving these trends
🎯 **Predictions**: Likely future scenarios based on data
⚠️ **Risk Factors**: What could disrupt these trends
📋 **Recommendations**: How to capitalize on or mitigate trends


Question: {question}

Trend Forecasting Analysis:"""
        )
        
        # Problem Diagnosis Prompt
        self.prompt_templates['problem_diagnosis'] = PromptTemplate(
            input_variables=["context", "question", "problem_area"],
            template="""You are a Business Problem Analyst for InsightForge AI. Diagnose issues in {problem_area} and provide solutions.

Problem Context Data:
{context}

Diagnostic Framework:
🔍 **Problem Identification**: What exactly is the issue
📊 **Root Cause Analysis**: Why is this happening (data-driven)
📈 **Impact Assessment**: How significant is this problem
💡 **Solution Options**: Multiple approaches to address the issue
🎯 **Recommended Action**: Best solution with implementation steps
📋 **Success Metrics**: How to measure improvement


Question: {question}

Problem Diagnosis & Solution:"""
        )
    
    def _initialize_chain_prompts(self):
        """Initialize chain prompts for multi-step analysis."""
        
        # Multi-Step Analysis Chain
        self.chain_prompts['multi_step_analysis'] = {
            'step1': PromptTemplate(
                input_variables=["context", "question"],
                template="""Step 1 - Data Overview: Based on this context, what are the key data points and metrics available?

Context: {context}


Question: {question}

Provide a structured overview of available data."""
            ),
            'step2': PromptTemplate(
                input_variables=["context", "step1_result", "question"],
                template="""Step 2 - Pattern Analysis: Based on the data overview and context, what patterns and trends do you identify?

Previous Analysis: {step1_result}
Context: {context}


Question: {question}

Identify key patterns, trends, and relationships."""
            ),
            'step3': PromptTemplate(
                input_variables=["context", "step1_result", "step2_result", "question"],
                template="""Step 3 - Strategic Insights: Based on all previous analysis, what are the strategic insights and recommendations?

Data Overview: {step1_result}
Pattern Analysis: {step2_result}
Context: {context}


Question: {question}

Provide final strategic insights and actionable recommendations."""
            )
        }
        
        # Deep Dive Analysis Chain
        self.chain_prompts['deep_dive'] = {
            'explore': PromptTemplate(
                input_variables=["context", "focus_area"],
                template="""Deep Dive Exploration: Thoroughly analyze {focus_area} from this data.

Data: {context}

Provide comprehensive analysis of {focus_area} including metrics, trends, and insights."""
            ),
            'correlate': PromptTemplate(
                input_variables=["context", "exploration_result", "focus_area"],
                template="""Correlation Analysis: How does {focus_area} relate to other business metrics?

Exploration Results: {exploration_result}
Full Context: {context}

Identify correlations and interdependencies."""
            ),
            'recommend': PromptTemplate(
                input_variables=["exploration_result", "correlation_result", "focus_area"],
                template="""Strategic Recommendations: Based on the deep dive analysis of {focus_area}, what actions should be taken?

Exploration: {exploration_result}
Correlations: {correlation_result}

Provide specific, actionable recommendations."""
            )
        }

    
    def _initialize_few_shot_examples(self):
        """Initialize few-shot learning examples for consistent responses."""
        
        self.few_shot_examples['sales_analysis'] = [
            {
                "question": "What are our total sales figures?",
                "context": "Total sales: $125,000, Average sale: $85, Best month: March",
                "answer": "**Sales Performance Summary:**\n\n💰 **Total Sales**: $125,000\n📊 **Average Sale Size**: $85\n📈 **Peak Performance**: March showed the strongest sales\n\n**Key Insight**: Your business achieved solid revenue with consistent average transaction values. March's strong performance suggests seasonal opportunities worth investigating."
            },
            {
                "question": "Which products perform best?",
                "context": "Product A: $45,000 revenue, Product B: $30,000 revenue, Product C: $50,000 revenue",
                "answer": "**Product Performance Ranking:**\n\n🥇 **Top Performer**: Product C ($50,000 - 40% of total revenue)\n🥈 **Strong Contributor**: Product A ($45,000 - 36% of total revenue)\n🥉 **Growth Opportunity**: Product B ($30,000 - 24% of total revenue)\n\n**Strategic Recommendation**: Focus marketing efforts on Product C while analyzing what makes it successful to apply those learnings to Product B."
            }
        ]
    


    def detect_query_type(self, question: str, context: List[Document]) -> str:
        """
        Automatically detect the type of business intelligence query to use appropriate prompt.
        
        Args:
            question: User's question
            context: Retrieved context documents
            
        Returns:
            str: Detected query type
        """
        question_lower = question.lower()
        
        # Sales performance keywords
        if any(keyword in question_lower for keyword in ['sales', 'revenue', 'performance', 'monthly', 'quarterly']):
            return 'sales_performance'
        
        # Product/regional keywords
        elif any(keyword in question_lower for keyword in ['product', 'region', 'geographic', 'area', 'location']):
            return 'product_regional'
        
        # Customer segmentation keywords
        elif any(keyword in question_lower for keyword in ['customer', 'demographic', 'age', 'gender', 'segment']):
            return 'customer_segmentation'
        
        # Statistical analysis keywords
        elif any(keyword in question_lower for keyword in ['average', 'median', 'standard deviation', 'correlation', 'trend']):
            return 'statistical_analysis'
        
        # Comparison keywords
        elif any(keyword in question_lower for keyword in ['compare', 'vs', 'versus', 'difference', 'better', 'worse']):
            return 'comparative_analysis'
        
        # Forecasting keywords
        elif any(keyword in question_lower for keyword in ['forecast', 'predict', 'future', 'trend', 'projection']):
            return 'trend_forecasting'
        
        # Problem-solving keywords
        elif any(keyword in question_lower for keyword in ['problem', 'issue', 'why', 'fix', 'improve', 'solution']):
            return 'problem_diagnosis'
        
        # Default to general BI
        else:
            return 'general_bi'
        

    
    def enhance_context_with_metadata(self, context: List[Document]) -> str:
        """
        Enhance context by organizing it with metadata information.
        
        Args:
            context: List of retrieved documents
            
        Returns:
            str: Enhanced context string
        """
        enhanced_context = []
        
        # Group by source type
        sales_data = []
        pdf_documents = []
        
        for doc in context:
            metadata = doc.metadata
            content = doc.page_content
            
            if metadata.get('source_type') == 'pdf':
                source_file = metadata.get('source_file', 'Unknown PDF')
                pdf_documents.append(f"📄 From {source_file}:\n{content}")
            elif metadata.get('section'):
                section = metadata['section']
                sales_data.append(f"📊 Sales Data ({section}):\n{content}")
            else:
                sales_data.append(f"📊 Sales Summary:\n{content}")
        
        # Organize enhanced context
        if sales_data:
            enhanced_context.append("**SALES DATA INSIGHTS:**")
            enhanced_context.extend(sales_data)
        
        if pdf_documents:
            enhanced_context.append("\n**DOCUMENT INSIGHTS:**")
            enhanced_context.extend(pdf_documents)
        
        return "\n\n".join(enhanced_context)
    


    def get_prompt(self, query_type: str, context: List[Document], question: str, **kwargs) -> PromptTemplate:
        """
        Get the appropriate prompt template for a query type.
        
        Args:
            query_type: Type of query (detected or specified)
            context: Retrieved context documents
            question: User's question
            **kwargs: Additional parameters for specific prompts
            
        Returns:
            PromptTemplate: Configured prompt template
        """
        if query_type not in self.prompt_templates:
            query_type = 'general_bi'
        
        prompt_template = self.prompt_templates[query_type]
        
        # Enhance context with metadata
        enhanced_context = self.enhance_context_with_metadata(context)
        
        # Add any additional variables needed for specific prompts
        prompt_vars = {
            'context': enhanced_context,
            'question': question
        }
        
        # Add specific variables for certain prompt types
        if query_type == 'sales_performance':
            prompt_vars['time_period'] = kwargs.get('time_period', 'Current Period')
        elif query_type == 'comparative_analysis':
            prompt_vars['comparison_type'] = kwargs.get('comparison_type', 'Performance')
        elif query_type == 'problem_diagnosis':
            prompt_vars['problem_area'] = kwargs.get('problem_area', 'Business Operations')
        
        return prompt_template, prompt_vars
    


    def execute_chain_prompt(self, chain_type: str, context: List[Document], question: str, llm) -> Dict[str, str]:
        """
        Execute a multi-step chain prompt analysis.
        
        Args:
            chain_type: Type of chain ('multi_step_analysis' or 'deep_dive')
            context: Retrieved context documents
            question: User's question
            llm: Language model instance
            
        Returns:
            Dict containing results from each step
        """
        if chain_type not in self.chain_prompts:
            raise ValueError(f"Chain type '{chain_type}' not found")
        
        chain = self.chain_prompts[chain_type]
        results = {}
        enhanced_context = self.enhance_context_with_metadata(context)
        
        if chain_type == 'multi_step_analysis':
            # Step 1: Data Overview
            step1_prompt = chain['step1'].format(context=enhanced_context, question=question)
            results['step1'] = llm(step1_prompt)
            
            # Step 2: Pattern Analysis
            step2_prompt = chain['step2'].format(
                context=enhanced_context, 
                step1_result=results['step1'], 
                question=question
            )
            results['step2'] = llm(step2_prompt)
            
            # Step 3: Strategic Insights
            step3_prompt = chain['step3'].format(
                context=enhanced_context,
                step1_result=results['step1'],
                step2_result=results['step2'],
                question=question
            )
            results['step3'] = llm(step3_prompt)
            
        elif chain_type == 'deep_dive':
            focus_area = self._extract_focus_area(question)
            
            # Exploration
            explore_prompt = chain['explore'].format(context=enhanced_context, focus_area=focus_area)
            results['explore'] = llm(explore_prompt)
            
            # Correlation
            correlate_prompt = chain['correlate'].format(
                context=enhanced_context,
                exploration_result=results['explore'],
                focus_area=focus_area
            )
            results['correlate'] = llm(correlate_prompt)
            
            # Recommendations
            recommend_prompt = chain['recommend'].format(
                exploration_result=results['explore'],
                correlation_result=results['correlate'],
                focus_area=focus_area
            )
            results['recommend'] = llm(recommend_prompt)
        
        return results
    


    def _extract_focus_area(self, question: str) -> str:
        """Extract the main focus area from a question for deep dive analysis."""
        question_lower = question.lower()
        
        if 'sales' in question_lower or 'revenue' in question_lower:
            return 'Sales Performance'
        elif 'product' in question_lower:
            return 'Product Analysis'
        elif 'customer' in question_lower:
            return 'Customer Behavior'
        elif 'region' in question_lower or 'geographic' in question_lower:
            return 'Regional Performance'
        else:
            return 'Business Performance'
    


    def get_available_prompt_types(self) -> List[str]:
        """Get list of available prompt types."""
        return list(self.prompt_templates.keys())
    
    def get_available_chain_types(self) -> List[str]:
        """Get list of available chain prompt types."""
        return list(self.chain_prompts.keys())




def create_business_intelligence_prompt(context: List[Document], question: str, query_type: str = None) -> str:
    """
    Convenience function to create a business intelligence prompt.
    
    Args:
        context: Retrieved context documents
        question: User's question
        query_type: Optional specific query type
        
    Returns:
        str: Formatted prompt ready for LLM
    """
    prompt_engine = InsightForgePromptEngineering()
    
    if query_type is None:
        query_type = prompt_engine.detect_query_type(question, context)
    
    prompt_template, prompt_vars = prompt_engine.get_prompt(query_type, context, question)
    return prompt_template.format(**prompt_vars)


