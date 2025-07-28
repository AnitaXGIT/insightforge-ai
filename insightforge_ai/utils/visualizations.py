# utils/visualizations.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json


class InsightForgeVisualizer:
    """
    Comprehensive data visualization system for InsightForge AI.
    Creates interactive charts and dashboards for business intelligence insights.
    """
    
    def __init__(self, save_plots: bool = True, output_directory: str = None):
        """
        Initialize the visualization system.
        
        Args:
            save_plots: Whether to save generated plots
            output_directory: Directory to save plot files
        """
        self.save_plots = save_plots
        self.output_directory = output_directory or self._get_default_output_dir()
        
        # Set styling for consistent look
        self._setup_styling()
        
        # Color palettes for different chart types
        self.color_palettes = {
            'primary': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            'business': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592941'],
            'performance': ['#00A86B', '#FFD700', '#FF6347', '#4169E1', '#9370DB'],
            'gradient': px.colors.sequential.Viridis
        }
        
        print(f"📊 InsightForge Visualizer initialized")
        print(f"💾 Plots will be saved to: {self.output_directory}")
    
    def _get_default_output_dir(self) -> str:
        """Get default output directory for visualizations."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(os.path.dirname(current_dir), "data", "visualizations")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def _setup_styling(self):
        """Setup consistent styling for visualizations."""
        # Matplotlib styling
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Plotly default template
        import plotly.io as pio
        pio.templates.default = "plotly_white"
    
    def create_sales_trends_chart(self, sales_data: pd.DataFrame, chart_type: str = "interactive") -> str:
        """
        Create sales trends over time visualization.
        
        Args:
            sales_data: DataFrame with Date and Sales columns
            chart_type: 'static' or 'interactive'
            
        Returns:
            str: Path to saved chart file
        """
        print("📈 Creating sales trends chart...")
        
        # Prepare data
        if 'Date' in sales_data.columns:
            sales_data['Date'] = pd.to_datetime(sales_data['Date'])
            monthly_sales = sales_data.groupby(sales_data['Date'].dt.to_period('M'))['Sales'].sum().reset_index()
            monthly_sales['Date'] = monthly_sales['Date'].dt.to_timestamp()
        else:
            # Create mock monthly data if Date column not available
            monthly_sales = self._create_mock_monthly_data(sales_data)
        
        if chart_type == "interactive":
            return self._create_interactive_sales_trends(monthly_sales)
        else:
            return self._create_static_sales_trends(monthly_sales)
    
    def _create_mock_satisfaction_data(self) -> pd.DataFrame:
        """Create mock customer satisfaction data."""
        satisfaction_levels = ['Low', 'Below Avg', 'Average', 'Good', 'Excellent']
        sales_sum = [8000, 15000, 35000, 45000, 22000]
        sales_mean = [65, 75, 85, 95, 110]
        count = [123, 200, 412, 474, 200]
        
        return pd.DataFrame({
            'Customer_Satisfaction': satisfaction_levels,
            'sum': sales_sum,
            'mean': sales_mean,
            'count': count
        })
    
    def create_evaluation_performance_chart(self, evaluation_results: List[Any]) -> str:
        """
        Create evaluation performance visualization.
        
        Args:
            evaluation_results: List of evaluation results
            
        Returns:
            str: Path to saved chart file
        """
        print("📊 Creating evaluation performance chart...")
        
        # Extract metrics from evaluation results
        if not evaluation_results:
            print("⚠️ No evaluation results provided")
            return ""
        
        metrics_data = {
            'Question': [f"Q{i+1}" for i in range(len(evaluation_results))],
            'Accuracy': [r.accuracy_score for r in evaluation_results],
            'Relevance': [r.relevance_score for r in evaluation_results],
            'Completeness': [r.completeness_score for r in evaluation_results],
            'Response_Time': [r.response_time for r in evaluation_results]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Create comprehensive evaluation dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Quality Metrics by Question', 'Average Performance', 'Response Time Analysis', 'Performance Distribution'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "histogram"}]]
        )
        
        colors = self.color_palettes['performance']
        
        # Quality metrics by question
        fig.add_trace(
            go.Scatter(x=metrics_df['Question'], y=metrics_df['Accuracy'], 
                      mode='lines+markers', name='Accuracy', line=dict(color=colors[0])),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=metrics_df['Question'], y=metrics_df['Relevance'], 
                      mode='lines+markers', name='Relevance', line=dict(color=colors[1])),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=metrics_df['Question'], y=metrics_df['Completeness'], 
                      mode='lines+markers', name='Completeness', line=dict(color=colors[2])),
            row=1, col=1
        )
        
        # Average performance
        avg_metrics = ['Accuracy', 'Relevance', 'Completeness']
        avg_values = [metrics_df[metric].mean() for metric in avg_metrics]
        
        fig.add_trace(
            go.Bar(x=avg_metrics, y=avg_values, name='Average Performance',
                   marker_color=colors[:3],
                   hovertemplate='<b>%{x}</b><br>Score: %{y:.3f}<extra></extra>'),
            row=1, col=2
        )
        
        # Response time analysis
        fig.add_trace(
            go.Scatter(x=metrics_df['Question'], y=metrics_df['Response_Time'], 
                      mode='lines+markers', name='Response Time',
                      line=dict(color=colors[3]),
                      hovertemplate='<b>%{x}</b><br>Time: %{y:.2f}s<extra></extra>'),
            row=2, col=1
        )
        
        # Performance distribution
        all_scores = (metrics_df['Accuracy'].tolist() + 
                     metrics_df['Relevance'].tolist() + 
                     metrics_df['Completeness'].tolist())
        
        fig.add_trace(
            go.Histogram(x=all_scores, name='Score Distribution',
                        marker_color=colors[4], opacity=0.7,
                        hovertemplate='Score Range: %{x}<br>Count: %{y}<extra></extra>'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="InsightForge AI - Evaluation Performance Dashboard",
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        # Update y-axes
        fig.update_yaxes(range=[0, 1], title_text="Score", row=1, col=1)
        fig.update_yaxes(range=[0, 1], title_text="Score", row=1, col=2)
        fig.update_yaxes(title_text="Time (seconds)", row=2, col=1)
        fig.update_xaxes(title_text="Score", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        
        # Save chart
        filename = f"evaluation_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = os.path.join(self.output_directory, filename)
        fig.write_html(filepath)
        
        print(f"✅ Evaluation performance chart saved: {filepath}")
        return filepath
    
    def generate_insights_summary(self, sales_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a summary of key insights from the data.
        
        Args:
            sales_data: Sales dataset
            
        Returns:
            Dict containing key insights and metrics
        """
        insights = {
            'data_overview': {
                'total_records': len(sales_data),
                'date_range': self._get_date_range(sales_data),
                'data_quality': self._assess_data_quality(sales_data)
            },
            'sales_insights': {},
            'customer_insights': {},
            'product_insights': {},
            'regional_insights': {}
        }
        
        # Sales insights
        if 'Sales' in sales_data.columns:
            insights['sales_insights'] = {
                'total_sales': f"${sales_data['Sales'].sum():,.0f}",
                'average_sale': f"${sales_data['Sales'].mean():.0f}",
                'median_sale': f"${sales_data['Sales'].median():.0f}",
                'sales_std': f"${sales_data['Sales'].std():.0f}",
                'top_sale': f"${sales_data['Sales'].max():,.0f}",
                'growth_trend': self._calculate_growth_trend(sales_data)
            }
        
        # Customer insights
        if 'Customer_Age' in sales_data.columns:
            insights['customer_insights'] = {
                'average_age': f"{sales_data['Customer_Age'].mean():.1f} years",
                'age_range': f"{sales_data['Customer_Age'].min()}-{sales_data['Customer_Age'].max()} years",
                'most_common_age_group': self._get_most_common_age_group(sales_data)
            }
        
        if 'Customer_Gender' in sales_data.columns:
            gender_dist = sales_data['Customer_Gender'].value_counts(normalize=True)
            insights['customer_insights']['gender_distribution'] = {
                gender: f"{pct:.1%}" for gender, pct in gender_dist.items()
            }
        
        # Product insights
        if 'Product' in sales_data.columns:
            product_sales = sales_data.groupby('Product')['Sales'].sum().sort_values(ascending=False)
            insights['product_insights'] = {
                'top_product': product_sales.index[0],
                'top_product_sales': f"${product_sales.iloc[0]:,.0f}",
                'product_count': len(product_sales),
                'performance_gap': f"{((product_sales.iloc[0] - product_sales.iloc[-1]) / product_sales.iloc[-1] * 100):.1f}%"
            }
        
        # Regional insights
        if 'Region' in sales_data.columns:
            regional_sales = sales_data.groupby('Region')['Sales'].sum().sort_values(ascending=False)
            insights['regional_insights'] = {
                'top_region': regional_sales.index[0],
                'top_region_sales': f"${regional_sales.iloc[0]:,.0f}",
                'region_count': len(regional_sales),
                'regional_concentration': f"{(regional_sales.iloc[0] / regional_sales.sum() * 100):.1f}%"
            }
        
        return insights
    
    def _get_date_range(self, sales_data: pd.DataFrame) -> str:
        """Get date range from sales data."""
        if 'Date' in sales_data.columns:
            try:
                dates = pd.to_datetime(sales_data['Date'])
                return f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}"
            except:
                pass
        return "Date range not available"
    
    def _assess_data_quality(self, sales_data: pd.DataFrame) -> str:
        """Assess data quality."""
        total_cells = sales_data.shape[0] * sales_data.shape[1]
        missing_cells = sales_data.isnull().sum().sum()
        quality_score = (total_cells - missing_cells) / total_cells
        
        if quality_score >= 0.95:
            return "Excellent"
        elif quality_score >= 0.85:
            return "Good"
        elif quality_score >= 0.70:
            return "Fair"
        else:
            return "Needs Improvement"
    
    def _calculate_growth_trend(self, sales_data: pd.DataFrame) -> str:
        """Calculate growth trend."""
        if 'Date' in sales_data.columns and len(sales_data) > 1:
            try:
                sales_data['Date'] = pd.to_datetime(sales_data['Date'])
                monthly_sales = sales_data.groupby(sales_data['Date'].dt.to_period('M'))['Sales'].sum()
                if len(monthly_sales) >= 2:
                    growth_rate = (monthly_sales.iloc[-1] - monthly_sales.iloc[0]) / monthly_sales.iloc[0]
                    if growth_rate > 0.1:
                        return "Strong Growth"
                    elif growth_rate > 0:
                        return "Moderate Growth"
                    elif growth_rate > -0.1:
                        return "Stable"
                    else:
                        return "Declining"
            except:
                pass
        return "Trend unavailable"
    
    def _get_most_common_age_group(self, sales_data: pd.DataFrame) -> str:
        """Get most common age group."""
        try:
            age_groups = pd.cut(sales_data['Customer_Age'], 
                              bins=[0, 25, 35, 45, 55, 100], 
                              labels=['18-25', '26-35', '36-45', '46-55', '55+'])
            return age_groups.value_counts().index[0]
        except:
            return "Age group analysis unavailable"
    
    def export_visualizations_report(self, sales_data: pd.DataFrame) -> str:
        """
        Export a comprehensive visualizations report.
        
        Args:
            sales_data: Sales dataset
            
        Returns:
            str: Path to exported report
        """
        print("📋 Generating comprehensive visualizations report...")
        
        # Generate all visualizations
        charts_created = []
        
        try:
            # Create all chart types
            sales_chart = self.create_sales_trends_chart(sales_data, "interactive")
            charts_created.append(("Sales Trends", sales_chart))
            
            product_chart = self.create_product_performance_chart(sales_data, "interactive")
            charts_created.append(("Product Performance", product_chart))
            
            regional_chart = self.create_regional_analysis_chart(sales_data, "interactive")
            charts_created.append(("Regional Analysis", regional_chart))
            
            demographics_chart = self.create_customer_demographics_chart(sales_data, "interactive")
            charts_created.append(("Customer Demographics", demographics_chart))
            
            dashboard = self.create_comprehensive_dashboard(sales_data)
            charts_created.append(("Comprehensive Dashboard", dashboard))
            
            # Generate insights summary
            insights = self.generate_insights_summary(sales_data)
            
            # Create HTML report
            report_html = self._create_html_report(charts_created, insights)
            
            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = os.path.join(self.output_directory, f"visualization_report_{timestamp}.html")
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_html)
            
            print(f"✅ Comprehensive visualization report saved: {report_file}")
            return report_file
            
        except Exception as e:
            print(f"❌ Error creating visualization report: {e}")
            return ""
    
    def _create_html_report(self, charts: List[Tuple[str, str]], insights: Dict[str, Any]) -> str:
        """Create HTML report with all visualizations and insights."""
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>InsightForge AI - Visualization Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header {{ text-align: center; background: linear-gradient(135deg, #2E86AB, #A23B72); color: white; padding: 20px; border-radius: 10px; }}
        .section {{ background: white; margin: 20px 0; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .insights-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .insight-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #2E86AB; }}
        .chart-link {{ display: inline-block; background: #2E86AB; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin: 5px; }}
        .chart-link:hover {{ background: #A23B72; }}
        .metric {{ font-size: 1.2em; font-weight: bold; color: #2E86AB; }}
        .timestamp {{ text-align: center; color: #666; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 InsightForge AI</h1>
        <h2>Comprehensive Business Intelligence Report</h2>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>📊 Executive Summary</h2>
        <div class="insights-grid">
            <div class="insight-card">
                <h3>📈 Sales Overview</h3>
                <p><span class="metric">Total Sales:</span> {insights.get('sales_insights', {}).get('total_sales', 'N/A')}</p>
                <p><span class="metric">Average Sale:</span> {insights.get('sales_insights', {}).get('average_sale', 'N/A')}</p>
                <p><span class="metric">Growth Trend:</span> {insights.get('sales_insights', {}).get('growth_trend', 'N/A')}</p>
            </div>
            <div class="insight-card">
                <h3>👥 Customer Insights</h3>
                <p><span class="metric">Average Age:</span> {insights.get('customer_insights', {}).get('average_age', 'N/A')}</p>
                <p><span class="metric">Gender Split:</span> {self._format_gender_distribution(insights.get('customer_insights', {}))}</p>
                <p><span class="metric">Top Age Group:</span> {insights.get('customer_insights', {}).get('most_common_age_group', 'N/A')}</p>
            </div>
            <div class="insight-card">
                <h3>📦 Product Performance</h3>
                <p><span class="metric">Top Product:</span> {insights.get('product_insights', {}).get('top_product', 'N/A')}</p>
                <p><span class="metric">Top Sales:</span> {insights.get('product_insights', {}).get('top_product_sales', 'N/A')}</p>
                <p><span class="metric">Performance Gap:</span> {insights.get('product_insights', {}).get('performance_gap', 'N/A')}</p>
            </div>
            <div class="insight-card">
                <h3>🗺️ Regional Analysis</h3>
                <p><span class="metric">Top Region:</span> {insights.get('regional_insights', {}).get('top_region', 'N/A')}</p>
                <p><span class="metric">Top Region Sales:</span> {insights.get('regional_insights', {}).get('top_region_sales', 'N/A')}</p>
                <p><span class="metric">Market Share:</span> {insights.get('regional_insights', {}).get('regional_concentration', 'N/A')}</p>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>📈 Interactive Visualizations</h2>
        <p>Click on the links below to view detailed interactive charts:</p>
        <div>
"""
        
        # Add chart links
        for chart_name, chart_path in charts:
            chart_filename = os.path.basename(chart_path)
            html_template += f'<a href="{chart_filename}" class="chart-link">{chart_name}</a>\n'
        
        html_template += f"""
        </div>
    </div>
    
    <div class="section">
        <h2>📋 Data Quality Assessment</h2>
        <div class="insight-card">
            <p><span class="metric">Total Records:</span> {insights.get('data_overview', {}).get('total_records', 'N/A'):,}</p>
            <p><span class="metric">Date Range:</span> {insights.get('data_overview', {}).get('date_range', 'N/A')}</p>
            <p><span class="metric">Data Quality:</span> {insights.get('data_overview', {}).get('data_quality', 'N/A')}</p>
        </div>
    </div>
    
    <div class="timestamp">
        <p>Report generated by InsightForge AI Business Intelligence System</p>
        <p>© 2024 InsightForge AI - Advanced Analytics & Visualization Platform</p>
    </div>
</body>
</html>
"""
        
        return html_template
    
    def _format_gender_distribution(self, customer_insights: Dict[str, Any]) -> str:
        """Format gender distribution for display."""
        gender_dist = customer_insights.get('gender_distribution', {})
        if gender_dist:
            return ", ".join([f"{gender}: {pct}" for gender, pct in gender_dist.items()])
        return "N/A"