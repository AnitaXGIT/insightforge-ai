
# loaders/data_loader.py

import pandas as pd
import numpy as np
import os


def generate_advanced_data_summary(df: pd.DataFrame) -> str:
    """
    Given a sales DataFrame, calculate business metrics and return a detailed summary as a string.
    """

    # Convert the 'Date' column to a datetime object and extract the month
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M')

    # Key sales metrics
    total_sales = df['Sales'].sum()
    avg_sale    = df['Sales'].mean()
    median_sale = df['Sales'].median()
    sales_std   = df['Sales'].std()

    # Monthly performance
    monthly_sales = df.groupby('Month', observed=False)['Sales'].sum()
    best_month, worst_month = monthly_sales.idxmax(), monthly_sales.idxmin()

    # Product analysis
    product_stats = df.groupby('Product', observed=False)['Sales'].agg(['sum','count'])
    top_product        = product_stats['sum'].idxmax()
    most_freq_product  = product_stats['count'].idxmax()

    # Regional performance
    region_sales = df.groupby('Region', observed=False)['Sales'].sum()
    best_region, worst_region = region_sales.idxmax(), region_sales.idxmin()

    # Customer demographics
    age_bins   = [0,25,35,45,55,100]
    age_labels = ['18-25','26-35','36-45','46-55','55+']
    df['Age_Group'] = pd.cut(df['Customer_Age'], bins=age_bins, labels=age_labels, right=False)

    best_age_group = df.groupby('Age_Group', observed=False)['Sales'].mean().idxmax()

    gender_sales = df.groupby('Customer_Gender', observed=False)['Sales'].mean()

    # Create formatted summary text
    summary = f'''
Advanced Sales Data Summary:

– Total sales : ${total_sales:,.2f}
– Average / Median sale : ${avg_sale:.2f} / ${median_sale:.2f}
– (Sales) : ${sales_std:.2f}

Time window
  – Best month : {best_month}  Worst month : {worst_month}

Product
  – Highest revenue : {top_product}  Most frequently sold : {most_freq_product}

Regions
  – Best : {best_region}  Worst : {worst_region}

Customers
  – Avg satisfaction : {df.Customer_Satisfaction.mean():.2f} (±{df.Customer_Satisfaction.std():.2f})
  – Best age group : {best_age_group}
  – Avg sale by gender : Male ${gender_sales.get("Male", 0):.2f} / Female ${gender_sales.get("Female", 0):.2f}
'''
    return summary


def load_and_process_sales_data(
    csv_path: str = "../data/sales_data.csv",
    summary_output_path: str = "../data/sales_summary.txt"
) -> str:
    """
    Load the sales data CSV, generate summary insights, save them to a text file,
    and return the summary string for later use (LLM, RAG, etc.).
    """



def load_and_process_sales_data(
    csv_path: str = None,
    summary_output_path: str = None
)-> str:
    """
    Load the sales data CSV, generate summary insights, save them to a text file,
    and return the summary string for later use (LLM, RAG, etc.).
    """
    if csv_path is None:
        # Get the directory where this script is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up to project root, then into data folder
        csv_path = os.path.join(os.path.dirname(current_dir), "data", "sales_data.csv")
    
    if summary_output_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        summary_output_path = os.path.join(os.path.dirname(current_dir), "data", "sales_summary.txt")

    
    # Step 1: Load data
    df = pd.read_csv(csv_path)

    # Step 2: Generate the summary
    summary_text = generate_advanced_data_summary(df)

    # Step 3: Save the summary to a text file
    os.makedirs(os.path.dirname(summary_output_path), exist_ok=True)
    with open(summary_output_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    # Step 4: Return summary for use in the app
    return summary_text