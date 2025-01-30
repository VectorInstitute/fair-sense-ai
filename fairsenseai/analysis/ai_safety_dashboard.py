"""
Tool to visualize AI safety risks and mitigation strategies.
"""

import pandas as pd
import plotly.express as px
from plotly.graph_objs import Figure
from typing import Tuple

ai_safety_risks = [
    {
        "Risk": "Disinformation Spread",
        "Category": "Information Integrity",
        "Percentage": 20,
        "Severity": 8,
        "Likelihood": 7,
        "Impact": "High",
        "Description": "AI-generated content can spread false information rapidly.",
        "Mitigation": "Develop AI tools for fact-checking and verification."
    },
    {
        "Risk": "Algorithmic Bias",
        "Category": "Fairness and Bias",
        "Percentage": 18,
        "Severity": 7,
        "Likelihood": 8,
        "Impact": "High",
        "Description": "AI systems may perpetuate or amplify societal biases.",
        "Mitigation": "Implement fairness-aware algorithms and diverse datasets."
    },
    {
        "Risk": "Privacy Invasion",
        "Category": "Data Privacy",
        "Percentage": 15,
        "Severity": 6,
        "Likelihood": 6,
        "Impact": "Medium",
        "Description": "AI can infer personal information without consent.",
        "Mitigation": "Adopt privacy-preserving techniques like differential privacy."
    },
    {
        "Risk": "Lack of Transparency",
        "Category": "Explainability",
        "Percentage": 12,
        "Severity": 5,
        "Likelihood": 5,
        "Impact": "Medium",
        "Description": "Complex models can be opaque, making decisions hard to understand.",
        "Mitigation": "Use explainable AI methods to increase transparency."
    },
    {
        "Risk": "Security Vulnerabilities",
        "Category": "Robustness",
        "Percentage": 10,
        "Severity": 6,
        "Likelihood": 5,
        "Impact": "Medium",
        "Description": "AI systems may be susceptible to adversarial attacks.",
        "Mitigation": "Employ robust training methods and continuous monitoring."
    },
    {
        "Risk": "Job Displacement",
        "Category": "Economic Impact",
        "Percentage": 8,
        "Severity": 7,
        "Likelihood": 6,
        "Impact": "High",
        "Description": "Automation may lead to loss of jobs in certain sectors.",
        "Mitigation": "Promote reskilling and education programs."
    },
    {
        "Risk": "Ethical Dilemmas",
        "Category": "Ethics",
        "Percentage": 7,
        "Severity": 5,
        "Likelihood": 4,
        "Impact": "Medium",
        "Description": "AI may make decisions conflicting with human values.",
        "Mitigation": "Incorporate ethical guidelines into AI development."
    },
    {
        "Risk": "Autonomous Weapons",
        "Category": "Physical Safety",
        "Percentage": 5,
        "Severity": 9,
        "Likelihood": 3,
        "Impact": "Critical",
        "Description": "AI could be used in weapons without human oversight.",
        "Mitigation": "Establish international regulations and oversight."
    },
    {
        "Risk": "Environmental Impact",
        "Category": "Sustainability",
        "Percentage": 3,
        "Severity": 4,
        "Likelihood": 5,
        "Impact": "Low",
        "Description": "High energy consumption in AI training affects the environment.",
        "Mitigation": "Optimize models and use renewable energy sources."
    },
    {
        "Risk": "Misuse for Surveillance",
        "Category": "Human Rights",
        "Percentage": 2,
        "Severity": 8,
        "Likelihood": 2,
        "Impact": "High",
        "Description": "AI can be used for mass surveillance violating privacy rights.",
        "Mitigation": "Enforce laws protecting individual privacy."
    },
]

def display_ai_safety_dashboard() -> Tuple[Figure, Figure, Figure, pd.DataFrame]:
    """
    Creates visualizations for AI safety risks.

    Returns
    -------
    Tuple[Figure, Figure, Figure, pd.DataFrame]
        A tuple containing the bar chart, pie chart, scatter plot, and the DataFrame of AI safety risks.

    Example
    -------
    >>> fig_bar, fig_pie, fig_scatter, df = display_ai_safety_dashboard()
    """

    df = pd.DataFrame(ai_safety_risks)

    # Bar Chart
    fig_bar = px.bar(
        df,
        x='Risk',
        y='Percentage',
        color='Category',
        text='Percentage',
        title='Percentage Distribution of AI Safety Risks',
        labels={'Percentage': 'Percentage (%)'},
        hover_data=['Description', 'Mitigation']
    )
    fig_bar.update_layout(
        xaxis_tickangle=-45,
        template='plotly_white',
        height=500,
        legend_title_text='Risk Category'
    )
    fig_bar.update_traces(texttemplate='%{text}%', textposition='outside')

    # Pie Chart
    category_counts = df['Category'].value_counts().reset_index()
    category_counts.columns = ['Category', 'Count']
    fig_pie = px.pie(
        category_counts,
        names='Category',
        values='Count',
        title='Proportion of Risk Categories',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    # Scatter Plot
    fig_scatter = px.scatter(
        df,
        x='Likelihood',
        y='Severity',
        size='Percentage',
        color='Impact',
        hover_name='Risk',
        title='Severity vs. Likelihood of AI Risks',
        labels={'Severity': 'Severity (1-10)', 'Likelihood': 'Likelihood (1-10)'},
        size_max=20
    )
    fig_scatter.update_layout(template='plotly_white', height=500)

    return fig_bar, fig_pie, fig_scatter, df