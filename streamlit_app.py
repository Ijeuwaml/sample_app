import os
os.environ['STREAMLIT_WATCHED_FILES_IGNORE'] = '*'

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from scipy.stats import norm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Set page configuration to wide mode
st.set_page_config(layout="wide")

# Custom CSS for compact layout
st.markdown("""
    <style>
    .reportview-container .main .block-container {
        padding: 2px;
        max-width: 100%;
    }
    div[data-testid="stVerticalBlock"] > div {
        padding: 2px;
        
    }
    h1, h2, h3, h4, h5, h6 {
        margin: 2px;
    }
    </style>
    """, unsafe_allow_html=True)

# ─── Load Small Language Model for Recommendation (as LangGraph proxy) ─────
@st.cache_resource(show_spinner=False)
def load_language_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model

tokenizer, model = load_language_model()

def generate_professional_recommendation(analysis: dict, method: str) -> str:
    """
    Generate a human-like professional recommendation based on simulation analysis.
    The prompt assumes the role of a civil engineer with over 15 years of project management experience.
    The recommendation focuses on the optimal totals from the analysis and explains why the chosen mix is best.
    The output will be at least two coherent lines of text.
    """
    optimal_total = analysis.get('optimal_total', 'N/A')
    prompt = (
        f"As a civil engineer with over 15 years of project management experience, "
        f"analyze the following simulation results using the '{method}' optimization method. "
        f"The optimal total weight is {optimal_total}. The details are as follows:\n\n"
        f"- Safety: {analysis['safety_value']} (Baseline: {analysis['base_safety']})\n"
        f"- Reliability: {analysis['reliability_value']} (Baseline: {analysis['base_reliability']})\n"
        f"- Skills: {analysis['skill_value']}\n"
        f"- Cost: {analysis['cost_value']}\n"
        f"- Total Variation: {analysis['variation_score']}\n"
        f"- Case Type: {analysis['case_type']}\n\n"
        "Generate a professional recommendation that explains why this mix is optimal compared to the baseline. "
        "Discuss key tradeoffs and potential implementation risks, and suggest actionable monitoring metrics. "
        "The recommendation should consist of at least two coherent lines of text."
    )
    #My attempt to log my Prompts
    print("Prompt:", prompt)
    
    # inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    # outputs = model.generate(
      #  **inputs,
       # max_length=400,
        # num_beams=5,
        #do_sample=True,
       # temperature=0.7,
        #repetition_penalty=1.25,
       # no_repeat_ngram_size=3,
        #early_stopping=True
   # )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        **inputs,
        max_length=500,           # Increased max_length
        num_beams=5,
        do_sample=True,
        temperature=0.8,          # Increased temperature for diversity
        top_p=0.9,                # Added top_p sampling
        repetition_penalty=1.25,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    print("Output:", outputs[outputs])

    raw_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Post-process to ensure coherent sentences and at least two lines.
    sentences = [s.strip().capitalize() for s in raw_text.replace('\n', ' ').split('. ') if s.strip()]
    recommendation = '. '.join(sentences)
    if not recommendation.endswith('.'):
        recommendation += '.'
    # Ensure there is at least one newline to separate two lines
    if recommendation.count('\n') < 1:
        recommendation = recommendation.replace('. ', '.\n', 1)
    return recommendation

# ─── Default Base Criteria Weights ─────────────────────────────────────────
default_weights = {
    'is_safe_enough': 0.41,
    'reliab_center_m': 0.26,
    'skilled_and_trained': 0.19,
    'cost_optimized': 0.14
}

# ─── Expanded Simulation Function ──────────────────────────────────────────
def generate_expanded_simulations(base_weights):
    simulations = []
    adjustment_options = [-0.01, 0.01]  # Allowed deltas

    for n_adj in [1, 2, 3, 4]:
        for criteria_comb in itertools.combinations(base_weights.keys(), n_adj):
            for deltas in itertools.product(adjustment_options, repeat=n_adj):
                new_weights = base_weights.copy()
                valid = True
                for i, criterion in enumerate(criteria_comb):
                    new_value = base_weights[criterion] + deltas[i]
                    if new_value < 0 or abs(deltas[i]) > 0.01:
                        valid = False
                        break
                    new_weights[criterion] = new_value
                if valid:
                    simulations.append({
                        **new_weights,
                        'case_type': f'{n_adj}_criteria',
                        'adjusted_criteria': ', '.join(criteria_comb),
                        'adjustment_values': ', '.join([f"{d:+.2f}" for d in deltas])
                    })

    df = pd.DataFrame(simulations)
    df['total_variation'] = df.apply(lambda row: sum(abs(row[crit] - base_weights[crit]) for crit in base_weights.keys()), axis=1)
    weight_cols = list(base_weights.keys())
    df['total_sum'] = df[weight_cols].sum(axis=1)
    df = df.round(4)
    return df

# ─── Analysis Preparation Functions ─────────────────────────────────────────
def prepare_wsa_analysis(df: pd.DataFrame, base_weights):
    df_opt = df[df['total_sum'] > 1.0]
    if df_opt.empty:
        return None
    optimal = df_opt.sort_values(by='total_sum', ascending=False).iloc[0]
    return {
        'safety_value': optimal['is_safe_enough'],
        'reliability_value': optimal['reliab_center_m'],
        'skill_value': optimal['skilled_and_trained'],
        'cost_value': optimal['cost_optimized'],
        'base_safety': base_weights['is_safe_enough'],
        'base_reliability': base_weights['reliab_center_m'],
        'variation_score': optimal['total_variation'],
        'case_type': optimal['case_type'],
        'optimal_total': optimal['total_sum']
    }

def prepare_variation_analysis(df: pd.DataFrame, base_weights):
    df_opt = df[df['total_sum'] > 1.0]
    if df_opt.empty:
        return None
    optimal = df_opt.sort_values(by='total_variation').iloc[0]
    return {
        'safety_value': optimal['is_safe_enough'],
        'reliability_value': optimal['reliab_center_m'],
        'skill_value': optimal['skilled_and_trained'],
        'cost_value': optimal['cost_optimized'],
        'base_safety': base_weights['is_safe_enough'],
        'base_reliability': base_weights['reliab_center_m'],
        'variation_score': optimal['total_variation'],
        'case_type': optimal['case_type'],
        'optimal_total': optimal['total_sum']
    }

# ─── Visualization Functions ─────────────────────────────────────────────
def create_total_sum_histogram(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df['total_sum'], bins=15, kde=True, ax=ax)
    ax.set_title("Total Weight Distribution")
    ax.set_xlabel("Total Sum of Weights")
    ax.set_ylabel("Frequency")
    return fig

def create_criteria_radar(df: pd.DataFrame, base_weights):
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'polar': True})
    categories = list(base_weights.keys())
    values = df[categories].mean().tolist()
    values += values[:1]
    
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    ax.plot(angles, values, color='blue', linewidth=2, linestyle='solid')
    ax.fill(angles, values, color='blue', alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title("Average Criteria Weight Distribution")
    return fig

def create_total_sum_violin(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    palette = sns.color_palette("Set2", n_colors=len(df['case_type'].unique()))
#    sns.violinplot(x='case_type', y='total_sum', data=df, palette=sns.color_palette("Set2", n_colors=len(df['case_type'].unique())), ax=ax, legend=False)
    sns.violinplot(x='case_type', y='total_sum', data=df, hue='case_type', palette=sns.color_palette("Set2", n_colors=len(df['case_type'].unique())), ax=ax, legend=False)
    ax.set_title("Weight Distribution by Case Type")
    ax.set_xlabel("Case Type")
    ax.set_ylabel("Total Sum")
    return fig

def create_exploded_pie_chart(df: pd.DataFrame):
    group = df.groupby('case_type')['total_sum'].sum()
    explode = [0.1 if value == group.max() else 0 for value in group]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(group, labels=group.index, autopct='%1.1f%%', startangle=90, explode=explode)
    ax.set_title("Case Type Distribution")
    return fig

# ─── Streamlit UI ─────────────────────────────────────────────
def main():
    st.title("Infrastructure Project Optimizer")
    st.markdown("""
    **AI-Powered Multi-Criteria Decision Support System**  
    1. Set base weights in the sidebar.  
    2. Choose an optimization method.  
    3. Generate expanded simulations with ±0.01 adjustments on 1-4 criteria.  
    4. View analysis visualizations.  
    5. Receive a professional, human-like recommendation based on the optimal totals.
    """)

    # Sidebar for controls
    with st.sidebar:
        st.header("Project Parameters")
        base_weights = {
            'is_safe_enough': st.number_input("Safety", 0.0, 1.0, default_weights['is_safe_enough'], 0.01),
            'reliab_center_m': st.number_input("Reliability", 0.0, 1.0, default_weights['reliab_center_m'], 0.01),
            'skilled_and_trained': st.number_input("Skills", 0.0, 1.0, default_weights['skilled_and_trained'], 0.01),
            'cost_optimized': st.number_input("Cost", 0.0, 1.0, default_weights['cost_optimized'], 0.01)
        }
        method = st.radio("Optimization Method", ("Weighted Sum", "Variation Minimization"))
        run_sim = st.button("🚀 Run Simulation Analysis")
    
    if run_sim:
        with st.spinner("Generating simulations..."):
            sim_df = generate_expanded_simulations(base_weights)
        st.success(f"Generated {len(sim_df)} simulations")
        
        with st.expander("📁 Simulation Data"):
            fmt_dict = {col: "{:.4f}" for col in sim_df.select_dtypes(include=["number"]).columns}
            st.dataframe(sim_df.style.format(fmt_dict), height=300)
        
        st.header("Analysis Visualizations")
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(create_total_sum_histogram(sim_df))
            st.pyplot(create_total_sum_violin(sim_df))
        with col2:
            st.pyplot(create_criteria_radar(sim_df, base_weights))
            st.pyplot(create_exploded_pie_chart(sim_df))
        
        st.header("Professional Recommendation")
        if method == "Weighted Sum":
            analysis = prepare_wsa_analysis(sim_df, base_weights)
        else:
            analysis = prepare_variation_analysis(sim_df, base_weights)
        
        if analysis:
            rec_text = generate_professional_recommendation(analysis, method)
            st.markdown(f'<div style="text-align: justify; font-size: 16px">{rec_text}</div>', unsafe_allow_html=True)
        else:
            st.warning("No valid configurations found meeting the criteria.")
    else:
        st.write("Awaiting simulation run...")

if __name__ == "__main__":
    main()
