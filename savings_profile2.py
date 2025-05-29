import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import os
import openai
import requests
from fpdf import FPDF
import base64
from io import BytesIO
import json

# --- Configuration ---
st.set_page_config(
    page_title="Savings & Debt Advisor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
CURRENCIES = ["USD", "JPY", "EUR", "GBP", "KES", "CAD", "AUD"]
PROFILE_TYPES = ["Individual", "Household/Couple", "Group"]
SAVINGS_APPROACHES = ["Aggressive", "Moderate", "Easy"]
REPAYMENT_FREQUENCIES = ["Daily", "Weekly", "Bi-Weekly", "Monthly"]

# --- AI Configuration ---
AI_PROVIDERS = ["ChatGPT", "DeepSeek", "Local Model"]
DEFAULT_CHATGPT_MODEL = "gpt-4-turbo"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# Add this helper function for text cleaning
def clean_text(text):
    """
    Replace problematic Unicode characters with ASCII equivalents
    """
    replacements = {
        '\u2013': '-',   # en dash
        '\u2014': '--',  # em dash
        '\u2018': "'",   # left single quote
        '\u2019': "'",   # right single quote
        '\u201c': '"',   # left double quote
        '\u201d': '"',   # right double quote
        '\u2026': '...', # ellipsis
        '\u00a0': ' ',   # non-breaking space
        '\u00ae': '(R)', # registered trademark
        '\u00a9': '(c)', # copyright
        '\u2122': '(TM)',# trademark
    }
    for orig, replacement in replacements.items():
        text = text.replace(orig, replacement)
    return text

# Update the PDF creation function to use clean_text
def create_pdf_document(plan_data):
    """Create a professional PDF financial plan document"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Helper function to clean and encode text
    def add_text(text, style='', size=12):
        text = clean_text(str(text))
        if style == 'B':
            pdf.set_font("Arial", 'B', size)
        else:
            pdf.set_font("Arial", '', size)
        pdf.multi_cell(0, 10, text)
    
    # Add header
    pdf.set_font("Arial", 'B', 24)
    pdf.cell(0, 10, "Financial Action Plan", 0, 1, 'C')
    pdf.ln(10)
    
    # Add metadata
    add_text(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add_text(f"Profile Type: {plan_data['profile_type']}")
    add_text(f"Financial Goal: {plan_data['goal']}")
    add_text(f"AI Provider: {plan_data['ai_provider']}")
    pdf.ln(5)
    
    # Profile Information
    add_text("Profile Information", 'B', 16)
    for i, member in enumerate(plan_data['members']):
        add_text(f"Member {i+1}: {member['name']} - Income: {plan_data['currency']} {member['income']:,.2f}")
    add_text(f"Total Monthly Income: {plan_data['currency']} {plan_data['total_income']:,.2f}")
    pdf.ln(5)
    
    # Financial Summary
    add_text("Financial Summary", 'B', 16)
    add_text(f"Total Monthly Expenses: {plan_data['currency']} {plan_data['total_expenses']:,.2f}")
    add_text(f"Monthly Surplus: {plan_data['currency']} {plan_data['surplus']:,.2f}")
    pdf.ln(5)
    
    # Goal Details
    add_text(f"{plan_data['goal']} Details", 'B', 16)
    if plan_data['goal'] == "Savings Goal":
        add_text(f"Target Amount: {plan_data['currency']} {plan_data['goal_amount']:,.2f}")
        add_text(f"Savings Approach: {plan_data['approach']}")
        add_text(f"Savings Frequency: {plan_data['frequency']}")
        add_text(f"Savings Period: {plan_data['period']} months")
        add_text(f"Required {plan_data['frequency']} Saving: {plan_data['currency']} {plan_data['required_amount']:,.2f}")
    else:
        add_text(f"Debt Amount: {plan_data['currency']} {plan_data['debt_amount']:,.2f}")
        add_text(f"Repayment Approach: {plan_data['approach']}")
        add_text(f"Repayment Frequency: {plan_data['frequency']}")
        add_text(f"Repayment Period: {plan_data['period']} months")
        add_text(f"Required {plan_data['frequency']} Payment: {plan_data['currency']} {plan_data['required_amount']:,.2f}")
        if plan_data['late_payment']:
            add_text("⚠️ Late payment history noted")
    pdf.ln(10)
    
    # Expense Breakdown
    add_text("Expense Breakdown", 'B', 16)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(100, 10, "Expense", 1, 0)
    pdf.cell(40, 10, "Amount", 1, 1)
    pdf.set_font("Arial", '', 12)
    
    for expense in plan_data['expenses']:
        pdf.cell(100, 10, clean_text(expense['name']), 1, 0)
        pdf.cell(40, 10, f"{plan_data['currency']} {expense['amount']:,.2f}", 1, 1)
    pdf.ln(10)
    
    # AI Recommendations
    add_text("AI Recommendations", 'B', 16)
    add_text(plan_data['recommendation'])
    
    # Member-specific recommendations
    if 'member_recommendations' in plan_data:
        pdf.ln(5)
        add_text("Member-Specific Recommendations", 'B', 14)
        for member in plan_data['member_recommendations']:
            add_text(f"{member['name']}:", 'B', 12)
            add_text(member['recommendation'])
            pdf.ln(3)
    
    # Footer
    pdf.set_y(-15)
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 10, "Generated by AI Financial Advisor | Confidential Financial Document", 0, 0, 'C')
    
    return pdf.output(dest='S').encode('latin1', 'replace')

# --- Helper Functions ---
def calculate_savings_plan(total_income, total_expenses, goal_amount, savings_period, frequency):
    surplus = total_income - total_expenses
    periods = {
        "Daily": savings_period * 30,
        "Weekly": savings_period * 4,
        "Bi-Weekly": savings_period * 2,
        "Monthly": savings_period
    }
    
    required_saving = goal_amount / periods.get(frequency, savings_period)
    return surplus, required_saving, periods.get(frequency, 1)

def calculate_debt_plan(total_income, total_expenses, debt_amount, repayment_period, frequency):
    surplus = total_income - total_expenses
    periods = {
        "Daily": repayment_period * 30,
        "Weekly": repayment_period * 4,
        "Bi-Weekly": repayment_period * 2,
        "Monthly": repayment_period
    }
    
    required_payment = debt_amount / periods.get(frequency, repayment_period)
    return surplus, required_payment, periods.get(frequency, 1)

def generate_ai_recommendation(prompt, provider="ChatGPT"):
    try:
        if provider == "ChatGPT":
            client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            response = client.chat.completions.create(
                model=DEFAULT_CHATGPT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        
        elif provider == "DeepSeek":
            # DeepSeek API implementation
            headers = {
                "Authorization": f"Bearer {st.secrets['DEEPSEEK_API_KEY']}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": DEEPSEEK_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1000,
                "stream": False
            }
            
            response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        
    except Exception as e:
        st.error(f"AI service error: {str(e)}")
    
    # Fallback to local knowledge base
    fallback_responses = {
        "saving": (
            "Based on your financial situation:\n\n"
            "1. Track all expenses meticulously for 30 days\n"
            "2. Reduce discretionary spending by 20-30%\n"
            "3. Consider side hustles to increase income\n"
            "4. Automate savings transfers on payday\n"
            "5. Review subscriptions and cancel unused services\n"
            "6. Negotiate better rates on utilities and insurance\n"
            "7. Implement the 50/30/20 rule: 50% needs, 30% wants, 20% savings"
        ),
        "debt": (
            "Debt reduction strategy:\n\n"
            "1. Prioritize high-interest debts first (avalanche method)\n"
            "2. Negotiate lower interest rates with creditors\n"
            "3. Consider debt consolidation for simpler management\n"
            "4. Allocate at least 20% of income to debt repayment\n"
            "5. Temporarily pause non-essential subscriptions\n"
            "6. Generate extra income through side gigs\n"
            "7. Build a small emergency fund to avoid new debt"
        )
    }
    
    if "saving" in prompt.lower():
        return fallback_responses["saving"]
    elif "debt" in prompt.lower():
        return fallback_responses["debt"]
    return "Financial advice: Create a detailed budget, track spending, and set clear financial goals."

# --- UI Components ---
def main_header():
    st.title("💰 AI-Powered Savings & Debt Advisor")
    st.markdown("""
    <style>
    .big-font { font-size:18px !important; }
    .highlight { background-color: #f0f2f6; border-radius: 5px; padding: 10px; }
    .stDownloadButton button { 
        background-color: #4CAF50 !important; 
        color: white !important; 
        font-weight: bold !important;
    }
    .stButton>button {
        background-color: #4285F4;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    cols = st.columns(3)
    with cols[0]:
        st.subheader("📊 Personalized Plans")
    with cols[1]:
        st.subheader("🤖 AI Recommendations")
    with cols[2]:
        st.subheader("📈 Interactive Projections")

# --- Main Application ---
def main():
    main_header()
    
    with st.sidebar:
        st.header("Configuration")
        profile_type = st.selectbox("Select Profile Type", PROFILE_TYPES, index=0)
        currency = st.selectbox("Select Currency", CURRENCIES, index=0)
        goal = st.radio("Financial Goal", ["Savings Goal", "Debt Clearance"])
        ai_provider = st.selectbox("AI Provider", AI_PROVIDERS, index=0)
        
        st.markdown("---")
        st.info("💡 This tool helps you create personalized savings or debt repayment plans with AI-powered recommendations")
    
    # Initialize session state variables
    if "expenses" not in st.session_state:
        st.session_state.expenses = []
    if "recommendation" not in st.session_state:
        st.session_state.recommendation = ""
    if "member_recommendations" not in st.session_state:
        st.session_state.member_recommendations = []
    
    # --- Profile Setup ---
    st.subheader("👤 Profile Information")
    members = []
    
    if profile_type == "Individual":
        cols = st.columns(2)
        with cols[0]:
            income = st.number_input("Monthly Income", min_value=0, value=3000, step=100)
        with cols[1]:
            balance = st.number_input("Current Account Balance", min_value=0, value=5000, step=100)
        members.append({"name": "Individual", "income": income})
        
    elif profile_type in ["Household/Couple", "Group"]:
        member_count = st.number_input("Number of Members", min_value=2, max_value=10, value=2, step=1)
        
        for i in range(member_count):
            st.markdown(f"### Member {i+1}")
            cols = st.columns(3)
            with cols[0]:
                name = st.text_input(f"Name", value=f"Member {i+1}", key=f"name_{i}")
            with cols[1]:
                income = st.number_input(f"Monthly Income", min_value=0, value=2000, step=100, key=f"income_{i}")
            with cols[2]:
                balance = st.number_input(f"Account Balance", min_value=0, value=3000, step=100, key=f"balance_{i}")
            members.append({"name": name, "income": income})
    
    total_income = sum(member["income"] for member in members)
    st.markdown(f"**Total Monthly Income: {currency} {total_income:,.2f}**")
    
    # --- Expense Tracking ---
    st.subheader("💸 Expense Tracking")
    expense_cols = st.columns([3, 2, 1])
    with expense_cols[0]:
        expense_name = st.text_input("Expense Name", value="Rent")
    with expense_cols[1]:
        expense_amount = st.number_input("Amount", min_value=0, value=1200, step=50)
    with expense_cols[2]:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Add Expense"):
            st.session_state.expenses.append({"name": expense_name, "amount": expense_amount})
    
    if st.session_state.expenses:
        expense_df = pd.DataFrame(st.session_state.expenses)
        st.dataframe(expense_df, hide_index=True)
        
        # Expense visualization
        fig = px.pie(expense_df, names="name", values="amount", title="Expense Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        total_expenses = expense_df["amount"].sum()
        st.markdown(f"**Total Monthly Expenses: {currency} {total_expenses:,.2f}**")
        surplus = total_income - total_expenses
        st.markdown(f"**Monthly Surplus: {currency} {surplus:,.2f}**")
    else:
        total_expenses = 0
        surplus = total_income
        st.info("No expenses added yet")
    
    # --- Goal Configuration ---
    st.subheader("🎯 Goal Details")
    late_payment = False
    
    if goal == "Savings Goal":
        savings_cols = st.columns(2)
        with savings_cols[0]:
            goal_amount = st.number_input("Savings Goal Amount", min_value=100, value=10000, step=500)
        with savings_cols[1]:
            savings_approach = st.selectbox("Savings Approach", SAVINGS_APPROACHES, index=1)
        
        freq_cols = st.columns(2)
        with freq_cols[0]:
            savings_frequency = st.selectbox("Savings Frequency", REPAYMENT_FREQUENCIES, index=3)
        with freq_cols[1]:
            savings_period = st.slider("Savings Period (months)", 3, 36, 12)
        
        # Calculate savings plan
        _, required_saving, total_periods = calculate_savings_plan(
            total_income, total_expenses, goal_amount, savings_period, savings_frequency
        )
        
        st.markdown(f"""
        <div class="highlight">
            <h4>Savings Plan Summary</h4>
            <p>Required {savings_frequency.lower()} saving: <b>{currency} {required_saving:,.2f}</b></p>
            <p>Current surplus: <b>{currency} {surplus:,.2f}</b></p>
            <p>Plan feasibility: <b>{' Achievable' if surplus >= required_saving else '⚠️ Needs adjustment'}</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Interactive adjustment
        st.subheader("🔧 Plan Adjustments")
        adj_cols = st.columns(2)
        with adj_cols[0]:
            new_frequency = st.selectbox("Adjust Frequency", REPAYMENT_FREQUENCIES, index=REPAYMENT_FREQUENCIES.index(savings_frequency))
        with adj_cols[1]:
            new_period = st.slider("Adjust Period (months)", 3, 36, savings_period)
        
        # Recalculate with adjustments
        _, adj_required_saving, adj_total_periods = calculate_savings_plan(
            total_income, total_expenses, goal_amount, new_period, new_frequency
        )
        
        st.markdown(f"**Adjusted {new_frequency.lower()} saving required: {currency} {adj_required_saving:,.2f}**")
        
        # Timeline visualization
        timeline_data = {
            "Period": [f"Month {i+1}" for i in range(new_period)],
            "Amount Saved": [adj_required_saving * (i+1) * (4 if new_frequency == "Weekly" else 1) for i in range(new_period)]
        }
        timeline_df = pd.DataFrame(timeline_data)
        fig = px.line(timeline_df, x="Period", y="Amount Saved", title="Savings Projection")
        st.plotly_chart(fig, use_container_width=True)
        
        # Generate AI recommendation
        prompt = f"""
        You are a financial advisor helping a {profile_type.lower()} with a {savings_approach.lower()} savings approach.
        Financial profile:
        - Total monthly income: {currency} {total_income:,.2f}
        - Total monthly expenses: {currency} {total_expenses:,.2f}
        - Monthly surplus: {currency} {surplus:,.2f}
        - Savings goal: {currency} {goal_amount:,.2f}
        - Timeframe: {savings_period} months
        - Savings frequency: {savings_frequency}
        - Required {savings_frequency.lower()} saving: {currency} {required_saving:,.2f}
        
        Provide a detailed savings plan with:
        1. Specific expense reduction strategies tailored to their spending
        2. Income enhancement opportunities
        3. Behavioral tips to maintain savings discipline
        4. Timeline milestones
        5. Contingency planning
        
        Format your response with clear sections, give practical and actionable steps, simplified for users with no finance knowledge.
        """
        
        goal_type = "saving"
        approach = savings_approach
        frequency = savings_frequency
        period = savings_period
        required_amount = required_saving
        
    else:  # Debt Clearance
        debt_cols = st.columns(2)
        with debt_cols[0]:
            debt_amount = st.number_input("Total Debt Amount", min_value=100, value=15000, step=500)
        with debt_cols[1]:
            debt_approach = st.selectbox("Repayment Approach", SAVINGS_APPROACHES, index=1)
        
        freq_cols = st.columns(2)
        with freq_cols[0]:
            repayment_frequency = st.selectbox("Repayment Frequency", REPAYMENT_FREQUENCIES, index=3)
        with freq_cols[1]:
            repayment_period = st.slider("Repayment Period (months)", 1, 36, 18)
        
        # Payment history
        late_payment = st.checkbox("Had late payments in the past 1-4 months?")
        
        # Calculate repayment plan
        _, required_payment, total_periods = calculate_debt_plan(
            total_income, total_expenses, debt_amount, repayment_period, repayment_frequency
        )
        
        st.markdown(f"""
        <div class="highlight">
            <h4>Debt Repayment Summary</h4>
            <p>Required {repayment_frequency.lower()} payment: <b>{currency} {required_payment:,.2f}</b></p>
            <p>Current surplus: <b>{currency} {surplus:,.2f}</b></p>
            <p>Plan feasibility: <b>{'✅ Achievable' if surplus >= required_payment else '⚠️ Needs adjustment'}</b></p>
            {f"<p>⚠️ Late payment history may affect credit score</p>" if late_payment else ""}
        </div>
        """, unsafe_allow_html=True)
        
        # Interactive adjustment
        st.subheader("🔧 Plan Adjustments")
        adj_cols = st.columns(2)
        with adj_cols[0]:
            extra_payment = st.number_input("Extra Payment Amount", min_value=0, value=0, step=50)
        with adj_cols[1]:
            new_frequency = st.selectbox("Adjust Frequency", REPAYMENT_FREQUENCIES, index=REPAYMENT_FREQUENCIES.index(repayment_frequency))
        
        # Recalculate with adjustments
        adj_required_payment = (debt_amount / total_periods) + extra_payment
        adj_repayment_period = max(1, int(debt_amount / adj_required_payment))
        
        st.markdown(f"**Estimated repayment period with adjustments: {adj_repayment_period} months**")
        
        # Timeline visualization
        timeline_data = {
            "Period": [f"Month {i+1}" for i in range(adj_repayment_period)],
            "Remaining Debt": [debt_amount - (adj_required_payment * (i+1) * (4 if new_frequency == "Weekly" else 1)) 
                              for i in range(adj_repayment_period)]
        }
        timeline_df = pd.DataFrame(timeline_data)
        fig = px.area(timeline_df, x="Period", y="Remaining Debt", title="Debt Reduction Projection")
        st.plotly_chart(fig, use_container_width=True)
        
        # Generate AI recommendation
        prompt = f"""
        You are a financial advisor helping a {profile_type.lower()} with a {debt_approach.lower()} debt repayment approach.
        Financial profile:
        - Total monthly income: {currency} {total_income:,.2f}
        - Total monthly expenses: {currency} {total_expenses:,.2f}
        - Monthly surplus: {currency} {surplus:,.2f}
        - Total debt: {currency} {debt_amount:,.2f}
        - Repayment timeframe: {repayment_period} months
        - Repayment frequency: {repayment_frequency}
        - Required {repayment_frequency.lower()} payment: {currency} {required_payment:,.2f}
        - Late payment history: {'Yes' if late_payment else 'No'}
        
        Provide a detailed debt repayment plan with:
        1. Debt prioritization strategy (avalanche vs snowball method)
        2. Specific expense reduction strategies based on the users spending.
        3. Negotiation tactics with creditors
        4. Credit score improvement tips
        5. Timeline milestones
        6. Contingency planning for financial emergencies
        
        Format your response with clear sections and actionable steps. Avoid cliche/commonly used recommendations. Simplify the response for users with no finance knowledge.
        """
        
        goal_type = "debt"
        approach = debt_approach
        frequency = repayment_frequency
        period = repayment_period
        required_amount = required_payment
    
    # --- AI Recommendations Section ---
    st.subheader("🤖 AI Recommendations")
    if st.button("Generate Personalized Advice", key="generate_advice"):
        with st.spinner("Generating AI recommendations..."):
            recommendation = generate_ai_recommendation(prompt, ai_provider)
            st.session_state.recommendation = recommendation
            st.session_state.member_recommendations = []
            
            # Generate member-specific recommendations for groups
            if profile_type in ["Household/Couple", "Group"] and len(members) > 1:
                for member in members:
                    member_prompt = f"""
                    Create personalized financial recommendations for {member['name']} who earns {member['income']} {currency}.
                    Focus on how they can contribute to the overall {goal_type} goal.
                    Provide 2-3 specific action items tailored to their situation.
                    """
                    with st.spinner(f"Generating tips for {member['name']}..."):
                        member_rec = generate_ai_recommendation(member_prompt, ai_provider)
                        st.session_state.member_recommendations.append({
                            "name": member['name'],
                            "recommendation": member_rec
                        })
    
    if st.session_state.recommendation:
        st.markdown(f"""
        <div style="background-color:#f0f2f6; padding:20px; border-radius:10px; margin-top:20px;">
            <h4 style="color:#2E86C1;">✨ Personalized Recommendations</h4>
            <p style="white-space: pre-wrap;">{st.session_state.recommendation}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display member-specific recommendations
        if st.session_state.member_recommendations:
            st.subheader("👥 Member-Specific Suggestions")
            for member in st.session_state.member_recommendations:
                with st.expander(f"Recommendations for {member['name']}"):
                    st.info(member['recommendation'])
    
    # --- Plan Export ---
    st.markdown("---")
    st.subheader("📥 Export Financial Plan")
    
    if st.button("Generate PDF Report", key="generate_pdf"):
        # Prepare data for PDF
        plan_data = {
            "profile_type": profile_type,
            "currency": currency,
            "goal": goal,
            "ai_provider": ai_provider,
            "members": members,
            "total_income": total_income,
            "expenses": st.session_state.expenses,
            "total_expenses": total_expenses,
            "surplus": surplus,
            "recommendation": st.session_state.recommendation,
            "late_payment": late_payment,
            "approach": approach,
            "frequency": frequency,
            "period": period
        }
        
        if goal == "Savings Goal":
            plan_data["goal_amount"] = goal_amount
            plan_data["required_amount"] = required_saving
        else:
            plan_data["debt_amount"] = debt_amount
            plan_data["required_amount"] = required_payment
            
        if st.session_state.member_recommendations:
            plan_data["member_recommendations"] = st.session_state.member_recommendations
        
        # Generate PDF
        try:
            pdf_bytes = create_pdf_document(plan_data)
            
            # Create download button
            st.download_button(
                label="Download Financial Plan (PDF)",
                data=pdf_bytes,
                file_name="financial_plan.pdf",
                mime="application/pdf"
            )
            st.success("PDF report generated successfully! Click the download button above.")
        except Exception as e:
            st.error(f"Failed to generate PDF: {str(e)}")

if __name__ == "__main__":
    main()