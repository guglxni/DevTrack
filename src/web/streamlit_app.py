import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional

# Define API base URL
API_BASE_URL = "http://localhost:8000"  # Change as needed

# Set page configuration
st.set_page_config(
    page_title="ASD Developmental Milestone Assessment",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "assessment_started" not in st.session_state:
    st.session_state.assessment_started = False
if "current_milestone" not in st.session_state:
    st.session_state.current_milestone = None
if "responses" not in st.session_state:
    st.session_state.responses = []
if "scores" not in st.session_state:
    st.session_state.scores = []
if "domain_quotients" not in st.session_state:
    st.session_state.domain_quotients = {}
if "assessment_complete" not in st.session_state:
    st.session_state.assessment_complete = False

# API interaction functions
def reset_assessment():
    """Reset the assessment in the backend"""
    try:
        response = requests.post(f"{API_BASE_URL}/reset")
        response.raise_for_status()
        st.session_state.assessment_started = False
        st.session_state.current_milestone = None
        st.session_state.responses = []
        st.session_state.scores = []
        st.session_state.domain_quotients = {}
        st.session_state.assessment_complete = False
        return True
    except Exception as e:
        st.error(f"Error resetting assessment: {str(e)}")
        return False

def start_assessment(age: int):
    """Start a new assessment"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/set-child-age",
            json={"age": age}
        )
        response.raise_for_status()
        st.session_state.assessment_started = True
        st.session_state.child_age = age
        # Get the first milestone
        get_next_milestone()
        return True
    except Exception as e:
        st.error(f"Error starting assessment: {str(e)}")
        return False

def get_next_milestone():
    """Get the next milestone to assess"""
    try:
        response = requests.get(f"{API_BASE_URL}/next-milestone")
        response.raise_for_status()
        data = response.json()
        
        if data.get("complete", False):
            st.session_state.assessment_complete = True
            generate_report()
            return None
        
        st.session_state.current_milestone = data
        return data
    except Exception as e:
        st.error(f"Error getting next milestone: {str(e)}")
        return None

def submit_response(response_text: str):
    """Submit a response for scoring"""
    if not st.session_state.current_milestone:
        st.error("No active milestone to score")
        return None
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/score-response",
            json={
                "response": response_text,
                "milestone_behavior": st.session_state.current_milestone["behavior"]
            }
        )
        response.raise_for_status()
        result = response.json()
        
        # Store the response and score
        st.session_state.responses.append({
            "milestone": st.session_state.current_milestone,
            "response": response_text,
            "score": result
        })
        
        # Move to next milestone
        get_next_milestone()
        return result
    except Exception as e:
        st.error(f"Error submitting response: {str(e)}")
        return None

def generate_report():
    """Generate the assessment report"""
    try:
        response = requests.get(f"{API_BASE_URL}/generate-report")
        response.raise_for_status()
        data = response.json()
        st.session_state.scores = data["scores"]
        st.session_state.domain_quotients = data["domain_quotients"]
        return data
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")
        return None

# UI Components

# Sidebar
st.sidebar.title("ASD Assessment Tool")
st.sidebar.subheader("Developmental Milestone Tracking")

# Start a new assessment
if not st.session_state.assessment_started:
    st.sidebar.subheader("Start New Assessment")
    child_age = st.sidebar.slider("Child's Age (months)", 0, 36, 24)
    if st.sidebar.button("Start Assessment"):
        start_assessment(child_age)

# Reset assessment
if st.session_state.assessment_started:
    if st.sidebar.button("Reset Assessment"):
        reset_assessment()

# Domain color mapping for consistent visualization
domain_colors = {
    "GM": "#FF9999",  # Gross Motor - Red
    "FM": "#FFCC99",  # Fine Motor - Orange
    "ADL": "#FFFF99", # Activities of Daily Living - Yellow
    "RL": "#99FF99",  # Receptive Language - Green
    "EL": "#99FFFF",  # Expressive Language - Cyan
    "COG": "#9999FF", # Cognitive - Blue
    "SOC": "#FF99FF", # Social - Pink
    "EMO": "#CC99FF"  # Emotional - Purple
}

# Domain full names for better readability
domain_names = {
    "GM": "Gross Motor",
    "FM": "Fine Motor",
    "ADL": "Activities of Daily Living",
    "RL": "Receptive Language",
    "EL": "Expressive Language",
    "COG": "Cognitive",
    "SOC": "Social",
    "EMO": "Emotional"
}

# Main content area
st.title("ASD Developmental Milestone Assessment")

if not st.session_state.assessment_started:
    # Welcome screen
    st.write("""
    ## Welcome to the ASD Developmental Milestone Assessment Tool
    
    This tool helps assess a child's developmental progress across 8 domains:
    
    1. **Gross Motor** - Large movements like walking, running, jumping
    2. **Fine Motor** - Small movements like grasping, writing, cutting
    3. **Activities of Daily Living** - Self-care skills like eating, dressing
    4. **Receptive Language** - Understanding language
    5. **Expressive Language** - Using language to communicate
    6. **Cognitive** - Thinking, learning, problem-solving
    7. **Social** - Interacting with others
    8. **Emotional** - Expressing and managing feelings
    
    To begin, set the child's age using the slider in the sidebar and click "Start Assessment".
    """)
    
    # Display sample radar chart
    st.subheader("Sample Development Profile")
    
    # Create sample data
    sample_data = {
        "GM": 85,
        "FM": 70,
        "ADL": 90,
        "RL": 60,
        "EL": 50,
        "COG": 75,
        "SOC": 65,
        "EMO": 80
    }
    
    # Create sample radar chart
    categories = list(domain_names.values())
    values = list(sample_data.values())
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Development Profile',
        line_color='rgba(75, 192, 192, 0.8)',
        fillcolor='rgba(75, 192, 192, 0.2)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False,
        title="Sample Development Profile"
    )
    
    st.plotly_chart(fig)
    
elif st.session_state.assessment_complete:
    # Display assessment results
    st.header("Assessment Complete!")
    st.subheader(f"Development Profile for Child (Age: {st.session_state.child_age} months)")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Development Profile", "Domain Scores", "Response History"])
    
    with tab1:
        # Radar chart of domain quotients
        if st.session_state.domain_quotients:
            # Prepare data
            categories = [domain_names[domain] for domain in st.session_state.domain_quotients.keys()]
            values = list(st.session_state.domain_quotients.values())
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                line_color='rgba(75, 192, 192, 0.8)',
                fillcolor='rgba(75, 192, 192, 0.2)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            overall_avg = sum(values) / len(values)
            
            st.subheader("Development Interpretation")
            if overall_avg >= 80:
                st.success(f"Overall Development Score: {overall_avg:.1f}% - Development is progressing well")
            elif overall_avg >= 60:
                st.info(f"Overall Development Score: {overall_avg:.1f}% - Development is generally on track with some areas to monitor")
            else:
                st.warning(f"Overall Development Score: {overall_avg:.1f}% - Some developmental areas need support")
            
            # Areas of strength and concern
            sorted_domains = sorted(st.session_state.domain_quotients.items(), key=lambda x: x[1], reverse=True)
            
            st.markdown("#### Areas of Strength:")
            strengths = [f"**{domain_names[domain]}** ({score:.1f}%)" for domain, score in sorted_domains[:3]]
            st.markdown(", ".join(strengths))
            
            st.markdown("#### Areas to Monitor:")
            concerns = [f"**{domain_names[domain]}** ({score:.1f}%)" for domain, score in sorted_domains[-3:]]
            st.markdown(", ".join(concerns))
    
    with tab2:
        # Bar chart of domain scores
        if st.session_state.domain_quotients:
            df = pd.DataFrame({
                'Domain': [domain_names[domain] for domain in st.session_state.domain_quotients.keys()],
                'Score': list(st.session_state.domain_quotients.values()),
                'Color': [domain_colors[domain] for domain in st.session_state.domain_quotients.keys()]
            })
            
            fig = px.bar(
                df, 
                x='Domain', 
                y='Score',
                color='Domain',
                color_discrete_sequence=df['Color'].tolist(),
                title="Domain Quotient Scores",
                labels={'Score': 'Score (%)', 'Domain': 'Developmental Domain'}
            )
            
            fig.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed score table
            st.subheader("Detailed Domain Scores")
            scores_df = pd.DataFrame(st.session_state.scores)
            if not scores_df.empty:
                # Add domain full names
                scores_df['Domain Name'] = scores_df['domain'].apply(lambda x: domain_names.get(x, x))
                
                # Customize display
                display_df = scores_df[['Domain Name', 'milestone', 'score_label', 'age_range']]
                display_df.columns = ['Domain', 'Milestone', 'Score', 'Age Range']
                
                st.dataframe(display_df, use_container_width=True)
    
    with tab3:
        # Display response history
        st.subheader("Response History")
        
        for i, response_data in enumerate(st.session_state.responses):
            milestone = response_data["milestone"]
            response = response_data["response"]
            score = response_data["score"]
            
            with st.expander(f"{i+1}. {milestone['domain']} - {milestone['behavior']} ({milestone['age_range']})"):
                st.markdown(f"**Criteria:** {milestone['criteria']}")
                st.markdown(f"**Response:** {response}")
                st.markdown(f"**Score:** {score['score_label']} ({score['score']})")
                
                # Color code the score
                score_value = score['score']
                if score_value == 4:
                    st.success("Independent skill")
                elif score_value == 3:
                    st.info("Requires support")
                elif score_value == 2:
                    st.warning("Emerging skill")
                else:
                    st.error("Needs development")
                    
else:
    # Assessment in progress
    # Display the current milestone
    if st.session_state.current_milestone:
        milestone = st.session_state.current_milestone
        
        # Display progress
        total_responses = len(st.session_state.responses)
        st.progress(total_responses / 30)  # Simplified progress indicator
        
        # Domain info
        domain_color = domain_colors.get(milestone["domain"], "#CCCCCC")
        domain_name = domain_names.get(milestone["domain"], milestone["domain"])
        
        # Create colored header
        st.markdown(
            f"""
            <div style="background-color: {domain_color}; padding: 10px; border-radius: 5px;">
                <h3 style="color: #333333;">{domain_name} - {milestone["behavior"]}</h3>
                <p><b>Age Range:</b> {milestone["age_range"]}</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        st.markdown(f"**Criteria:** {milestone['criteria']}")
        
        # Form for response
        with st.form("response_form"):
            response_text = st.text_area(
                "Caregiver's description of the child's behavior for this milestone:",
                height=150,
                placeholder="Describe how the child performs this skill..."
            )
            
            # Example responses to guide the user
            with st.expander("Example Responses"):
                st.markdown("""
                **Examples of different skill levels:**
                
                * **Independent:** "She does this consistently without any help. She can easily stack blocks and makes towers of 6-7 blocks independently."
                
                * **With Support:** "He can do this with some guidance. He stacks blocks when I help him get started, but needs encouragement to continue."
                
                * **Emerging:** "She's just starting to show this skill. Sometimes she puts one block on another but doesn't build towers yet."
                
                * **Lost Skill:** "He used to be able to stack blocks a few months ago, but doesn't seem interested anymore and has stopped trying."
                
                * **Cannot Do:** "She doesn't stack blocks at all. When given blocks, she just holds them or drops them."
                """)
            
            # Response buttons
            submitted = st.form_submit_button("Submit Response")
            
            if submitted and response_text:
                result = submit_response(response_text)
                if result:
                    st.success("Response recorded!")
                    # Force a rerun to update the UI with the next milestone
                    st.rerun()

        # Quick response buttons (for faster assessment)
        st.markdown("### Quick Response Options")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        if col1.button("Cannot Do"):
            result = submit_response("Child cannot do this skill at all.")
            if result:
                st.rerun()
                
        if col2.button("Lost Skill"):
            result = submit_response("Child used to have this skill but has lost it.")
            if result:
                st.rerun()
                
        if col3.button("Emerging"):
            result = submit_response("Child is beginning to show this skill sometimes.")
            if result:
                st.rerun()
                
        if col4.button("With Support"):
            result = submit_response("Child can do this with help and support.")
            if result:
                st.rerun()
                
        if col5.button("Independent"):
            result = submit_response("Child does this independently and consistently.")
            if result:
                st.rerun()
                
        # Display response history
        if st.session_state.responses:
            with st.expander("Previous Responses"):
                for i, response_data in enumerate(st.session_state.responses[-5:]):
                    milestone = response_data["milestone"]
                    response = response_data["response"]
                    score = response_data["score"]
                    
                    st.markdown(f"**{milestone['domain']} - {milestone['behavior']}**")
                    st.markdown(f"Response: {response}")
                    st.markdown(f"Score: {score['score_label']}")
                    st.markdown("---")
    else:
        st.warning("No active milestone. The assessment may be complete.")
        if st.button("View Results"):
            st.session_state.assessment_complete = True
            generate_report()
            st.rerun()

# Footer
st.markdown("---")
st.markdown("© 2023 ASD Developmental Assessment Tool") 