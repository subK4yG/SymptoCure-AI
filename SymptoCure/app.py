import streamlit as st
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from sklearn.exceptions import InconsistentVersionWarning
import warnings
import base64
import time
import os

# Ignore warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Set paths to data files
DATA_DIR = "datasets"
MODEL_PATH = "models/svc.pkl"

# Custom CSS with animations
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

:root {
    --primary: #2563eb;
    --secondary: #7c3aed;
    --dark: #1e293b;
    --light: #f8fafc;
    --medical-red: #ff4b4b;
    --medical-blue: #1a73e8;
}

* {
    font-family: 'Poppins', sans-serif;
}

body {
    background-color: var(--light);
}

.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}

/* Header styles */
.header {
    text-align: center;
    padding: 2rem 0;
    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
    color: white;
    border-radius: 15px;
    margin-bottom: 2rem;
    box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    animation: fadeIn 1s ease-in-out;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Card styles */
.card {
    background: white;
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
}

/* Medical card */
.medical-card {
    background: white;
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    border-left: 4px solid var(--medical-blue);
}

/* Button styles */
.stButton>button {
    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(37, 99, 235, 0.4);
}

/* Input styles */
.stTextInput>div>div>input {
    border-radius: 8px !important;
    padding: 10px 15px !important;
}

/* Tab styles */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    padding: 10px 20px;
    border-radius: 8px !important;
    transition: all 0.3s ease;
}

.stTabs [aria-selected="true"] {
    background-color: var(--primary) !important;
    color: white !important;
}

/* Animation classes */
.animate-bounce {
    animation: bounce 2s infinite;
}

@keyframes bounce {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-10px); }
}

.animate-pulse {
    animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* Feature highlights */
.feature {
    display: flex;
    align-items: center;
    margin-bottom: 1rem;
    padding: 1rem;
    border-radius: 10px;
    background: rgba(255,255,255,0.7);
}

.feature-icon {
    font-size: 2rem;
    margin-right: 1rem;
    color: var(--primary);
}

/* Cover image styling */
.cover-img {
    width: 100%;
    border-radius: 15px;
    box-shadow: 0 15px 30px rgba(0,0,0,0.1);
    margin: 2rem 0;
    animation: float 6s ease-in-out infinite;
}

@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-15px); }
    100% { transform: translateY(0px); }
}

/* Stethoscope animation */
@keyframes stethAnimation {
    0% { transform: scale(0.5) rotate(-30deg); opacity: 0; }
    50% { transform: scale(1.2) rotate(10deg); opacity: 1; }
    100% { transform: scale(1) rotate(0deg); opacity: 0; }
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .header {
        padding: 1rem 0;
    }
}
</style>
"""

# Stethoscope animation HTML
STETHOSCOPE_HTML = """
<div id="stethoscope-container" style="
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    pointer-events: none;
    z-index: 9999;
">
    <div id="stethoscope" style="
        font-size: 100px;
        animation: stethAnimation 2s ease-out;
    ">🩺</div>
</div>

<style>
    @keyframes stethAnimation {
        0% { transform: scale(0.5) rotate(-30deg); opacity: 0; }
        50% { transform: scale(1.2) rotate(10deg); opacity: 1; }
        100% { transform: scale(1) rotate(0deg); opacity: 0; }
    }
</style>

<script>
    // Remove the element after animation completes
    setTimeout(function() {
        var element = document.getElementById('stethoscope-container');
        if (element) {
            element.remove();
        }
    }, 2000);
</script>
"""

# HTML templates
WELCOME_HTML = """
<div class="header">
    <h1>Welcome to SymptoCure</h1>
    <p>Your AI-powered health companion</p>
</div>

<div class="card">
    <h3>✨ How it works</h3>
    <div class="feature">
        <span class="feature-icon">💡</span>
        <div>Enter your symptoms and let our AI analyze potential conditions</div>
    </div>
    <div class="feature">
        <span class="feature-icon">🧠</span>
        <div>Get accurate predictions with detailed health recommendations</div>
    </div>
    <div class="feature">
        <span class="feature-icon">🍎</span>
        <div>Personalized diet plans, medications, and workout routines</div>
    </div>
</div>
"""

PREDICTION_HTML = """
<div class="header">
    <h1>🔍 Disease Predictor</h1>
    <p>Enter your symptoms to get started</p>
</div>
"""

DISEASE_INFO_HTML = """
<div class="header">
    <h1>📋 Disease Information</h1>
    <p>Comprehensive health resources at your fingertips</p>
</div>
"""

# Disease name cleaning function
def clean_disease_name(name):
    """Standardize disease names for comparison"""
    name = str(name).lower().strip()
    name = name.replace("(", "").replace(")", "")
    name = name.replace("paroymsal", "paroxysmal")  # Fix common typo
    name = name.replace("  ", " ")  # Remove double spaces
    return name

# Cached model and data loading
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}")
        return None
    return pickle.load(open(MODEL_PATH, 'rb'))

@st.cache_data
def load_data():
    data_files = {
        'symptom_severity': 'Symptom-severity.csv',
        'symptoms_df': 'symtoms_df.csv',
        'diets_df': 'diets.csv',
        'meds_df': 'medications.csv',
        'workout_df': 'workout_df.csv',
        'precautions_df': 'precautions_df.csv',
        'desc_df': 'description.csv'
    }
    
    data = {}
    for key, filename in data_files.items():
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            st.error(f"Data file not found: {filepath}")
            continue
        
        df = pd.read_csv(filepath)
        
        # Clean disease names in all dataframes
        if 'Disease' in df.columns:
            df['Disease'] = df['Disease'].str.strip()
            df['Disease'] = df['Disease'].str.replace(r'\s+', ' ', regex=True)  # Remove extra spaces
            
        data[key] = df
    
    return data

# Function to display cover image with animation
def render_cover_image():
    cover_path = "cover.jpg"
    if os.path.exists(cover_path):
        with open(cover_path, "rb") as f:
            encoded_image = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <div style="text-align: center;">
                <img class="cover-img" src="data:image/jpeg;base64,{encoded_image}" alt="Health Cover Image">
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("Cover image not found")

# Main app function
def main():
    # Apply CSS
    st.markdown(CSS, unsafe_allow_html=True)
    
    # Load all required datasets
    model = load_model()
    data = load_data()
    
    # Check if data loaded successfully
    if not data or model is None:
        st.error("Failed to load required data or model. Please check file paths.")
        return
    
    # Extract dataframes
    symptom_severity = data.get('symptom_severity', pd.DataFrame())
    symptoms_df = data.get('symptoms_df', pd.DataFrame())
    diets_df = data.get('diets_df', pd.DataFrame())
    meds_df = data.get('meds_df', pd.DataFrame())
    workout_df = data.get('workout_df', pd.DataFrame())
    precautions_df = data.get('precautions_df', pd.DataFrame())
    desc_df = data.get('desc_df', pd.DataFrame())

    # Build mappings
    symptom_cols = ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']
    disease_symptoms = defaultdict(set)
    symptom_diseases = defaultdict(set)

    for _, row in symptoms_df.iterrows():
        disease = row['Disease']
        symptoms = [row[col].strip() for col in symptom_cols if pd.notna(row[col]) and row[col] != '']
        disease_symptoms[disease].update(symptoms)
        for s in symptoms:
            symptom_diseases[s].add(disease)

    disease_symptoms = {k: sorted(v) for k, v in disease_symptoms.items()}
    symptom_diseases = {k: sorted(v) for k, v in symptom_diseases.items()}

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["🏠 Welcome", "🔍 Predict Disease", "📋 Disease Info", "📊 Visualizations", "ℹ️ About"])

    # Welcome Page
    if page == "🏠 Welcome":
        st.markdown(WELCOME_HTML, unsafe_allow_html=True)
        render_cover_image()
        
        st.markdown(
            """
            <div class="card">
                <h3>🚀 Get Started</h3>
                <p>Select <b>'Predict Disease'</b> from the sidebar to begin your health assessment.</p>
                <div class="animate-pulse" style="text-align: center; font-size: 2rem;">
                    ↓
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Prediction Page
    elif page == "🔍 Predict Disease":
        st.markdown(PREDICTION_HTML, unsafe_allow_html=True)
        
        with st.form("prediction_form"):
            # Improved input section with better styling
            col1, col2 = st.columns([3, 1])
            with col1:
                symptoms_input = st.text_input(
                    'Enter your symptoms (comma separated)',
                    'itching, skin rash, fatigue',
                    help="Example: headache, fever, cough"
                )
            with col2:
                st.write("")  # Spacer
                st.write("")  # Spacer
                submitted = st.form_submit_button(
                    "🔍 Predict Disease",
                    use_container_width=True
                )
            
            if submitted:
                with st.spinner('Analyzing symptoms...'):
                    time.sleep(1)  # Simulate processing time
                    
                    input_symptoms = [s.strip().lower() for s in symptoms_input.split(',')]
                    input_vector = np.zeros(len(model.feature_names_in_))
                    
                    # Create input vector for model
                    for s in input_symptoms:
                        if s in model.feature_names_in_:
                            idx = np.where(model.feature_names_in_ == s)[0][0]
                            input_vector[idx] = 1
                    
                    # Make prediction
                    prediction_idx = model.predict([input_vector])[0]
                    
                    # Get disease names
                    disease_names = sorted(disease_symptoms.keys())
                    try:
                        prediction = disease_names[prediction_idx]
                    except IndexError:
                        prediction = f"Unknown Disease (Code: {prediction_idx})"
                    
                    # Display results with improved layout
                    success_container = st.container()
                    with success_container:
                        col_a, col_b = st.columns([1, 3])
                        with col_a:
                            st.markdown(STETHOSCOPE_HTML, unsafe_allow_html=True)
                        with col_b:
                            st.success(f"**Predicted Disease:** {prediction}")
                    
                    # Show symptom matches with improved visualization
                    st.subheader("🔍 Symptom-based Matches", divider="blue")
                    matches = []
                    for disease, symptoms in disease_symptoms.items():
                        match_count = len(set(symptoms) & set(input_symptoms))
                        if match_count:
                            matches.append((disease, match_count, len(symptoms)))
                    
                    # Sort by match percentage
                    matches_sorted = sorted(
                        matches,
                        key=lambda x: (x[1]/x[2], x[1]),
                        reverse=True
                    )[:5]  # Show top 5
                    
                    for disease, count, total in matches_sorted:
                        percentage = int((count / total) * 100)
                        with st.expander(f"{disease} - {percentage}% match", expanded=True):
                            st.markdown(
                                f"""
                                <div style="margin-bottom: 10px;">
                                    <p><b>{count} out of {total}</b> symptoms matched</p>
                                    <progress value="{percentage}" max="100" style="width: 100%; height: 10px;"></progress>
                                </div>
                                <div>
                                    <p><b>Common symptoms:</b></p>
                                    <ul>
                                """,
                                unsafe_allow_html=True
                            )
                            
                            # Show matching symptoms
                            matched_symptoms = set(disease_symptoms[disease]) & set(input_symptoms)
                            for symptom in matched_symptoms:
                                st.markdown(f"- ✅ {symptom}")
                            
                            # Show missing symptoms
                            missing_symptoms = set(disease_symptoms[disease]) - set(input_symptoms)
                            if missing_symptoms:
                                st.markdown("<p><b>Other symptoms to watch for:</b></p>", unsafe_allow_html=True)
                                for symptom in missing_symptoms:
                                    st.markdown(f"- ◻️ {symptom}")
                            
                            st.markdown("</ul></div>", unsafe_allow_html=True)
                    
                    # Add recommendation section
                    st.markdown("---")
                    st.subheader("📋 Recommended Actions")
                    
                    if len(input_symptoms) < 3:
                        st.warning("For more accurate results, please enter at least 3 symptoms")
                    
                    cols = st.columns(2)
                    with cols[0]:
                        st.markdown("""
                        <div class="card">
                            <h4>🩺 Next Steps</h4>
                            <ul>
                                <li>Monitor your symptoms</li>
                                <li>Note any changes in severity</li>
                                <li>Track symptom duration</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with cols[1]:
                        st.markdown("""
                        <div class="card">
                            <h4>⚠️ When to Seek Help</h4>
                            <ul>
                                <li>Difficulty breathing</li>
                                <li>Severe pain</li>
                                <li>Symptoms worsening</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)

    # Disease Info Page
    elif page == "📋 Disease Info":
        st.markdown(DISEASE_INFO_HTML, unsafe_allow_html=True)
        
        selected_disease = st.selectbox("Select a disease", sorted(disease_symptoms.keys()))
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Symptoms", "Description", "Diet", "Medications", "Precautions"])
        
        # Clean the selected disease name
        clean_selected = clean_disease_name(selected_disease)
        
        with tab1:
            st.subheader(f"Symptoms of {selected_disease}")
            if selected_disease in disease_symptoms:
                for symptom in disease_symptoms[selected_disease]:
                    st.markdown(f"- {symptom}")
            else:
                st.warning("No symptom information available for this disease")
        
        with tab2:  # Description
            desc_match = desc_df[desc_df['Disease'].apply(clean_disease_name) == clean_selected]
            if not desc_match.empty:
                st.info(desc_match['Description'].values[0])
            else:
                st.warning(f"No description found for: {selected_disease}")
        
        with tab3:  # Diet
            diet_match = diets_df[diets_df['Disease'].apply(clean_disease_name) == clean_selected]
            if not diet_match.empty:
                st.dataframe(diet_match.drop(columns=['Disease']), hide_index=True)
            else:
                st.warning(f"No diet info found for: {selected_disease}")
        
        with tab4:  # Medications
            meds_match = meds_df[meds_df['Disease'].apply(clean_disease_name) == clean_selected]
            if not meds_match.empty:
                st.dataframe(meds_match.drop(columns=['Disease']), hide_index=True)
            else:
                st.warning(f"No medications found for: {selected_disease}")
        
        with tab5:  # Precautions
            precautions_match = precautions_df[precautions_df['Disease'].apply(clean_disease_name) == clean_selected]
            if not precautions_match.empty:
                st.dataframe(precautions_match.drop(columns=['Disease']), hide_index=True)
            else:
                st.warning(f"No precautions found for: {selected_disease}")

    # Visualizations Page
    elif page == "📊 Visualizations":
        st.title("📊 Health Insights")
        
        # Symptom Frequency Chart
        st.subheader("Symptom Frequency Chart")
        if not symptoms_df.empty:
            symptom_counts = symptoms_df[symptom_cols].stack().value_counts().head(10)
            st.bar_chart(symptom_counts)
        else:
            st.warning("No symptom data available for visualization")
        
        # Disease Prevalence
        st.subheader("Disease Prevalence")
        if not symptoms_df.empty and 'Disease' in symptoms_df.columns:
            disease_counts = symptoms_df['Disease'].value_counts().head(10)
            st.bar_chart(disease_counts)
        else:
            st.warning("No disease data available for visualization")
    
    # About Page
    elif page == "ℹ️ About":
        st.title("About SymptoCure")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("https://img.icons8.com/color/96/medical-doctor.png", width=100)
        
        with col2:
            st.markdown("""
            <div class="medical-card">
                <h3 style="color: var(--medical-blue);">Our Mission</h3>
                <p>SymptoCure bridges the gap between patients and healthcare knowledge using AI technology. 
                We're making medical information accessible to everyone, anywhere.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Dynamic features section
        features = [
            {"icon": "⚡", "title": "Instant Analysis", "desc": "Get potential diagnoses in seconds"},
            {"icon": "🔍", "title": "Comprehensive Info", "desc": "Detailed disease information at your fingertips"},
            {"icon": "💊", "title": "Treatment Plans", "desc": "Personalized medication and diet recommendations"},
            {"icon": "📊", "title": "Health Insights", "desc": "Visual data to understand symptom patterns"}
        ]
        
        st.subheader("Key Features")
        cols = st.columns(2)
        for i, feature in enumerate(features):
            with cols[i%2]:
                st.markdown(f"""
                <div class="card" style="padding: 15px;">
                    <div style="display: flex; align-items: center; gap: 15px;">
                        <span style="font-size: 30px;">{feature['icon']}</span>
                        <div>
                            <h4 style="margin: 0; color: var(--medical-blue);">{feature['title']}</h4>
                            <p style="margin: 5px 0 0;">{feature['desc']}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Team section
        st.subheader("Our Team")
        team = [
            {"name": "Dr. Smith", "role": "Medical Advisor", "avatar": "👨‍⚕️"},
            {"name": "AI Team", "role": "Machine Learning", "avatar": "🤖"},
            {"name": "Dev Team", "role": "Application Development", "avatar": "💻"}
        ]
        
        cols = st.columns(3)
        for i, member in enumerate(team):
            with cols[i]:
                st.markdown(f"""
                <div style="text-align: center;">
                    <div style="font-size: 40px; margin: 10px 0;">{member['avatar']}</div>
                    <h4 style="margin: 5px 0; color: var(--medical-blue);">{member['name']}</h4>
                    <p style="margin: 0; color: #666;">{member['role']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Dynamic stats
        st.subheader("Our Impact")
        stats = [
            {"value": "10,000+", "label": "Daily Users"},
            {"value": "500+", "label": "Conditions Covered"},
            {"value": "98%", "label": "Accuracy Rate"}
        ]
        
        cols = st.columns(3)
        for i, stat in enumerate(stats):
            with cols[i]:
                st.markdown(f"""
                <div style="text-align: center; background: #f0f5ff; border-radius: 10px; padding: 20px;">
                    <h2 style="margin: 0; color: var(--medical-blue);">{stat['value']}</h2>
                    <p style="margin: 5px 0 0; font-size: 14px;">{stat['label']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Disclaimer
        st.markdown("""
        <div class="card" style="margin-top: 30px; background: #fff8f8; border-left: 4px solid var(--medical-red);">
            <h4 style="color: var(--medical-red); margin-top: 0;">Important Disclaimer</h4>
            <p style="margin-bottom: 0;">
            SymptoCure is an AI-powered informational tool only. It is not a substitute for professional 
            medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider 
            for any health concerns.
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()