import streamlit as st
import os
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="Mentaid Prototype",
    page_icon="🧠",
    layout="wide"
)

def save_to_local_storage(key, value):
    """Save data to browser local storage"""
    try:
        st.session_state[key] = json.dumps(value)
        return True
    except Exception as e:
        st.error(f"Error saving to local storage: {str(e)}")
        return False

def load_from_local_storage(key):
    """Load data from browser local storage"""
    try:
        return json.loads(st.session_state.get(key, "{}"))
    except Exception as e:
        st.error(f"Error loading from local storage: {str(e)}")
        return {}

def user_interface():
    st.title("Mentaid - User Dashboard")
    
    # Mood rating
    st.subheader("How are you feeling today?")
    mood = st.selectbox(
        "Select your mood",
        ["😀 Very Happy", "🙂 Happy", "😐 Neutral", "😔 Sad", "😢 Very Sad"]
    )
    
    # Journal entry
    st.subheader("Write your journal entry")
    journal_text = st.text_area("Share your thoughts...", height=200)
    
    if st.button("Save Entry"):
        if journal_text:
            entry = {
                "mood": mood,
                "text": journal_text,
                "timestamp": datetime.now().isoformat(),
                "user_id": "user_1"
            }
            # Save to local storage
            entries = load_from_local_storage("journal_entries")
            entries.append(entry)
            save_to_local_storage("journal_entries", entries)
            st.success("Journal entry saved successfully!")
        else:
            st.warning("Please write something in the journal entry!")

from models.svm_model import SVMModel
from models.nlp_model import NLPModel
from text_utils import chunk_text_svm, chunk_text_nlp, average_predictions
import plotly.graph_objects as go
import shap
import lime.lime_text
from lime.lime_text import LimeTextExplainer
import numpy as np
import matplotlib.pyplot as plt
import cloudinary.uploader
import cloudinary.api

# Initialize models
svm_model = SVMModel()
nlp_model = NLPModel()

# Initialize explainers
svm_explainer = shap.Explainer(svm_model.model)
nlp_explainer = LimeTextExplainer(class_names=['Not Severe', 'Severe'])

def clinician_dashboard():
    st.title("Mentaid - Clinician Dashboard")
    
    # Load latest journal entry from local storage
    entries = load_from_local_storage("journal_entries")
    if entries:
        latest_entry = entries[-1]
        st.subheader("Latest Journal Entry")
        st.write(latest_entry["text"])
        
        # Get chunks for both models
        svm_chunks = chunk_text_svm(latest_entry["text"])
        nlp_chunks = chunk_text_nlp(latest_entry["text"])
        
        # Get predictions
        svm_predictions = svm_model.predict_chunks(svm_chunks)
        nlp_predictions = nlp_model.predict_chunks(nlp_chunks)
        
        # Calculate ensemble prediction
        ensemble_prediction = (np.mean(svm_predictions) + np.mean(nlp_predictions)) / 2
        
        # Display predictions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("SVM Prediction", f"{np.mean(svm_predictions):.2f}")
            st.plotly_chart(create_prediction_chart(svm_predictions), use_container_width=True)
            
        with col2:
            st.metric("NLP Prediction", f"{np.mean(nlp_predictions):.2f}")
            st.plotly_chart(create_prediction_chart(nlp_predictions), use_container_width=True)
            
        with col3:
            st.metric("Ensemble Prediction", f"{ensemble_prediction:.2f}")
            st.plotly_chart(create_prediction_chart([ensemble_prediction]), use_container_width=True)
            
        # Display SHAP explanation
        st.subheader("SHAP Explanation (SVM)")
        
        # Preprocess and vectorize the text for SHAP, just like for prediction
        processed_text = svm_model.preprocess_text(latest_entry["text"])
        vectorized_text = svm_model.vectorizer.transform([processed_text])
        
        # Get SHAP values for the vectorized text
        shap_values = svm_explainer(vectorized_text)
        
        # Use the first explanation for the bar plot
        shap.plots.bar(shap_values[0])
        plt.savefig("shap_plot.png")
        st.image("shap_plot.png")
        
        # Display LIME explanation
        st.subheader("LIME Explanation (NLP)")
        exp = nlp_explainer.explain_instance(latest_entry["text"], 
                                            nlp_model.predict, 
                                            num_features=10)
        exp.save_to_file("lime.html")
        st.components.v1.html(open("lime.html").read(), height=800)
    else:
        st.warning("No journal entries found in local storage!")

def create_prediction_chart(predictions):
    """Create a plotly chart showing prediction distribution"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=predictions,
        mode='lines+markers',
        name='Predictions'
    ))
    fig.update_layout(
        title='Prediction Distribution',
        yaxis_title='Prediction Score',
        xaxis_title='Chunk Number'
    )
    return fig

def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["User Interface", "Clinician Dashboard"])
    
    if page == "User Interface":
        user_interface()
    else:
        clinician_dashboard()

if __name__ == "__main__":
    main()
