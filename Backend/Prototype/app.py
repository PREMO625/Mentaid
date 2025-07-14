import streamlit as st
import os
from datetime import datetime
import json
import plotly.express as px
import pandas as pd
from liwc_analyzer import LIWCAnalyzer

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

# Page configuration
st.set_page_config(
    page_title="Mentaid Prototype",
    page_icon="🧠",
    layout="wide"
)

# Initialize LIWC analyzer
liwc_analyzer = LIWCAnalyzer()

def analyze_text(text):
    """
    Function to analyze text and return psychological category analysis
    """
    if not text:
        return None
    
    # Get detailed analysis
    return liwc_analyzer.get_category_stats(text)

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
            entries = load_from_local_storage("journal_entries") or []
            # Ensure it's a list (handles old dict storage bug)
            if not isinstance(entries, list):
                entries = [entries]
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
import pandas as pd
import matplotlib.pyplot as plt
import cloudinary.uploader
import cloudinary.api

# Initialize models
svm_model = SVMModel()
nlp_model = NLPModel()

# Initialize explainers
# Explainability tools will be initialized lazily in the dashboard to save memory

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

        # Determine class labels (0: Ctrl, 1: SCZ)
        svm_label = 1 if np.mean(svm_predictions) >= 0.5 else 0
        nlp_label = 1 if np.mean(nlp_predictions) >= 0.5 else 0
        # Calculate ensemble prediction
        ensemble_prediction = (np.mean(svm_predictions) + np.mean(nlp_predictions)) / 2
        # Threshold adapted: require both models high or average >0.6
        ensemble_label = 1 if ensemble_prediction >= 0.6 else 0
        
        # Show chunking details
        st.markdown(f"**SVM chunks:** {len(svm_chunks)} | **NLP chunks:** {len(nlp_chunks)}")
        
        # Per-chunk tables
        svm_df = pd.DataFrame({"chunk": list(range(1, len(svm_chunks)+1)), "svm_prob": svm_predictions})
        nlp_df = pd.DataFrame({"chunk": list(range(1, len(nlp_chunks)+1)), "nlp_prob": nlp_predictions})
        cols = st.columns(2)
        with cols[0]:
            st.subheader("Per-chunk SVM probs")
            st.dataframe(svm_df, height=250)
        with cols[1]:
            st.subheader("Per-chunk NLP probs")
            st.dataframe(nlp_df, height=250)
        
        # Display predictions
        st.subheader("Model Predictions")
        st.markdown(f"**SVM probability:** {np.mean(svm_predictions):.2f} → **Label:** {'SCZ (1)' if svm_label else 'Ctrl (0)'}")
        st.markdown(f"**NLP probability:** {np.mean(nlp_predictions):.2f} → **Label:** {'SCZ (1)' if nlp_label else 'Ctrl (0)'}")
        st.markdown(f"**Ensemble probability:** {ensemble_prediction:.2f} → **Label:** {'SCZ (1)' if ensemble_label else 'Ctrl (0)'}")
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
            
        # ---- Explainability for SVM ----
        st.subheader("Explainability for SVM")

        # Wrapper that returns probability of class 1 from SVM
        def svm_proba(text_list):
            X = svm_model.vectorizer.transform(text_list)
            if getattr(svm_model.model, "probability", False):
                return svm_model.model.predict_proba(X)[:, 1]
            scores = svm_model.model.decision_function(X)
            return 1 / (1 + np.exp(-scores))

        # ---------- SHAP ----------
        if "svm_shap_explainer" not in st.session_state:
            # Initialize text masker explainer once
            masker = shap.maskers.Text()
            st.session_state["svm_shap_explainer"] = shap.Explainer(svm_proba, masker)
        shap_explainer = st.session_state["svm_shap_explainer"]
        shap_exp = shap_explainer([latest_entry["text"]], max_evals=2000)
        shap_vals = shap_exp[0].values
        tokens = shap_exp[0].data

        top_idx = np.argsort(np.abs(shap_vals))[::-1][:20]
        shap_df = pd.DataFrame({
            "token": np.array(tokens)[top_idx],
            "shap": shap_vals[top_idx]
        })
        st.markdown("**Top-20 SHAP tokens**")
        abs_max = np.max(np.abs(shap_vals))
        styled = shap_df.style.background_gradient(cmap="RdBu", vmin=-abs_max, vmax=abs_max)
        st.dataframe(styled)

                # ---------- LIME ----------
        st.subheader("LIME (SVM)")
        lime_explainer = LimeTextExplainer(class_names=["Ctrl", "SCZ"])

        def svm_proba_lime(text_list):
            p1 = svm_proba(text_list)
            return np.column_stack((1 - p1, p1))  # shape (n_samples, 2)

        lime_exp = lime_explainer.explain_instance(
            latest_entry["text"],
            svm_proba_lime,
            num_features=15,
            num_samples=1000,
            labels=[1],
        )
        st.components.v1.html(lime_exp.as_html(), height=350, scrolling=True)

        # ---- Final flag message ----
        st.markdown("### Final Assessment")
        flag_text = "**Flagged as SCZ (1)**" if ensemble_label else "**Labelled Ctrl (0)**"
        st.markdown(flag_text)
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

    return None
        
    # Get detailed analysis
    return liwc_analyzer.get_category_stats(text)

def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    # Main content
    st.title("Text Analysis Dashboard")
    
    # Text input
    text_input = st.text_area("Enter text to analyze:", height=200)
    
    if st.button("Analyze Text"):
        if text_input:
            analysis = analyze_text(text_input)
            if analysis:
                st.subheader("Analysis Results")
                
                # Display basic statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Words", analysis['total_words'])
                    st.metric("Average Word Length", f"{analysis['avg_word_length']:.2f}")
                with col2:
                    st.metric("Total Sentences", analysis['total_sentences'])
                    st.metric("Most Prominent Emotion", analysis['most_prominent_emotion'])
                
                # Display emotion scores
                st.subheader("Emotion Analysis")
                emotion_categories = [cat for cat in analysis['category_scores'] if "emotion" in cat]
                emotion_scores = {cat: analysis['category_scores'][cat] for cat in emotion_categories}
                
                # Create bar chart for emotions
                emotions_df = pd.DataFrame(list(emotion_scores.items()), columns=['Emotion', 'Score'])
                fig = px.bar(emotions_df, x='Emotion', y='Score', title='Emotion Distribution')
                st.plotly_chart(fig)
                
                # Display other category scores
                st.subheader("Psychological Categories")
                other_categories = [cat for cat in analysis['category_scores'] if cat not in emotion_categories]
                other_scores = {cat: analysis['category_scores'][cat] for cat in other_categories}
                
                # Create radar chart for other categories
                other_df = pd.DataFrame(list(other_scores.items()), columns=['Category', 'Score'])
                fig = px.line_polar(other_df, r='Score', theta='Category', line_close=True, title='Psychological Profile')
                st.plotly_chart(fig)
                
                # Display top categories
                st.subheader("Top Categories")
                sorted_categories = sorted(analysis['category_scores'].items(), key=lambda x: x[1], reverse=True)[:5]
                for cat, score in sorted_categories:
                    st.write(f"- {cat}: {score:.2f}%")
        else:
            st.warning("Please enter some text to analyze")
    page = st.sidebar.radio("Go to", ["User Interface", "Clinician Dashboard"])
    
    if page == "User Interface":
        user_interface()
    else:
        clinician_dashboard()

if __name__ == "__main__":
    main()
