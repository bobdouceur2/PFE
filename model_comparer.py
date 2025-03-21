import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import classification_report, accuracy_score
import torch
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import streamlit.components.v1 as components
from bs4 import BeautifulSoup


# Streamlit Configuration
st.title("Comparison: Pre-trained DistilBERT Classifier vs Fine-tuned DistilBERT Classifier")
st.write("Comparison of the performance of multiple text classification models.")

# Load dataset
def load_data():
    dataset = load_dataset("dair-ai/emotion")
    test_data = dataset['test']
    return test_data

test_data = load_data()
test_texts = test_data['text']
test_labels = test_data['label']


# Load models
def load_models():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    distilbert_pretrained_classifier = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=6)
    finetuned_model = DistilBertForSequenceClassification.from_pretrained(r"C:\Users\thinn\PycharmProjects\MasterProject\accuracy-92")

    return tokenizer, distilbert_pretrained_classifier, finetuned_model


tokenizer, distilbert_pretrained_classifier, distilbert_finetuned_classifier = load_models()


# Model performance comparison
st.write("### Model Evaluation")


# Function to perform batch predictions
def batch_predict(model, tokenizer, texts, batch_size=128):
    dataloader = DataLoader(texts, batch_size=batch_size, shuffle=False)
    predictions = []

    # Progress bar for predictions
    progress_bar = st.progress(0)
    total_batches = len(dataloader)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            logits = model(**inputs).logits
            batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()  # Convert to numpy on CPU
            predictions.extend(batch_predictions)
            progress_bar.progress((i + 1) / total_batches)

    progress_bar.empty()
    return predictions

# Use session variables to store predictions
if 'predictions_pretrained' not in st.session_state:
    st.write("Generating predictions with the pre-trained model...")
    st.session_state.predictions_pretrained = batch_predict(distilbert_pretrained_classifier, tokenizer, test_texts)
    st.session_state.accuracy_distilbert_pretrained = accuracy_score(test_labels, st.session_state.predictions_pretrained)

if 'predictions_finetuned' not in st.session_state:
    st.write("Generating predictions with the fine-tuned model...")
    st.session_state.predictions_finetuned = batch_predict(distilbert_finetuned_classifier, tokenizer, test_texts)
    st.session_state.accuracy_distilbert_finetuned = accuracy_score(test_labels, st.session_state.predictions_finetuned)

st.write(f"Accuracy of Pre-trained DistilBERT Classifier: {st.session_state.accuracy_distilbert_pretrained * 100:.2f}%")
st.write(f"Accuracy of Fine-tuned DistilBERT Classifier: {st.session_state.accuracy_distilbert_finetuned * 100:.2f}%")


# LIME for prediction explanation
st.write("### Explication avec LIME")
def explain_and_visualize_with_lime(text, model_type, tokenizer, model, class_names):
    explainer = LimeTextExplainer(class_names=class_names)

    def predictor(texts):
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1).numpy()
        return probabilities

    probabilities = predictor([text])[0]
    predicted_label = np.argmax(probabilities)

    st.write(f"Predicted Sentiment: {class_names[predicted_label]}")

    fig, ax = plt.subplots()
    ax.barh(class_names, probabilities, color="orange")
    ax.set_xlabel("Probability")
    ax.set_title("Prediction Probabilities")
    st.pyplot(fig)

    st.write("### Interpretability")
    Interpretability = explainer.explain_instance(text, predictor, top_labels=1, num_features=10, num_samples=500)
    fig_exp = Interpretability.as_pyplot_figure(label=predicted_label)
    html_content = Interpretability.as_html()
    components.html(html_content, height=600)
    return fig_exp


class_names = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]

st.write("### Test a custom sentence")
example_text = "Nice! I am still struggling with my homeworks." #I like eating cake, but when I do, I get sick.
user_input = st.text_area("Enter your sentence here:", "Nice! I am still struggling with my homeworks.")

if st.button("Analyze with LIME"):
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Pre-trained DistilBERT Classifier")
        pretrained_explanation = explain_and_visualize_with_lime(user_input, "distilbert_pretrained", tokenizer, distilbert_pretrained_classifier, class_names)
        st.pyplot(pretrained_explanation)

    with col2:
        st.write("### Fine-tuned DistilBERT Classifier")
        finetuned_explanation = explain_and_visualize_with_lime(user_input, "distilbert_finetuned", tokenizer, distilbert_finetuned_classifier, class_names)
        st.pyplot(finetuned_explanation)

# Afficher l'explication initiale avec LIME
if 'initial_explanation_done' not in st.session_state:
    col1, col2 = st.columns(2)

    with col1:
        st.write("### DistilBERT Classifier pré-entraîné")
        pretrained_explanation = explain_and_visualize_with_lime(example_text, "distilbert_pré-entraîné", tokenizer, distilbert_pretrained_classifier, class_names)
        st.pyplot(pretrained_explanation)

    with col2:
        st.write("### DistilBERT Classifier fine-tuné")
        finetuned_explanation = explain_and_visualize_with_lime(example_text, "distilbert_finetuned", tokenizer, distilbert_finetuned_classifier, class_names)
        st.pyplot(finetuned_explanation)

    st.session_state.initial_explanation_done = True




