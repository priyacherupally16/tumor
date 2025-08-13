import streamlit as st
import numpy as np
from PIL import Image
import joblib
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import pickle
import os

# Class names
CLASSES = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']

# Load model & feature extractor
@st.cache_resource(hash_funcs={Model: lambda _: None})
MODEL_PATH = "rf_model.pkl"

@st.cache_resource(hash_funcs={Model: lambda _: None})
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Upload it to your repo or download it dynamically.")
        st.stop()

    # Use pickle instead of joblib if it was saved with pickle
    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)
    return clf, feature_extractor


def extract_features(img, feature_extractor):
    img = img.resize((224, 224), Image.LANCZOS)
    img_array = np.array(img)
    if img_array.shape[-1] == 4:  # Convert RGBA to RGB if needed
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = feature_extractor.predict(img_array)
    features = features.reshape(features.shape[0], -1)
    return features

def predict(img, clf, feature_extractor):
    features = extract_features(img, feature_extractor)
    pred = clf.predict(features)[0]
    proba = clf.predict_proba(features)[0]
    return pred, proba

def home_page():
    st.title("Brain Tumor Detection Application")
    st.header("Brain Tumor Education and Awareness")
    st.markdown("""
    ### What is a Brain Tumor?
    A brain tumor is an abnormal growth of cells within the brain or central spinal canal. They can be **benign** or **malignant**.
    
    **Types:**
    - **Glioma** – Tumors from glial cells, can be aggressive.
    - **Meningioma** – Usually benign, from brain/spinal cord membranes.
    - **Pituitary Tumor** – In pituitary gland, affecting hormones.
    - **No Tumor** – Normal brain MRI.
    
    **Common Symptoms:**
    Headaches, seizures, vision problems, nausea, balance issues, speech changes, memory problems, weakness, personality changes, fatigue.
    """)

def prediction_page(clf, feature_extractor):
    st.title("MRI Image Tumor Prediction")
    uploaded_file = st.file_uploader("Upload MRI Brain Scan Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded MRI Image", use_column_width=True)
        
        if st.button("Predict Tumor Type"):
            with st.spinner("Analyzing image..."):
                pred, proba = predict(img, clf, feature_extractor)
                st.success(f"Predicted Tumor Type: **{CLASSES[pred]}**")
                
                # Pie chart
                fig, ax = plt.subplots()
                ax.pie(proba, labels=CLASSES, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors[:len(CLASSES)])
                ax.axis('equal')
                st.pyplot(fig)
                plt.close(fig)

def self_assessment_page():
    st.title("Self Assessment: Brain Tumor Risk Analysis")
    st.markdown("Answer the questions below with Yes or No to evaluate potential risk based on symptoms.")
    
    symptoms = [
        "Persistent headaches",
        "Seizures or convulsions",
        "Vision problems",
        "Nausea or vomiting",
        "Difficulty with balance or walking",
        "Changes in speech or hearing",
        "Memory problems or confusion",
        "Weakness or numbness in limbs",
        "Personality or behavior changes",
        "Fatigue or drowsiness",
    ]
    
    responses = {symptom: st.radio(symptom, options=["No", "Yes"], index=0, horizontal=True) for symptom in symptoms}
    
    if st.button("Calculate Risk"):
        score = sum(1 for answer in responses.values() if answer == "Yes")
        st.write(f"Total symptoms reported as 'Yes': {score}")
        
        if score == 0:
            st.success("Low risk. No significant symptoms detected.")
        elif 1 <= score <= 3:
            st.warning("Moderate risk. Consider consulting a healthcare professional if symptoms persist.")
        else:
            st.error("High risk. Please consult a doctor immediately.")

def main():
    st.sidebar.title("Navigation")
    options = ["Home", "MRI Prediction", "Self Assessment"]
    choice = st.sidebar.radio("Select a page", options)
    
    clf, feature_extractor = load_model()
    
    if choice == "Home":
        home_page()
    elif choice == "MRI Prediction":
        prediction_page(clf, feature_extractor)
    else:
        self_assessment_page()

if __name__ == "__main__":
    main()







