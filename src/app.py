from train_and_save import PREPROCESSOR
import matplotlib.pyplot as plt
import plotly.express as px
import tensorflow as tf
import streamlit as st
import pandas as pd
import numpy as np
import time
import cv2
import os

BASE_DIR = os.getcwd()
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

MODELS = [
    'EfficientNetB5',
    'DenseNet121',
    'ResNet50V2',
    'MobileNetV2',
]

CLASS_LABELS = [
    'buildings', 
    'forest', 
    'glacier', 
    'mountain', 
    'sea', 
    'street'
]

def preprocess_image(uploaded_image, model_name, IMG_SIZE=(224, 224)):
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    resized_image = cv2.resize(image_rgb, IMG_SIZE)

    preprocess_fn = PREPROCESSOR[model_name]
    preprocessed_image = preprocess_fn(resized_image)

    image = np.expand_dims(preprocessed_image, axis=0)
    
    return image

@st.cache_resource(show_spinner=False)
def load_all_models():
    models = {}
    for model_name in MODELS:
        models[model_name] = tf.keras.models.load_model(os.path.join(CHECKPOINT_DIR, f'{model_name}_finetune.keras'))
    return models

def predict_label(models_dict, model_name, image):
    model = models_dict[model_name]

    start_time = time.time()
    prediction = model.predict(image)[0]
    end_time = time.time()
    inference_time = end_time - start_time

    index = np.argmax(prediction)
    predicted_class = CLASS_LABELS[index]
    
    confidence = prediction[index] * 100

    return predicted_class, confidence, inference_time

def build_UI():
    models_dict = load_all_models()

    st.set_page_config(page_title='Image Classification - Compare 4 Models')
    st.title(body='Image Classification - Compare 4 Models')

    uploaded_image = st.file_uploader(
        label='Upload your image here',
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=False,
    )

    if uploaded_image is None:
        st.warning(body='You must upload your image to continue!')
        st.stop()
    
    with st.spinner('Uploading...'):
        st.image(uploaded_image, caption='Your uploaded image', width='stretch')

    predict = st.button(label='Predict', width='stretch')

    if not predict:
        st.stop()

    with st.spinner(text='Predicting...'):
        image = preprocess_image(uploaded_image=uploaded_image, model_name='EfficientNetB5')
        
        results = []
        for model_name in MODELS:
            predicted_class, confidence, inference_time = predict_label(models_dict=models_dict, model_name=model_name, image=image)
            results.append({
                "Model": model_name,
                "Predicted Class": predicted_class,
                "Confidence (%)": confidence,
                "Inference Time (s)": inference_time,
            })
        results_df = pd.DataFrame(results)
        st.header(body='Inference Results')
        results_df

        best_index = results_df['Confidence (%)'].idxmax()
        best_result = results_df.loc[best_index]
        best_model = best_result['Model']
        best_class = best_result['Predicted Class']
        best_confidence = best_result['Confidence (%)']
        st.success(
            f'Best Prediction: {best_model} -> {best_class} ({best_confidence:.2f}%)'
        )

    with st.spinner(text='Preparing...'):
        fig = px.bar(
            data_frame=results_df,
            x='Model',
            y='Confidence (%)',
            color='Model',
            title='Model Confidence Comparison',
            text='Confidence (%)',
        )

        fig.update_traces(
            texttemplate='%{text:.2f}%',
            textposition='outside',
        )

        fig.update_layout(
            yaxis_range=[0, 110],
            template='plotly_dark',
        )

        st.header(body='Confidence Comparison Among Models')
        st.plotly_chart(fig)

    with st.spinner(text='Preparing...'):
        fig = px.bar(
            data_frame=results_df,
            x='Model',
            y='Inference Time (s)',
            color='Model',
            title='Model Inference Time Comparison',
            text='Inference Time (s)',
        )

        fig.update_traces(
            texttemplate='%{text:.2f}s',
            textposition='outside',
        )

        fig.update_layout(
            yaxis_range=[0, 10],
            template='plotly_dark',
        )

        st.header(body='Inference Time Comparison Among Models')
        st.plotly_chart(fig)

if __name__ == '__main__':
    build_UI()