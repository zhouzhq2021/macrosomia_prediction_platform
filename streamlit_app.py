import streamlit as st
import joblib
import pandas as pd
import numpy as np
from PIL import Image

# Load models
@st.cache_resource
def load_models():
    model = joblib.load('stacking_model.pkl')
    scaler = joblib.load('z_score_scaler.pkl')
    return model, scaler

model, scaler = load_models()

# Sidebar
with st.sidebar:
    st.title("Navigation")
    page = st.radio("", ["App Introduction", 
                        "Model Prediction", 
                        "User Guide"])

    st.markdown("---")
    st.markdown("**About the Study**")
    st.markdown("""
    This research focuses on developing an ensemble learning model to predict macrosomia risk 
    using maternal and fetal characteristics during pregnancy.
    """)

# Page routing
if page == "🏠 App Introduction":
    st.title("📈 Macrosomia Risk Prediction System")
    st.markdown("### 🔍 Research Background")
    st.markdown("""
    Macrosomia (birth weight ≥4000g) poses significant risks to both mothers and infants. 
    This ensemble learning model integrates multiple clinical parameters to provide early risk assessment.
    """)
    
    st.markdown("### 🧠 Model Architecture")
    try:
        img = Image.open('images/model_structure.png')  # Prepare your model structure image
        st.image(img, caption='Stacking Ensemble Model Structure')
    except:
        pass
    
    st.markdown("### 🏋️ Training Process")
    st.markdown("""
    - Dataset: Clinical data from 5000 pregnancies
    - Features: 14 key maternal/fetal parameters
    - Algorithm: Stacking ensemble of XGBoost, Random Forest, and Logistic Regression
    - Validation: 5-fold cross-validation (AUC: 0.92)
    """)

elif page == "🤖 Model Prediction":
    st.title("Risk Prediction Interface")
    
    # Create input columns
    col1, col2 = st.columns(2)
    
    # Continuous features
    with col1:
        st.header("Continuous Features")
        cont_features = {
            "BMI": st.number_input("BMI", 15.0, 40.0, 25.0),
            "怀孕孕周": st.number_input("Gestational Weeks", 20, 42, 32),
            "空腹葡萄糖": st.number_input("Fasting Glucose (mmol/L)", 3.0, 10.0, 5.0),
            "25-32周婴儿胎盘厚": st.number_input("Placental Thickness (mm)", 1.0, 5.0, 2.5),
            "25-32周婴儿腹围": st.number_input("Abdominal Circumference (mm)", 20.0, 40.0, 30.0),
            "10-20周游离FT4": st.number_input("Free FT4 (pmol/L)", 5.0, 20.0, 12.0),
            "25-32周婴儿双顶径": st.number_input("Biparietal Diameter (mm)", 6.0, 10.0, 8.0),
            "25-32周婴儿股骨长": st.number_input("Femur Length (mm)", 4.0, 8.0, 6.0)
        }

    # Categorical features
    with col2:
        st.header("Categorical Features")
        cat_features = {
            "婴儿性别": st.selectbox("Fetal Gender", ["Male", "Female"]),
            "孕妇产次": st.selectbox("Parity", [0, 1, 2, 3]),
            "25-32周婴儿胎位": st.selectbox("Fetal Position", ["Cephalic", "Non-Cephalic"]),
            "1-20周甲状腺过氧化物酶抗体": st.selectbox("TPOAb (IU/mL)", ["Negative", "Positive"]),
            "1-20周抗甲状腺过氧化物酶抗体": st.selectbox("Anti-TPO (IU/mL)", ["Negative", "Positive"]),
            "1-32周尿葡萄糖最大阳性强度": st.selectbox("Urine Glucose", ["Negative", "+", "++", "+++","++++"])
        }

    # Create feature dataframe
    feature_mapping = {
        # Continuous features
        "BMI": "BMI",
        "Gestational Weeks": "怀孕孕周",
        "Fasting Glucose (mmol/L)": "空腹葡萄糖",
        "Placental Thickness (mm)": "25-32周婴儿胎盘厚",
        "Abdominal Circumference (mm)": "25-32周婴儿腹围",
        "Free FT4 (pmol/L)": "10-20周游离FT4",
        "Biparietal Diameter (mm)": "25-32周婴儿双顶径",
        "Femur Length (mm)": "25-32周婴儿股骨长",
        
        # Categorical features
        "Fetal Gender": ("婴儿性别", {"Male": 1, "Female": 0}),
        "Parity": "孕妇产次",
        "Fetal Position": ("25-32周婴儿胎位", {"Cephalic": 0, "Non-Cephalic": 1}),
        "TPOAb (IU/mL)": ("1-20周甲状腺过氧化物酶抗体", {"Negative": 0, "Positive": 1}),
        "Anti-TPO (IU/mL)": ("1-20周抗甲状腺过氧化物酶抗体", {"Normal": 0, "Positive": 1}),
        "Urine Glucose": ("1-32周尿葡萄糖最大阳性强度", {"Negative": 0, "+": 1, "++": 2, "+++": 3, "++++": 4})
    }

    # Convert categorical features
    processed_features = {}
    for feature, value in cat_features.items():
        if isinstance(feature_mapping[feature], tuple):
            chinese_name, mapping = feature_mapping[feature]
            processed_features[chinese_name] = mapping[value]
        else:
            processed_features[feature_mapping[feature]] = value

    # Combine all features
    all_features = {**cont_features, **processed_features}
    input_df = pd.DataFrame([all_features])

    # Normalization and prediction
    if st.button("Predict Risk"):
        try:
            # Normalization
            scaled_features = scaler.transform(input_df)
            
            # Prediction
            risk_prob = model.predict_proba(scaled_features)[0][1]
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            # Visual display
            col_result, col_gauge = st.columns(2)
            with col_result:
                st.metric("Macrosomia Risk Probability", f"{risk_prob*100:.1f}%")
                
            with col_gauge:
                # Create a simple gauge visualization
                gauge_html = f"""
                <div style="width: 100%; background: #f0f2f6; border-radius: 10px; padding: 20px;">
                    <div style="width: {risk_prob*100}%; height: 20px; background: {'#ff4b4b' if risk_prob > 0.5 else '#4CAF50'}; 
                        border-radius: 5px; transition: 0.3s;"></div>
                    <p style="text-align: center; margin-top: 10px;">Risk Level Indicator</p>
                </div>
                """
                st.markdown(gauge_html, unsafe_allow_html=True)
            
            # Add interpretation
            st.markdown("---")
            if risk_prob > 0.7:
                st.error("🚨 High Risk: Recommend clinical consultation and further monitoring")
            elif risk_prob > 0.4:
                st.warning("⚠️ Moderate Risk: Suggest increased monitoring frequency")
            else:
                st.success("✅ Low Risk: Maintain routine prenatal care")

        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")

elif page == "📖 User Guide":
    st.title("📝 User Manual")
    
    st.markdown("### 🖥 Interface Overview")
    st.image("images/interface_demo.png", caption="Prediction Interface Layout")  # Prepare demo image
    
    st.markdown("### 📋 Input Guidelines")
    st.markdown("""
    1. **Continuous Parameters**  
    - Obtain from clinical measurements  
    - Input exact numerical values
    
    2. **Categorical Parameters**  
    - Select from standardized clinical reports  
    - Use most recent measurement values
    """)
    
    st.markdown("### ⚠️ Precautions")
    st.markdown("""
    - All data should be collected between 25-32 weeks gestation
    - Measurement methods must follow standard protocols
    - Results should be interpreted by qualified clinicians
    """)

# Add footer
st.markdown("---")
footer = """<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: white;
    color: black;
    text-align: center;
}
</style>
<div class="footer">
<p>Developed by AIMSLab - Macrosomia Prediction System © 2025</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
