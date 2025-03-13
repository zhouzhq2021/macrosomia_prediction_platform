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
    page = st.radio("Page Selection",  
                   ["App Introduction", 
                    "Model Prediction", 
                    "User Guide"],
                   label_visibility="collapsed")  

    st.markdown("---")
    st.markdown("**About the Study**")
    st.markdown("""
    This study aims to integrate a series of physical records, biochemical tests, 
                imaging tests and other clinical examination results during pregnancy, 
                and on the basis of mining the key influencing factors of macrosomia, 
                combine multiple machine learning and deep learning models, 
                and introduce the Ensemble Learning (EL) method to establish a more objective and comprehensive, 
                accurate and efficient prediction model of macrosomia, 
                and overcome the low model accuracy of existing research methods.
    """)

# Page routing
if page == "App Introduction":
    st.title("Ensemble Learning-based Risk Prediction of Macrosomia Occurrence Study")
    st.markdown("### Research Background")
    st.markdown("""
        Macrosomia is one of the most common complications of pregnancy. In clinical practice, 
        macrosomia is typically defined as a birth weight of 4000 grams or more, regardless of gestational age.
        As a pregnancy complication, macrosomia poses significant health and life threats to both the mother and the fetus. 
        
        \n
         
        To date, the accurate diagnosis of macrosomia still relies primarily on the measurement of the infant's weight after birth. 
        Techniques such as two-dimensional ultrasound, three-dimensional ultrasound, and magnetic resonance imaging (MRI) 
        have limitations in accurately estimating fetal weight. Thus, effectively utilizing existing maternal examination 
        results to accurately predict the probability of macrosomia is crucial for improving pregnancy outcomes and 
        safeguarding the health of both mother and fetus.

        
    """)
    
    st.markdown("### Model Architecture")
    # st.markdown(""" """)


    img = Image.open('Stacking.png')  
    st.image(img, caption='Stacking Ensemble Model Structure', width=400)
    

elif page == "Model Prediction":
    st.title("Risk Prediction Interface")
    
    # Create input columns
    col1, col2 = st.columns(2)

    feature_mapping = {
        "BMI": "BMI",
        "怀孕孕周": "Gestational Weeks",
        "空腹葡萄糖": "Fasting Glucose (mmol/L)",
        "25-32周婴儿胎盘厚": "Placental Thickness (mm)",
        "25-32周婴儿腹围": "Abdominal Circumference (mm)",
        "10-20周游离FT4": "Free FT4 (pmol/L)",
        "25-32周婴儿双顶径": "Biparietal Diameter (mm)",
        "25-32周婴儿股骨长": "Femur Length (mm)",
        "婴儿性别": "Fetal Gender",
        "孕妇产次": "Parity",
        "25-32周婴儿胎位": "Fetal Position",
        "1-20周甲状腺过氧化物酶抗体": "TPOAb (IU/mL)",
        "1-20周抗甲状腺过氧化物酶抗体": "Anti-TPO (IU/mL)",
        "1-32周尿葡萄糖最大阳性强度": "Urine Glucose"
    }

    # Continuous features: Using sliders for better UI
    with col1:
        st.header("Continuous Features")

        bmi = st.number_input("BMI", 15.0, 40.0, 25.0)
        gestational_weeks = st.number_input("Gestational Weeks", 20, 42, 32)
        fasting_glucose = st.number_input("Fasting Glucose (mmol/L)", 3.0, 10.0, 5.0)
        placental_thickness = st.number_input("Placental Thickness (mm)", 10.0, 50.0, 25.0)
        abdominal_circumference = st.number_input("Abdominal Circumference (mm)", 200.0, 400.0, 300.0)
        ft4 = st.number_input("Free FT4 (pmol/L)", 5.0, 20.0, 12.0)
        biparietal_diameter = st.number_input("Biparietal Diameter (mm)", 100.0, 500.0, 300.0)
        femur_length = st.number_input("Femur Length (mm)", 100.0, 500.0, 300.0)

# 分类变量
        
    with col2:
        st.header("Categorical Features")
        
        gender = st.selectbox("Fetal Gender", ["Male", "Female"], index=0)
        parity = st.selectbox("Parity", [0, 1, 2, 3], index=0)
        fetal_position = st.selectbox("Fetal Position", ["Cephalic", "Non-Cephalic"], index=0)
        TPOAb = st.selectbox("TPOAb (IU/mL)", ["Negative", "Positive"], index=0)
        anti_tpo = st.selectbox("Anti-TPO (IU/mL)", ["Negative", "Positive"], index=0)
        urine_glucose = st.selectbox("Urine Glucose", ["Negative", "+", "++", "+++", "++++"], index=0)
    
    gender_map = {"Male": 0, "Female": 1}
    fetal_position_map = {"Cephalic": 0, "Non-Cephalic": 1}
    tpoab_map = {"Negative": 0, "Positive": 1}
    anti_tpo_map = {"Negative": 0, "Positive": 1}
    urine_glucose_map = {"Negative": 0, "+": 1, "++": 2, "+++": 3, "++++": 4}

    columns_to_normalize = ['病人年龄', '怀孕孕周', 'BMI', '空腹葡萄糖', '10-20周游离FT4',
                        '25-32周婴儿双顶径', '25-32周婴儿头围', '25-32周婴儿腹围', '25-32周婴儿股骨长', 
                        '25-32周婴儿胎盘厚', '25-32周婴儿脐动脉S/D', '25-32周婴儿胎心', '1小时葡萄糖', '2小时葡萄糖']

    data_to_normalize = {
        '病人年龄': 0,
        "怀孕孕周": gestational_weeks,
        'BMI':bmi,
        "空腹葡萄糖": fasting_glucose,
        "10-20周游离FT4": ft4,
        "25-32周婴儿双顶径": biparietal_diameter,
        "25-32周婴儿头围": 0,
        "25-32周婴儿腹围": abdominal_circumference,
        "25-32周婴儿股骨长": femur_length,
        "25-32周婴儿胎盘厚": placental_thickness,
        '25-32周婴儿脐动脉S/D': 0,
        '25-32周婴儿胎心': 0,
        '1小时葡萄糖': 0,
        '2小时葡萄糖': 0,
    }

    other_data = {
        "婴儿性别": gender_map[gender],  # 映射性别
        "孕妇产次": parity,
        "25-32周婴儿胎位": fetal_position_map[fetal_position],  # 映射胎位
        "1-20周甲状腺过氧化物酶抗体": tpoab_map[TPOAb],  # 映射TPOAb
        "1-20周抗甲状腺过氧化物酶抗体": anti_tpo_map[anti_tpo],  # 映射Anti-TPO
        "1-32周尿葡萄糖最大阳性强度": urine_glucose_map[urine_glucose]  # 映射尿糖
    }

    data_to_normalize_df = pd.DataFrame([data_to_normalize])

    scaled_features = scaler.transform(data_to_normalize_df)

    input_df_1 = pd.DataFrame(scaled_features, columns=data_to_normalize_df.columns)[
        ["BMI", "怀孕孕周", "空腹葡萄糖", "25-32周婴儿胎盘厚", "25-32周婴儿腹围", "10-20周游离FT4", "25-32周婴儿双顶径", "25-32周婴儿股骨长"]
    ]

    other_data_df = pd.DataFrame([other_data])

    input_df = pd.concat([input_df_1, other_data_df], axis=1)
    
    # Normalization and prediction
    if st.button("Predict Risk"):
        try:
            print(input_df)
            risk_prob = model.predict_proba(input_df)[0][1]
                
            # Display results
            st.markdown("---")
            st.subheader("📊 Prediction Results")
                
            # Visual display
            col_result, col_gauge = st.columns(2)
            with col_result:
                st.metric("Macrosomia Risk Probability", f"{risk_prob*100:.1f}%")
                    
                with col_gauge:
                    gauge_html = f"""
                    <div style="width: 100%; background: #f0f2f6; border-radius: 10px; padding: 20px;">
                        <div style="width: {risk_prob*100}%; height: 20px; background: {'#ff4b4b' if risk_prob > 0.5 else '#4CAF50'}; 
                            border-radius: 5px; transition: 0.3s;"></div>
                        <p style="text-align: center; margin-top: 10px;">Risk Level Indicator</p>
                    </div>
                    """
                    st.markdown(gauge_html, unsafe_allow_html=True)
                
            st.markdown("---")
            if risk_prob > 0.7:
                st.error("🚨 High Risk: Recommend clinical consultation and further monitoring. Further tests such as XYZ may be required.")
            elif risk_prob > 0.4:
                st.warning("⚠️ Moderate Risk: Suggest increased monitoring frequency, and consider additional tests like ABC.")
            else:
                st.success("✅ Low Risk: Maintain routine prenatal care. Regular check-ups are recommended.")

        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")

elif page == "User Guide":
    st.title("How to use this Macrosomia Predictive System ")

    
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
    - All data should be collected before 32 weeks gestation
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
<p>Developed by zhouzhq2021@lzu.edu.cn © 2025</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)

# <p>Developed by AIMSLab - Macrosomia Prediction System © 2025</p>
