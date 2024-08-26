#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


# In[2]:


st.title('mortality Risk Prediction for Patients with severe TCSCI')


# In[5]:


#1.Age
Age= st.number_input(label='Age(y)')
#2 CCI
CCI=st.slider(label='Charlson Comorbidity Index')
#3.ISS
ISS=st.slider(label='Injury severity score')
#4 Thoracic and abdominal organs damage
Thoracic_abdominal_organs_damage=st.radio(label='Thoracic abdominal organs damage',options=['Non-damage','Single','multiple'])
#5 Cervical fracture
Cervical_fracture=st.radio(label='Cervical fracture',options=['Non-fractures','Upper(C1-2)','Lower(C3-7)'])
#6.NLI
NLI=st.radio(label='NLI',options=['C1-C4','C5-C8'])
#7 Time of injury
Time_injury=st.slider(label='Time of injury')
#8 Surgery timing
Surgery_timing=st.radio(label='Surgery Timing',
                   options=['Non-surgery','Early','Delay'])
#9.Transfusion
Transfusion=st.radio(label='Transfusion',options=['No surgery','Surgery with transfusion','Surgery without transfusion'])
#10.Critical care
Critical_care=st.radio(label='Critical care',options=['No','Yes'])
#11. Nourishment
Nourishment=st.radio(label='Nourishment',
                   options=['Normal','Enteral','Parenteral'])


# In[7]:


#采集数据
if st.button("Predict"):
    # Unpickle classifier
    RSF = joblib.load("RSF.pkl")
    explainer=joblib.load("Explainer.pkl")
    # Store inputs into dataframe
    X = pd.DataFrame([[Age,CCI,ISS,Thoracic_abdominal_organs_damage,Cervical_fracture,NLI,Time_injury,
                    Surgery_timing,Transfusion,Critical_care,Nourishment]], 
                    columns = ['Age', 'CCI','ISS','Thoracic and abdominal organs damage','Cervical fracture',
                              'NLI','Time of injury','Surgery timing','Transfusion','Critical care','Nourishment'])
    X = X.replace(["C1-C4", "C5-C8"], [1, 0])
    X = X.replace(["Yes", "No"], [1, 0])
    X = X.replace(['Non-damage','Single','multiple'], [0,1,2])
    X = X.replace(['Non-fractures','Upper(C1-2)','Lower(C3-7)'], [0,1,2])
    X = X.replace(['Non-surgery','Early','Delay'], [0,1,2])
    X = X.replace(['Normal','Enteral','Parenteral'], [0,1,2])
    X = X.replace(['No surgery','Surgery with transfusion','Surgery without transfusion'],[0,1,2])
    #结果
    def survival_time(model,patient):
        va_times=np.arange(0,60)
        chf_funcs=model.predict_cumulative_hazard_function(patient)
        Time=()
        for fn in chf_funcs:#
            if fn(va_times[-1])<0.5:#在最后的预测时间内死亡全部累计概率不到0.6
                time_value=999
                Time=('This patient had no predicted death for 60 months')
                return Time
            else:
                for time in va_times:
                    if fn(time)>1:
                        time_value=time#发生结局的最短时间
                        break
                Time=('The prognosis survival time of the patients was expected to be {} months'.format(time)) 
                return Time
    prediction = RSF.predict(X)[0]
    patient = X[X.index==0]
    ST = survival_time(RSF,patient)
    
    def risk_groups(model,patient):
        y_risk=model.predict(patient)
        group=()
        for fn in y_risk:#
            if fn<20:
                group=('Low-risk group')
                return group
            if 20<=fn<45.5:
                group=('Medium-risk group')
                return group
            if fn>=45.5:
                group=('High-risk group')
                return group 
    #预测死亡时间
    patient = X[X.index==0]
    rg=risk_groups(RSF,patient)
    shap_values = explainer(patient[0:1])
    shap.plots.force(shap_values,matplotlib=True,show=False,contribution_threshold=0.01)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    # Output prediction
    st.header('outcome prediction')
    st.text(f"mortality risk:{rg}")
    st.text(f"Predicting Outcomes:{ST}")
    st.image("shap_force_plot.png")


# In[ ]:




