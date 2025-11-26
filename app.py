import streamlit as st
import pandas as pd
import numpy as np
import pickle


reg=pickle.load(open('factors_affecting_heartattack.pkl','rb'))
ohe=pickle.load(open('OneHot_Encoder.pkl','rb'))

## for title heading
st.title("HEART ATTACK PREDICTION")
st.write("This app predicts the risk of heart attack based on various health parameters.")
#if st.sidebar.button("Basic Information"):
 #       choice = st.radio("What do you want to enter?", ["Name", "Gender", "Age"])
        #a=st.selectbox("Select to Enter", [" ","Name", "Gender","Age"])
  #      if choice=="Name":
   #         name=st.text_input("Enter Your Name")
    #    elif choice=="Gender":
     #       gender=st.selectbox("Select Gender",ohe.categories_[0])
      #  elif choice=="Age":
       #     age=st.slider("Age(in years)",0,100)
        
#if st.sidebar.selectbox("Basic Information",[" ","Name","Gender","Age"]):
name = st.text_input("Enter Your Name")
gender = st.selectbox("Select Gender", ohe.categories_[0])
age = st.slider("Age (in years)", 0, 100)
    
    
    
        
#name=st.text_input("Enter Your Name")
#gender=st.selectbox("Select Gender",ohe.categories_[0])

#age=st.slider("Age(in years)",0,100)])
## dropdowm for gender selection


cholestrol=st.number_input("Enter Cholestrol Level",min_value=0,max_value=500)
high_bp=st.number_input("Enter High Blood Pressure Level",min_value=120,max_value=250)
low_bp=st.number_input("Enter Low Blood Pressure Level",min_value=20,max_value=80)
heart_rate=st.number_input("Enter Heart Rate",min_value=40,max_value=200)
diabetes=st.select_slider("Diabetes",options=['Yes','No'])
if diabetes=='Yes':
    diabetes=1
else:
    diabetes=0
smoking=st.select_slider("Smoking",options=['Yes','No'])
if smoking=='Yes':
    smoking=1
else:
    smoking=0
family_history=st.radio("Do you have family history of heart attack?",['Yes','No'])
if family_history=='Yes':
    family_history=1
else:
    family_history=0
previous_heart_problems=st.select_slider("Previous Heart Problems",options=['Yes','No'])
if previous_heart_problems=='Yes':
    previous_heart_problems=1
else:
    previous_heart_problems=0
stress_level=st.slider("Stress Level(1-10)",1,10)

encoded_model=ohe.transform([[gender]])[0].toarray()
#encoded_model_np=np.array([[encoded_model]])
#st.dataframe
#input_data=[age,cholestrol,high_bp,low_bp,heart_rate,diabetes,smoking,previous_heart_problems,stress_level]
#input_data.extend(encoded_model)
#input_data_np=np.array([[input_data]]).reshape(1,-1)
#st.write=(input_data_np.shape)
#st.write=(encoded_model.shape)
#st.dataframe(input_data_np)
input_data_np=np.array([[age,cholestrol,high_bp,low_bp,heart_rate,family_history,diabetes,smoking,previous_heart_problems,stress_level]])
another_in=np.concatenate([input_data_np,encoded_model],axis=1)

#print(another_in)

if st.button("Predict Heart Attack Risk"):
    #input_data=[[encoded_model,mileage,age]]
    prediction=reg.predict(another_in)
    if prediction==1:
        st.warning(f"{name} You are at a higher risk of heart attack. Please consult a doctor immediately!")
    else:
        st.warning(f"{name} You are at a lower risk of heart attack. Maintain a healthy lifestyle!")

