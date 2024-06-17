#web app
import streamlit as st
st.title("Credit Card Fraud Detection Model")
input_df = st.text_input("Enter All Required Features Values")
input_df_splited = input_df.split(',')

Submit = st.button("Submit")

if Submit:
    features = np.asanyarray(input_df_splited,dtype=np.float64)
    prediction = model.predict(features.reshape(1,-1))

    if prediction[0]==0:
        st.write('Legitimate Transaction')
    else:
        st.write('Fradulant Transaction') 
           
