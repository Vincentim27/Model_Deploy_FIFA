import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# Load all files
with open('model_lin_reg.pkl', 'rb') as file_1:
    model_lin_reg = pickle.load(file_1)
with open('model_scaler.pkl', 'rb') as file_2:
    model_scaler = pickle.load(file_2)
with open('model_encoder.pkl', 'rb') as file_3:
    model_encoder = pickle.load(file_3)
with open('list_num_cols.txt', 'r') as file_4:
    list_num_cols = json.load(file_4)
with open('list_cat_cols.txt', 'r') as file_5:
    list_cat_cols = json.load(file_5)

def run():
    # Membuat form
    with st.form(key='form parameters'):
        name = st.text_input('Name',value='Dwi')
        age = st.number_input('Age', min_value=0, max_value=60,value=0,step=1,help='Usia Pemain')
        weight = st.number_input('Weight', min_value=50, max_value=150,value=70)
        height = st.slider('Height', 50, 250, 170)
        price = st.number_input('Price', min_value=0,max_value=1000000000,value=0)
        st.markdown('---')

        attackingworkrate = st.selectbox('Attacking Work Rate',('Low','Medium','High'),index=1)
        defensiveworkrate = st.selectbox('Defensive Work Rate',('Low','Medium','High'), index=1)
        st.markdown('---')

        pace = st.number_input('Pace', min_value=0, max_value=100, value=50)
        shooting = st.number_input('Shooting', min_value=0, max_value=100, value=50)
        passing = st.number_input('Passing', min_value=0, max_value=100, value=50)
        dribbling = st.number_input('Dribbling', min_value=0, max_value=100, value=50)
        defending = st.number_input('Defending', min_value=0, max_value=100, value=50)
        psysicality = st.number_input('Pysicality', min_value=0, max_value=100, value=50)

        submitted = st.form_submit_button('Predict')

    data_inf = {
        'Name': name, 
        'Age': age, 
        'Height': height, 
        'Weight': weight, 
        'Price': price, 
        'AttackingWorkRate': attackingworkrate,
        'DefensiveWorkRate': defensiveworkrate, 
        'PaceTotal': pace, 
        'ShootingTotal': shooting, 
        'PassingTotal': passing,
        'DribblingTotal': dribbling, 
        'DefendingTotal': defending, 
        'PhysicalityTotal': psysicality
    }
    data_inf = pd.DataFrame([data_inf])
    st.dataframe(data_inf)

    if submitted:
        # Split between num col and cat col
        data_inf_num = data_inf[list_num_cols]
        data_inf_cat = data_inf[list_cat_cols]

        # Feature scaling and Feature encoding
        data_inf_num_scaled = model_scaler.transform(data_inf_num)
        data_inf_cat_encoded = model_encoder.transform(data_inf_cat)

        # Concat
        data_inf_final = np.concatenate([data_inf_num_scaled, data_inf_cat_encoded],axis=1)

        # Predict using Linear Regression
        y_pred_inf = model_lin_reg.predict(data_inf_final)

        st.write('# Rating :', str(int(y_pred_inf)))

if __name__ == '__main__':
    run()