import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

st.set_page_config(
    page_title ='Fifa 2022 - EDA',
    layout='wide',
    initial_sidebar_state='expanded'
)

def run():
    # Membuat Title
    st.title('Fifa 2022 Player Rating Prediction')

    # Membuat Sub Header
    st.subheader('EDA untuk Analis Dataset FIFA 2022')

    # Menambahkan gambar
    st.image('https://ichef.bbci.co.uk/news/1024/branded_pidgin/a617/live/4c8ad930-65d9-11ed-a6af-4f332dcec329.jpg',
             caption='FIFA 2022')

    # Menambahkan deskripsi
    st.write('Page ini dibuat oleh Vincent')
    st.write('# Halo')
    st.write('## Halo')
    st.write('### Halo')

    # Membuat garis lurus
    st.markdown('---')

    # Magic syntax
    '''
    Pada page kali ini, penulis akan melakukan eksplorasi sederhana,
    Dataset yang digunakan adalah dataset FIFA 2022.
    Dataset ini berasal dari web sofifa.com
    '''

    # Show DataFrame
    df = pd.read_csv('https://raw.githubusercontent.com/FTDS-learning-materials/phase-1/master/w1/P1W1D1PM%20-%20Machine%20Learning%20Problem%20Framing.csv')
    st.dataframe(df)

    # Membuat Barplot
    st.write('#### Plot AttackingWorkRate')
    fig = plt.figure(figsize=(15,5))
    sns.countplot(x='AttackingWorkRate',data=df)
    st.pyplot(fig)

    # Membuat Histogram berdasarkan input user
    st.write('#### Histogram berdasarkan Input User')
    pilihan = st.selectbox('Pilih kolom:',('Age','Weight','Height','ShootingTotal'))
    fig = plt.figure(figsize=(15,5))
    sns.histplot(df[pilihan], bins=30, kde=True)
    st.pyplot(fig)

    # Membuat Plotly Plot
    st.write('#### Plotly Plot - ValueEur dengan Overall')
    fig = px.scatter(df,x='ValueEUR',y='Overall', hover_data=['Name','Age'])
    st.plotly_chart(fig)

if __name__== '__main__':
    run()
