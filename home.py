import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def chart(String):
    chart = st.selectbox('Silahkan pilih :' , ['line chart' , 'area chart', 'bar chart'])
    if chart=='line chart':
        st.write('Grafik line ')
        st.line_chart(String)
    elif chart=='area chart':
        st.write('Grafik area ')
        st.area_chart(String)
    elif chart=='bar chart':
        st.write('Grafik bar ')
        st.bar_chart(String)


page = st.sidebar.selectbox('Select Page',['Home','Exploratory data','Visualisasi data utama','Model dan prediksi'])

model = pickle.load(open('model_prediksi_harga_rumah.sav', 'rb'))
df_house = pd.read_csv('housing_price_dataset.csv')
if page=='Home':
    st.title('Prediksi Harga Rumah')
    st.header('Judul')
    st.write("House Prediction")
    st.header("Sumber dan Alasan")
    st.write("sumber berasal dari kaggle.com dan alasannya karena pertema ditemukan di kaggle.com")

elif page=='Exploratory data':
    st.title("Exploratory house data")
    st.header("House dataframe")
    st.dataframe(df_house)
    st.header("Check null value")
    st.dataframe(df_house.isnull().sum())
    st.header("Describe dataframe")
    st.dataframe(df_house.describe())

    

elif page=='Visualisasi data utama':
    st.title("visualisasi data")
    grafik = st.selectbox('Pilih Grafik',df_house.columns)
    if grafik==df_house.columns[0]:
        chart(df_house[df_house.columns[0]])
    elif grafik==df_house.columns[1]:
        chart(df_house[df_house.columns[1]])
    elif grafik==df_house.columns[2]:
        chart(df_house[df_house.columns[2]])
    elif grafik==df_house.columns[3]:
        chart(df_house[df_house.columns[3]])
    elif grafik==df_house.columns[4]:
        chart(df_house[df_house.columns[4]])
    elif grafik==df_house.columns[5]:
        chart(df_house[df_house.columns[5]])


elif page=='Model dan prediksi':
    st.title('Model and Prediction')
        
    x = df_house[['Bedrooms','Bathrooms','SquareFeet']]
    y = df_house['Price']

    #13
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

    model_regresi = LinearRegression()
    model_regresi.fit(x_train,y_train)

    filename = 'model_prediksi_harga_rumah.sav'
    pickle.dump(model_regresi,open(filename,'wb'))

    model = pickle.load(open('model_prediksi_harga_rumah.sav','rb'))

    st.title('house prediction')

    Bedrooms = st.number_input("Bedrooms",2,5)
    Bathrooms = st.number_input("Bathrooms",1,3)
    SquareFeet = st.number_input("SquareFeet",1000,3000)

    if st.button('prediksi'):
        car_prediction = model.predict([[Bedrooms,Bathrooms,SquareFeet]])

        harga_mobil_str = np.array(car_prediction)
        harga_mobil_float = float(harga_mobil_str[0])

        harga_mobil_formatted = st.write('jadi harga house adalah = $ ',"{:,.3f}".format(harga_mobil_float))


