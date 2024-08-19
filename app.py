import streamlit as st
import numpy as np
import pandas as pd
import pickle


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from keras.src.saving.saving_api import load_model
from PIL import Image


loaded_model = load_model("neural_cnn.h5")


poly = PolynomialFeatures(degree = 4)

pickle_in_pol = open("poly_reg.pkl","rb")
poly_reg = pickle.load(pickle_in_pol)

pickle_in_lin = open("reg_all.pkl","rb")
reg_all = pickle.load(pickle_in_lin)

pickle_in = open("lin_reg.pkl","rb")
lin_reg = pickle.load(pickle_in)

def predict_1(ON_STREAM_HRS, AVG_DOWNHOLE_TEMPERATURE, AVG_ANNULUS_PRESS, AVG_CHOKE_SIZE_P, AVG_WHP_P, AVG_WHT_P, DP_CHOKE_SIZE, BORE_WAT_VOL):
    try:
        
        prediction_1 = reg_all.predict([[ON_STREAM_HRS, AVG_DOWNHOLE_TEMPERATURE, AVG_ANNULUS_PRESS, AVG_CHOKE_SIZE_P, AVG_WHP_P, AVG_WHT_P, DP_CHOKE_SIZE, BORE_WAT_VOL]])
        return float(prediction_1[0])
    except Exception as e:
        print(f"Error in predict_1: {e}")
        return None

def predict_2(ON_STREAM_HRS, AVG_DOWNHOLE_TEMPERATURE, AVG_ANNULUS_PRESS, AVG_CHOKE_SIZE_P, AVG_WHP_P, AVG_WHT_P, DP_CHOKE_SIZE, BORE_WAT_VOL):
    try:
        value = poly_reg.transform([[ON_STREAM_HRS, AVG_DOWNHOLE_TEMPERATURE, AVG_ANNULUS_PRESS, AVG_CHOKE_SIZE_P, AVG_WHP_P, AVG_WHT_P, DP_CHOKE_SIZE, BORE_WAT_VOL]])
        prediction_2 = lin_reg.predict(value)
        if prediction_2<0 :
            prediction_2 = ["Not defined"]
        return float(prediction_2[0])
    except Exception as e:
        print(f"Error in predict_2: {e}")
        return None

def neural_predict(a, b, c, d, e, f, g, h):
    try:
        input_data  =np.array([[a,b,c,d,e,f,g,h]])
        prediction_3 = loaded_model.predict(input_data)
        
        return float(prediction_3[0])
    except Exception as e:
        print(f"Error in neural_predict: {e}")
        return None

st.title("Get your prediction here")
st.sidebar.title("Input Production Parameters")



ON_STREAM_HRS = st.sidebar.number_input('ON_STREAM_HRS', min_value=0.0, max_value=30.0, value=0.960000, step=0.00001, format="%.5f")
AVG_DOWNHOLE_TEMPERATURE = st.sidebar.number_input('AVG_DOWNHOLE_TEMPERATURE', min_value=0.0, max_value=200.0, value= 0.963303, step=0.00001, format="%.5f")
AVG_ANNULUS_PRESS = st.sidebar.number_input('AVG_ANNULUS_PRESSURE', min_value=0.0, max_value=50.0, value=0.000000, step=0.00001, format="%.5f")
AVG_CHOKE_SIZE_P = st.sidebar.number_input('AVG_CHOKE_SIZE_PRESSURE', min_value=0.0, max_value=100.0, value=0.360000, step=0.00001, format="%.5f")
AVG_WHP_P = st.sidebar.number_input('AVG_WHP_PRESSURE', min_value=0.0, max_value=200.0, value=0.583942, step=0.00001, format="%.5f")
AVG_WHT_P = st.sidebar.number_input('AVG_WHT_PRESSURE', min_value=0.0, max_value=100.0, value=0.797872, step=0.00001, format="%.5f")
DP_CHOKE_SIZE = st.sidebar.number_input('DP_CHOKE_SIZE', min_value=0.0, max_value=200.0, value=47.000000, step=0.00001, format="%.5f")
BORE_WAT_VOL = st.sidebar.number_input('BORE_WAT_VOL', min_value=-500.0, max_value=3500.0, value=2.000000, step=0.00001, format="%.5f")



result_1 = ""
result_2 = ""
result_3 = ""

if st.button('Predict Oil Production'):
    result_1 = predict_1(ON_STREAM_HRS ,AVG_DOWNHOLE_TEMPERATURE,AVG_ANNULUS_PRESS,AVG_CHOKE_SIZE_P, AVG_WHP_P,AVG_WHT_P, DP_CHOKE_SIZE, BORE_WAT_VOL)
    result_2 = predict_2(ON_STREAM_HRS ,AVG_DOWNHOLE_TEMPERATURE,AVG_ANNULUS_PRESS,AVG_CHOKE_SIZE_P, AVG_WHP_P,AVG_WHT_P, DP_CHOKE_SIZE, BORE_WAT_VOL)
    result_3 = neural_predict(ON_STREAM_HRS ,AVG_DOWNHOLE_TEMPERATURE,AVG_ANNULUS_PRESS,AVG_CHOKE_SIZE_P, AVG_WHP_P,AVG_WHT_P, DP_CHOKE_SIZE, BORE_WAT_VOL)
    if result_2 == 0:
        result_2="undefined behaviour"
        
    st.write(
    pd.DataFrame(
        {
            "Algorithm use": ["Linear Regression", "Polynomial Regression","Neural Network(CNN)"],
            "Oil Production": [result_1,result_2,result_3]
            
        }
     )
   )
    st.markdown(
    """
    **Note:** Machine learning model used here is based on models
    trained on the Volve production dataset. To know more about the Volve field,
    go to [Volve Field Facts](https://www.norskpetroleum.no/en/facts/field/volve/).
    
    This is a sample of the training model on well 5351.
    """
)
    # Load the image file
    image = Image.open('my_plot.png')

# Display the image in the Streamlit app
    st.image(image, caption='My Matplotlib Plot', use_column_width=True)


