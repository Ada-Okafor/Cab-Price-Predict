import streamlit as st
import pandas as pd
import numpy as np
from pickle import dump
import pickle
from PIL import Image

st.markdown("<h1 style='text-align: center; color: black;'>Predict Your Cab Price</h1>", unsafe_allow_html=True)

photo = Image.open('cab-booking-app.jpg')
st.image(photo.resize([700,300]))

model = pickle.load(open('1_model.pkl', 'rb'))

Scaler = pickle.load(open('1_scaler.pkl', 'rb'))

encoder = pickle.load(open('1_enc.pkl', 'rb'))

st.divider()

st.markdown("<h2 style='text-align: center; color: black;'>Fill Relevant Details</h1>", unsafe_allow_html=True)




def data_input():
    c1, c2,c3 = st.columns(3)

    with c1:
        DIS = st.number_input('Distance from source to Destination')
        SM = st.number_input('Surge multiplier for trip', step =1)
        DES = st.selectbox('Select Destination', ['North Station', 'Northeastern University', 'West End',
       'Haymarket Square', 'Fenway', 'South Station', 'Theatre District',
       'Beacon Hill', 'North End', 'Boston University', 'Back Bay',
       'Financial District'])
        SOU = st.selectbox('Select Source Location', ['Haymarket Square', 'Back Bay', 'North End', 'North Station',
       'Boston University', 'Fenway', 'Theatre District', 'West End',
       'Beacon Hill', 'Financial District', 'South Station',
       'Northeastern University'])
        NAM = st.selectbox('Select Cab Category', ['Lux', 'Lux Black XL', 'WAV', 'Shared', 'Lyft', 'Lux Black',
       'UberXL', 'Black', 'Lyft XL', 'Black SUV', 'UberX', 'UberPool'])
        
    with c2:
        CT = st.selectbox('Select Cab type', ['Lyft', 'Uber'])
        TEMP = st.number_input('Enter temperature of source location', step =10.0 )
        CLD = st.number_input('Enter cloud reading of source location')
        PRE = st.number_input('Enter pressure reading of source location', step =100.0)
        RAIN = st.number_input('Enter rain reading of source location')


    with c3:
            HUM = st.number_input('Enter humidity reading of source location')
            WIN = st.number_input('Enter wind reading of source location')
            DAY = st.selectbox('Select Day of Week', range(0,7))
            HOUR = st.selectbox('Select Hour of the day', range(0,24))
            td= DIS*HOUR   
            TD = st.number_input('Time*Distance', td)
            

    feat = np.array([DIS,SM,DES,SOU,NAM,CT,TEMP,CLD,PRE,RAIN,HUM,WIN,DAY,HOUR,TD]).reshape(1,-1)

    cat = ['distance', 'surge_multiplier', 'destination', 'source', 'name',  'cab_type', 
           'temp', 'clouds', 'pressure', 'rain', 'humidity', 'wind', 'day', 'hour', 'time_distance']
    

    
    table = pd.DataFrame(feat,columns=cat)

    

   
    return table


#function to apply the scaler and encoders to the table

def process (df):       
    
    enc_data = pd.DataFrame(encoder.transform(np.array(df[['destination', 'source', 
                                                                        'name', 'cab_type']])).toarray(), 
                                                                        columns=encoder.get_feature_names_out(['destination', 'source', 'name', 'cab_type']))
   
       
    # Concatenate the encoded features
    df = pd.concat([df, enc_data], axis=1) 

    # Drop the original categorical columns
    df.drop(['destination', 'source', 'name', 'cab_type'], axis=1, inplace=True)



    col = df.columns
    df=Scaler.transform(df)
    df=pd.DataFrame(df,columns=col)
            

    df.drop('hour', axis=1, inplace=True)

    return df




#calling our functions
frame = data_input()

st.divider()


if st.button('Predict Price'):
    frame2 = process(frame)
    pred = model.predict(frame2)
    rd = round(float(pred[0]), 2)
    st.write ('Cost of ride :')
    st.write(f'${rd}')
    
    

st.divider()




       