from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.response import Response
from rest_framework.decorators import api_view, renderer_classes
from rest_framework.renderers import JSONRenderer, TemplateHTMLRenderer
import requests
import os
from datetime import datetime,timedelta
import math
import os
import smtplib
import numpy as np
from email.mime.text import MIMEText
import pyowm
import pandas
import datetime
import os 
import pytz

os.environ['TF_CPP_MIN_LOG_LEVEL']= '0'
os.environ['TF_ENABLE_ONEDNN_OPTS']= '0'
from tensorflow.keras.models import load_model
import logging
logging.disable(logging.WARNING)
#import pandas as pd
import joblib
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
#from xgboost import XGBRegressor
import json 

from pandasql import sqldf

cities = ['mumbai','delhi','hyderabad','pune','jaipur','bengaluru','nagpur','kanpur']

mlp_models = {}
for i in cities:
    mlp_models[i]=load_model(f'static/mlp_models/{i}_mlp_model.h5')

wind_model_l = load_model(f'static/mlp_models/100_2PLSTM.h5')
wind_model_s = load_model(f'static/mlp_models/100_011PLSTM.h5')

def get_wind_prediction_l(X,Tk,RH,P):

    # Assuming standardized values for Cp, R, and other parameters
    Cp = 0.45  # average efficiency of a wind turbine lies between 0.35 to 0.45
    R = 50  # average blade length of small scale residential turbines in meters

    #calculating air density of given region

    Tc = Tk - 273.15  #Temperature in Celcius
    p1 = 6.1078 * math.pow(10,((7.5*Tc)/(Tc+237.3)))
    pv = p1*RH*0.01  #vapor pressure
    pd = P-pv   #dry air pressure
    Rd = 287.058 #J/kg.K #specific gas constant for dry air
    Rv = 461.495 #J/kg.K #specific gas constant for vapour
    p = (pd/(Rd*Tk)) + (pv/(Rv*Tk))
    p= p*100
    rho = p #1.225  # air density in kg/m^3 (standard value)
    pi = np.pi
    y_theoretical = 0.001 * 0.5 * Cp * rho * pi * R**2 * X**3  # Calculate theoretical power output

    theoretical_power = y_theoretical

    # Create a numpy array with the user input
    user_input = np.array([[X, theoretical_power]])

    user_input_normalized = user_input#scaler.transform(user_input)

    # Reshape the input for LSTM
    user_input_reshaped = user_input_normalized.reshape((user_input_normalized.shape[0], 1, user_input_normalized.shape[1]))

    # Make predictions
    predicted_active_power = wind_model_l.predict(user_input_reshaped)
    
    pap = predicted_active_power[0][0]

    pap=abs(pap)*0.01 if predicted_active_power[0][0]< 0 else pap
    fpap = (0.75*y_theoretical)+(0.25*pap)
    
    fpap=round(fpap,2)
    return fpap,y_theoretical

def get_wind_prediction_s(X,Tk,RH,P):

    # Assuming standardized values for Cp, R, and other parameters
    Cp = 0.45  # average efficiency of a wind turbine lies between 0.35 to 0.45
    R = 5  # average blade length of small scale residential turbines in meters

    #calculating air density of given region

    Tc = Tk - 273.15  #Temperature in Celcius
    p1 = 6.1078 * math.pow(10,((7.5*Tc)/(Tc+237.3)))
    pv = p1*RH*0.01  #vapor pressure
    pd = P-pv   #dry air pressure
    Rd = 287.058 #J/kg.K #specific gas constant for dry air
    Rv = 461.495 #J/kg.K #specific gas constant for vapour
    p = (pd/(Rd*Tk)) + (pv/(Rv*Tk))
    p= p*100
    rho = p #1.225  # air density in kg/m^3 (standard value)
    pi = np.pi
    y_theoretical = 0.001 * 0.5 * Cp * rho * pi * R**2 * X**3  # Calculate theoretical power output

    theoretical_power = y_theoretical

    # Create a numpy array with the user input
    user_input = np.array([[X, theoretical_power]])

    user_input_normalized = user_input#scaler.transform(user_input)

    # Reshape the input for LSTM
    user_input_reshaped = user_input_normalized.reshape((user_input_normalized.shape[0], 1, user_input_normalized.shape[1]))

    # Make predictions
    predicted_active_power = wind_model_s.predict(user_input_reshaped)

    pap = predicted_active_power[0][0]

    pap=abs(pap)*0.01 if predicted_active_power[0][0]< 0 else pap
    fpap = (0.75*y_theoretical)+(0.25*pap)
    
    fpap=round(fpap,2)
    return fpap,y_theoretical

def get_city_output(city):
    ct = city
    #setmodel(f'static/mlp_models/{city}_mlp_model.h5')
    setmodel(city)
    df = pandas.read_csv(f'static/data/{city}_mod.csv', header=0)
    now = datetime.datetime.now()
    mon = now.strftime("%m")
    tim = now.strftime("%X")
    day = now.strftime("%d")
    query = f"SELECT avg(mintempC), avg(maxtempC),avg(humidity),avg(cloudcover),avg(windspeedKmph),avg(winddirDegree),avg(pressure),avg(FeelslikeC) FROM df where month = '{mon}' and date like '{day}%' and time like '{tim[:2]}%'"

    stats = sqldf(query)

    daily_query = f"SELECT avg(mintempC), avg(maxtempC),avg(humidity),avg(cloudcover),avg(windspeedKmph),avg(winddirDegree),avg(pressure),avg(FeelslikeC) FROM df where month = '{mon}' and date like '{day}%'"

    daily_stats = sqldf(query)

    api_data = weatherInfo(ct)
    loc_data = locationInfo(ct)
    year =  float(now.strftime("%Y"))
    month = float(now.strftime("%m"))
    day = float(now.strftime("%d"))
    hour = float(now.strftime("%H"))
    temp_c = round(((api_data['main']['feels_like']) - 273.15),2)
    temp_e = round(stats.iat[0, 7],2)
    wse = round(stats.iat[0, 4]*0.277778,2)
   # hour = 13.00
    current = np.array([year, month , day ,hour, 30, temp_c , round(api_data['main']['humidity'],2), round(api_data['main']['pressure'],2), round(api_data['wind']['deg'],2), round(api_data['wind']['speed'],2)])
    expected = np.array([year, month , day ,hour , 30, temp_e , round(stats.iat[0, 2],2), round(stats.iat[0, 6],2), round(stats.iat[0, 5],2), round(stats.iat[0, 4],2)])
    fullday_expected = np.array([year, month , day ,hour , 30,round(daily_stats.iat[0, 7],2) , round(daily_stats.iat[0, 2],2), round(daily_stats.iat[0, 6],2), round(daily_stats.iat[0, 5],2), round(daily_stats.iat[0, 4],2)])
    Tk = (api_data['main']['temp_min']+api_data['main']['temp_max'])/2.0
    Tke = (round(stats.iat[0, 1],2)+round(stats.iat[0, 0],2))/2.0
    FTke = (round(daily_stats.iat[0, 1],2)+round(daily_stats.iat[0, 0],2))/2.0
    #fpap_l = get_wind_prediction_l(round(api_data['wind']['speed'],2),Tk,round(api_data['main']['humidity'],2),round(api_data['main']['pressure'],2))
    fpap_s,s_theo = get_wind_prediction_s(round(api_data['wind']['speed'],2),Tk,round(api_data['main']['humidity'],2),round(api_data['main']['pressure'],2))
    #fpap_le = get_wind_prediction_l(wse,Tke+273.15,round(stats.iat[0, 2],2),round(stats.iat[0, 6],2))
    #fpap_se = get_wind_prediction_s(wse,Tke+273.15,round(stats.iat[0, 2],2),round(stats.iat[0, 6],2))

    
    fullday_fpap_s,fs_theo = get_wind_prediction_s(round(daily_stats.iat[0, 4]*0.277778,2),FTke+273.15,round(daily_stats.iat[0, 2],2),round(daily_stats.iat[0, 6],2))
    
    test = np.array([2024, 2, 21, 12, 30, 37, 82, 1020, 60, 3.8])
    #year =  2024
    #month = 5
    #day = 4
    
    #temp_c = 34.54
    #temp_e = 54.23
   
   
    current_irr = round(predict_with_mlp(current)[0][0],2)
    current_irr = 0 if current_irr==-0.0 else current_irr
    #expected_irr = 0 if current_irr==0 else round(predict_with_mlp(expected)[0][0],2)
    #expected_irr = 0 if expected_irr== -0.0 else expected_irr
    fullday_irr = round(predict_with_mlp(fullday_expected)[0][0],2)
    fullday_irr = 0 if fullday_irr==-0.0 else fullday_irr
    curr_input_power = pandas.DataFrame({
        'Year': [year],
        'Month': [month],
        'Day': [day],
        'Hour': [hour],
        'Minute': [30],
        'GHI': [current_irr],
        'Temperature': [temp_c]
    })
    """exp_input_power = pandas.DataFrame({
        'Year': [year],
        'Month': [month],
        'Day': [day],
        'Hour': [hour],
        'Minute': [30],
        'GHI': [expected_irr],
        'Temperature': [temp_e]
    })"""

    fullday_input_power = pandas.DataFrame({
        'Year': [year],
        'Month': [month],
        'Day': [day],
        'Hour': [hour],
        'Minute': [30],
        'GHI': [fullday_irr],
        'Temperature': [round(daily_stats.iat[0, 7],2)]
    })

    features_df = create_new_features(curr_input_power)
    
    current_pac = predict_pac(features_df)
    current_pac = round(current_pac.iat[0, 12],2)


    #features_df = create_new_features(exp_input_power)

    #expected_pac = predict_pac(features_df)
    #expected_pac = round(expected_pac.iat[0, 12],2)

    features_df = create_new_features(fullday_input_power)

    fullday_pac = predict_pac(features_df)
    fullday_pac = round(fullday_pac.iat[0, 12],2)

    
    IST_time = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))
    GMT_time = datetime.datetime.now(pytz.timezone('GMT'))
   
    


    output = [{

            
            "Solar Power":[

                {
                    "Name":"Irradiation",
                    "Icon":"Irradiation",
                    "current": round(current_irr),
                    
                },

                {
                    "Name":"Solar Power",
                    "Icon":"Power",
                    "current": round(current_pac*0.001,2),
                    
                    "fullday": round(fullday_pac*0.001,2)
                },
                
            ],
            "Wind Power":[


                {
                    "Name":"Wind Power(Residential)",
                    "Icon":"Power",
                    "current":fpap_s,
                
                    "fullday":fullday_fpap_s
                    
                }
            



    ],
    "Time":[
                {
                    "Name":"GMT(+0:00)",
                    "Date":GMT_time.strftime("%d %b %Y %a"),
                    "Time": str(GMT_time.time())[0:8]
                },
                {
                    "Name": "IST(+5:30)",
                    "Date":IST_time.strftime("%d %b %Y %a"),
                    "Time": str(IST_time.time())[0:8]
                }
                
    ]

          
   
            
      },
      
      ]
  
    
    return output

def get_city(city):
    ct = city
    #setmodel(f'static/mlp_models/{city}_mlp_model.h5')
    setmodel(city)
    df = pandas.read_csv(f'static/data/{city}_mod.csv', header=0)
    now = datetime.datetime.now()
    mon = now.strftime("%m")
    tim = now.strftime("%X")
    day = now.strftime("%d")
    query = f"SELECT avg(mintempC), avg(maxtempC),avg(humidity),avg(cloudcover),avg(windspeedKmph),avg(winddirDegree),avg(pressure),avg(FeelslikeC) FROM df where month = '{mon}' and time like '{tim[:2]}%'"

    stats = sqldf(query)

    daily_query = f"SELECT avg(mintempC), avg(maxtempC),avg(humidity),avg(cloudcover),avg(windspeedKmph),avg(winddirDegree),avg(pressure),avg(FeelslikeC) FROM df where month = '{mon}' and date like '{day}%'"

    daily_stats = sqldf(query)

    api_data = weatherInfo(ct)
    loc_data = locationInfo(ct)
    year =  float(now.strftime("%Y"))
    month = float(now.strftime("%m"))
    day = float(now.strftime("%d"))
    hour = float(now.strftime("%H"))
    temp_c = round(((api_data['main']['feels_like']) - 273.15),2)
    temp_e = round(stats.iat[0, 7],2)
    wse = round(stats.iat[0, 4]*0.277778,2)
   # hour = 13.00
    current = np.array([year, month , day ,hour, 30, temp_c , round(api_data['main']['humidity'],2), round(api_data['main']['pressure'],2), round(api_data['wind']['deg'],2), round(api_data['wind']['speed'],2)])
    expected = np.array([year, month , day ,hour , 30, temp_e , round(stats.iat[0, 2],2), round(stats.iat[0, 6],2), round(stats.iat[0, 5],2), round(stats.iat[0, 4],2)])
    fullday_expected = np.array([year, month , day ,hour , 30,round(daily_stats.iat[0, 7],2) , round(daily_stats.iat[0, 2],2), round(daily_stats.iat[0, 6],2), round(daily_stats.iat[0, 5],2), round(daily_stats.iat[0, 4],2)])
    Tk = (api_data['main']['temp_min']+api_data['main']['temp_max'])/2.0
    Tke = (round(stats.iat[0, 1],2)+round(stats.iat[0, 0],2))/2.0
    FTke = (round(daily_stats.iat[0, 1],2)+round(daily_stats.iat[0, 0],2))/2.0
    fpap_l,l_theo = get_wind_prediction_l(round(api_data['wind']['speed'],2),Tk,round(api_data['main']['humidity'],2),round(api_data['main']['pressure'],2))
    fpap_s,s_theo = get_wind_prediction_s(round(api_data['wind']['speed'],2),Tk,round(api_data['main']['humidity'],2),round(api_data['main']['pressure'],2))
    fpap_le,le_theo = get_wind_prediction_l(wse,Tke+273.15,round(stats.iat[0, 2],2),round(stats.iat[0, 6],2))
    fpap_se,se_theo = get_wind_prediction_s(wse,Tke+273.15,round(stats.iat[0, 2],2),round(stats.iat[0, 6],2))

    
    fullday_fpap_s,fs_theo = get_wind_prediction_s(round(daily_stats.iat[0, 4]*0.277778,2),FTke+273.15,round(daily_stats.iat[0, 2],2),round(daily_stats.iat[0, 6],2))
    
    test = np.array([2024, 2, 21, 12, 30, 37, 82, 1020, 60, 3.8])
    #year =  2024
    #month = 5
    #day = 4
    
    #temp_c = 34.54
    #temp_e = 54.23
   
   
    current_irr = round(predict_with_mlp(current)[0][0],2)
    current_irr = 0 if current_irr==-0.0 else current_irr
    expected_irr = 0 if current_irr==0 else round(predict_with_mlp(expected)[0][0],2)
    expected_irr = 0 if expected_irr== -0.0 else expected_irr
    fullday_irr = round(predict_with_mlp(fullday_expected)[0][0],2)
    fullday_irr = 0 if fullday_irr==-0.0 else fullday_irr
    curr_input_power = pandas.DataFrame({
        'Year': [year],
        'Month': [month],
        'Day': [day],
        'Hour': [hour],
        'Minute': [30],
        'GHI': [current_irr],
        'Temperature': [temp_c]
    })
    exp_input_power = pandas.DataFrame({
        'Year': [year],
        'Month': [month],
        'Day': [day],
        'Hour': [hour],
        'Minute': [30],
        'GHI': [expected_irr],
        'Temperature': [temp_e]
    })

    fullday_input_power = pandas.DataFrame({
        'Year': [year],
        'Month': [month],
        'Day': [day],
        'Hour': [hour],
        'Minute': [30],
        'GHI': [fullday_irr],
        'Temperature': [round(daily_stats.iat[0, 7],2)]
    })

    features_df = create_new_features(curr_input_power)
    
    current_pac = predict_pac(features_df)
    current_pac = round(current_pac.iat[0, 12],2)


    features_df = create_new_features(exp_input_power)

    expected_pac = predict_pac(features_df)
    expected_pac = round(expected_pac.iat[0, 12],2)

    features_df = create_new_features(fullday_input_power)

    fullday_pac = predict_pac(features_df)
    fullday_pac = round(fullday_pac.iat[0, 12],2)

    
    IST_time = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))
    GMT_time = datetime.datetime.now(pytz.timezone('GMT'))
    N = dayOfYear(str(IST_time.date()))
    LST = 4*(float(loc_data[0]["lon"])-82.50)+EOT(N)
    offset = 330+int(LST)
    LST = IST_time + timedelta(minutes=LST)
    


    output = [{
            
            "Maximum Temperature":{
                "current": round(((api_data['main']['temp_max']) - 273.15),2),
                "expected":round(stats.iat[0, 1],2)
            }, 
            "Minimum Temperature": {
                 "current": round(((api_data['main']['temp_min']) - 273.15),2),
                "expected": round(stats.iat[0, 0],2)
            },
            "Feels Like": {
                 "current": round(((api_data['main']['feels_like']) - 273.15),2),
                "expected": round(stats.iat[0, 7],2)
            },
            "Humidity": {
                "current": round(api_data['main']['humidity'],2),
                "expected":round(stats.iat[0, 2],2)
            },
            "Cloud Cover":{
                 "current": round(api_data['clouds']['all'],2),
                "expected":round(stats.iat[0, 3],2)
            },
            "Pressure":{
                "current": round(api_data['main']['pressure'],2),
                "expected":round(stats.iat[0, 6],2)
            },
            "Wind Speed":{
                "current": round(api_data['wind']['speed'],2),
                "expected":wse
            },
            "Wind Direction":{
                "current": round(api_data['wind']['deg'],2),
                "expected": round(stats.iat[0, 5],2)
            },
            
            "Solar Power":[

                {
                    "Name":"Irradiation",
                    "Icon":"Irradiation",
                    "current": round(current_irr),
                    "expected":  round(expected_irr)
                },

                {
                    "Name":"Solar Power",
                    "Icon":"Power",
                    "current": round(current_pac*0.001,2),
                    "expected": round(expected_pac*0.001,2),
                    "fullday": round(fullday_pac*0.001,2)
                },
                
            ],
            "Wind Power":[

                {
                    "Name":"Wind Power(Large Scale)",
                    "Icon":"Power",
                    "current":fpap_l,
                    "expected":fpap_le
                        
                    
                },

                {
                    "Name":"Wind Power(Residential)",
                    "Icon":"Power",
                    "current":fpap_s,
                    "expected":fpap_se,
                    "fullday":fullday_fpap_s
                    
                }



    ],

            "Location Info": loc_data[0],
            "Time":[
                {
                    "Name":"GMT(+0:00)",
                    "Date":GMT_time.strftime("%d %b %Y %a"),
                    "Time": str(GMT_time.time())[0:8]
                },
                {
                    "Name": "IST(+5:30)",
                    "Date":IST_time.strftime("%d %b %Y %a"),
                    "Time": str(IST_time.time())[0:8]
                },
                {
                    "Name": f"LST(+{offset//60}:{offset%60:02d})",
                    "Date":LST.strftime("%d %b %Y %a"),
                    "Time": str(LST.time())[0:8]
                }
                
    ]
            
      }]
  
    
    return output
def create_new_features(sensor_df):
    df = sensor_df.copy()
    # df = df[df['GHI'] != 0]
    df['datetime'] = pandas.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    hours = df['datetime'].dt.hour + df['datetime'].dt.minute / 60
    # df = df[(hours > 6) & (hours <= 17)]
    df.rename(columns={'GHI': 'Gir', 'Temperature': 'Ta'}, inplace=True)
    df['hours'] = hours
    df['Gir^3'] = np.power(df['Gir'], 3)
    df['Gir^2'] = np.power(df['Gir'], 2)
    df['Ta^2'] = np.power(df['Ta'], 2)
    df['Gir^2.Ta'] = df['Gir^2'] * df['Ta']
    df['Gir.Ta^2'] = df['Gir'] * df['Ta^2']
    df['Gir.Ta'] = df['Gir'] * df['Ta']
    df = df[['Year', 'Month', 'Day', 'Hour', 'Minute', 'hours', 'Gir^3', 'Gir^2', 'Gir^2.Ta', 'Gir.Ta^2', 'Gir.Ta', 'Gir']]
    #print(df)
    return df

def predict_pac(df, model_path=f'static/mlp_models/linear_regression_model_new.pkl'):
    loaded_model = joblib.load(model_path)
    X = df.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute', 'hours']).values
    #print(X)
    # Predict pac using the loaded linear regression model
    pac_predicted = loaded_model.predict(X)

    # Add the predicted pac values to the DataFrame
    df['pac_predicted'] = pac_predicted

    return df


model = mlp_models.get("mumbai")

def setmodel(modelpath):
    #model = load_model(modelpath)
    model = mlp_models.get(modelpath)


def predict_with_mlp( custom_input):
    
    prediction = model.predict(custom_input.reshape(1, -1))
  
    return prediction

def weatherInfo(location):
    user_api = "bb50e1ce47cf6b9e361bd82996b9c93c" 
    complete_api_link = "https://api.openweathermap.org/data/2.5/weather?q=" + location + "&appid=" + user_api
    api_link = requests.get(complete_api_link)
    api_data = api_link.json()
    return api_data

def dayOfYear(date):
    days = [0,31,28,31,30,31,30,31,31,30,31,30,31]
    d = list(map(int,date.split("-")))
    if d[0] % 400 == 0:
        days[2]+=1
    elif d[0]%4 == 0 and d[0]%100!=0:
        days[2]+=1
    for i in range(1,len(days)):
        days[i]+=days[i-1]
    return days[d[1]-1]+d[2]

def EOT(N):
  B= (360*(N-81))/365
  D = 6.24004077 + 0.01720197*(365.25*(2024-2000)+N)
  t = -7.659*math.sin(D)+9.863*math.sin(2*D+3.5932)
  E = (9.87*math.sin(2*B))-(7.53*math.cos(B))-(1.5*math.sin(B))
  return t

def locationInfo(location):
    complete_api_link = "https://nominatim.openstreetmap.org/search?format=json&limit=1&q=" + location 
    api_link = requests.get(complete_api_link)
    api_data = api_link.json()
    return api_data

# Create your views here.
@api_view(['GET'])
def getRoutes(request):
   send_url = "http://api.ipstack.com/check?access_key=83e8f3bb7c31bc1c1097d0c87b7ea803"
   geo_req = requests.get(send_url)
   geo_json = json.loads(geo_req.text)
   latitude = geo_json['latitude']
   longitude = geo_json['longitude']
   city = geo_json['city']
   

   api_data = weatherInfo(city)
   return Response(api_data)



output = []

@api_view(['GET'])
def citywise_poweroutputs(request):
    output = []
    IST_time = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))
    GMT_time = datetime.datetime.now(pytz.timezone('GMT'))
    N = dayOfYear(str(IST_time.date()))
    
    for i in range(len(cities)):
        po = get_city_output(cities[i])
        
        so = po[0]["Solar Power"][1]["current"]
        wo = po[0]["Wind Power"][0]["current"]
        fso = po[0]["Solar Power"][1]["fullday"]
        fwo = po[0]["Wind Power"][0]["fullday"]

        d = {
            "city":cities[i],
            "Solar Power":round(so*30*12,2),
            "Solar Power Fullday":round(fso*30*12,2),
            "Wind Power Fullday":round(fwo*24,2),
            "Wind Power":round(wo*24,2)
        }
        print(f"{cities[i]} done")
        output.append(d)
    res = {
        "Main":output,
        
        "Time":[
            {
                "Name":"GMT(+0:00)",
                "Date":GMT_time.strftime("%d %b %Y %a"),
                "Time": str(GMT_time.time())[0:8]
            },
            {
                "Name": "IST(+5:30)",
                "Date":IST_time.strftime("%d %b %Y %a"),
                "Time": str(IST_time.time())[0:8]
            }
                
    ]
    }
    return Response(res)

@api_view(['GET'])
def citywise_bar(request,city):
    #output=[]
    loc_data = locationInfo(city)
    po = get_city(city)
    IST_time = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))
    GMT_time = datetime.datetime.now(pytz.timezone('GMT'))
    N = dayOfYear(str(IST_time.date()))
    LST = 4*(float(loc_data[0]["lon"])-82.50)+EOT(N)
    offset = 330+int(LST)
    LST = IST_time + timedelta(minutes=LST)
    
    so = po[0]["Solar Power"][1]["current"]
    fso = po[0]["Solar Power"][1]["fullday"]
    wo = po[0]["Wind Power"][1]["current"]
    fwo = po[0]["Wind Power"][1]["fullday"]
    output = {"Main":[

                {
            "Source":"Avg Daily Consumption(30 Flats)",
            "Solar":20*30,
            "Wind": 0
                },

                {
            "Source":"Daily Energy Production (at current rate)",
            "Solar":round(so*12*30,2),
            "Wind":round(wo*24,2),
                },

                
                {
            "Source":"Daily Energy Production (at avg rate)",
            "Solar":round(fso*12*30,2),
            "Wind":round(fwo*24,2),
                },

                
            ],
            "Location Info": loc_data[0],
            "Time":[
                {
                    "Name":"GMT(+0:00)",
                    "Date":GMT_time.strftime("%d %b %Y %a"),
                    "Time": str(GMT_time.time())[0:8]
                },
                {
                    "Name": "IST(+5:30)",
                    "Date":IST_time.strftime("%d %b %Y %a"),
                    "Time": str(IST_time.time())[0:8]
                },
                {
                    "Name": f"LST(+{offset//60}:{offset%60:02d})",
                    "Date":LST.strftime("%d %b %Y %a"),
                    "Time": str(LST.time())[0:8]
                }
                
    ]
            }
    

    return Response(output)



@api_view(['GET'])
def city(request, city):
    ct = city
    #setmodel(f'static/mlp_models/{city}_mlp_model.h5')
    setmodel(city)
    df = pandas.read_csv(f'static/data/{city}_mod.csv', header=0)
    now = datetime.datetime.now()
    nb = 21
    dya = 18
    mon = now.strftime("%m")
    tim = now.strftime("%X")
    day = now.strftime("%d")
    query = f"SELECT avg(mintempC), avg(maxtempC),avg(humidity),avg(cloudcover),avg(windspeedKmph),avg(winddirDegree),avg(pressure),avg(FeelslikeC) FROM df where month_year like '{mon}%' and date like '{day}%' and time like '{tim[:2]}%'"
   
    IST_time = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))
    GMT_time = datetime.datetime.now(pytz.timezone('GMT'))
    #ist = pytz.timezone('Asia/Kolkata')
    #GMT_time = datetime.datetime(2024, 3, dya, nb, 30, 0, tzinfo=pytz.utc)
    #IST_time = GMT_time.astimezone(ist)

    stats = sqldf(query)
    api_data = weatherInfo(ct)
    loc_data = locationInfo(ct)
    year =  float(now.strftime("%Y"))
    month = float(now.strftime("%m"))
    day = float(now.strftime("%d"))
    hour = float(now.strftime("%H"))
    temp_c = round(((api_data['main']['feels_like']) - 273.15),2)
    temp_e = round(stats.iat[0, 7],2)
    wse = round(stats.iat[0, 4]*0.277778,2)
   # hour = 13.00
    current = np.array([year, month , day ,hour, 30, temp_c , round(api_data['main']['humidity'],2), round(api_data['main']['pressure'],2), round(api_data['wind']['deg'],2), round(api_data['wind']['speed'],2)])
    expected = np.array([year, month , day ,hour , 30, temp_e , round(stats.iat[0, 2],2), round(stats.iat[0, 6],2), round(stats.iat[0, 5],2), round(stats.iat[0, 4],2)])
    Tk = (api_data['main']['temp_min']+api_data['main']['temp_max'])/2.0
    Tke = (round(stats.iat[0, 1],2)+round(stats.iat[0, 0],2))/2.0
    fpap_l,l_theo = get_wind_prediction_l(round(api_data['wind']['speed'],2),Tk,round(api_data['main']['humidity'],2),round(api_data['main']['pressure'],2))
    fpap_s,s_theo = get_wind_prediction_s(round(api_data['wind']['speed'],2),Tk,round(api_data['main']['humidity'],2),round(api_data['main']['pressure'],2))
    fpap_le,le_theo = get_wind_prediction_l(wse,Tke+273.15,round(stats.iat[0, 2],2),round(stats.iat[0, 6],2))
    fpap_se,se_theo = get_wind_prediction_s(wse,Tke+273.15,round(stats.iat[0, 2],2),round(stats.iat[0, 6],2))
    
    test = np.array([2024, 2, 21, 12, 30, 37, 82, 1020, 60, 3.8])
    #year =  2024
    #month = 5
    #day = 4
    
    #temp_c = 34.54
    #temp_e = 54.23
   
   
    current_irr = round(predict_with_mlp(current)[0][0],2)
    current_irr = 0 if current_irr==-0.0 else current_irr
    expected_irr = 0 if current_irr!=0 else round(predict_with_mlp(expected)[0][0],2)
    expected_irr = 0 if expected_irr== -0.0 else expected_irr
    print(hour)
    curr_input_power = pandas.DataFrame({
        'Year': [year],
        'Month': [month],
        'Day': [day],
        'Hour': [hour],
        'Minute': [30],
        'GHI': [current_irr],
        'Temperature': [temp_c]
    })
    exp_input_power = pandas.DataFrame({
        'Year': [year],
        'Month': [month],
        'Day': [day],
        'Hour': [hour],
        'Minute': [30],
        'GHI': [expected_irr],
        'Temperature': [temp_e]
    })

    features_df = create_new_features(curr_input_power)
    
    current_pac = predict_pac(features_df)
    current_pac = round(current_pac.iat[0, 12],2)
    features_df = create_new_features(exp_input_power)

    expected_pac = predict_pac(features_df)
    expected_pac = round(expected_pac.iat[0, 12],2)

    

    
    
    N = dayOfYear(str(IST_time.date()))
    LST = 4*(float(loc_data[0]["lon"])-82.50)+EOT(N)
    offset = 330+int(LST)
    LST = IST_time + timedelta(minutes=LST)
    


    output = [{
            "theoretical":{
                "large":le_theo,
                "small":se_theo
            },
            "Maximum Temperature":{
                "current": round(((api_data['main']['temp_max']) - 273.15),2),
                "expected":round(stats.iat[0, 1],2)
            }, 
            "Minimum Temperature": {
                 "current": round(((api_data['main']['temp_min']) - 273.15),2),
                "expected": round(stats.iat[0, 0],2)
            },
            "Feels Like": {
                 "current": round(((api_data['main']['feels_like']) - 273.15),2),
                "expected": round(stats.iat[0, 7],2)
            },
            "Humidity": {
                "current": round(api_data['main']['humidity'],2),
                "expected":round(stats.iat[0, 2],2)
            },
            "Cloud Cover":{
                 "current": round(api_data['clouds']['all'],2),
                "expected":round(stats.iat[0, 3],2)
            },
            "Pressure":{
                "current": round(api_data['main']['pressure'],2),
                "expected":round(stats.iat[0, 6],2)
            },
            "Wind Speed":{
                "current": round(api_data['wind']['speed'],2),
                "expected":wse
            },
            "Wind Direction":{
                "current": round(api_data['wind']['deg'],2),
                "expected": round(stats.iat[0, 5],2)
            },
            
            "Solar Power":[

                {
                    "Name":"Irradiation",
                    "Icon":"Irradiation",
                    "current": round(current_irr),
                    "expected":  round(expected_irr)
                },

                {
                    "Name":"Solar Power",
                    "Icon":"Power",
                    "current": round(current_pac*0.001,2),
                    "expected": round(expected_pac*0.001,2)
                }
            ],
            "Wind Power":[

                {
                    "Name":"Wind Power(Large Scale)",
                    "Icon":"Power",
                    "current":fpap_l,
                    "expected":fpap_le
                        
                    
                },

                {
                    "Name":"Wind Power(Residential)",
                    "Icon":"Power",
                    "current":fpap_s,
                    "expected":fpap_se
                        
                    
                }



    ],

            

            "Location Info": loc_data[0],
            "Time":[
                {
                    "Name":"GMT(+0:00)",
                    "Date":GMT_time.strftime("%d %b %Y %a"),
                    "Time": str(GMT_time.time())[0:8]
                },
                {
                    "Name": "IST(+5:30)",
                    "Date":IST_time.strftime("%d %b %Y %a"),
                    "Time": str(IST_time.time())[0:8]
                },
                {
                    "Name": f"LST(+{offset//60}:{offset%60:02d})",
                    "Date":LST.strftime("%d %b %Y %a"),
                    "Time": str(LST.time())[0:8]
                }
                
    ]
            
      }]
  
    

    """
    #temp_max_exp = sqlcontext.sql(f"SELECT avg(maxtempC) FROM city where month_year like '{mon}%' and time like '{tim[:2]}%'")
    #temp_max_exp = temp_max_exp.toPandas()
    query = f"SELECT avg(maxtempC) FROM df where month_year like '{mon}%' and time like '{tim[:2]}%'"
    temp_max_exp = sqldf(query)
    temp_max_exp = float(temp_max_exp.iat[0, 0])

    #hmdt_exp = sqlcontext.sql(f"SELECT avg(humidity) FROM city where month_year like '{mon}%' and time like '{tim[:2]}%'")
    #hmdt_exp = hmdt_exp.toPandas()
    query = f"SELECT avg(humidity) FROM df where month_year like '{mon}%' and time like '{tim[:2]}%'"
    hmdt_exp = sqldf(query)
    hmdt_exp = float(hmdt_exp.iat[0, 0])

    #cc_exp = sqlcontext.sql(f"SELECT avg(cloudcover) FROM city where month_year like '{mon}%' and time like '{tim[:2]}%'")
    query = f"SELECT avg(cloudcover) FROM df where month_year like '{mon}%' and time like '{tim[:2]}%'"
    cc_exp = sqldf(query)
    cc_exp = float(cc_exp.iat[0, 0])

    #wind_speed_exp = sqlcontext.sql(f"SELECT avg(windspeedKmph) FROM city where month_year like '{mon}%' and time like '{tim[:2]}%'")
    query = f"SELECT avg(windspeedKmph) FROM df where month_year like '{mon}%' and time like '{tim[:2]}%'"
    #wind_speed_exp = wind_speed_exp.toPandas()
    wind_speed_exp = sqldf(query)
    wind_speed_exp = float(wind_speed_exp.iat[0, 0])

    #wind_dir_exp = sqlcontext.sql(f"SELECT avg(winddirDegree) FROM city where month_year like '{mon}%' and time like '{tim[:2]}%'")
    #wind_dir_exp = wind_dir_exp.toPandas()
    query = f"SELECT avg(winddirDegree) FROM df where month_year like '{mon}%' and time like '{tim[:2]}%'"
    wind_dir_exp = sqldf(query)
    wind_dir_exp = float(wind_dir_exp.iat[0, 0])

    output = {'wind_speed': wind_spd, 'wind_speed_exp': round(wind_speed_exp, 2), 'wind_dir': wind_dir,
                   'wind_dir_exp': round(wind_dir_exp, 2), 'cloud_cover': cloud_cover,
                   'cloud_cover_exp': round(cc_exp, 1), 'humidity_curr': round(hmdt, 2),
                   'humidity_exp': round(hmdt_exp, 2), 'minTemp': round(temp_min, 2), 'maxTemp': round(temp_max, 2),
                   'minTempEx': round(temp_min_exp, 2), 'maxTempEx': round(temp_max_exp, 2)} 
   
   """
   
    #json_object = json.dumps(output, indent = 4)
    return Response(output)