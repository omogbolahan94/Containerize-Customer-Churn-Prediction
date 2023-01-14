import pickle

with open('model_c=0.1.bin', 'rb') as f:
    dv, model = pickle.load(f)

# predicting if a  new customer will churn or not
customer = {
 "gender": "female",
 "partner": "yes",
 "dependents": "no",
 "phoneservice": "no",
 "multiplelines": "no_phone_service",
 "internetservice": "dsl",
 "onlinesecurity": "no",
 "onlinebackup": "yes",
 "deviceprotection": "no",
 "techsupport": "no",
 "streamingtv": "no",
 "streamingmovies": "no",
 "contract": "month-to-month",
 "paperlessbilling": "yes",
 "paymentmethod": "electronich_check",
 "seniorcitizen": 0,
 "tenure": 1,
 "monthlycharges": 29.85,
 "totalcharges": 29.85
}

# transform the new data
X = dv.transform([customer])

# predict
prob = model.predict_proba(X)[0][1]
if prob > 0.5:
    print(f'Customer will CHURN with a probability of {round(prob, 3)}!!!')
else:
    print(f'Customer will not churn.')


