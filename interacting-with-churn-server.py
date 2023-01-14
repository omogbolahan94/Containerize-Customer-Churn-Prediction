import requests

# connect the local machine port (5000) to docker port (8080)
# which is already running at at http://0.0.0.0:8080 and ready to listen
# to all available network interfaces on this local machine port

url = "http://127.0.0.1:5000/predict"

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

response = requests.post(url, json=customer)
print(f"Request result: {response}")
print(response.json())

