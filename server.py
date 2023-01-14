from flask import Flask, request, jsonify
import pickle
import waitress

with open('model_c=0.1.bin', 'rb') as f:
    dv, model = pickle.load(f)

app = Flask('churn_classifier')


@app.route('/predict', methods=['post'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0][1]

    churn = y_pred >= 0.5

    result = {
        "Churn Probability": round(float(y_pred), 2),
        "Churn": bool(churn)
    }
    return jsonify(result)


# if __name__ == '__main__':
#     app.run(debug=True, host='127.0.0.1', port=5000)

    # or using waitress to run the flask application
    # waitress.serve(app, host='127.0.0.1', port=5000)
    # or use waitress.serve(app, listen='127.0.0.1:5000')


