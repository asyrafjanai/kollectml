import os
import numpy as np
import pandas as pd
import pendulum
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
root_dir = os.path.dirname(os.path.abspath('kollect'))
cache_dir = os.path.join(root_dir, 'storage')

@app.route('/')
def index():
    return 'Server Works!'

@app.route('/greet')
def say_hello():
    # Rec_class.train_model()
    return 'Hello from Server'

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Sales should be $ {}'.format(output))

@app.route('/results',methods=['POST'])
def results():
    model_path = os.path.join(cache_dir, 'model.sav')
    model = joblib.load(model_path)
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

@app.route('/api')
def recommend_api():
    '''
    /api?age=45&gender=4&marital=2&residence=2&exceed=1&advance=0&pass_due=1
    description:
    output:
    '''
    try:
        ticktok = pendulum.now()
        # try using request.form()
        # print("get in")
        age = int(request.args.get('age'))
        gender = int(request.args.get('gender'))
        marital = int(request.args.get('marital'))
        residence = int(request.args.get('residence'))
        exceed = int(request.args.get('exceed'))
        advance = int(request.args.get('advance'))
        pass_due = int(request.args.get('pass_due'))

        features = [age,gender,marital,residence,exceed,advance,pass_due]
        model_path = os.path.join(cache_dir, 'model.sav')
        model = joblib.load(model_path)
        temp = np.array(features).reshape(1, -1)
        col = ['age', 'gender_id', 'marital_status_id',
               'residence_or_business_country_id', 'exceed_original_bal',
               'advance_payment', 'ltd_pass_due']
        X = pd.DataFrame(temp, columns=col)
        prediction = model.predict(X)

        if prediction is not None:
            return jsonify(
                meta={
                    'is_success': True,
                    'exec_time': ticktok.diff(pendulum.now()).in_words(),
                },
                results=f'The prediction is {prediction}'
            )

        else:
            return jsonify(
                meta={
                    'message': 'No prediction can be made',
                    'is_success': False,
                    'exec_time': ticktok.diff(pendulum.now()).in_words(),
                }
            )
    except:
        return jsonify(
            meta={
                'message': 'Check input data',
                'is_success': False,
                'exec_time': ticktok.diff(pendulum.now()).in_words(),
            }
        )
#
if __name__ == '__main__':
    app.run(debug=True)
