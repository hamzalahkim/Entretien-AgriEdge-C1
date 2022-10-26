import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)
model = joblib.load('pipe.pkl')

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/classify',methods=['POST'])
def classify():
    features = request.form.to_dict()
    features = list(features.values())
    features = np.array(features).reshape(1,-1)
    test =pd.DataFrame(data=features, columns=['Annee', 'ID', 'Var1', 'Var3', 'Var4', 'Var5', 'Var6'])
    print(test)
    res = model.predict(test)

    return render_template('form.html', result=f'{str(res[0])}')

if __name__ == "__main__":
    app.run(debug=True)