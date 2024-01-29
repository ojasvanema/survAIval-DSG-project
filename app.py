from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import joblib


coxph_model = joblib.load('coxph_model.pkl')


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        rx = int(request.form['rx'])
        age = int(request.form['age'])
        adhere = int(request.form['adhere'])
        nodes = int(request.form['nodes'])
        extent = int(request.form['extent'])
        surg = int(request.form['surg'])
        node4 = int(request.form['node4'])


        x_new = pd.DataFrame({'rx': [rx], 'age': [age], 'adhere': [adhere],
                              'nodes': [nodes], 'extent': [extent], 'surg': [surg], 'node4': [node4]})


        pred_surv = coxph_model.predict_survival_function(x_new)


        time_points = np.arange(1, 3000)

        plt.figure(figsize=(8, 6))
        for i, surv_func in enumerate(pred_surv):
            plt.step(time_points, surv_func(time_points), where="post", label=f"Sample {i + 1}")

        plt.ylabel("Estimated Probability of Survival")
        plt.xlabel("Time in Days")
        plt.legend()


        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)


        img_base64 = base64.b64encode(img.read()).decode('utf-8')

        return render_template('result.html', img_base64=img_base64)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
