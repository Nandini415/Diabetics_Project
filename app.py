from flask import Flask, render_template ,request,redirect, url_for
import os
import numpy as np
import pandas as pd
import pickle

diabetes_model_path = 'rf_pipeline.pkl'
diabetes_model = pickle.load(open(diabetes_model_path, 'rb'))
app = Flask(__name__)
app.app_context().push()

#==============================Controllers=================================

@app.route('/',methods=['GET','POST'])
def home():
    if request.method == 'POST':
        f = [float(x) for x in request.form.values()]
        data = [np.array(f)]
        print(data)
        my_prediction = diabetes_model.predict(data)
        if(my_prediction[0] == 1):
            final_prediction = 'Yes'
            output = 'For women diagnosed with diabetes, its essential to take medications as prescribed and regularly monitor blood sugar levels using a home device. A balanced diet with limited sugars and regular meals is crucial, as is aiming for at least 150 minutes of moderate activity each week. Routine medical check-ups, including eye and foot exams, are vital for early detection of potential complications. Managing stress through techniques like meditation can be beneficial. If considering pregnancy, its imperative to closely monitor blood sugar levels. Limiting alcohol and smoking intake is advised, and staying updated on vaccinations, especially for the flu and pneumonia, is essential. Always consult with a healthcare provider for personalized advice.'
        else:
            final_prediction = 'No'
            output = 'To prevent diabetes in women, maintaining a healthy weight is key. This involves regular exercise, like brisk walking for 150 minutes weekly, and a diet rich in whole grains and lean proteins, minimizing processed foods and sugars. Periodic blood sugar tests are essential for early detection. Avoiding smoking, limiting alcohol, and managing stress are also vital. Being aware of family diabetes history provides added insight. In essence, a combination of a mindful diet, consistent exercise, and healthy lifestyle choices can significantly reduce the risk of diabetes in women.'
        return render_template('index.html', prediction_text=final_prediction,output=output)
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)