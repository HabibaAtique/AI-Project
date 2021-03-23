import os
import numpy as np
from flask import Flask, render_template, request
import pickle


os.system('python predictor.py')
fileobj = 'Dataset.pkl'
classifier= pickle.load(open(fileobj, 'rb'))
with open(fileobj, 'rb') as fs:
    classifier = pickle.load(fs)


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predicting', methods=['POST'])
def predicting():
    a=0
    if request.method == 'POST':
        age = int(request.form['age'])
        dpf = float(request.form['dpf'])
        bmi = float(request.form['bmi'])
        insulin = int(request.form['insulin'])
        st = int(request.form['skinthickness'])
        bp = int(request.form['bloodpressure'])
        glucose = int(request.form['glucose'])
        preg = int(request.form['pregnancies'])
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        result = classifier.predict(data)
        accuracies=pickle.load( open('Accuracies.pkl', 'rb'))
        predicting.farray=[]
        for i in accuracies:
            predicting.farray.append(i[1])
        for i in accuracies:
            if i[1]==max(predicting.farray):
                maximum=a

            a=a+1

        return render_template('result.html', result=result,accuracies=accuracies,maximum=accuracies[maximum])


if __name__ == '__main__':
	app.run(debug=True)


