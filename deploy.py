from flask import Flask, render_template, request
import pickle

app = Flask(__name__, template_folder='templates', static_url_path='/static')
#load the model
model = pickle.load(open('diabetes_model.sav.', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/predict', methods=['POST', 'GET'])
def predict():
    ss1 = float(request.form['ss1'])
    ss2 = float(request.form['ss2'])
    ss3 = float(request.form['ss3'])
    ss4 = float(request.form['ss4'])
    ss5 = float(request.form['ss5'])
    ss6 = float(request.form['ss6'])
    ss7 = float(request.form['ss7'])
    ss8 = float(request.form['ss8'])
    result = model.predict([[ss1,ss2,ss3,ss4,ss5,ss6,ss7,ss8]])[0]

    
    return render_template ('index.html', **locals())
   

if __name__ == '__main__':
    app.run(debug=True)