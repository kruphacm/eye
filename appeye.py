import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from flask import Markup
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('HTMLeye.html')
@app.route('/predict',methods=['POST'])
def predict():
    res1={'Cloudy vision/Blurred Vision':1.0,'Sensitivity to light':2.0,'Poor vision at night':3.0,'Double vision':4.0,'Eye pain':5.0,"Red-eye":6.0,'Vision Loss':7.0,'Dry or watery eyes':8.0,'burning or itching':9.0,'Tired':10.0,'Blurred vision for distant objects, near objects, or both':11.0,'Headache':12.0,'Irritation, itching':13.0,'Foreign body sensation eye discomfort':14.0,'Burning':15.0,'Itching':16.0,'Loss of vision':17.0,'Blurred Vision':18.0,'Discharge or stickiness':19.0}
    int_features = [res1[x] for x in request.form.values()]
    print(int_features)
    final_features = [np.array(int_features)]
    print(final_features)
    prediction = model.predict(final_features)
    output = abs(int(prediction[0]))
    res={0:'<p>Predicted Disease is Cataract<br><br>Treatment:Surgery</p>',1:'<p>Predicted Disease is Glaucoma<br><br>Treatment:Medical and Surgery</p>',2:'<p>Predicted Disease is Eye strain<br><br>Treatment:Artificial tears</p>',3:'<p>Predicted Disease is Refractive Errors<br><br>Treatment:Eyeglasses,Contact lenses,Refractive surgery</p>',4:'<p>Predicted Disease is Dry eye Syndrome<br><br>Treatment:Artificial tears</p>',5:'<p>Predicted Disease is Diabetic Retinopathy<br><br>Treatment:Management of Diabetes,laser treatment,Surgery</p>',6:'<p>Predicted Disease is Conjunctivitis<br><br>Treatment:Antibiotic/ antihistaminic eye drops or ointments</p>'}
    output=res[output]
    output=Markup(output)
    return render_template('HTMLeye.html', prediction_text=output)

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)