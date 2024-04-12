import tensorflow 
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import numpy as np

model = tensorflow.keras.models.load_model("garv.h5")

app = Flask(__name__, template_folder="templates")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == 'POST':
        return predict()
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    CustomerId = int(request.form.get("CustomerId"))
    CreditScore = int(request.form.get("CreditScore"))  
    Age = int(request.form.get("Age")) 
    Tenure = int(request.form.get("Tenure")) 
    Balance = float(request.form.get("Balance") or 0) 
    NumOfProducts = int(request.form.get("NumOfProducts")) 
    EstimatedSalary = float(request.form.get("EstimatedSalary")) 
    Geography = request.form.get("Geography")
    
    if Geography and Geography.lower() == "france":
        Geography = 0
    elif Geography and Geography.lower() == 'germany':
        Geography = 1
    else:
        Geography = 2
            
    HasCrCard = request.form.get("HasCrCard")
    if HasCrCard and HasCrCard.lower() == "yes":
        HasCrCard = 1
    else:
        HasCrCard = 0
            
    IsActiveMember = request.form.get("IsActiveMember")   
    if IsActiveMember and IsActiveMember.lower() == "yes":
        IsActiveMember = 1
    else:
        IsActiveMember = 0
            
    Gender = request.form.get("Gender")
    if Gender and Gender.lower() == "male":
        Gender = 1 
    else:
        Gender = 0
            
    features = np.array([CustomerId, CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary])
    features = features.reshape(1, -1)
        
    ss = StandardScaler()
    features1 = ss.fit_transform(features)
        
    prediction = model.predict(features1)
        
    return render_template('index.html', result=prediction)

if __name__ == "__main__":
    app.run(debug=True,port=5000)
