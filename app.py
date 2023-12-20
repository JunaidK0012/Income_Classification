from flask import Flask,render_template,request
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

app = Flask(__name__)

@app.route("/")
def welcome():
    return render_template('index.html')

@app.route('/predictdata',methods = ['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template('pred.html')
    else:
        data = CustomData ( 
            age = int(request.form.get('age')),
            workclass = request.form.get('workclass'),
            education_num = int(request.form.get('education-num')),
            marital_status = request.form.get('marital-status'),
            occupation = request.form.get('occupation'),
            relationship = request.form.get('relationship'),
            race = request.form.get('race'),
            sex = request.form.get('sex'),
            capital_gain = float(request.form.get('capital-gain')),
            capital_loss = float(request.form.get('capital-loss')),
            hours_per_week = float(request.form.get('hours-per-week')),
            native_country = request.form.get('native-country')
            
        )

        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template('pred.html',results=results[0])






if __name__=="__main__":
    app.run(debug=True)