import sys
from flask import Flask, request, render_template
from src.CreditCardDefaulter.pipelines.prediction_pipeline import CustomDataset, PredictPipeline
from src.CreditCardDefaulter.exception import CustomException
from src.CreditCardDefaulter.logger import logging

app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')



@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoints():
    try:
        if request.method == 'GET':
            return render_template('form.html')
        else:
            # Extract form data
            form_data = {
                'LIMIT_BAL': request.form.get('LIMIT_BAL'),
                'SEX': request.form.get('SEX'),
                'EDUCATION': request.form.get('EDUCATION'),
                'MARRIAGE': request.form.get('MARRIAGE'),
                'AGE': request.form.get('AGE'),
                'PAY_0': request.form.get('PAY_0'),
                'PAY_2': request.form.get('PAY_2'),
                'PAY_3': request.form.get('PAY_3'),
                'PAY_4': request.form.get('PAY_4'),
                'PAY_5': request.form.get('PAY_5'),
                'PAY_6': request.form.get('PAY_6'),
                'BILL_AMT1': request.form.get('BILL_AMT1'),
                'BILL_AMT2': request.form.get('BILL_AMT2'),
                'BILL_AMT3': request.form.get('BILL_AMT3'),
                'BILL_AMT4': request.form.get('BILL_AMT4'),
                'BILL_AMT5': request.form.get('BILL_AMT5'),
                'BILL_AMT6': request.form.get('BILL_AMT6'),
                'PAY_AMT1': request.form.get('PAY_AMT1'),
                'PAY_AMT2': request.form.get('PAY_AMT2'),
                'PAY_AMT3': request.form.get('PAY_AMT3'),
                'PAY_AMT4': request.form.get('PAY_AMT4'),
                'PAY_AMT5': request.form.get('PAY_AMT5'),
                'PAY_AMT6': request.form.get('PAY_AMT6')
            }

            # Check for missing fields
            missing_fields = [key for key, value in form_data.items() if value is None]
            if missing_fields:
                error_message = f"Missing fields: {', '.join(missing_fields)}"
                return render_template('form.html', error_message=error_message)

            try:
                data = CustomDataset(
                    LIMIT_BAL=float(form_data['LIMIT_BAL']),
                    SEX=int(form_data['SEX']),
                    EDUCATION=int(form_data['EDUCATION']),
                    MARRIAGE=int(form_data['MARRIAGE']),
                    AGE=int(form_data['AGE']),
                    PAY_0=int(form_data['PAY_0']),
                    PAY_2=int(form_data['PAY_2']),
                    PAY_3=int(form_data['PAY_3']),
                    PAY_4=int(form_data['PAY_4']),
                    PAY_5=int(form_data['PAY_5']),
                    PAY_6=int(form_data['PAY_6']),
                    BILL_AMT1=float(form_data['BILL_AMT1']),
                    BILL_AMT2=float(form_data['BILL_AMT2']),
                    BILL_AMT3=float(form_data['BILL_AMT3']),
                    BILL_AMT4=float(form_data['BILL_AMT4']),
                    BILL_AMT5=float(form_data['BILL_AMT5']),
                    BILL_AMT6=float(form_data['BILL_AMT6']),
                    PAY_AMT1=float(form_data['PAY_AMT1']),
                    PAY_AMT2=float(form_data['PAY_AMT2']),
                    PAY_AMT3=float(form_data['PAY_AMT3']),
                    PAY_AMT4=float(form_data['PAY_AMT4']),
                    PAY_AMT5=float(form_data['PAY_AMT5']),
                    PAY_AMT6=float(form_data['PAY_AMT6'])
                )
            except ValueError as ve:
                error_message = "Invalid input data. Please enter correct values."
                return render_template('form.html', error_message=error_message)

            final_data = data.get_data_as_dataframe()
            predict_pipeline = PredictPipeline()
            prediction = predict_pipeline.predict(final_data)
            result = prediction.tolist()

            if result[0] == 1:
                final_result = "The credit card holder will be a Defaulter in the next month."
                final_result_class = "negative"
            else:
                final_result = "The credit card holder will not be a Defaulter in the next month."
                final_result_class = "positive"

            return render_template('result.html', final_result=final_result, final_result_class=final_result_class)

    except Exception as e:
        logging.info("An exception has occurred in predict_datapoints.")
        raise CustomException(e, sys)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
