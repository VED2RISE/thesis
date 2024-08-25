from flask import Flask, render_template, request
import numpy as np
import pickle
import joblib
import shap
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from lime.lime_tabular import LimeTabularExplainer
import pandas as pd

app = Flask(__name__)

### LOADING MODEL AND PRE-TRAINED STANDARD SCALER###
with open('model1.pkl', 'rb') as file:
    model = pickle.load(file)
scaler = joblib.load('scaler2.pkl')
X_train = pd.read_csv('X_train_preprocessed.csv')
feature_names = X_train.columns.tolist()

lime_explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    class_names=['Denied', 'Approved'],
    mode='classification'
)

@app.route('/', methods=['GET', 'POST'])
### GATHERING DATA FTOM THE FORM ###
def survey():
    if request.method == 'POST':
        duration = int(request.form['duration'])
        credit_amount = float(request.form['credit_amount'])
        installment_commitment = int(request.form['installment_commitment'])
        residence_since = int(request.form['residence_since'])
        age = int(request.form['age'])
        existing_credits = int(request.form['existing_credits'])
        num_dependents = int(request.form['num_dependents'])
        own_telephone = int(request.form['own_telephone'])  # 0 for No, 1 for Yes
        foreign_worker = int(request.form['foreign_worker'])  # 0 for No, 1 for Yes

        categorical_vector = np.zeros(50, dtype=int)

        categorical_map = {
            'checking_status': {
                '0 <= ... < £87.45': 0,
                '< £0': 1,
                '>= £87.45 / salary assignments for at least 1 year': 2,
                'no checking account': 3
            },
            'credit_history': {
                'all credits at this bank paid back duly': 4,
                'critical account/ other credits existing (not at this bank)': 5,
                'delay in paying off in the past': 6,
                'existing credits paid back duly till now': 7,
                'no credits taken/ all credits paid duly': 8
            },
            'purpose': {
                'business': 9,
                'car (new)': 10,
                'car (used)': 11,
                'domestic appliances': 12,
                'education': 13,
                'furniture/equipment': 14,
                'others': 15,
                'radio/television': 16,
                'repairs': 17,
                'retraining': 18
            },
            'savings_status': {
                '£43.46 <= ... < £217.30': 19,
                '£217.30 <= ... < £434.60': 20,
                '< £43.46': 21,
                '>= £434.60': 22,
                'unknown/ no savings account': 23
            },
            'employment': {
                '1 <= ... < 4 years': 24,
                '4 <= ... < 7 years': 25,
                '< 1 year': 26,
                '>= 7 years': 27,
                'unemployed': 28
            },
            'personal_status': {
                'female : divorced/separated/married': 29,
                'male : divorced/separated': 30,
                'male : married/widowed': 31,
                'male : single': 32
            },
            'other_parties': {
                'co-applicant': 33,
                'guarantor': 34,
                'none': 35
            },
            'property_magnitude': {
                'if not A121 : building society savings agreement/ life insurance': 36,
                'if not A121/A122 : car or other, not in attribute 6': 37,
                'real estate': 38,
                'unknown / no property': 39
            },
            'other_payment_plans': {
                'bank': 40,
                'none': 41,
                'stores': 42
            },
            'housing': {
                'for free': 43,
                'own': 44,
                'rent': 45
            },
            'job': {
                'management/ self-employed/ highly qualified employee/ officer': 46,
                'skilled employee / official': 47,
                'unemployed/ unskilled - non-resident': 48,
                'unskilled - resident': 49
            }
        }

        ### GETTING A FINAL FEATURE VECTOR TO BE USED IN THE MDODEL ###
        for field, options in categorical_map.items():
            user_input = request.form[field]
            if user_input in options:
                categorical_vector[options[user_input]] = 1

        numerical_features = np.array([
            duration, credit_amount, installment_commitment, residence_since, 
            age, existing_credits, num_dependents
        ]).reshape(1, -1)

        standardized_numerical_features = scaler.transform(numerical_features)

        feature_vector = np.concatenate([
            standardized_numerical_features.flatten(),
            [own_telephone, foreign_worker],
            categorical_vector
        ])

        proba = model.predict_proba([feature_vector])[0][1]

        threshold = 0.45

        decision = "Approved" if proba > threshold else "Denied"

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values([feature_vector])[0]

        shap_values_grouped = {
            'Duration': shap_values[0],
            'Credit Amount': shap_values[1],
            'Installment Commitment': shap_values[2],
            'Residence Since': shap_values[3],
            'Age': shap_values[4],
            'Existing Credits': shap_values[5],
            'Number of Dependents': shap_values[6],
            'Own Telephone': shap_values[7],
            'Foreign Worker': shap_values[8],
            'Checking Status': shap_values[9:13].sum(),
            'Credit History': shap_values[13:18].sum(),
            'Purpose': shap_values[18:28].sum(),
            'Savings Status': shap_values[28:33].sum(),
            'Employment': shap_values[33:38].sum(),
            'Personal Status': shap_values[38:43].sum(),
            'Other Parties': shap_values[43:46].sum(),
            'Property Magnitude': shap_values[46:50].sum(),
            'Other Payment Plans': shap_values[50:53].sum(),
            'Housing': shap_values[53:56].sum(),
            'Job': shap_values[56:60].sum()
        }

        final_feature_names = list(shap_values_grouped.keys())
        final_shap_values = np.array(list(shap_values_grouped.values()))

        plt.figure(figsize=(10, 8)) 
        plt.barh(final_feature_names, final_shap_values, color='blue')
        plt.xlabel("SHAP Value", fontsize=12)
        plt.title("Contribution of each feature (SHAP)", fontsize=14)

        plt.text(0.5, -0.1, f'Predicted Probability: {proba:.3f}', ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.5, -0.15, f'Model Threshold: {threshold}', ha='center', va='center', fontsize=12, color='red', transform=plt.gca().transAxes)

        plt.tight_layout(pad=2.0) 
        plt.subplots_adjust(left=0.3) 

        # saving SHAP plot in memory 
        img = BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        shap_plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close()

        ### LIME ###
        lime_exp = lime_explainer.explain_instance(
            data_row=feature_vector,
                        predict_fn=model.predict_proba,
            num_features=feature_vector.shape[0]
        )

        #Filtration of important features
        min_weight = 0.01
        filtered_explanation = [
            (feature, weight) for feature, weight in lime_exp.as_list() if abs(weight) >= min_weight
        ]

        img = BytesIO()
        fig = lime_exp.as_pyplot_figure()
        plt.clf() 
        plt.barh(
            [feature for feature, _ in filtered_explanation],
            [weight for _, weight in filtered_explanation],
            color=['green' if weight > 0 else 'red' for _, weight in filtered_explanation]
        )
        plt.xlabel("Lime Feature Weight", fontsize=12)
        plt.title("Significant Features (Lime Explanation)", fontsize=14)

        plt.tight_layout()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        lime_plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close()

        return render_template(
            'result.html',
            decision=decision,
            shap_explanation=f'<img src="data:image/png;base64,{shap_plot_url}" alt="SHAP explanation" />',
            lime_explanation=f'<img src="data:image/png;base64,{lime_plot_url}" alt="Lime explanation" />'
        )

    return render_template('home.html')

@app.route('/home')
def home():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)