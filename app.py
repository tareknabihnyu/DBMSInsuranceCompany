import random
from datetime import datetime

from flask import Flask, jsonify, flash, redirect, render_template, request, session, url_for
import pandas as pd
import pypyodbc as odbc
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


connection_string = 'Driver={ODBC Driver 18 for SQL Server};Server=tcp:echang.database.windows.net,1433;Database=car_insurance_db;Uid=TARE_NABI;Pwd=DMSFALL2024%-;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;'

conn = odbc.connect(connection_string)

app = Flask(__name__)
app.secret_key = 'secret_key'

@app.route('/')
def login():
    return render_template('login.html')

@app.post('/login') 
def validate():
    ssn = request.form['ssn']
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM Customer WHERE CustomerSSN = ?', [ssn])
    
    customer = cursor.fetchone()
    if not customer:
        flash('Invalid SSN.', 'error')
        return redirect(url_for('login'))

    customer_info = {
        'SSN': customer[0],
        'FirstName': customer[1],
        'LastName': customer[2], 
        'DOB': customer[3],
        'Gender': customer[4],
        'Email': customer[5],
        'Phone': customer[6],
        'LicenseNumber': customer[7],
    }
    print('Customer info:', customer_info)
    session['customer_info'] = customer_info

    # Fetch driving history, vehicle, and address
    cursor.execute('SELECT * FROM Driving_History WHERE CustomerSSN = ?', [ssn])
    driving_history = cursor.fetchone()

    cursor.execute('SELECT * FROM Vehicle WHERE CustomerSSN = ?', [ssn])
    vehicle = cursor.fetchone()

    cursor.execute('SELECT * FROM Address WHERE CustomerSSN = ?', [ssn])
    address = cursor.fetchone()

    print("Driving history:", driving_history)
    print("Vehicle info:", vehicle)

    # Calculate driver age
    dob = customer[3]
    current_date = datetime.now()
    driver_age = current_date.year - dob.year - ((current_date.month, current_date.day) < (dob.month, dob.day))
    driving_profile = {
        'age': driver_age,
        'gender': customer[4],
        'mileage': vehicle[5],
        'trafficviolations': driving_history[1],
        'accidents': driving_history[2],
        'drivingexperience': driving_history[3]
    }

    address_info = {
        'AddressLine1': address[0],
        'Zip': address[1],
        'AddressLine2': address[4],
        'State': address[5],
        'City': address[6]
    }

    vehicle_info = {
        'VIN': vehicle[0],
        'Brand': vehicle[1],
        'Model': vehicle[2],
        'Year': vehicle[3],
        'LicensePlate': vehicle[4],
        'Mileage': vehicle[5],
        'VehicleType': vehicle[6],
    }

    # Retrieve the highest premium policy
    cursor.execute('SELECT * FROM Contract WHERE CustomerSSN = ? ORDER BY MonthlyPrice DESC', [ssn])
    policy = cursor.fetchone()

    policy_info = {
        'ContractID': policy[0],
        'CoverageType': policy[1],
        'MaxCoverage': policy[2],
        'MonthlyPremium': policy[5]
    }
    
    cursor.execute('SELECT * FROM Company WHERE CompanyCode = ?', [policy[6]])
    company = cursor.fetchone() 
    company_info = {
        'CompanyCode': company[0],
        'CompanyName': company[1],
    }

    session['company_info'] = company_info
    session['policy_info'] = policy_info
    session['vehicle_info'] = vehicle_info
    session['driving_profile'] = driving_profile
    session['address_info'] = address_info
    print("Driving profile:", driving_profile)
    return redirect(url_for('home'))

@app.route('/home')
def home():
    customer_info = session.get('customer_info')
    print('Customer info:', customer_info)
    return render_template('home.html', user=customer_info)

@app.route('/profile')
def profile():
    customer_info = session.get('customer_info')
    vehicle_info = session.get('vehicle_info')
    driver_profile = session.get('driving_profile')
    address_info = session.get('address_info')
    policy_info = session.get('policy_info')
    return render_template('profile.html', policy=policy_info, user=customer_info, vehicle=vehicle_info, profile=driver_profile, address=address_info)

@app.route('/file-claim')
def file_claim():
    return render_template('file_claim.html')

@app.post('/file-claim')
def file_claim_post():
    data = request.get_json()
    print(data)
    accident_date = data.get('accidentDate')
    accident_desc = data.get('accidentDesc')
    claim_amount = data.get('claimAmount')

    claim_id = random.randint(300, 10000)
    status = 'Pending'
    ssn = session.get('customer_info')['SSN']
    contract_id = session.get('policy_info')['ContractID']

    new_claim = [claim_id, accident_date, accident_desc, claim_amount, status, ssn, contract_id]
    print("Claim Info:", new_claim)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO Claim (ClaimID, AccidentDate, AccidentDesc, ClaimAmount, Status, CustomerSSN, ContractID) VALUES (?, ?, ?, ?, ?, ?, ?)', new_claim)
    conn.commit()
    
    return jsonify({'message': 'Claim filed successfully!'})

@app.route('/generate-quote')
def generate_quote():
    driving_profile = session.get('driving_profile')
    premium_pred = round(predict_premium(quote_calculation_model, driving_profile)[0], 2)
    company_info = session.get('company_info')
    return render_template('generate_quote.html', quote=premium_pred, company=company_info)

def learning_model():
    sql_query = "SELECT * FROM Profile"
    df_profile = pd.read_sql_query(sql_query, conn)
    print(df_profile)
    df_profile['dob'] = pd.to_datetime(df_profile['dob'])
    df_profile['age'] = df_profile['dob'].apply(lambda x: datetime.now().year - x.year)

    features = ['age', 'customergender', 'mileage', 'trafficviolations', 'accidents', 'drivingexperience']
    X = df_profile[features]
    y = df_profile['monthlypremium']

    numeric_features = ['age', 'mileage', 'trafficviolations', 'accidents', 'drivingexperience']
    categorical_features = ['customergender']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ]
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(objective='reg:squarederror', random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [3, 5, 7],
        'regressor__learning_rate': [0.01, 0.1]
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error with Best Model: {mse}")

    for actual, predicted in zip(y_test[:10], y_pred[:10]):
        print(f"Actual: {actual}, Predicted: {predicted}")

    return best_model

def predict_premium(model, driving_profile):
    data = pd.DataFrame({
        'age': [driving_profile['age']],
        'customergender': [driving_profile['gender']],
        'mileage': [driving_profile['mileage']],
        'trafficviolations': [driving_profile['trafficviolations']],
        'accidents': [driving_profile['accidents']],
        'drivingexperience': [driving_profile['drivingexperience']]
    })

    premium_pred = model.predict(data)
    print(f"Predicted Monthly Premium for age {driving_profile['age']}: ${premium_pred[0]:.2f}")
    return premium_pred

if __name__ == '__main__':
    quote_calculation_model = learning_model()
    app.run(debug=True, port=3000)

