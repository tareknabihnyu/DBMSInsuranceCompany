﻿# Database Systems NYU 2024 - Final Project

**Author:** Tarek Nabih

## Demo

![Demo GIF](demo.gif)

## How to Run the Application

### 1. Install the ODBC Driver for SQL Server

Begin by installing the [ODBC Driver for SQL Server](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server?view=sql-server-ver16).

### 2. Install Required Python Packages

This application utilizes the `Flask` framework to serve static files. To install the necessary Python packages, open your terminal or command prompt and execute the following command:

```bash
pip install Flask pypyodbc pandas xgboost scikit-learn
```

### 3. Navigate to the Application Directory

Change your current directory to the main folder where the application is located:

```bash
cd path/to/app
```

### 4. Run the Application

Start the application by running the following command in your terminal:

```bash
python3 app.py
```

* The application should now be accessible in your web browser at: `http://127.0.0.1:3000/`
* For testing purposes, use the SSN `107-10-9797` from the `CUSTOMER` table to sign in.
* **Note:** Ensure you have Python 3 installed, as the command `python3 app.py` assumes the use of Python 3.

### Additional Information

The application relies on various libraries for data processing and machine learning. If you encounter errors related to missing packages, verify that all required packages are installed correctly.

## Use Cases

### Viewing Insurance Profile

Customers registered in the database through their auto insurance providers can enter their SSN (note: using SSN is not recommended for real-world applications) to view detailed insurance profile information. This includes:

- **Account Holder Information:** SSN, Last Name, First Name, Date of Birth, Gender, Email, Phone, License Number
- **Vehicle Details:** VIN, Brand, Model, Year, License Plate, Mileage, Vehicle Type
- **Current Contract Information:** Contract ID, Coverage Type, Maximum Coverage, Monthly Premium
- **Driving History:** Traffic Violations, Accidents, Driving Experience

### Generating a Quote

Customers can receive a real-time estimate of their monthly auto insurance premium based on their personal and vehicle information. The quote is generated using a supervised machine learning model trained on a dataset of insurance profiles, considering factors such as:

- Driver Age
- Gender
- Vehicle Mileage
- Traffic Violations
- Accidents
- Driving Experience


### Filing a Claim

Customers can file an auto insurance claim online by filling out a virtual claim form, eliminating the need for traditional paperwork. A representative agent will respond within 3-5 business days. For this project, the claim submission process has been simplified for demonstration purposes.

### Future Use Cases

* **Online Payments:** Enable users to make monthly premium and deductible payments online.

## Using `pyodbc`

For this project, we utilized `pyodbc` to connect to the Azure SQL Server, which hosts our Auto Insurance database tables. Below are examples of how `pyodbc` is used in the application:

### Retrieving CUSTOMER Information with SELECT

Using a `SELECT` query with `pyodbc`, the Flask app retrieves customer information (and other related data) for processing.

##### Example from `app.py`

```python
cursor = conn.cursor()
cursor.execute('SELECT * FROM Customer WHERE CustomerSSN = ?', [ssn])  
customer = cursor.fetchone()
```

### Submitting a New CLAIM with INSERT

When a user submits a new claim, the client sends a `POST` request to the `/file-claim` API endpoint. The server processes this data in JSON format and uses an `INSERT` query with `pyodbc` to add a new `CLAIM` record to the Azure SQL Database.

##### Example from `app.py`

```python
cursor.execute('INSERT INTO Claim (ClaimID, AccidentDate, AccidentDesc, ClaimAmount, Status, CustomerSSN, ContractID) VALUES (?, ?, ?, ?, ?, ?, ?)', new_claim)
conn.commit()
```
````
