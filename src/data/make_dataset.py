import pandas as pd

def load_and_clean_data(data_path):
    # Load dataset
    df = pd.read_csv(data_path)

    # convert columns to object type
    df['Credit_History'] = df['Credit_History'].astype('object')
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].astype('object')
    #Categorical variables
    df['Gender'].fillna('Male', inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

    #Numerical variable
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    df.drop(columns='Credit_History', axis=1, inplace=True)
    df = df.drop('Loan_ID', axis=1)

    return df