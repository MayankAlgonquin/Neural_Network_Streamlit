import pandas as pd

# create dummy features
def create_dummy_vars(df):
    
    df = pd.get_dummies(df, columns=['Gender', 'Married', 'Dependents','Education','Self_Employed','Property_Area'], dtype=int)
    df['Loan_Approved'] = df['Loan_Approved'].replace({'Y':1, 'N':0})
    # store the processed dataset in data/processed
    df.to_csv('data/processed/Processed_credit.csv', index=None)

    # Separate the input features and target variable
    x = df.drop('Loan_Approved',axis=1)
    y = df.Loan_Approved

    return x, y