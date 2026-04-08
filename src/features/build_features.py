import pandas as pd

# create dummy features
def create_dummy_vars(df):
    
    df = pd.get_dummies(df, columns=['University_Rating','Research'],dtype='int')
    x = df.drop(['Admit_Chance'], axis=1)
    y = df['Admit_Chance']
    return x, y