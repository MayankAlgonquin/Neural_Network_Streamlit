#Importing pandas library for data processing
import pandas as pd

#Loading and cleaning the dataset.
def load_and_clean_data(data_path):
    # Load dataset
    data = pd.read_csv(data_path)

    # Converting the target variable into a categorical variable
    data['Admit_Chance']=(data['Admit_Chance'] >=0.8).astype(int)
    data = data.drop(['Serial_No'], axis=1)
    data['University_Rating'] = data['University_Rating'].astype('object')
    data['Research'] = data['Research'].astype('object')

    return data