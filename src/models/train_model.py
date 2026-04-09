#Importing relevant models for training models
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import pickle


# Function to train the model
def train_RFmodel(X, y):
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # Train the MLP Classifier
    

    MLP = MLPClassifier(hidden_layer_sizes=(3), batch_size=50, max_iter=200, random_state=123)
    MLP.fit(X_train_scaled ,y_train)
   
    
    # Save the trained model
    with open('models/mlpmodel.pkl', 'wb') as f:
        pickle.dump(MLP, f)
    
    with open("models/columns.pkl", "wb") as f:
        pickle.dump(X.columns.tolist(), f)
        
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return MLP, X_test_scaled, y_test
