#Importing functions from different modules. 
#This file is the data pipeline to ingest a dataset, prepare the data, train and evaluate the model.

from src.data.make_dataset import load_and_clean_data
from src.visualization.visualize import plot_correlation_heatmap, plot_feature_importance, plot_confusion_matrix
from src.features.build_features import create_dummy_vars
from src.models.train_model import train_RFmodel
from src.models.predict_model import evaluate_model
import logging


import os

os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)

#Implementing logger to track the main pipeline.
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "DEBUG"),
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    
    # Step 1: Load data
    try:
        data_path = "/Users/mayankarora/Documents/BISI /Term 2/Neural Netoworks Main/data/raw/Admission.csv"
        df = load_and_clean_data(data_path)
        logger.info("Data loaded successfully")
    except FileNotFoundError:
        logger.error("File not found. Please enter a valid file.")
        
    except Exception as e:
        logger.exception(f"Error while loading data: {e}")
        

    # Step 2: Feature engineering
    try:
        X, y = create_dummy_vars(df)
        logger.info("Feature engineering completed")
    except Exception as e:
        logger.exception(f"Error in creating dummy variables: {e}")
        

    # Step 3: Train model
    try:
        model, X_test_scaled, y_test = train_RFmodel(X, y)
        logger.info("Model trained successfully")
    except Exception as e:
        logger.exception(f"Error during model training: {e}")
        

    # Step 4: Visualization
    try:
        plot_feature_importance(model, X_test_scaled, y_test, X.columns)
        plot_correlation_heatmap(df)
        logger.info("Feature importance plotted")
    except Exception as e:
        logger.exception(f"Error in feature importance: {e}")

    # Step 5: Evaluation
    try:
        metrics = evaluate_model(model, X_test_scaled, y_test)
        logger.info("Model evaluated successfully")
        logger.info(f"Metrics: {metrics}")
    except Exception as e:
        logger.exception(f"Error during evaluation: {e}")
        
    #Print all logs
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)
    
