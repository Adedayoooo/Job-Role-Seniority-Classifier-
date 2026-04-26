import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import json
import os
import joblib
from src.config import HANDLABELED_DATA_PATH,HF_TOKEN_PATH,MODEL_PATH
from huggingface_hub import HfApi,login
import logging

logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger=logging.getLogger(__name__)

token=HF_TOKEN_PATH.read_text().strip() if HF_TOKEN_PATH.exists() else None

def train_data()->tuple[pd.Series,pd.Series]:
    try:
        logger.info("Loading hand labeled training data...")
        training_data=pd.read_csv(HANDLABELED_DATA_PATH)
        logger.info("Training data loaded")
        train_df = pd.DataFrame(training_data)
        X_train = train_df["text"]
        y_train = train_df["level"]
        return X_train,y_train
    except Exception as e:
        logger.error(f"An error occurred:{e}")
        raise 

def vectorize_and_train(X_train:pd.Series,y_train:pd.Series)->tuple[TfidfVectorizer,LogisticRegression]:
    try:
        vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        X_train_vec = vectorizer.fit_transform(X_train)
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_vec, y_train)
        logger.info("Model trained")
        
        logger.info("Saving model...")
        joblib.dump((vectorizer,model),MODEL_PATH)
        return vectorizer,model
    except Exception as e:
        logger.error(f"An error occurred:{e}")
        raise 

def push_model_to_huggingface(model,repo_name:str,token:str=None):
  try:
    logger.info("Logging into Hugging Face Hub...")
    if token:
      login(token=token)
    else:
      login()
    model_filename=MODEL_PATH
    logger.info(f"Saving best model to {model_filename}...")
    joblib.dump(model,model_filename)
    logger.info(f"Model saved locally as {model_filename}")
    api = HfApi()
    api.upload_file(path_or_fileobj=model_filename,path_in_repo=model_filename.name,repo_id=repo_name,repo_type="model",commit_message="Upload trained Job_Role_Seniority_Classifier model")
    logger.info(f"Model successfully pushed to Hugging Face Hub: https://huggingface.co/{repo_name}")
  except Exception as e:
    logger.error(f"Failed to push model to HuggingFace due to the following: {e}")
    raise
         
if __name__=="__main__":
    X_train,y_train=train_data()
    vectorizer,model=vectorize_and_train(X_train,y_train)
    push_model_to_huggingface(model,"Job_Role_Seniority_Classifier_Model")