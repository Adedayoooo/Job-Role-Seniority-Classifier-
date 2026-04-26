from pathlib import Path

ROOT_DIR=Path(__file__).parent.parent

APP_DIR=ROOT_DIR/"app"
DATA_DIR=ROOT_DIR/"data"
MODEL_DIR=ROOT_DIR/"model"
REQUIREMENTS_PATH=ROOT_DIR/"requirements.txt"
HF_TOKEN_PATH=ROOT_DIR/"hf_token.txt"
SRC_DIR=ROOT_DIR/"src"

#Data files
#Sample dataset created for training model
HANDLABELED_DATA_PATH=DATA_DIR/"handlabeled_training_data.csv"
#Data gotten from scraping 
SCRAPED_DATA_PATH=DATA_DIR/"scraped_jobs.csv"
#Final data after preprocessing scraped data 
FINAL_JOBS_DATA=DATA_DIR/"final_jobs_data.csv" # 

#App file
APP_PATH=APP_DIR/"app.py"

#Model files
MODEL_PATH=MODEL_DIR/"saved_model.pkl"

#Training file
TRAIN_PATH=SRC_DIR/"classify.py"

#Data gotten from scraping 
SCRAPE_PATH=SRC_DIR/"scrape.py"/

#Final dataset path(after preprocessing scraped data)
FINAL_DATASET_PATH=SRC_DIR/"final_dataset.py"

#Test code files
RF_TEST_PATH=SRC_DIR/"RandomForest"/"predict_rf.py"
