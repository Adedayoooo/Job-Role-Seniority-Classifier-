import pandas as pd  
from src.scrape import scrape_remoteok,save_scraped_jobs
from src.config import SCRAPED_DATA_PATH,FINAL_JOBS_DATA
from src.classify import vectorize_and_train
import logging

logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger=logging.getLogger(__name__)

scraped_jobs=scrape_remoteok()
save_scraped_jobs(scraped_jobs)
vectorizer,model=vectorize_and_train(X_train,y_train)

def process_scraped_data():
    try:
        df = pd.read_csv(SCRAPED_DATA_PATH)
        df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")
        X_test_vec = vectorizer.transform(df["text"])
        df["predicted_level"] = model.predict(X_test_vec)
        
        df = df[["title", "company", "description", "salary", "url", "date_posted", "predicted_level"]]
        df = df.drop_duplicates(subset=["url"])
        df.to_csv(FINAL_JOBS_DATA, index=False)
        
        logger.info(f"Created final_jobs_data.csv with {len(df)} jobs and seniority labels")
        logger.info("Columns include predicted_level (Intern/Junior/Mid/Senior/Lead)")
        return df
    except Exception as e:
        logger.error(f"An error occurred:{e}")
        raise 

if __name__=="__main__":
    process_scraped_data()