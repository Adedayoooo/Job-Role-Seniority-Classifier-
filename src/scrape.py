import os 
import pandas as pd 
import requests
import time
import logging
from src.config import SCRAPED_DATA_PATH

logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s",datefmt='%Y-%m-%d %H:%M:%S')
logger=logging.getLogger(__name__)

HEADERS={
    "User-Agent":"Mozilla/5.0 (compatible; MyRecruiterBot/1.0; +mailto:adebayoadedayo23@gmail.com)"
}

RATE_LIMIT_SECONDS=1

def scrape_remoteok():
    try:
        jobs=[]
        page=1
        max_pages=30
        skipped=0
        
        logger.info("Starting scrape from RemoteOk API...")
        while len(jobs)<250 and page<=max_pages:
            url=f"https://remoteok.com/api?page={pages}"
            response=requests.get(url,headers=HEADERS)
            if response.status_code!=200:
                logger.info(f"Scraping stopped at page {page}/nStatus {response.status_code})")
                break 
            data=response.json()
            if not isinstance(data,list) or len(data)==0:
                break
            for job in data:
                if not isinstance(job,dict) or "position" not in job:
                    continue 
                jobs.append({
                    "title": job.get("position", ""),
                    "company": job.get("company", ""),
                    "description": job.get("description", ""),
                    "salary": f"{job.get('salary_min', 0)} - {job.get('salary_max', 0)}" if job.get("salary_min") or job.get("salary_max") else "Not listed",
                    "url": job.get("url") or job.get("apply_url", ""),
                    "date_posted": job.get("date", "")
                })
                logger.info(f"Page {page} → {len(data)} jobs (total so far: {len(jobs)})")
            for job in data:
                if "position" not in job:
                    skipped += 1
                continue
            logger.info(f"Skipped {skipped} bad records")
            time.sleep(RATE_LIMIT_SECONDS)
            page += 1
            logger.info("Web scraping using RemoteOk.com completed!")
        return jobs 
    except Exception as e:
        logger.error(f"An error occurred:{e}")
        raise

def save_scraped_jobs(jobs:list(str))->pd.DataFrame:
    try:
        logger.info("Saving scraped jobs from RemoteOk.com...")
        os.makedirs("data", exist_ok=True)
        df = pd.DataFrame(jobs)
        df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
        df.to_csv(SCRAPED_DATA_PATH,index=False)
        logger.info(f"Scraped {len(df)} unique jobs")
        return df
    except Exception as e:
        logger.error(f"An error occurred:{e}")
        raise
    
if __name__ == "__main__":
    scraped_jobs=scrape_remoteok()
    save_scraped_jobs(scraped_jobs)