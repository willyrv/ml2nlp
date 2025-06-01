#!/usr/bin/env python3

from Bio import Entrez
from Bio import Medline
import json
import os
from datetime import datetime
import calendar
from tqdm import tqdm
import time
import pandas as pd
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import logging
import math

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Configure Entrez
Entrez.email = os.getenv('PUBMED_EMAIL')
Entrez.api_key = os.getenv('PUBMED_API_KEY')

# Create necessary directories
DOWNLOADS_DIR = Path("downloaded_articles")
PDF_DIR = Path("pdf_files")
DOWNLOADS_DIR.mkdir(exist_ok=True)
PDF_DIR.mkdir(exist_ok=True)

# Constants
BATCH_SIZE = 1000  # Number of IDs to retrieve per batch
RETMAX = 9000     # Maximum number of articles to retrieve in total per search
SLEEP_TIME = 0.5  # Time to sleep between API calls

def construct_monthly_queries():
    """Construct a list of monthly queries from January 2024 to current month."""
    terms = [
        '"Artificial Intelligence"',
        '"Large Language Models"',
        '"AI-assisted diagnosis"'
    ]
    base_query = " OR ".join(terms)
    
    queries = []
    current_date = datetime.now()
    
    # Start from January 2024
    year = 2024
    month = 1
    
    while (year < current_date.year) or (year == current_date.year and month <= current_date.month):
        # Get the last day of the month
        _, last_day = calendar.monthrange(year, month)
        
        # Construct date restriction for this month
        date_restriction = f"{year}/{month:02d}/01:{year}/{month:02d}/{last_day}[Date - Publication]"
        
        # Create the full query
        monthly_query = {
            'query': f"({base_query}) AND ({date_restriction})",
            'year': year,
            'month': month,
            'description': datetime(year, month, 1).strftime("%B %Y")
        }
        
        queries.append(monthly_query)
        
        # Move to next month
        month += 1
        if month > 12:
            month = 1
            year += 1
    
    return queries

def get_article_count(query):
    """Get the total number of articles matching the search criteria."""
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=0)
        result = Entrez.read(handle)
        handle.close()
        return int(result["Count"])
    except Exception as e:
        logging.error(f"Error getting article count: {e}")
        return 0

def get_article_ids_with_history(query, start=0, max_results=RETMAX):
    """Get article IDs using PubMed's WebEnv feature."""
    try:
        # Initial search to get WebEnv and query_key
        handle = Entrez.esearch(
            db="pubmed",
            term=query,
            usehistory="y",
            retmax=0
        )
        search_results = Entrez.read(handle)
        handle.close()

        web_env = search_results["WebEnv"]
        query_key = search_results["QueryKey"]
        count = min(int(search_results["Count"]), max_results)

        # Fetch IDs in batches
        all_ids = []
        for start_idx in range(start, count, BATCH_SIZE):
            end = min(count, start_idx + BATCH_SIZE)
            fetch_size = end - start_idx
            
            try:
                fetch_handle = Entrez.esearch(
                    db="pubmed",
                    term=query,
                    retstart=start_idx,
                    retmax=fetch_size,
                    webenv=web_env,
                    query_key=query_key
                )
                fetch_results = Entrez.read(fetch_handle)
                fetch_handle.close()
                
                batch_ids = fetch_results["IdList"]
                all_ids.extend(batch_ids)
                
                time.sleep(SLEEP_TIME)
                
            except Exception as e:
                logging.error(f"Error fetching batch starting at {start_idx}: {e}")
                continue

        return list(set(all_ids))
    except Exception as e:
        logging.error(f"Error in get_article_ids_with_history: {e}")
        return []

def save_progress(downloaded_ids, year, month):
    """Save the list of downloaded article IDs to a progress file."""
    progress_file = f"download_progress_{year}_{month:02d}.json"
    with open(progress_file, "w") as f:
        json.dump({"downloaded_ids": list(downloaded_ids)}, f)

def load_progress(year, month):
    """Load the list of previously downloaded article IDs."""
    progress_file = f"download_progress_{year}_{month:02d}.json"
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            data = json.load(f)
            return set(data.get("downloaded_ids", []))
    return set()

def extract_pdf_link(pmid):
    """Try to extract PDF link from PubMed Central."""
    try:
        # First check if article is in PubMed Central
        handle = Entrez.elink(dbfrom="pubmed", db="pmc", id=pmid)
        results = Entrez.read(handle)
        handle.close()

        if not results[0]["LinkSetDb"]:
            return None

        pmc_id = results[0]["LinkSetDb"][0]["Link"][0]["Id"]
        
        # Get the PMC page
        url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for PDF link
        pdf_link = soup.find('a', {'class': 'int-view-pdf'})
        if pdf_link:
            return f"https://www.ncbi.nlm.nih.gov{pdf_link['href']}"
    except Exception as e:
        logging.warning(f"Could not extract PDF link for PMID {pmid}: {e}")
    return None

def download_pdf(url, pmid):
    """Download PDF file if available."""
    try:
        response = requests.get(url)
        if response.status_code == 200 and response.headers['Content-Type'] == 'application/pdf':
            pdf_path = PDF_DIR / f"article_{pmid}.pdf"
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            return str(pdf_path)
    except Exception as e:
        logging.warning(f"Failed to download PDF for PMID {pmid}: {e}")
    return "not available"

def get_article_details(pmid):
    """Get detailed article information."""
    try:
        handle = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="text")
        records = Medline.parse(handle)
        article = next(records)
        handle.close()

        # Extract and format authors
        authors = "; ".join(article.get("AU", []))
        
        # Get PDF if available
        pdf_link = extract_pdf_link(pmid)
        pdf_local_path = download_pdf(pdf_link, pmid) if pdf_link else "not available"
        
        return {
            'PMID': pmid,
            'Title': article.get("TI", ""),
            'Authors': authors,
            'Abstract': article.get("AB", ""),
            'Keywords': "; ".join(article.get("MH", [])),
            'Journal': article.get("JT", ""),
            'Publication Date': article.get("DP", ""),
            'DOI': article.get("LID", ""),
            'PubMed Link': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            'PDF Path': pdf_local_path
        }
    except Exception as e:
        logging.error(f"Error getting article details for PMID {pmid}: {e}")
        return None

def download_monthly_articles(query_info):
    """Download articles for a specific month."""
    query = query_info['query']
    year = query_info['year']
    month = query_info['month']
    description = query_info['description']
    
    logging.info(f"\nProcessing articles for {description}")
    
    # Get total number of articles for this month
    total_articles = get_article_count(query)
    logging.info(f"Found {total_articles:,} articles for {description}")
    
    if total_articles == 0:
        return
    
    # Load previously downloaded articles for this month
    downloaded_ids = load_progress(year, month)
    
    # Calculate how many articles we've already downloaded
    start_index = len(downloaded_ids)
    
    # Get article IDs for this batch
    article_ids = get_article_ids_with_history(query, start=start_index)
    
    if not article_ids:
        logging.info(f"No new articles to download for {description}")
        return
    
    # Filter out already downloaded articles
    remaining_ids = [pid for pid in article_ids if pid not in downloaded_ids]
    
    if not remaining_ids:
        logging.info(f"All articles have already been downloaded for {description}")
        return
    
    logging.info(f"Downloading {len(remaining_ids):,} new articles for {description}...")
    
    # List to store article details for CSV
    articles_data = []
    
    # Download articles with progress bar
    for pmid in tqdm(remaining_ids, desc=f"Downloading {description}"):
        # Get article details
        article_details = get_article_details(pmid)
        
        if article_details:
            article_details['Month'] = description  # Add month information
            articles_data.append(article_details)
            downloaded_ids.add(pmid)
            save_progress(downloaded_ids, year, month)
        
        time.sleep(SLEEP_TIME)
    
    # Update or create CSV file
    csv_path = "articles_metadata.csv"
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        new_df = pd.DataFrame(articles_data)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.drop_duplicates(subset=['PMID'], keep='last', inplace=True)
        combined_df.to_csv(csv_path, index=False)
    else:
        pd.DataFrame(articles_data).to_csv(csv_path, index=False)

def main():
    # Check environment variables
    if not Entrez.email or not Entrez.api_key:
        logging.error("Please set PUBMED_EMAIL and PUBMED_API_KEY in your .env file!")
        return
    
    # Get all monthly queries
    queries = construct_monthly_queries()
    logging.info(f"Created {len(queries)} monthly queries from January 2024 to present")
    
    # Process each month automatically
    for query_info in queries:
        description = query_info['description']
        try:
            download_monthly_articles(query_info)
            logging.info(f"Completed download for {description}")
        except KeyboardInterrupt:
            logging.info("\nDownload interrupted by user")
            return
        except Exception as e:
            logging.error(f"Error processing {description}: {e}")
            continue
        
        print("\n" + "="*50 + "\n")  # Visual separator between months

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("\nScript interrupted by user")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        logging.info("\nScript completed") 