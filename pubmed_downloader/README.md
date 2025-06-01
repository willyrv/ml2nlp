# PubMed Article Downloader

This script downloads PubMed articles related to Artificial Intelligence, Large Language Models, and AI-assisted diagnosis published from January 2024 to the present date. The script processes articles month by month to handle large numbers of results efficiently.

## Features

- Automatic monthly download of articles from January 2024 to present
- Download articles in XML format
- Download PDF files when available
- Export comprehensive article metadata to CSV format
- Resume capability if download is interrupted
- Progress tracking per month
- Rate limiting to avoid server issues

## Requirements

- Python 3.x
- Required packages are listed in `requirements.txt`
- PubMed API key (get it from your NCBI account)

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Set up your credentials:
   - Copy `env.template` to `.env`
   - Edit `.env` and add your PubMed email and API key
   ```
   PUBMED_EMAIL=your.email@example.com
   PUBMED_API_KEY=your_api_key_here
   ```

## Usage

Simply run the script:
```bash
python pubmed_downloader.py
```

The script will automatically:
1. Create queries for each month from January 2024 to present
2. Download articles for each month sequentially
3. Save article metadata and PDFs
4. Track progress in monthly progress files
5. Continue from where it left off if interrupted

## Output

The script creates several output files and directories:

- `articles_metadata.csv`: Contains detailed information about each article including:
  - PMID
  - Title
  - Authors
  - Abstract
  - Keywords
  - Journal
  - Publication Date
  - DOI
  - PubMed Link
  - PDF Path (local path or "not available")
  - Month (when the article was published)
- `pdf_files/`: Directory containing downloaded PDF files (when available)
- `downloaded_articles/`: Directory containing article metadata
- `download_progress_YYYY_MM.json`: Progress tracking files for each month

## Resuming Downloads

If the script is interrupted, simply run it again. It will automatically:
1. Check which articles have already been downloaded for each month
2. Continue from where it left off
3. Skip previously downloaded articles

## Notes

- PDF downloads are only available for articles in PubMed Central with open access
- The script uses rate limiting to avoid overwhelming the PubMed servers
- All errors and warnings are logged for troubleshooting
- The script can be interrupted at any time with Ctrl+C
- Progress is saved after each article download 