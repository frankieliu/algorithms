import requests
from bs4 import BeautifulSoup
from argparse import ArgumentParser
import pandas
import pickle
import re

def get_url_text(url):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text(separator='\n', strip=True)

def main(input_file, output_dir, debug):
    with open(input_file, 'rb') as f:
        df = pickle.load(f)
    if debug:
        for val in df.values[:,0]:
            print(val) 
        return
    for val in df.values[:,0]:
        url = f"https://jobs.apple.com/en-us/details/{val}"
        text = re.sub(r'[^\x00-\x7F]+', '', get_url_text(url))
        out_file = f"{output_dir}{val}.txt"
        with open(out_file, "w") as f:
            f.write(text)

if __name__=="__main__":
    parser = ArgumentParser(description="Read df from pickle and get data from website")
    parser.add_argument("-i", "--input_file", default="Data/250602_job_number.pkl")
    parser.add_argument("-o", "--output_dir", default="Data/Scrape/")
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    args = parser.parse_args()
    main(args.input_file, args.output_dir, args.debug)
