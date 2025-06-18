import requests
from bs4 import BeautifulSoup
from argparse import ArgumentParser
import pandas as pd
import pickle
import re

def get_url_text(url):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text(separator='\n', strip=True)

def main(previous, current, output_file, combined_file, debug):
    with open(previous, 'rb') as f:
        df_old = pickle.load(f)
    with open(current, 'rb') as f:
        df_new = pickle.load(f)
   
    # job number, title, team 
    merged_df = df_new.merge(
       df_old[["job number"]], how="left",
       on="job number",
       indicator=True)
    merged_new_df = merged_df[merged_df['_merge'] == 'left_only'].drop('_merge', axis=1)
    if debug:
        print(merged_new_df)
        return
    print(f"Before: {len(df_new)}, After: {len(merged_new_df)}, Dups: {len(df_new) - len(merged_new_df)}") 
    merged_new_df.to_pickle(output_file)
    combined = pd.concat([merged_new_df, df_old], axis = 0)
    print(f"Cummulative: {len(combined)}")
    combined.to_pickle(combined_file)
 
if __name__=="__main__":
    data_dir = "../data"
    date_old = "250602"
    date_new = "250617"
    parser = ArgumentParser(description="Dedups any duplication from old in new")
    parser.add_argument("-p", "--previous", default=f"{data_dir}/{date_old}_job_number.pkl")
    parser.add_argument("-c", "--current", default=f"{data_dir}/{date_new}_job_number.pkl")
    parser.add_argument("-o", "--output", default=f"{data_dir}/{date_new}_job_number_dedup.pkl")
    parser.add_argument("-m", "--merged", default=f"{data_dir}/{date_new}_job_number_cum.pkl")
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    args = parser.parse_args()
    main(args.previous, args.current, args.output, args.merged, args.debug)
