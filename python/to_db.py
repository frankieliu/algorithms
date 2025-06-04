import pandas as pd
import pickle
import sqlite3

def save_to_db(df, table_name, dbpath):
    conn = sqlite3.connect(dbpath)
    df.to_sql('jobs.2025.06.02', conn, if_exists='replace', index=False)
    conn.close()

def get_df(apath):
    with open(apath, 'rb') as f:
        df = pickle.load(f)
    return df

def main():
    dbpath = "Data/apple.db"
    job_number_path = "Data/250602_job_number.pkl"
    job_desc_path = "Data/250602_job_description.pkl"
    df1 = get_df(job_number_path)
    # job number, title, team 
    df1.rename(columns={'job number':'number', 'title': 'title-short'}, inplace=True)
    # title, location, posted, number, summary, description, mininum, pref, pay_min, pay_max
    df2 = get_df(job_desc_path)
    for c in ['pay_min','pay_max']:
        df2[c] = df2[c].str.replace(',','')
        df2[c] = pd.to_numeric(df2[c], errors="coerce")
    df = pd.merge(df1, df2, on='number', how="outer")
    save_to_db(df, 'jobs.25.06.02', dbpath) 
    # print(df.columns)

if __name__=="__main__":
    main()