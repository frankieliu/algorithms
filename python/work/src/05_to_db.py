import pandas as pd
import pickle
from argparse import ArgumentParser
import sqlite3


def save_to_db(df, dbpath, table_name):
    conn = sqlite3.connect(dbpath)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()


def get_df(apath):
    with open(apath, 'rb') as f:
        df = pickle.load(f)
    return df


def main(job_number_path, job_desc_path, dbpath, table_name, debug):
    df1 = get_df(job_number_path)
    # job number, title, team
    df1.rename(columns={'job number': 'number',
               'title': 'title-short'}, inplace=True)
    # title, location, posted, number, summary, description, mininum, pref, pay_min, pay_max
    df2 = get_df(job_desc_path)
    for c in ['pay_min', 'pay_max']:
        df2[c] = df2[c].str.replace(',', '')
        df2[c] = pd.to_numeric(df2[c], errors="coerce")
    df = pd.merge(df1, df2, on='number', how="outer")
    save_to_db(df, dbpath, table_name)
    # print(df.columns)


if __name__ == "__main__":
    data_dir = "../data"
    date = "250713"
    dbpath = f"{data_dir}/apple.db"
    job_number_path = f"{data_dir}/{date}_02_job_number_dedup.pkl"
    job_desc_path = f"{data_dir}/{date}_04_job_description.pkl"
    table_name = f"jobs_{date}"
    parser = ArgumentParser(description="Save to database")
    parser.add_argument("-n", "--job_number", default=job_number_path)
    parser.add_argument("-p", "--description", default=job_desc_path)
    parser.add_argument("-o", "--output", default=dbpath)
    parser.add_argument("-t", "--table", default=table_name)
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    args = parser.parse_args()
    main(args.job_number,
         args.description,
         args.output,
         args.table,
         args.debug)
