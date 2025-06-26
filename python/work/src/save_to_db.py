import sqlite3

def df_to_db(df, dbpath, table_name):
    conn = sqlite3.connect(dbpath)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()