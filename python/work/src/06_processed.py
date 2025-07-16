# Add a column for processed
from argparse import ArgumentParser
import re
import pandas as pd
from save_to_db import df_to_db


def get_jobs(input):
    """ get the jobs numbers from file """
    out = []
    with open(input, "r") as f:
        for line in f.readlines():
            mo = re.search(
                r"https://jobs.apple.com/en-us/details/(?P<job_number>\d+)(?:$|/.+)", line)
            if mo:
                if mo.group("job_number"):
                    print(mo["job_number"])
                    out.append(int(mo["job_number"]))
                else:
                    # pass
                    print(f"Not found: {line}")
            else:
                pass
                # print(f"Not found: {line}")
    return out


if __name__ == "__main__":
    file = "actions"
    date = "250713"
    data_dir = "../data/"
    db = "../data/apple.db"
    table = "jobs_250617_action"
    parser = ArgumentParser(description="Read processed jobs")
    parser.add_argument("-i", "--input_file", default=f"{data_dir}/{file}.md")
    parser.add_argument("-o", "--output_db", default=f"{data_dir}/{db}")
    parser.add_argument("-t", "--table_name", default=table)
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    args = parser.parse_args()
    jobs = get_jobs(args.input_file)
    jobs_df = pd.DataFrame({"number": jobs})
    jobs_df["action"] = 1
    df_to_db(jobs_df, args.output_db, args.table_name)
