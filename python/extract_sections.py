from argparse import ArgumentParser
from collections import defaultdict
from read_files import list_files

import re
headings = [("pay", "Pay \& Benefits"),
            ("pref", "Preferred Qualifications"),
            ("min", "Minimum Qualifications"),
            ("desc", "Description"),
            ("sum", "Summary"),
            ("title", "Search"),
            ("end", "Apple employees")]
regex = "|".join(f"(?P<{name}>{value})" for name,value in headings)

def extract(text):
    # Note the section titles come after the section
    # so we save the lines until we encounter a 
    col = {}
    prev = None
    saved = []
    for line in text.splitlines():
        line = line.strip()
        # looking for headings
        match_ = re.match(regex, line)
        if match_:
            # don't save prev
            if prev:
                col[prev] = saved
                saved = []
            prev = match_.lastgroup
        saved.append(line)
    if len(saved) > 0:
        col[prev] = saved
    print(col)

def main(input_dir, output_file, debug):
    files = list_files(input_dir)
    for file in files:
        with open(file, "r") as fp:
            text = fp.readlines()
            print(text[0])

if __name__=="__main__":
    parser = ArgumentParser(description="Reads text files to extract sections")
    parser.add_argument("-i", "--input_dir", default="Data/Scrape/")
    parser.add_argument("-o", "--output_file", default="Data/250602_job_description.pkl")
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    args = parser.parse_args()
    main(args.input_dir, args.output_file, args.debug) 


