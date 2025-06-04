from argparse import ArgumentParser
from collections import defaultdict
from read_files import list_files
import pandas as pd
import re
headings = [("pay", "Pay \& Benefits"),
            ("pref", "Preferred Qualifications"),
            ("min", "Minimum Qualifications"),
            ("desc", "Description"),
            ("sum", "Summary"),
            ("title", "Search"),
            ("end", "(Apple employees)|(Apple is an equal)")]
regex = "|".join(f"(?P<{name}>{value})" for name,value in headings)

def extract(text):
    col = defaultdict(list)
    cur_section = None
    for line in text:
        line = line.strip()
        # looking for headings
        match_ = re.match(regex, line)
        if match_:
            cur_section = match_.lastgroup
        else:
            if cur_section:
                col[cur_section].append(line)
    return col

def refine_summary(summary):
    col = dict()
    col['posted'] = col['number'] = None
    i = j = 0
    if 'Posted:' in summary:
        i = summary.index('Posted:') + 1
        col['posted'] = summary[i]
    if 'Role Number:' in summary:
        j = summary.index('Role Number:') + 1
        col['number'] = summary[j]
    col['summary'] = "\n".join(summary[max(i,j)+1:])
    return col

def refine(prev):
    col = dict()
    col['title'] = prev['title'][0]
    col['location'] = prev['title'][1]
    col |= refine_summary(prev['sum'])
    col['description'] = "\n".join(prev['desc'])
    col['minimum'] = "\n".join('* '+x for x in prev['min'])
    col['pref'] = "\n".join('* '+x for x in prev['pref'])

    if len(prev['pay']) == 0:
        col['pay_min'] = None
        col['pay_max'] = None
        # print(prev)
    else:
        match_ = re.search(r"\$([0-9,.]+) and \$([0-9,.]+)", prev['pay'][0])
        if not match_:
            col['pay_min'] = None
            col['pay_max'] = None
            print(prev['pay'])
        else:
            col['pay_min'] = match_.group(1)
            col['pay_max'] = match_.group(2)
    return col

def main(input_dir, output_file, debug, debug_files=None):
    if debug and debug_files:
        files = debug_files
    else:
        files = list_files(input_dir)
    out = []
    for file in files:
        with open(file, "r") as fp:
            text = fp.readlines()
            sections = extract(text)
            if debug:
                print("----- SECTIONS ------")
                print(sections)
            sections = refine(sections)
            if debug:
                print("------ REFINE ------")
                print(sections)
            out.append(sections)
    df = pd.DataFrame(out)
    with open(output_file, "wb") as f:
        df.to_pickle(f)

if __name__=="__main__":
    parser = ArgumentParser(description="Reads text files to extract sections")
    parser.add_argument("-i", "--input_dir", default="Data/Scrape/")
    parser.add_argument("-o", "--output_file", default="Data/250602_job_description.pkl")
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    parser.add_argument("-f", "--files_to_debug", nargs='*', default=["Data/Scrape/200604244.txt"])
    args = parser.parse_args()
    main(args.input_dir, args.output_file, args.debug,
        debug_files=args.files_to_debug) 


