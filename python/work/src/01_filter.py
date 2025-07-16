import argparse
import re
import pandas as pd
import numpy as np


def main(input_file, output_file, debug=False, exclude=""):
    out = []
    with open(input_file) as f:
        l = f.readlines()
        for el in l:
            el = el.strip()
            m = re.split(r"\/+|\?team=", el)
            if debug:
                if len(m) != 7:
                    print(el)
                continue
            if exclude:
                match = re.search(exclude, m[5])
                if match:
                    continue
            out.append(m)
    out = np.array(out)
    # print(out[:,[4,5,6]])
    df = pd.DataFrame(out[:, [4, 5, 6]],
                      columns=["job number", "title", "team"])
    df.to_pickle(output_file)


if __name__ == "__main__":
    date = "250713"
    parser = argparse.ArgumentParser(description="Get job ids")
    parser.add_argument("-i", "--input_file", default=f"../data/{date}.md")
    parser.add_argument("-o", "--output_file",
                        default=f"../data/{date}_01_job_number.pkl")
    parser.add_argument("-d", "--debug", dest="debug",
                        action="store_true", default=False)
    parser.add_argument("-e", "--exclude", default=r"manager")
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.debug, args.exclude)
