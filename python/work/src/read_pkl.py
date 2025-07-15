
import sys 
import pickle

def read_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def check_for_duplicates(df, column):
    # keep = False marks all duplicates as True
    print(df[df[column].duplicated(keep=False)])

if __name__=="__main__":
    df = read_pickle(sys.argv[1])
    check_for_duplicates(df, 'job number' )