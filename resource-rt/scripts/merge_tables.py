import sys, argparse
import pandas as pd

argparser = argparse.ArgumentParser(description='Does an inner merge of two space-delimited data tables.')
argparser.add_argument('f1', type=str, nargs=1, help='Path to first input data table')
argparser.add_argument('f2', type=str, nargs=1, help='Path to second input data table')
argparser.add_argument('key_cols', metavar='key', type=str, nargs='+', \
    help='Merge key fields')
args, unknown = argparser.parse_known_args()

def main():
    data1 = pd.read_csv(args.f1[0],sep=' ',skipinitialspace=True)
    data2 = pd.read_csv(args.f2[0],sep=' ',skipinitialspace=True)

    no_dups = [c for c in data2.columns.values if c not in data1.columns.values]

    data2_cols = args.key_cols + no_dups

    merged = pd.merge(data1, data2.filter(items=data2_cols), how='inner', on=args.key_cols)
    merged = merged * 1 # convert boolean to [1,0]
    merged.to_csv(sys.stdout, ' ', index=False, na_rep='nan')
      
main()   