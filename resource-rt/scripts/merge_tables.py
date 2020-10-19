import sys, argparse
import pandas as pd

argparser = argparse.ArgumentParser(description='Does an inner merge of two space-delimited data tables.')
argparser.add_argument('f1', type=str, nargs=1, help='Path to first input data table')
argparser.add_argument('f2', type=str, nargs=1, help='Path to second input data table')
argparser.add_argument('key_cols', metavar='key', type=str, nargs='+', \
    help='Merge key fields')
argparser.add_argument('-o', '--outer', dest='outer', action='store_true', help='Use outer merge (defaults to inner)')
argparser.add_argument('-f', '--fallback', type=int, default=1, help='Which data table to return in case of key match failure. Either 1 or 2.')
args, unknown = argparser.parse_known_args()

merge_how = 'outer' if args.outer else 'inner'

def main():
    data1 = pd.read_csv(args.f1[0],sep=' ',skipinitialspace=True)
    data2 = pd.read_csv(args.f2[0],sep=' ',skipinitialspace=True)

    if len(set(data2.columns) & set(args.key_cols)) == 0:
        sys.stderr.write("No overlap between second data table and key fields.\n")
        if args.fallback == 1:
            sys.stderr.write("Returning first data table.\n")
            merged = data1
        elif args.fallback == 2:
            sys.stderr.write("Return second data table.\n")
            merged = data2
    else:
        no_dups = [c for c in data2.columns.values if c not in data1.columns.values]
        data2_cols = args.key_cols + no_dups
        merged = pd.merge(data1, data2.filter(items=data2_cols), how=merge_how, on=args.key_cols, suffixes=('', '_2'))
        merged = merged * 1 # convert boolean to [1,0]
        if 'subject' in merged.columns:
            merged.sort_values(['subject'] + args.key_cols, inplace=True)

    merged.to_csv(sys.stdout, ' ', index=False, na_rep='NaN')
      
main()   
