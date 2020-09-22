import sys, argparse, pandas as pd, numpy as np
from mvpa2.misc.data_generators import double_gamma_hrf as hrf


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Convolve data table using HRF')
    argparser.add_argument('data', type=str, help='Path to data table')
    argparser.add_argument('-s', '--step', type=float, default=2.0, help='Step size (in seconds) between fMRI samples')
    argparser.add_argument('-S', '--start', type=float, default=0.0, help='Start time (time of first scan)')
    args = argparser.parse_args()

    df = pd.read_csv(args.data,sep=' ',skipinitialspace=True)
    df['rate'] = 1.
   
    if 'sentid' in df.columns:
        df.sentid = df.sentid.astype('category')
    if 'docid' in df.columns:
        df.docid = df.docid.astype('category')
    else:
        df['docid'] = '1'
    if 'rolled' in df.columns:
        df.rolled = df.rolled.astype('category')

    cols = [x for x in df.select_dtypes([np.number]).columns if x != 'time']

    gb = df.groupby('docid')
    series = [x[1] for x in gb]
    series_names = [x[0] for x in gb]

    out = []
    for i in range(len(series)):
        df_cur = series[i]
        X = df_cur[cols]
        impulse_times = df_cur.time.values
        max_response_time = int(np.ceil(df_cur.time.max()))
        if max_response_time % 2 != 0:
           max_response_time += 1
        response_times = np.arange(0, max_response_time+args.step, args.step) + args.start
        D = response_times[..., None] - impulse_times[None, ...]
        G_mask = D >= 0
        G = hrf(D)
        G = np.where(G_mask, G, 0)
        X_conv = np.dot(G, X)
        X_conv = pd.DataFrame(X_conv, columns=cols)
        X_conv['time'] = response_times
        X_conv['docid'] = series_names[i]
        out.append(X_conv)

    out = pd.concat(out, axis=0)
    out.reset_index(drop=True, inplace=True)
    out['sampleid'] = 1
    out.sampleid = out.groupby(['docid']).sampleid.cumsum()
    sampleid_format = '{0:05d}'
    out.sampleid = out.docid.astype('str').str.cat(out.sampleid.apply(sampleid_format.format), sep='-')
    out.to_csv(sys.stdout, ' ', index=False, na_rep='NaN')

