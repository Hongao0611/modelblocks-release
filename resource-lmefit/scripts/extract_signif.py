import sys, re, argparse

argparser = argparse.ArgumentParser('''
Extracts recall values from a user-supplied list of lrtsignifs and outputs a space-delimited table of results.
''')
argparser.add_argument('lrtsignifs', type=str, nargs='+', help='One or more *.lrtsignif files from which to extract lme comparison results.')
args, unknown = argparser.parse_known_args()

val = re.compile('^.+: *([^ ]+)"$')
effectpair = re.compile('([^ ]+)-vs-([^ ]+)')

def compute_row(f, diamName=None, vs=None):
    row = {}
    line = f.readline()
    while line and not line.startswith('[1] "Main effect'):
        line = f.readline()
    assert line, 'Input not properly formatted'
    row['effect'] = val.match(line).group(1)
    line = f.readline()
    assert line.startswith('[1] "Corpus'), 'Input not properly formatted'
    row['corpus'] = val.match(line).group(1)
    line = f.readline()
    assert line.startswith('[1] "Effect estimate'), 'Input not properly formatted'
    row['estimate'] = '%.5g'%(float(val.match(line).group(1)))
    line = f.readline()
    assert line.startswith('[1] "t value'), 'Input not properly formatted'
    row['t value'] = '%.5g'%(float(val.match(line).group(1)))
    line = f.readline()
    assert line.startswith('[1] "Significance (Pr(>Chisq))'), 'Input not properly formatted'
    row['signif'] = '%.5g'%(float(val.match(line).group(1)))
    line = f.readline()
    assert line.startswith('[1] "Relative gradient (baseline)'), 'Input not properly formatted'
    row['rel_grad_base'] = '%.5g'%(float(val.match(line).group(1)))
    line = f.readline()
    assert line.startswith('[1] "Relative gradient (main effect)'), 'Input not properly formatted'
    row['rel_grad_main'] = '%.5g'%(float(val.match(line).group(1)))
    if diamName:
        row['diamondname'] = diamName
        left = effectpair.match(diamName).group(1)
        right = effectpair.match(diamName).group(2)
    if vs == 'base':
        row['pair'] = str(row['effect']) + '-vs-baseline'
    elif vs == 'both':
        if str(row['effect']) == left:
            base = right
        else:
            base = left
        row['pair'] = 'both-vs-' + base
    elif vs != None:
        row['pair'] = str(row['effect']) + '-vs-' + str(vs)
    return(row)

def print_row(row):
    out = [row['effect'], row['corpus'], row['estimate'], row['t value'], row['signif'], row['rel_grad_base'], row['rel_grad_main']]
    print(' '.join(out))

# Thanks to Daniel Sparks on StackOverflow for this one (post available at 
# http://stackoverflow.com/questions/5084743/how-to-print-pretty-string-output-in-python)
def getPrintTable(row_collection, key_list, field_sep=' '):
  return '\n'.join([field_sep.join([str(row[col]).ljust(width)
    for (col, width) in zip(key_list, [max(map(len, column_vector))
      for column_vector in [ [v[k]
        for v in row_collection if k in v]
          for k in key_list ]])])
            for row in row_collection])

pair_evals = [x for x in args.lrtsignifs if 'diamond' not in x]
diam_evals = [x for x in args.lrtsignifs if 'diamond' in x]

if len(pair_evals) > 0:
    print('===================================')
    print('Pairwise evaluation of significance')
    print('===================================')

    headers = ['effect', 'corpus', 'estimate', 't value', 'signif', 'rel_grad_base', 'rel_grad_main']
    
    header_row = {}
    for h in headers:
        header_row[h] = h

    rows = []

    for path in pair_evals:
        with open(path, 'rb') as f:
            rows.append(compute_row(f))

    converged = [header_row] + sorted([x for x in rows if (float(x['rel_grad_base']) < 0.002 and float(x['rel_grad_main']) < 0.002)], \
                key = lambda y: float(y['signif']))
    nonconverged = [header_row] + sorted([x for x in rows if (float(x['rel_grad_base']) >= 0.002 or float(x['rel_grad_main']) >= 0.002)], \
                   key = lambda y: float(y['signif']))

    print(getPrintTable(converged, headers))

    if len(nonconverged) > 1: #First element is the header row
        print('-----------------------------------')
        print('Convergence failures')
        print('-----------------------------------')
        print(getPrintTable(nonconverged, headers))

    print ''
    print ''
        
if len(diam_evals) > 0:
    print('==================================')
    print('Diamond evaluation of significance')
    print('==================================')

    headers = ['effect', 'corpus', 'diamondname', 'pair', 'estimate', 't value', 'signif', 'rel_grad_base', 'rel_grad_main']

    header_row = {}
    for h in headers:
        header_row[h] = h

    rows = []

    for path in diam_evals:
        with open(path, 'rb') as f:
            line = f.readline()
            while line and not line.startswith('[1] "Diamond Anova'):
                line = f.readline()
            assert line, 'Input is not properly formatted'
            diamName = val.match(line).group(1)
            while line and not line.startswith('[1] "Effect 1 ('):
                line = f.readline()
            assert line, 'Input not properly formatted'
            rows.append(compute_row(f, diamName, 'baseline'))
            while line and not line.startswith('[1] "Effect 2 ('):
                line = f.readline()
            assert line, 'Input not properly formatted'
            rows.append(compute_row(f, diamName, 'baseline'))
            while line and not line.startswith('[1] "Both vs. Effect 1'):
                line = f.readline()
            assert line, 'Input not properly formatted'
            rows.append(compute_row(f, diamName, 'both'))
            while line and not line.startswith('[1] "Both vs. Effect 2'):
                line = f.readline()
            assert line, 'Input not properly formatted'
            rows.append(compute_row(f, diamName, 'both'))

    converged = [header_row] + sorted([x for x in rows if (float(x['rel_grad_base']) < 0.002 and float(x['rel_grad_main']) < 0.002)], \
                key = lambda y: float(y['signif']))
    nonconverged = [header_row] + sorted([x for x in rows if (float(x['rel_grad_base']) >= 0.002 or float(x['rel_grad_main']) >= 0.002)], \
                   key = lambda y: float(y['signif']))

    print(getPrintTable(converged, headers))

    if len(nonconverged) > 0:
        print('-----------------------------------')
        print('Convergence failures')
        print('-----------------------------------')
        print(getPrintTable(nonconverged, headers))

    print ''
    print ''   
 