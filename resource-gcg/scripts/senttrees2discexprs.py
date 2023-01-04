import sys
import re
import tree

VERBOSE = False    ## print debugging info.
for a in sys.argv:
  if a=='-d':
    VERBOSE = True


################################################################################
##
##  I. HELPER FUNCTIONS
##
################################################################################

########################################
#
#  IA. get number of local/nonlocal arguments...
#
########################################

def getNoloArity( cat ):
  cat = re.sub( '-x.*', '', cat )
  while '{' in cat:
    cat = re.sub('\{[^\{\}]*\}','X',cat)
  return len(re.findall('-[ghirv]',cat))


def getLocalArity( cat ):
  cat = re.sub( '-x.*', '', cat )
  while '{' in cat:
    cat = re.sub('\{[^\{\}]*\}','X',cat)
  return len(re.findall('-[ab]',cat))


########################################
#
#  IB. get predicate (lemma) from word...
#
########################################

def getLemma( c, w ):
  s = re.sub('-l.','',c) + ':' + w.lower()
  eqns = re.sub( '-x.*:', ':', s )
  for xrule in re.split( '-x', c )[1:] :
    ## apply compositional lex rules...
    m = re.search( '(.*)%(.*)%(.*)\|(.*)%(.*)%(.*)', xrule )
    if m is not None:
      eqns = re.sub( '^'+m.group(1)+'(.*)'+m.group(2)+'(.*)'+m.group(3)+'$', m.group(4)+'\\1'+m.group(5)+'\\2'+m.group(6), eqns )
      continue
    m = re.search( '(.*)%(.*)\|(.*)%(.*)', xrule )
    if m is not None:
      eqns = re.sub( '^'+m.group(1)+'(.*)'+m.group(2)+'$', m.group(3)+'\\1'+m.group(4), eqns )
      continue
    m = re.search( '.*%.*\|(.*)', xrule )
    if m is not None: eqns = m.group(1)
  return eqns


########################################
#
#  IC. set scopes and variable numbers...
#
########################################

def setHeadScopeAnaph( t, Scopes, Anaphs, nWord=0 ):

  ## Recurse...
  for st in t.ch:
    nWord = setHeadScopeAnaph( st, Scopes, Anaphs, nWord )

  ## Account head words as done in gcg annotation guidelines, in order to track scope...
  if len(t.ch) == 0:
    nWord += 1
    t.sVar = str(nWord)
  elif len(t.ch) == 1:
    t.sVar = t.ch[0].sVar
  elif len(t.ch) == 2:
    t.sVar = t.ch[0].sVar if '-lU' in t.ch[0].c else t.ch[1].sVar if '-lU' in t.ch[1].c else t.ch[0].sVar if '-l' not in t.ch[0].c else t.ch[1].sVar if '-l' not in t.ch[1].c else None
  else: print( '\nERROR: too many children in ', t )

  ## Store scopes and anaphora...
  if '-yQ' in t.c:
    Scopes[t.sVar] = '0'
  m = re.search( '-s([0-9][0-9])?([0-9][0-9])', t.c )
  if m != None:
    sDest = str(int(m.group(2)))
    if sDest not in Scopes: Scopes[sDest] = '0'
    Scopes[t.sVar] = sDest
  m = re.search( '-[nm]([0-9][0-9])?([0-9][0-9])', t.c )
  if m != None:
    Anaphs[t.sVar] = str(int(m.group(2)))

  t.bMax = True
  for st in t.ch:
    st.bMax = ( st.sVar!= t.sVar )

  return( nWord )


########################################
#
#  ID. mark sites for raising above all in-situs...
#
########################################

def markSites( t, Scopes, aboveAllInSitu=True ):

  ## Mark until un-scoped argument...
  t.aboveAllInSitu = aboveAllInSitu
  if len(t.ch) == 2 and aboveAllInSitu and getLocalArity(t.c)==0:
    if ('-lA' in t.ch[0].c or '-lU' in t.ch[0].c) and t.ch[0].sVar not in Scopes: aboveAllInSitu = False
    if ('-lA' in t.ch[1].c or '-lU' in t.ch[1].c) and t.ch[1].sVar not in Scopes: aboveAllInSitu = False

  ## Recurse...
  for st in t.ch:
    markSites( st, Scopes, aboveAllInSitu )


########################################
#
#  IE. recursively translate tree to logic...
#
########################################

indent = 0
def translate( t, Scopes, Anaphs, lsNolo=[] ):

  ## 1. Verbose reporting...
  global indent
  indent += 2
  if VERBOSE: print( ' '*indent, 'var,max,tree:', t.sVar, t.bMax, t )
  if VERBOSE: print( ' '*indent, 'non-locals:', lsNolo )

  ## 2.a. Store quantifier...
  t.qstore = []
#  ## If can scope in situ, remove from scopes and carry on translating...
##  if t.sVar in Scopes and t.sVar not in Raised and t.sVar not in Scopes.values():
#  if t.bMax and t.sVar in Scopes and t.sVar not in Scopes.values():
#    del Scopes[ t.sVar ]
  ## If scoped and cannot be in situ, store...
#  if t.sVar in Scopes and t.sVar not in Raised and t.sVar in Scopes.values():
  if t.bMax and t.sVar in Scopes and t.sVar in Scopes.values():
    markSites( t, Scopes )
#    s = translate( t, Scopes, Anaphs, Raised+[t.sVar], lsNolo )
#    t.qstore = [( t.qstore, s, t.sVar )]
#    output = [ 'RaiseTrace', 'x'+t.sVar ]
##    t.aboveAllInSitu = False
#
#  ## 2.b. Pre-terminal branch...
#  elif len(t.ch) == 1 and len(t.ch[0].ch) == 0:
  if len(t.ch) == 1 and len(t.ch[0].ch) == 0:
    pred = getLemma( t.c, t.ch[0].c )
    output = 'Ident' if pred == '' else '@'+pred

  ## 3.c. Unary branch...
  elif len(t.ch) == 1:
    if   '-lE' in t.ch[0].c and len(t.ch[0].c) >= len(t.c):  output = [ translate( t.ch[0], Scopes, Anaphs, lsNolo[:-1] ), lsNolo[-1] ]
    elif '-lE' in t.ch[0].c and len(t.ch[0].c) <  len(t.c):  output = [ 'Mod'+str(getLocalArity(t.ch[0].c)), translate( t.ch[0], Scopes, Anaphs, lsNolo[:-1] ), lsNolo[-1] ]
    elif '-lV' in t.ch[0].c:  output = [ 'Pasv', 'x'+ t.sVar, translate( t.ch[0], Scopes, Anaphs, [( 'Trace', 'x'+t.sVar )] + lsNolo ) ]
    elif '-lZ' in t.ch[0].c:  output = [ 'Prop', translate( t.ch[0], Scopes, Anaphs, lsNolo ) ]
    elif getLocalArity(t.c) < getLocalArity(t.ch[0].c):  output = [ translate( t.ch[0], Scopes, Anaphs, lsNolo ), 'Some' ]
    else:  output = translate( t.ch[0], Scopes, Anaphs, lsNolo )
    ## Propagate child stores...
    t.qstore = t.ch[0].qstore

  ## 2.d. Binary branch...
  elif len(t.ch) == 2:
    m = getNoloArity(t.ch[0].c)
    if VERBOSE: print( ' '*indent, 'child cats and nolos:', t.ch[0].c, t.ch[1].c, m, lsNolo[:m], lsNolo[m:] )
    ## In-situ...
    if   '-lD' in t.ch[0].c or t.ch[0].c[0] in ',;:.!?':  output = translate( t.ch[1], Scopes, Anaphs, lsNolo )
    elif '-lD' in t.ch[1].c or t.ch[1].c[0] in ',;:.!?':  output = translate( t.ch[0], Scopes, Anaphs, lsNolo )
    elif '-lA' in t.ch[0].c or '-lU' in t.ch[0].c:  output = [ translate( t.ch[1], Scopes, Anaphs, lsNolo[m:] ), translate( t.ch[0], Scopes, Anaphs, lsNolo[:m] ) ]
    elif '-lA' in t.ch[1].c or '-lU' in t.ch[1].c:  output = [ translate( t.ch[0], Scopes, Anaphs, lsNolo[:m] ), translate( t.ch[1], Scopes, Anaphs, lsNolo[m:] ) ]
    elif '-lI' in t.ch[0].c:  output = [ translate( t.ch[1], Scopes, Anaphs, lsNolo[m:] ), [ 'SelfStore', 'x'+t.ch[1].sVar, translate( t.ch[0], Scopes, Anaphs, [( 'Trace', 'x'+t.ch[1].sVar )] + lsNolo[:m] ) ] ]
    elif '-lI' in t.ch[1].c:  output = [ translate( t.ch[0], Scopes, Anaphs, lsNolo[:m] ), [ 'SelfStore', 'x'+t.ch[0].sVar, translate( t.ch[1], Scopes, Anaphs, [( 'Trace', 'x'+t.ch[0].sVar )] + lsNolo[m:] ) ] ]
    elif '-lM' in t.ch[0].c:  output = [ 'Mod'+str(getLocalArity(t.ch[1].c)), translate( t.ch[1], Scopes, Anaphs, lsNolo[m:] ), translate( t.ch[0], Scopes, Anaphs, lsNolo[:m] ) ]
    elif '-lM' in t.ch[1].c:  output = [ 'Mod'+str(getLocalArity(t.ch[0].c)), translate( t.ch[0], Scopes, Anaphs, lsNolo[:m] ), translate( t.ch[1], Scopes, Anaphs, lsNolo[m:] ) ]
    elif '-lC' in t.ch[0].c:  output = [ 'And'+str(getLocalArity(t.ch[0].c)), translate( t.ch[0], Scopes, Anaphs, lsNolo ), translate( t.ch[1], Scopes, Anaphs, lsNolo ) ]
    elif '-lC' in t.ch[1].c:  output = translate( t.ch[1], Scopes, Anaphs, lsNolo )
    elif '-lG' in t.ch[0].c:  output = [ 'Store', 'x'+t.ch[0].sVar, translate( t.ch[0], Scopes, Anaphs, lsNolo[:m] ), translate( t.ch[1], Scopes, Anaphs, [( 'Trace', 'x'+t.ch[0].sVar )] + lsNolo[m:] ) ]
    elif '-lH' in t.ch[1].c and getNoloArity(t.ch[1].c)==1:  output = [ 'Store', 'x'+t.ch[1].sVar, [ 'SelfStore', 'x'+t.ch[0].sVar, translate( t.ch[1], Scopes, Anaphs, lsNolo[m:] + [('Trace', 'x'+t.ch[0].sVar )] ) ],
                                                                        translate( t.ch[0], Scopes, Anaphs, [('Trace', 'x'+t.ch[1].sVar )] + lsNolo[:m] ) ]
    elif '-lH' in t.ch[1].c:  output = [ 'Store', 'x'+t.ch[1].sVar, translate( t.ch[1], Scopes, Anaphs, lsNolo[m:] ), translate( t.ch[0], Scopes, Anaphs, [( 'Trace', 'x'+t.ch[1].sVar )] + lsNolo[:m] ) ]
    elif '-lR' in t.ch[0].c:  output = [ 'Mod'+str(getLocalArity(t.ch[1].c)), translate( t.ch[1], Scopes, Anaphs, lsNolo ), translate( t.ch[0], Scopes, Anaphs, [( 'Trace', 'x'+t.ch[1].sVar )] ) ]
    elif '-lR' in t.ch[1].c:  output = [ 'Mod'+str(getLocalArity(t.ch[0].c)), translate( t.ch[0], Scopes, Anaphs, lsNolo ), translate( t.ch[1], Scopes, Anaphs, [( 'Trace', 'x'+t.ch[0].sVar )] ) ]
    else:  print( '\nERROR: unhandled rule from ' + t.c + ' to ' + t.ch[0].c + ' ' + t.ch[1].c )
    ## Propagate child stores...
    t.qstore += (t.ch[0].qstore if hasattr(t.ch[0],'qstore') else []) + (t.ch[1].qstore if hasattr(t.ch[1],'qstore') else [])

  ## 2.e. Fail...
  else: print( '\nERROR: too many children in ', t )

  '''
  ## 3. Mark anaphora...
  if t.bMax and t.sVar in Anaphs:
    output = [ 'Anaphor', t.sVar, output ]
  if t.bMax and t.sVar in Anaphs.values():
    output = [ 'Antecedent', t.sVar, output ]
  '''

  ## 4. Retrieve quantifier...
  if VERBOSE: print( ' '*indent, 'cat and scopes:', t.c, Scopes )
  if VERBOSE: print( ' '*indent, 'quant store: ', t.qstore )
  if t.aboveAllInSitu:
    while len(t.qstore) > 0:
      l = [ r for r in t.qstore if r[2] not in Scopes.values() ]
      if len(l) > 0:
        if VERBOSE: print( ' '*indent, 'retrieving:', l[0] )
        output = [ '\\r'+l[0][2], '\\s'+l[0][2], l[0][1], [ '\\x'+l[0][2], 'True' ], [ '\\x'+l[0][2], output, 'r'+l[0][2], 's'+l[0][2] ] ]
        del Scopes[ l[0][2] ]
        t.qstore += l[0][0]
        t.qstore.remove( l[0] )
      else:
        break

  ## 5. If scoped and cannot be in situ, store...
  ## If can scope in situ, remove from scopes and carry on translating...
  if t.bMax and t.sVar in Scopes and t.sVar not in Scopes.values():
    del Scopes[ t.sVar ]
  if t.bMax and t.sVar in Scopes and t.sVar in Scopes.values():
    t.qstore = [( t.qstore, output, t.sVar )]
    output = [ 'RaiseTrace', 'x'+t.sVar ]

  if VERBOSE: print( ' '*indent, 'returning:', output )
  indent -= 2
  return( output )


################################################################################
##
##  II. HELPER FUNCTIONS FOR MACRO SUBSTITUTION
##
################################################################################

########################################
#
#  IIA. Replace constants with lambda functions...
#
########################################

Univ = [ '\\z', 'True' ]

def unpack( t ):
  if not isinstance(t,str):  return([ unpack(st) for st in t ])
  elif t=='And0':  return( [ '\\f', '\\g',        '\\r', '\\s', ['^', [ 'g',      'r', 's' ], [ 'f',      'r', 's' ] ] ] )
  elif t=='And1':  return( [ '\\f', '\\g', '\\q', '\\r', '\\s', ['^', [ 'g', 'q', 'r', 's' ], [ 'f', 'q', 'r', 's' ] ] ] )
  elif t=='Mod0':  return( [ '\\f', '\\g',        '\\r', '\\s', [ 'f',      [ '\\x', [ '^', ['r', 'x' ], [ 'g', [ '\\t', '\\u', '^', ['t','x'], ['u','x'] ], Univ, Univ ] ] ], 's' ] ] )
  elif t=='Mod1':  return( [ '\\f', '\\g', '\\q', '\\r', '\\s', [ 'f', 'q', [ '\\x', [ '^', ['r', 'x' ], [ 'g', [ '\\t', '\\u', '^', ['t','x'], ['u','x'] ], Univ, Univ ] ] ], 's' ] ] )
  elif t=='Prop':  return( [ '\\p', '\\q', '\\r', '\\s', 'q', Univ, [ '\\x', [ 'p', Univ, ['\\y','Equal','y','x'] ] ] ] )
#  elif t=='Prop':  return( [ '\\p', '\\q', '\\r', '\\s', 'q', Univ, [ '\\x', '^', [ 'p', Univ, ['\\y','Equal','y','x'] ], ['r','x'] ] ] )
  elif t.split(':')[0] == '@N-aD':  return( [ '\\q', '\\r', '\\s', 'Some', [ '\\z', '^', [ 'Some', [ '\\e', t[1:],'e','z' ], Univ ], ['r','z'] ], 's' ] )
  elif t.split(':')[0] == '@N-b{N-aD}':  return( [ '\\f', '\\r', '\\s', t[1:], [ '\\x', '^', ['r','x'], ['f','Some',['\\xx','Equal','xx','x'],Univ] ], 's' ] )
  elif t.split(':')[0] == '@B-aN-b{A-aN}':  return( [ '\\f', '\\q', '\\r', '\\s', 'f', 'q', [ '\\e', '^', [t[1:],'e'], ['r','e'] ], 's' ] )
  elif t[0]=='@' and getLocalArity( t.split(':')[0] ) == 1:  return( [        '\\q', '\\r', '\\s', 'q', Univ, [ '\\x',                     'Some', [ '\\e', '^', [t[1:],'e','x'    ], ['r','e'] ], 's'   ] ] )
  elif t[0]=='@' and getLocalArity( t.split(':')[0] ) == 2:  return( [ '\\p', '\\q', '\\r', '\\s', 'q', Univ, [ '\\x', 'p', Univ, [ '\\y', 'Some', [ '\\e', '^', [t[1:],'e','x','y'], ['r','e'] ], 's' ] ] ] )
  else:  return( t )


########################################
#
#  IIB. Replace in beta reduce...
#
########################################

def replace( t, old, new ):
#  if VERBOSE: print( 'replacing:', old, 'with', new, 'in', t )
  if t == old:
    return( new )
  elif isinstance(t,str):
    return( t )
  elif any( [ st[0]=='\\' and st[1:]==old for st in t ] ):
    return( t )
  else:
    return( [ replace( st, old, new ) for st in t ] )


########################################
#
#  IIC. Beta reduce...
#
########################################

def betaReduce( t ):
  if VERBOSE: print( 'reducing:', t )
  ## If string, skip...
  if isinstance(t,str):
    return
  ## If initial term is string, betaReduce children...
  elif isinstance(t[0],str):
    for st in t:
      betaReduce( st )
  ## Flatten initial application... 
  elif t[0][0][0]!='\\':
    t[:] = t[0] + t[1:]
    betaReduce( t )
  ## Substitute second term for initial lambda variable of initial (abstraction) term...
  elif len(t) >= 2:
    t[:] = [ replace( t[0][1:], t[0][0][1:], t[1] ) ] + t[2:]
    betaReduce( t )
  else:
    t[:] = t[0]
    betaReduce( t )


########################################
#
#  IID. Conjunction elimination...
#
########################################

def simplify( t ):
  if isinstance(t,str):  return
  if len(t)==3 and t[0]=='^' and t[1]==['True']:
    t[:] = t[2]
    simplify( t )
  elif len(t)==3 and t[0]=='^' and t[2]==['True']:
    t[:] = t[1]
    simplify( t )
  elif len(t)==4 and t[0][0]=='\\' and t[1]=='^' and t[2]==['True']:
    t[:] = [ t[0], t[3] ]
    simplify( t )
  elif len(t)==4 and t[0][0]=='\\' and t[1]=='^' and t[3]==['True']:
    t[:] = [ t[0], t[2] ]
    simplify( t )
  else:
    for st in t:
      simplify( st )


################################################################################
##
##  III. MAIN LOOP
##
################################################################################

nSent = None
Anaphs = None
for nLine,line in enumerate( sys.stdin ):

  print( '========== line ' + str(nLine) + ' ==========' )
  print( line[:-1] )

  if '!ARTICLE' in line:
    nSent = 0
    Anaphs = {}

  else:
    nSent += 1
    Scopes = {}
    t = tree.Tree()
    t.read( line )

    print( '----------' )
    setHeadScopeAnaph( t, Scopes, Anaphs )
    markSites( t, Scopes )
    if VERBOSE: print( 'Scopes', Scopes )
    out = translate( t, Scopes, Anaphs )
    if t.qstore != []: print( '\nERROR: nothing in quant store', t.qstore, 'allowed by scope list', Scopes )
    print( out )

    print( '----------' )
    out = [ unpack(out), Univ, Univ ]
    betaReduce( out )
    simplify( out )
    print( out )

