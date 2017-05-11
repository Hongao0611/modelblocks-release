////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//  This file is part of ModelBlocks. Copyright 2009, ModelBlocks developers. //
//                                                                            //
//  ModelBlocks is free software: you can redistribute it and/or modify       //
//  it under the terms of the GNU General Public License as published by      //
//  the Free Software Foundation, either version 3 of the License, or         //
//  (at your option) any later version.                                       //
//                                                                            //
//  ModelBlocks is distributed in the hope that it will be useful,            //
//  but WITHOUT ANY WARRANTY; without even the implied warranty of            //
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             //
//  GNU General Public License for more details.                              //
//                                                                            //
//  You should have received a copy of the GNU General Public License         //
//  along with ModelBlocks.  If not, see <http://www.gnu.org/licenses/>.      //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#define ARMA_64BIT_WORD
#include <iostream>
#include <fstream>
#include <list>
#include <regex>
using namespace std;
#include <armadillo>
using namespace arma;
#define DONT_USE_UNMAPPABLE_TUPLES
#include <nl-randvar.h>
#include <nl-string.h>
#include <Delimited.hpp>
bool STORESTATE_TYPE = true;
bool STORESTATE_CHATTY = true;
#include <StoreState.hpp>
#include <Tree.hpp>

map<L,double> mldLemmaCounts;
const int MINCOUNTS = 100;

////////////////////////////////////////////////////////////////////////////////

int getArityGivenLabel ( const L& l ) {
  int depth = 0;
  int arity = 0;
  if ( l[0]=='N' ) arity++;
  for ( uint i=0; i<l.size(); i++ ) {
    if ( l[i]=='{' ) depth++;
    if ( l[i]=='}' ) depth--;
    if ( l[i]=='-' && l[i+1]>='a' && l[i+1]<='d' && depth==0 ) arity++;
  }
  return arity;
}

////////////////////////////////////////////////////////////////////////////////

O getOp ( const L& l, const L& lSibling, const L& lParent ) {
  if ( string::npos != l.find("-lN") ) return 'N';
  if ( string::npos != l.find("-lV") ) return 'V';
  if ( string::npos == l.find("-l")  || string::npos != l.find("-lS") || string::npos != l.find("-lC") ) return 'I';
  if ( string::npos != l.find("-lM") || string::npos != l.find("-lQ")      ) return 'M';
  if ( string::npos != l.find("-lA") && string::npos != lParent.find("\\") ) return '0'+getArityGivenLabel( string(lParent,lParent.find("\\")+1) );
  if ( string::npos != l.find("-lA") && string::npos == lParent.find('\\') ) return '0'+getArityGivenLabel( lSibling );
  cerr << "ERROR: unhandled -l tag in label \"" << l << "\"" << endl;
  return O();
}

////////////////////////////////////////////////////////////////////////////////

E getExtr ( const Tree& tr ) {
  N n =  T(L(tr).c_str()).getLastNonlocal();
  if ( n == N_NONE ) return 'N';
  if ( (tr.front().size()==0 || tr.front().front().size()==0) && n == N("-rN") ) return '0';
  if ( T(L(tr.front()).c_str()).getLastNonlocal()==n || T(L(tr.back() ).c_str()).getLastNonlocal()==n ) return 'N';
  return ( n.isArg() ) ? T(L(tr).c_str()).getArity()+'1' : 'M';
}

////////////////////////////////////////////////////////////////////////////////

T T_COLON ( "Pk" );                       // must be changed to avoid confusion with " : " delimiter in P params (where type occurs individually).
T T_CONTAINS_COMMA ( "!containscomma!" ); // must be changed to avoid confusion with "," delimiter in F,J params.

T getType ( const L& l ) {
  if ( l[0]==':' )                 return T_COLON;
  if ( l.find(',')!=string::npos ) return T_CONTAINS_COMMA;
  return string( string( l, 0, l.find("-l") ), 0, l.find("-o") ).c_str();
}

////////////////////////////////////////////////////////////////////////////////

pair<K,T> getPred ( const L& lP, const L& lW ) {
  T t = getType ( lP );   //string( lP, 0, lP.find("-l") ).c_str();

  // If punct, but not special !-delimited label...
  if ( ispunct(lW[0]) && ('!'!=lW[0] || lW.size()==1) ) return pair<K,T>(K::kBot,t);

  // Make predicate be lowercase...
  string sPred=lW;  transform(sPred.begin(), sPred.end(), sPred.begin(), [](unsigned char c) { return std::tolower(c); });
  string sBaseType = t.getString();

  // If preterm is morphrule-annotated, use baseform in pred...
  smatch m; for( string s=lP; regex_match(s,m,regex("^(.*?)-x([^-:|]*:|[^-%:|]*)([^-%:|]*?)%([^-%:|]*)[|](.)([^-:|]*:|[^-%:|]*)([^-%:|]*?)%([^-%:|]*)(.*)$")); s=m[9] ) {
    //cout<<"MATCH "<<string(m[1])<<" "<<string(m[2])<<" "<<string(m[3])<<" "<<string(m[4])<<" "<<string(m[5])<<" "<<string(m[6])<<" "<<string(m[7])<<" "<<string(m[8])<<" "<<string(m[9])<<endl;
    sPred = regex_replace( sPred, regex("^"+string(m[3])+"(.*)"+string(m[4])+"$"), string(m[7])+"$1"+string(m[8]) );
    sBaseType[0] = string(m[5])[0];
  }
  sBaseType = regex_replace( sBaseType, regex("-x.*"), string("") );
  lP = sBaseType;

  /*
  smatch m;  if ( regex_match( lP, m,regex("(.*?)-o.*[|]([^- ]*)") ) ) {
    sBaseType = (lP[0]=='V' || lP[0]=='B' || lP[0]=='L' || lP[0]=='G') ? "B" + string(m[1],1) + "-o" + string(m[2])
              : (lP[0]=='N')                                           ? "N" + string(m[1],1) + "-o" + string(m[2])
              : (lP[0]=='A' || lP[0]=='R')                             ? "A" + string(m[1],1) + "-o" + string(m[2])
                                                                       : "ERROR:UNDEFINED_BASE";
    sPred     = regex_replace( sPred, regex(string(m[4])+"$"), string(m[3]) );
  }
  */

  if ( mldLemmaCounts.find(sPred)==mldLemmaCounts.end() || mldLemmaCounts[sPred]<MINCOUNTS ) sPred = "!unk!";
  if ( isdigit(lW[0]) )                                                                      sPred = "!num!";
  return pair<K,T>( (sBaseType + ':' + sPred + '_' + ((lP[0]=='N') ? '1' : '0')).c_str(), t );
}

////////////////////////////////////////////////////////////////////////////////

void calcContext ( const Tree& tr, int s=1, int d=0, E e='N' ) {
  static F          f;
  static E          eF;
  static Sign       aPretrm;
  static StoreState q;

  // At unary preterminal...
  if ( tr.size()==1 && tr.front().size()==0 ) {

    f            = 1 - s;
    if( e=='N' ) eF = e = getExtr ( tr );
    pair<K,T> kt = getPred ( L(tr), L(tr.front()) );
    K k          = kt.first;
    T t          = kt.second;
    aPretrm      = Sign( k, t );

    // Print preterminal / fork-phase predictors...
    DelimitedList<psX,FPredictor,psComma,psX> lfp;  q.calcForkPredictors(lfp);
    cout<<"----"<<q<<endl;
    cout << "F "; for ( auto& fp : lfp ) { if ( &fp!=&lfp.front() ) cout<<","; cout<<fp<<"=1"; }  cout << " : " << FResponse(f,e,k) << endl;
    cout << "P " << q.calcPretrmTypeCondition(f,e,k) << " : " << t              << endl;
    cout << "W " << k << " " << t                    << " : " << L(tr.front())  << endl;
  }

  // At unary prepreterminal (prior to filling gaps)...
  else if ( tr.size()==1 && tr.front().size()==1 && tr.front().front().size()==0 ) {

    f            = 1 - s;
    if( e=='N' ) eF = e = getExtr ( tr );
    pair<K,T> kt = getPred ( L(tr.front()), L(tr.front().front()) );          // use lower category (with gap filled) for predicate.
    K k          = kt.first;
    //T t          = kt.second;
    aPretrm      = Sign( k, getPred(L(tr),L(tr.front().front())).second );    // use upper category (with gap empty) for sign.

    // Print preterminal / fork-phase predictors...
    DelimitedList<psX,FPredictor,psComma,psX> lfp;  q.calcForkPredictors(lfp);
    cout<<"----"<<q<<endl;
    cout << "F "; for ( auto& fp : lfp ) { if ( &fp!=&lfp.front() ) cout<<","; cout<<fp<<"=1"; }  cout << " : " << FResponse(f,e,k) << endl;
    cout << "P " << q.calcPretrmTypeCondition(f,e,k) << " : " << aPretrm.getType()     << endl;
    cout << "W " << k << " " << aPretrm.getType()    << " : " << L(tr.front().front()) << endl;
  }

  // At unary prepreterminal...
  else if ( tr.size()==1 ) {
    E e = getExtr ( tr );
    calcContext ( tr.front(), s, d, e );
  }

  // At binary nonterminal...
  else if ( tr.size()==2 ) {

    // Traverse left child...
    calcContext ( tr.front(), 0, d+s );

    J j          = s;
    //NOT USED! Sign aAncstr = q.getAncstr ( f );
    //NOT USED! Sign aLchildTmp;
    //NOT USED! Sign aLchild = q.getLchild ( aLchildTmp, f, aPretrm );
    e            = getExtr ( tr ) ;
    O oL         = getOp ( L(tr.front()), L(tr.back()),  L(tr) );
    O oR         = getOp ( L(tr.back()),  L(tr.front()), L(tr) );

    // Print binary / join-phase predictors...
    DelimitedList<psX,JPredictor,psComma,psX> ljp;  q.calcJoinPredictors(ljp,f,eF,aPretrm);
    cout << "==== " << L(tr) << " -> " << L(tr.front()) << " " << L(tr.back()) << endl;
    cout << "J ";  for ( auto& jp : ljp ) { if ( &jp!=&ljp.front() ) cout<<","; cout<<jp<<"=1"; }  cout << " : " << JResponse(j,e,oL,oR)  << endl;
    cout << "A " << q.calcApexTypeCondition(f,j,eF,e,oL,aPretrm)                   << " : " << getType(tr)         << endl;
    cout << "B " << q.calcBrinkTypeCondition(f,j,eF,e,oL,oR,getType(tr),aPretrm)   << " : " << getType(tr.back())  << endl;

    // Update storestate...
    q = StoreState ( q, f, j, eF, e, oL, oR, getType(tr), getType(tr.back()), aPretrm );

    // Traverse right child...
    calcContext ( tr.back(), 1, d );
  }

  // At abrupt terminal (e.g. 'T' discourse)...
  else if ( tr.size()==0 );

  else cerr<<"ERROR: non-binary non-unary-preterminal: " << tr << endl;
}

////////////////////////////////////////////////////////////////////////////////

int main ( int nArgs, char* argv[] ) {

  list<DelimitedPair<psX,Delimited<double>,psSpace,L,psX>> lLC;

  // For each command-line flag or model file...
  for ( int a=1; a<nArgs; a++ ) {
    //if ( 0==strcmp(argv[a],"t") ) STORESTATE_TYPE = true;
    //else {
      cerr << "Loading model " << argv[a] << "..." << endl;
      // Open file...
      ifstream fin (argv[a], ios::in );
      // Read model lists...
      int linenum = 0;
      while ( fin && EOF!=fin.peek() ) {
        fin >> *lLC.emplace(lLC.end()) >> "\n";
        if ( ++linenum%1000000==0 ) cerr << "  " << linenum << " items loaded..." << endl;
      }
      cerr << "Model " << argv[a] << " loaded." << endl;
      for ( auto& l : lLC ) mldLemmaCounts[l.second] = l.first;
    //}
  }

  int linenum = 0;
  while ( cin && EOF!=cin.peek() ) {
    linenum++;
    Tree t("T"); t.emplace_back(); t.emplace_back("T");
    cin >> t.front() >> "\n";
    cout.flush();
    cout << "TREE " << linenum << ": " << t << "\n";
    if ( t.front().size() > 0 ) calcContext ( t );
  }

  cerr << "F TOTALS: " << FPredictor::getDomainSize() << " predictors, " << FResponse::getDomain().getSize() << " responses." << endl;
  cerr << "J TOTALS: " << JPredictor::getDomainSize() << " predictors, " << JResponse::getDomain().getSize() << " responses." << endl;
}

