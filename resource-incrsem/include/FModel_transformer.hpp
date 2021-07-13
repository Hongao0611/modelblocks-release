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

int getDepth( const StoreState& ss ) {
    return ss.getDepth();
}


CVar getCatBase( const StoreState& ss ) {
    return ss.getBase().getCat();
}


HVec getHvB( const StoreState& ss ) {
    return (( ss.getBase().getHVec().size() > 0 ) ? ss.getBase().getHVec() : hvBot);
}


// returns a positional encoding for an embedding of dimensionality dim,
// at position wordIndex in sequence
vec getPositionalEncoding( uint dim, uint wordIndex ) {
  vec encoding = vec(dim);
  for ( uint i=0; i<dim; i++ ) {
    if ( i % 2 == 0 ) {
      encoding[i] = sin(wordIndex * exp(i * -log(10000) / dim));
    }
    else {
      encoding[i] = cos(wordIndex * exp((i-1) * -log(10000) / dim));
    }
  }
  return encoding;
}


class FPredictorVec {

  private:
    const HVec& hvA;
    const HVec& hvF;
    bool nullA;
    const StoreState& ss;

  public:
    FPredictorVec(  const HVec& hvAnt, bool nullAnt, const StoreState& sState )
      : hvA ((hvAnt.size() > 0) ? hvAnt : hvBot),
      hvF (( sState.getBase().getCat().getNoloArity() && sState.getNoloBack().getHVec().size() != 0 ) ? sState.getNoloBack().getHVec() : hvBot),
      nullA (nullAnt),
      ss (sState)
    {
    }

    bool getNullA() {
      return nullA;
    }

    const HVec getHvA() const {
      return hvA; //antecedent
    }

    const HVec getHvF() const {
      return hvF; //filler
    }

    const StoreState& getStoreState() const {
        return ss;
    }

    friend ostream& operator<< ( ostream& os, const FPredictorVec& fpv ) {
      //const StoreState ss = fpv.getBeamElement().getHidd().getStoreState();
      const int d = getDepth(fpv.ss);
      const CVar catBase = getCatBase(fpv.ss);
      const HVec hvB = getHvB(fpv.ss);
      const HVec hvF = fpv.getHvF();
        
      os << d << " " << catBase << " " << hvB << " " << hvF << " " << fpv.hvA << " " << fpv.nullA;
      return os;
    }
};



////////////////////////////////////////////////////////////////////////////////
class FModel {

  typedef DelimitedTrip<psX,F,psAmpersand,Delimited<EVar>,psAmpersand,Delimited<K>,psX> FEK;
  typedef DelimitedCol<psLBrack, double, psComma, psRBrack> CVec;
  typedef DelimitedCol<psLBrack, double, psComma, psRBrack> KDenseVec;

  private:
    static const uint FSEM_DIM_DEFAULT = 20;
    static const uint FSYN_DIM_DEFAULT = 20;
    static const uint FANT_DIM_DEFAULT = 20;
    uint FSEM_DIM;
    uint FSYN_DIM;
    uint FANT_DIM;

    map<CVar,vec> mcbv;                        // map between syntactic category and embeds
    map<KVec,vec> mkbdv;                  // map between KVec and embeds
    map<KVec,vec> mkfdv;
    map<KVec,vec> mkadv;

    map<FEK,unsigned int> mfeki;                // response indices
    map<unsigned int,FEK> mifek;

    unsigned int iNextResponse  = 0;

    // weights
    DelimitedVector<psX, double, psComma, psX> fwp; // pre-attention feedforward
    DelimitedVector<psX, double, psComma, psX> fwi; // attention input projection -- contains query, key, and value matrices
    DelimitedVector<psX, double, psComma, psX> fwo; // attention output projection
    DelimitedVector<psX, double, psComma, psX> fwf; // first feedforward
    DelimitedVector<psX, double, psComma, psX> fws; // second feedforward
    // biases
    DelimitedVector<psX, double, psComma, psX> fbp; // pre-attention feedforward
    DelimitedVector<psX, double, psComma, psX> fbi; // attention input projection
    DelimitedVector<psX, double, psComma, psX> fbo; // attention output projection
    DelimitedVector<psX, double, psComma, psX> fbf; // first feedforward
    DelimitedVector<psX, double, psComma, psX> fbs; // second feedforward

    mat fwpm;
    mat fwim;
    mat fwqm; // query
    mat fwkm; // key
    mat fwvm; // value
    mat fwom;
    mat fwfm;
    mat fwsm;

    vec fbpv;
    vec fbiv;
    vec fbqv; // query
    vec fbkv; // key
    vec fbvv; // value
    vec fbov;
    vec fbfv;
    vec fbsv;

    vec computeResult( vector<vec> attnInput, vec corefVector, bool verbose=0 ) const;

  public:

    FModel( ) 
      //: FSEM_DIM_DEFAULT (20),
      //: FSYN_DIM_DEFAULT (20),
      //FANT_DIM_DEFAULT (20) 
      { }

    FModel( istream& is )
      //: FSEM_DIM_DEFAULT (20),
      //: FSYN_DIM_DEFAULT (20),
      //FANT_DIM_DEFAULT (20) 
    {
      while ( is.peek()=='F' ) {
        Delimited<char> c;
        is >> "F " >> c >> " ";
        if (c == 'P') is >> fwp >> "\n";
        if (c == 'p') is >> fbp >> "\n"; 
        if (c == 'I') is >> fwi >> "\n";
        if (c == 'i') is >> fbi >> "\n"; 
        if (c == 'O') is >> fwo >> "\n";
        if (c == 'o') is >> fbo >> "\n"; 
        if (c == 'F') is >> fwf >> "\n";
        if (c == 'f') is >> fbf >> "\n"; 
        if (c == 'S') is >> fws >> "\n";
        if (c == 's') is >> fbs >> "\n"; 
      }

      FSYN_DIM = FSYN_DIM_DEFAULT;
      while ( is.peek()=='C' ) {
        Delimited<char> c;
        Delimited<CVar> cv;
        DelimitedVector<psX, double, psComma, psX> vtemp;  
        is >> "C " >> c >> " " >> cv >> " ";
        is >> vtemp >> "\n";
        //if (c == 'B') mcbv.try_emplace(cv,vtemp);
        assert (c == 'B');
        mcbv.try_emplace(cv,vtemp);
        FSYN_DIM=vtemp.size();
      }

      FSEM_DIM = FSEM_DIM_DEFAULT;
      FANT_DIM = FANT_DIM_DEFAULT;
      //zeroCatEmb=arma::zeros(FSYN_SIZE);
      while ( is.peek()=='K' ) {
        Delimited<char> c;
        Delimited<K> k;
        DelimitedVector<psX, double, psComma, psX> vtemp;
        is >> "K " >> c >> " " >> k >> " ";
        is >> vtemp >> "\n";
        if (c == 'B') { 
          mkbdv.try_emplace(k, vtemp);
          FSEM_DIM=vtemp.size();
        }
        else if (c == 'F') { 
          mkfdv.try_emplace(k, vtemp);
          FSEM_DIM=vtemp.size();
        }
        else if (c == 'A') {
          mkadv.try_emplace(k, vtemp);
          FANT_DIM=vtemp.size();
        }
      }
      while ( is.peek()=='f' ) {
        unsigned int i;
        is >> "f " >> i >> " ";
        is >> mifek[i] >> "\n";
        mfeki[mifek[i]] = i;
      }

      fwpm = fwp;
      fwim = fwi;
      fwom = fwo;
      fwfm = fwf;
      fwsm = fws;

      fbpv = fbp;
      fbiv = fbi;
      fbov = fbo;
      fbfv = fbf;
      fbsv = fbs;

      //cerr << "FSEM: " << FSEM_SIZE << " FSYN: " << FSYN_SIZE << " FANT: " << FANT_SIZE << endl;
      cerr << "FSEM: " << FSEM_DIM << " FSYN: " << FSYN_DIM << " FANT: " << FANT_DIM << endl;
      //FFULL_WIDTH = 8 + 2*FSEM_SIZE + FSYN_SIZE + FANT_SIZE;

      // reshape weight matrices
      uint pre_attn_dim = 7 + FSEM_DIM + FSYN_DIM;
      uint attn_dim = fwp.size()/pre_attn_dim;
      // output of attn layer is concatenated with hvF (dim=FSYN_DIM), 
      // hvAnt (dim = FANT_DIM) and nullA (dim = 1)
      uint post_attn_dim = attn_dim + FSEM_DIM + FANT_DIM + 1;
      uint hidden_dim = fwf.size()/post_attn_dim;
      uint output_dim = fws.size()/hidden_dim;

      fwpm.reshape(attn_dim, pre_attn_dim);

      // fwim contains query, key, and value projection matrices,
      // each of dimension attn_dim x attn_dim
      fwim.reshape(3*attn_dim, attn_dim);
      //fwqm = fwim.rows(0, attn_dim);
      //fwkm = fwim.rows(attn_dim, 2*attn_dim);
      //fwvm = fwim.rows(2*attn_dim, 3*attn_dim);
      fwqm = fwim.rows(0, attn_dim-1);
      fwkm = fwim.rows(attn_dim, 2*attn_dim-1);
      fwvm = fwim.rows(2*attn_dim, 3*attn_dim-1);

      fwom.reshape(attn_dim, attn_dim);
      fwfm.reshape(hidden_dim, post_attn_dim);
      fwsm.reshape(output_dim, hidden_dim);

      cerr << "fwp size: " << fwp.size() << endl;
      cerr << "pre_attn_dim: " << pre_attn_dim << endl;
      cerr << "attn dim: " << attn_dim << endl;
      cerr << "fbiv size: " << fbiv.size() << endl;

      // fbiv contains biases vectors for query, key, and value
      //fbqv = fbiv(span(0, attn_dim));
      //fbkv = fbiv(span(attn_dim, 2*attn_dim));
      //fbvv = fbiv(span(2*attn_dim, 3*attn_dim));
      fbqv = fbiv(span(0, attn_dim-1));
      fbkv = fbiv(span(attn_dim, 2*attn_dim-1));
      fbvv = fbiv(span(2*attn_dim, 3*attn_dim-1));
    }

    const FEK& getFEK( unsigned int i ) const {
      auto it = mifek.find( i );
      assert( it != mifek.end() );
      if (it == mifek.end()) { 
        cerr << "ERROR: FEK not defined in fmodel: no value found for: " << i << endl; 
      }
      return it->second;
    }

    const vec getCatEmbed( CVar i, Delimited<char> c) const {
      vec zeroCatEmb = zeros(FSYN_DIM);
      assert (c == 'B');
      auto it = mcbv.find( i );
      return ( ( it != mcbv.end() ) ? it->second : zeroCatEmb );
    }

    const vec getKVecEmbed( HVec hv, Delimited<char> c ) const {
      vec KVecEmbed;// = arma::zeros(FSEM_SIZE);
      if (c == 'B') {
        KVecEmbed = arma::zeros(FSEM_DIM);
        for ( auto& kV : hv.at(0) ) {
          if ( kV == K::kTop) {
            KVecEmbed += arma::ones(FSEM_DIM);
            continue;
          }
          auto it = mkbdv.find( kV );
          if ( it == mkbdv.end() ) {
            continue;
          } else {
            KVecEmbed += it->second;
          }
        }
      }
      else if (c == 'F') {
        KVecEmbed = arma::zeros(FSEM_DIM);
        for ( auto& kV : hv.at(0) ) {
          if ( kV == K::kTop) {
            KVecEmbed += arma::ones(FSEM_DIM);
            continue;
          }
          auto it = mkfdv.find( kV );
          if ( it == mkfdv.end() ) {
            continue;
          } else {
            KVecEmbed += it->second;
          }
        }
      }
      else if (c == 'A') {
        KVecEmbed = arma::zeros(FANT_DIM);
        for ( auto& kV : hv.at(0) ) {
          if ( kV == K::kTop) {
            KVecEmbed += arma::ones(FANT_DIM);
            continue;
          }
          auto it = mkadv.find( kV );
          if ( it == mkadv.end() ) {
            continue;
          } else {
            KVecEmbed += it->second;
          }
        }
      }
      else cerr << "ERROR: F model KVec position misspecified." << endl;
      return KVecEmbed;
    }

    unsigned int getResponseIndex( F f, EVar e, K k ) {
      const auto& it = mfeki.find( FEK(f,e,k) );  if( it != mfeki.end() ) return( it->second );
      mfeki[ FEK(f,e,k) ] = iNextResponse;  mifek[ iNextResponse ] = FEK(f,e,k);  return( iNextResponse++ );
    }

    unsigned int getResponseIndex( F f, EVar e, K k ) const {                  // const version with closed predictor domain
      const auto& it = mfeki.find( FEK(f,e,k) );
      return ( ( it != mfeki.end() ) ? it->second : uint(-1) );
    }

    vec calcResponses( FPredictorVec& lfpredictors ) const;

    void testCalcResponses() const;
};

vec FModel::calcResponses( FPredictorVec& lfpredictors ) const {
// return distribution over FEK indices
  const HVec hvF = lfpredictors.getHvF();
  const HVec hvA = lfpredictors.getHvA();
  const bool nullA = lfpredictors.getNullA();

  const vec hvFEmb = getKVecEmbed(hvF, 'F');
  const vec hvAEmb = getKVecEmbed(hvA, 'A');

  vec postAttnInput = join_cols(join_cols(hvFEmb, hvAEmb), zeros(1));
  if (nullA) postAttnInput(FANT_DIM + FSEM_DIM) = 1;

  const StoreState ss = lfpredictors.getStoreState();

  vector<vec> attnInput;

  CVar catB;
  HVec hvB;
  uint depth;
  vec catBEmb;
  vec hvBEmb;
  
  // TODO do we need to add a fake depth 0 sign to match the python?
  for (uint i=0; i<ss.getDepth(); i++)
  {
    const DerivationFragment& currDf = ss.back(i);
    // TODO eventually we should also include apex and fillers here, not just base
    const Sign& dfBase = currDf.base().back();
    catB = dfBase.getCat();
    hvB = dfBase.getHVec();
    depth = ss.getDepth() - i;
    // TODO continue here
    catBEmb = getCatEmbed(catB, 'B');
    hvBEmb = getKVecEmbed(hvB, 'B');
    vec currAttnInput = join_cols(join_cols(catBEmb, hvBEmb), zeros(7)); 
    currAttnInput(FSEM_DIM + FSYN_DIM + depth) = 1;
    attnInput.emplace_back(currAttnInput);
  }

  // Add a "Top" fragment at depth 0. This ensures that there's something
  // to feed into the attention layer at the beginning of a sentence when
  // the stack is empty
  catB = cTop;
  hvB = hvTop;
  depth = 0;
  catBEmb = getCatEmbed(catB, 'B');
  hvBEmb = getKVecEmbed(hvB, 'B');
  vec currAttnInput = join_cols(join_cols(catBEmb, hvBEmb), zeros(7)); 
  currAttnInput(FSEM_DIM + FSYN_DIM + depth) = 1;
  attnInput.emplace_back(currAttnInput);

  
//  for (const BeamElement<HiddState>* curr = &be; ( (curr != &BeamElement<HiddState>::beStableDummy) && (wordOffset < MAX_WINDOW_SIZE) ); curr=&curr->getBack(), wordOffset++) {
//    CVar catB = getCatBase(*curr);
//    HVec hvB = getHvB(*curr);
//    HVec hvF = getHvF(*curr);
//    uint d = getDepth(*curr);
//    
//    vec catBEmb = getCatEmbed(catB, 'B');
//    vec hvBEmb = getKVecEmbed(hvB, 'B');
//    vec hvFEmb = getKVecEmbed(hvF, 'F');
//
//    vec currAttnInput = join_cols(join_cols(join_cols(catBEmb, hvBEmb), hvFEmb), zeros(7)); 
//    currAttnInput(2*FSEM_DIM + FSYN_DIM + d) = 1;
//    // vector<> doesn't have an emplace_front method
//    attnInput.emplace_back(currAttnInput);
//  }
//
//  // reverse attnInput so that the last item is the most recent word
//  reverse(attnInput.begin(), attnInput.end());
  
  return computeResult(attnInput, postAttnInput);
}


// returns distribution over FEK indices
vec FModel::computeResult( vector<vec> attnInput, vec postAttnInput, bool verbose ) const {
  // attnInput.front is the deepest derivation fragment
  vec first = attnInput.front();
  //vec last = attnInput.back();
  //
  if ( verbose ) {
    cerr << "first: " << first << endl;
    cerr << "fwpm size: " << fwpm.size() << endl;
    cerr << "first size: " << first.size() << endl;
    cerr << "fbpv size: " << fbpv.size() << endl;
  }

  vec proj = fwpm*first + fbpv;

  if ( verbose ) {
    cerr << "F proj" << proj << endl;
  }
  const vec query = fwqm*proj + fbqv;
  // used to scale the attention softmax (see section 3.2.1 of Attention
  // Is All You Need)
  const double scalingFactor = sqrt(fbqv.size());

  vector<vec> values;
  vector<double> scaledDotProds;
  //int currIndex = wordIndex - attnInput.size() + 1;
  for ( vec curr : attnInput ) { 
    proj = fwpm*curr + fbpv;
//    if ( usePositionalEncoding ) {
//      proj = proj + getPositionalEncoding(fbpv.size(), currIndex++);
//    }
    vec key = fwkm*proj + fbkv;
    vec value = fwvm*proj + fbvv;
    values.emplace_back(value);
    scaledDotProds.emplace_back(dot(query, key)/scalingFactor);
  }

  vec sdp = vec(scaledDotProds.size());
  for ( uint i=0; (i < scaledDotProds.size()); i++ ) {
    sdp(i) = scaledDotProds[i];
  }

  vec sdpExp = exp(sdp);
  double norm = accu(sdpExp);
  vec sdpSoftmax = sdpExp/norm;

  // calculate scaled_softmax(QK)*V
  vec attnResult = zeros<vec>(fbvv.size());

  for ( uint i=0; (i < values.size()); i++ ) {
    double weight = sdpSoftmax(i);
    vec val = values[i];
    attnResult += weight*val;
  }

  vec attnOutput = fwom*attnResult + fbov;

  if ( verbose ) {
    cerr << "F attnOutput" << attnOutput << endl;
  }
  // final bit is for nullA
  vec hiddenInput = join_cols(attnOutput, postAttnInput);
  vec logScores = fwsm * relu(fwfm*hiddenInput + fbfv) + fbsv;
  vec scores = exp(logScores);
  double outputNorm = accu(scores);
  vec result = scores/outputNorm;
  if ( verbose ) {
    cerr << "F result" << result << endl;
  }
  return result;
} 


// pass vectors of 1s as inputs as a sanity check (for comparison with
// Python training code)
void FModel::testCalcResponses() const {
  //vec corefVec = ones<vec>(FANT_DIM+1);
  vec postAttnInput;
  vec currAttnInput;
  vector<vec> attnInput;
  uint attnInputDim = FSEM_DIM + FSYN_DIM + 7;
  uint postAttnDim = FANT_DIM + FSEM_DIM + 1;
  uint stackSize = 5;
  postAttnInput = vec(postAttnDim);
  postAttnInput.fill(1);
  // the ith dummy input will be a vector of 0.1*(i+1)
  // [0.1 0.1 0.1 ...] [0.2 0.2 0.2 ...] ...
  for ( uint i=0; (i<stackSize); i++ ) {
    currAttnInput = vec(attnInputDim);
    currAttnInput.fill(0.1 * (i+1));
    attnInput.emplace_back(currAttnInput);
  }
  cerr << "F ==== output ====" << endl;
  computeResult(attnInput, postAttnInput, 1);
}


  //=============================================================================

//  CVar catB = getCatBase(be);
//  HVec hvB = getHvB(be);
//  HVec hvF = getHvF(be);
//  uint d = getDepth(be);
//  
//  vec catBEmb = getCatEmbed(catB, 'B');
//  vec hvBEmb = getKVecEmbed(hvB, 'B');
//  vec hvFEmb = getKVecEmbed(hvF, 'F');
//
//  // populate input vector to pre-attn feedforward
//  vec preAttnInputs = join_cols(join_cols(join_cols(catBEmb, hvBEmb), hvFEmb), zeros(7)); 
//  preAttnInputs(2*FSEM_DIM + FSYN_DIM + d) = 1;
//
//  vec attnInputs = fwpm*preAttnInputs + fbpv;
//  // we only char about the query for the latest word, hence the const
//  const vec query = fwqm*attnInputs + fbqv;
//  vec key = fwkm*attnInputs + fbkv;
//  vec value = fwvm*attnInputs + fbvv;
//
//  // TODO not sure if this is the right way to get the length of a vec
//  const double scalingFactor = sqrt(fbqv.size());
//
//  list<vec> values;
//  // Q*K for each f decision being attended to, scaled by sqrt(attn_dim)
//  list<double> scaledDotProds;
//  values.emplace_front(value);
//  scaledDotProds.emplace_front(dot(query, key)/scalingFactor);
//
//  //const BeamElement<HiddState>* pbeAnt = &beDummy;
//  //const BeamElement<HiddState>* curr = &be.getBack();
//
//  //while (&curr != &BeamElement<HiddState>::beStableDummy)
//  //for ( int tAnt = t; (&pbeAnt->getBack() != &BeamElement<HiddState>::beStableDummy) && (int(t-tAnt)<=COREF_WINDOW); tAnt--, pbeAnt = &pbeAnt->getBack()) { 
//  
//  int wordOffset = 0;
//  // TODO limit how far back you go?
//  for (const BeamElement<HiddState>* curr = &be.getBack(); (curr != &BeamElement<HiddState>::beStableDummy); curr = &curr->getBack(), wordOffset++) {
//    catB = getCatBase(*curr);
//    hvB = getHvB(*curr);
//    hvF = getHvF(*curr);
//    d = getDepth(*curr);
//    
//    catBEmb = getCatEmbed(catB, 'B');
//    hvBEmb = getKVecEmbed(hvB, 'B');
//    hvFEmb = getKVecEmbed(hvF, 'F');
//
//    preAttnInputs = join_cols(join_cols(join_cols(catBEmb, hvBEmb), hvFEmb), zeros(7)); 
//    preAttnInputs(2*FSEM_DIM + FSYN_DIM + d) = 1;
//
//    attnInputs = fwpm*preAttnInputs + fbpv;
//    // TODO add positional encoding here, using wordIndex - wordOffset
//    key = fwkm*attnInputs + fbkv;
//    value = fwvm*attnInputs + fbvv;
//    values.emplace_front(value);
//    scaledDotProds.emplace_front(dot(query, key)/scalingFactor);
//
//    //curr = &curr->getBack();
//  }
//
//  // take softmax of scaled dot products
//  vec sdp = vec(scaledDotProds.size());
//  for ( uint i=0; (i < scaledDotProds.size()); i++ ) {
//    sdp(i) = scaledDotProds.front();
//    scaledDotProds.pop_front();
//  }
//  vec sdpExp = exp(sdp);
//  double norm = accu(sdpExp);
//  vec sdpSoftmax = sdpExp/norm;
//
//  // calculate scaled_softmax(QK)*V
//  vec attnResult = zeros<vec>(fbvv.size());
//
//  for ( uint i=0; (i < values.size()); i++ ) {
//    double weight = sdpSoftmax(i);
//    vec val = values.front();
//    values.pop_front();
//    attnResult = attnResult + weight*val;
//  }
//
//  vec attnOutput = fwom*attnResult + fbov;
//  // final bit is for nullA
//  //vec hiddenInput = join_cols(join_cols(attnResult, hvAEmb), zeros(1));
//  vec hiddenInput = join_cols(join_cols(attnOutput, hvAEmb), zeros(1));
//  //if (nullA) hiddenInput(attnResult.size() + hvAEmb.size()) = 1;
//  if (nullA) hiddenInput(attnOutput.size() + hvAEmb.size()) = 1;
//  vec logScores = fwsm * relu(fwfm*hiddenInput + fbfv) + fbsv;
//  vec scores = exp(logScores);
//  double outputNorm = accu(scores);
//  return scores/outputNorm;
//} 
