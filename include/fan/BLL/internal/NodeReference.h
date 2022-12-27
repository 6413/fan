#ifdef BLL_set_namespace
  namespace BLL_set_namespace {
#endif

#if BLL_set_Language == 1
  struct _P(t);
  struct _P(NodeReference_t);
  #if BLL_set_AreWeInsideStruct == 0
    static _P(NodeReference_t) _P(_NodeReference_Next)(_P(NodeReference_t) *, _P(t) *);
    static _P(NodeReference_t) _P(_NodeReference_Prev)(_P(NodeReference_t) *, _P(t) *);
    static _P(NodeReference_t) _P(gnric)();
  #endif
#endif


BLL_StructBegin(_P(NodeReference_t))
  BLL_set_type_node NRI;

  #if BLL_set_Language == 1
    bool operator==(_P(NodeReference_t) nr) {
      return NRI == nr.NRI;
    }
    bool operator!=(_P(NodeReference_t) nr) {
      return NRI != nr.NRI;
    }

    /* set invalid constant */
    void sic(){
      *this = _P(gnric)();
    }
    /* is invalid constant */
    /* check _BLL_POFTWBIT(inric) at rest.h for more info */
    bool iic(){
      return *this == _P(gnric)();
    }

    #if BLL_set_Link == 1
      _P(NodeReference_t) Next(_P(t) *list) { return _P(_NodeReference_Next)(this, list); }
      _P(NodeReference_t) Prev(_P(t) *list) { return _P(_NodeReference_Prev)(this, list); }
    #endif
  #endif

  #ifdef BLL_set_NodeReference_Overload_Declare
    _P(NodeReference_t)() = default;
    BLL_set_NodeReference_Overload_Declare
  #endif
BLL_StructEnd(_P(NodeReference_t))

/* get node reference invalid constant */
static
_P(NodeReference_t)
_P(gnric)
(
){
  _P(NodeReference_t) nr;
  nr.NRI = (BLL_set_type_node)-1;
  return nr;
}
/* is node reference invalid constant */
static
bool
_P(inric)
(
  _P(NodeReference_t) NodeReference
){
  _P(NodeReference_t) nric = _P(gnric)();
  return nric.NRI == NodeReference.NRI;
}

#ifdef BLL_set_namespace
  }
#endif
