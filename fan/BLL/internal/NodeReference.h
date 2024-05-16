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
    static void _P(snric)(_P(NodeReference_t) *);
  #endif
#endif

#pragma pack(push, 1)
BLL_StructBegin(_P(NodeReference_t))
  BLL_set_type_node NRI;

  #if BLL_set_Language == 1
    bool operator==(_P(NodeReference_t) nr) const{
      return NRI == nr.NRI;
    }
    bool operator!=(_P(NodeReference_t) nr) const{
      return NRI != nr.NRI;
    }

    /* set invalid constant */
    void sic(){
      _P(snric)(this);
    }

    #if BLL_set_CPP_nrsic == 1
      _P(NodeReference_t)(){
        sic();
      }
    #else
      _P(NodeReference_t)() = default;
      _P(NodeReference_t)(bool p){
        sic();
      }
    #endif
  
    /* is invalid constant */
    /* check _BLL_POFTWBIT(inric) at rest.h for more info */
    bool iic() const {
      return *this == _P(gnric)();
    }

    #if BLL_set_Link == 1
      _P(NodeReference_t) Next(_P(t) *list) { return _P(_NodeReference_Next)(this, list); }
      _P(NodeReference_t) Prev(_P(t) *list) { return _P(_NodeReference_Prev)(this, list); }
    #endif
  #endif

  #ifdef BLL_set_NodeReference_Overload_Declare
    BLL_set_NodeReference_Overload_Declare
  #endif
BLL_StructEnd(_P(NodeReference_t))
#pragma pack(pop)

__cta(sizeof(_P(NodeReference_t)) == sizeof(BLL_set_type_node));

/* set node reference invalid constant */
static
void
_P(snric)
(
  _P(NodeReference_t) *nr
){
  nr->NRI = (BLL_set_type_node)-1;
}
/* get node reference invalid constant */
static
_P(NodeReference_t)
_P(gnric)
(
){
  _P(NodeReference_t) nr;
  _P(snric)(&nr);
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
