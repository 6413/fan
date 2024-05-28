#if BLL_set_IntegerNR == 0
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
        explicit _P(NodeReference_t)(bool p){
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
#elif BLL_set_IntegerNR == 1
  typedef BLL_set_type_node _P(NodeReference_t);
#endif

/* get node reference integer */
static
BLL_set_type_node *
_P(gnrint)
(
  _P(NodeReference_t) *nr
){
  #if BLL_set_IntegerNR == 0
    return &nr->NRI;
  #elif BLL_set_IntegerNR == 1
    return nr;
  #endif
}

/* is node refence equal */
static
bool
_P(inre)
(
  _P(NodeReference_t) p0,
  _P(NodeReference_t) p1
){
  return *_P(gnrint)(&p0) == *_P(gnrint)(&p1);
}

/* set node reference invalid constant */
static
void
_P(snric)
(
  _P(NodeReference_t) *nr
){
  *_P(gnrint)(nr) = (BLL_set_type_node)-1;
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
  _P(NodeReference_t) nr
){
  return _P(inre)(nr, _P(gnric)());
}
