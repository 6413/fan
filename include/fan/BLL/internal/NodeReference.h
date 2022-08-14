#ifdef BLL_set_namespace
  namespace BLL_set_namespace {
#endif

BLL_StructBegin(_P(NodeReference_t))
  BLL_set_type_node NRI;

  /* for some relief */
  #if BLL_set_Language == 1
    bool operator==(_P(NodeReference_t) nr) {
      return NRI == nr.NRI;
    }
    bool operator!=(_P(NodeReference_t) nr) {
      return NRI != nr.NRI;
    }
  #endif

  #ifdef BLL_set_NodeReference_Overload_Declare
    BLL_set_NodeReference_Overload_Declare
  #endif
BLL_StructEnd(_P(NodeReference_t))

#ifdef BLL_set_namespace
  }
#endif
