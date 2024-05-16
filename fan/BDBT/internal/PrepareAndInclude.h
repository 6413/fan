#if BDBT_set_StructFormat == 0
  #define BDBT_StructBegin(n) typedef struct{
  #define BDBT_StructEnd(n) }n;
#elif BDBT_set_StructFormat == 1
  #define BDBT_StructBegin(n) struct n{
  #define BDBT_StructEnd(n) };
#endif

#if BDBT_set_declare_rest == 1
  #if defined(BDBT_set_lc)
    #define _BDBT_this This
    #define _BDBT_fdec(rtype, name, ...) static rtype name(_P(t) *_BDBT_this, ##__VA_ARGS__)
    #define _BDBT_fcall(name, ...) name(_BDBT_this, ##__VA_ARGS__)
  #elif defined(BDBT_set_lcpp)
    #define _BDBT_this this
    #define _BDBT_fdec(rtype, name, ...) rtype name(__VA_ARGS__)
    #define _BDBT_fcall(name, ...) name(__VA_ARGS__)
  #else
    #error ?
  #endif

  #include "rest.h"

  #undef _BDBT_this
  #undef _BDBT_fdec
  #undef _BDBT_fcall
#endif
#if BDBT_set_declare_Key == 1
  #include "Key/Key.h"
#endif

#undef BDBT_StructBegin
#undef BDBT_StructEnd
