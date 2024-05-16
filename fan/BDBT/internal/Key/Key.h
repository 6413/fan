#if !defined(BDBT_set_KeySize) && !defined(BDBT_set_lcpp)
  #error KeySize needs to be defined
#endif

#if defined(BDBT_set_KeySize)
  #if defined(BDBT_set_lcpp) && BDBT_set_KeySize != 0
    #error to make cpp key compiletime, dont define KeySize. to make cpp key runtime, define KeySize as 0. error(BDBT_set_KeySize != 0)
  #endif
#endif

#ifdef BDBT_set_namespace
  namespace BDBT_set_namespace {
#endif

#if defined(BDBT_set_lc)
  #if BDBT_set_KeySize == 0
    #error KeySize 0 is not implemented for c yet.
  #endif

  #if BDBT_set_KeySize <= 0xff
    typedef uint8_t _BDBT_P(KeySize_t);
  #elif BDBT_set_KeySize <= 0xffff
    typedef uint16_t _BDBT_P(KeySize_t);
  #elif BDBT_set_KeySize <= 0xffffffff
    typedef uint32_t _BDBT_P(KeySize_t);
  #else
    #error no
  #endif

  #define _BDBT_Key_ParameterKeySize(p0, p1)
  #define _BDBT_Key_GetKeySize BDBT_set_KeySize
  #define _BDBT_Key_PassKeySize
  #define _BDBT_Key_PrepareBeforeLast
  #define _BDBT_set_KeySize_BeforeLast (BDBT_set_KeySize - 8)
  #define _BDBT_Key_GetBeforeLast _BDBT_set_KeySize_BeforeLast

  #include "rest.h"

  #undef _BDBT_Key_GetBeforeLast
  #undef _BDBT_set_KeySize_BeforeLast
  #undef _BDBT_Key_PrepareBeforeLast
  #undef _BDBT_Key_PassKeySize
  #undef _BDBT_Key_GetKeySize
  #undef _BDBT_Key_ParameterKeySize
#elif defined(BDBT_set_lcpp)
  #include "cpp.h"
#endif

#ifdef BDBT_set_namespace
  }
#endif

#undef BDBT_set_KeySize
