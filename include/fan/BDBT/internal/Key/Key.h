#ifndef BDBT_set_KeySize
  #error KeySize needs to be defined
#endif

#ifdef BDBT_set_namespace
  namespace BDBT_set_namespace {
#endif

#if BDBT_set_KeySize != 0
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
#else
  #include "cpp.h"
#endif

#ifdef BDBT_set_namespace
  }
#endif

#undef BDBT_set_KeySize
