#pragma once

#ifndef ETC_WED_set_BaseLibrary
  #define ETC_WED_set_BaseLibrary 0
#endif

#ifndef ETC_WED_set_Prefix
  #error set Prefix
#endif

#ifndef ETC_WED_set_debug_InvalidCharacterAccess
  #define ETC_WED_set_debug_InvalidCharacterAccess 0
#endif
#ifndef ETC_WED_set_debug_InvalidLineAccess
  #define ETC_WED_set_debug_InvalidLineAccess 0
#endif
#ifndef ETC_WED_set_debug_InvalidCursorAccess
  #define ETC_WED_set_debug_InvalidCursorAccess 0
#endif

#ifndef ETC_WED_set_Abort
  #define ETC_WED_set_Abort() assert(0)
#endif

#ifndef ETC_WED_set_DataType
  #define ETC_WED_set_DataType wchar_t
#endif

#if ETC_WED_set_BaseLibrary == 0
  #define _ETC_WED_INCLUDE _WITCH_PATH
#elif ETC_WED_set_BaseLibrary == 1
  #define _ETC_WED_INCLUDE _FAN_PATH
#endif

#define _ETC_WED_P(p) CONCAT3(ETC_WED_set_Prefix,_,p)

#include "internal/Base.h"

#undef _ETC_WED_P

#undef _ETC_WED_INCLUDE

#undef ETC_WED_set_DataType

#undef ETC_WED_set_Abort

#undef ETC_WED_set_debug_InvalidCursorAccess
#undef ETC_WED_set_debug_InvalidLineAccess
#undef ETC_WED_set_debug_InvalidCharacterAccess

#undef ETC_WED_set_Prefix

#undef ETC_WED_set_BaseLibrary
