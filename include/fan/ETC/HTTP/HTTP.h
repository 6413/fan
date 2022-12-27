#ifndef ETC_HTTP_set_prefix
	#error ifndef ETC_HTTP_set_prefix
#endif
#ifndef ETC_HTTP_set_LineLengthLimit
	#define ETC_HTTP_set_LineLengthLimit 0x1000
#endif
#ifndef ETC_HTTP_set_PadStruct
	#define ETC_HTTP_set_PadStruct 1
#endif
#ifndef ETC_HTTP_set_NotFriendlyData
	#define ETC_HTTP_set_NotFriendlyData 0
#endif

#include _WITCH_PATH(ETC/HTTP/internal/rest.h)

#undef ETC_HTTP_set_prefix
#undef ETC_HTTP_set_LineLimit
#undef ETC_HTTP_set_PadStruct
#undef ETC_HTTP_set_NotFriendlyData
