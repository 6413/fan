#include _FAN_PATH(types/memory.h)

#ifndef BLL_set_prefix
	#error ifndef BLL_set_prefix
#endif
#ifndef BLL_set_declare_basic_types
	#define BLL_set_declare_basic_types 1
#endif
#ifndef BLL_set_declare_rest
	#define BLL_set_declare_rest 1
#endif
#ifndef BLL_set_type_node
	#define BLL_set_type_node uint32_t
#endif
#ifndef BLL_set_node_data
	#if BLL_set_declare_rest == 1
		#error ifndef BLL_set_node_data
	#endif
#endif
/* if you access next more than prev it can make performance difference */
#ifndef BLL_set_PreferNextFirst
	#define BLL_set_PreferNextFirst 1
#endif
#ifndef BLL_set_PadNode
	#define BLL_set_PadNode 0
#endif
#ifndef BLL_set_debug_InvalidAction
	#define BLL_set_debug_InvalidAction 0
#endif
#ifndef BLL_set_IsNodeUnlinked
	#if BLL_set_debug_InvalidAction == 1
		#define BLL_set_IsNodeUnlinked 1
	#else
		#define BLL_set_IsNodeUnlinked 0
	#endif
#endif

#if BLL_set_debug_InvalidAction == 1
	#if BLL_set_IsNodeUnlinked == 0
		#error BLL_set_IsNodeUnlinked cant be 0 when BLL_set_debug_InvalidAction is 1
	#endif
	#ifndef BLL_set_debug_InvalidAction_srcAccess
		#define BLL_set_debug_InvalidAction_srcAccess 1
	#endif
	#ifndef BLL_set_debug_InvalidAction_dstAccess
		#define BLL_set_debug_InvalidAction_dstAccess 1
	#endif
#else
	#ifndef BLL_set_debug_InvalidAction_srcAccess
		#define BLL_set_debug_InvalidAction_srcAccess 0
	#endif
	#ifndef BLL_set_debug_InvalidAction_dstAccess
		#define BLL_set_debug_InvalidAction_dstAccess 0
	#endif
#endif

#if BLL_set_declare_basic_types == 1
	#include _FAN_PATH(BLL/internal/basic_types.h)
#endif
#if BLL_set_declare_rest == 1
	#include _FAN_PATH(BLL/internal/rest.h)
#endif

#undef BLL_set_prefix
#undef BLL_set_declare_basic_types
#undef BLL_set_declare_rest
#undef BLL_set_type_node
#undef BLL_set_node_data
#undef BLL_set_PreferNextFirst
#undef BLL_set_PadNode
#undef BLL_set_debug_InvalidAction
#undef BLL_set_debug_InvalidAction_srcAccess
#undef BLL_set_debug_InvalidAction_dstAccess
#undef BLL_set_IsNodeUnlinked
