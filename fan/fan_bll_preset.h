#ifndef BLL_set_Language
  #define BLL_set_Language 1
#endif
#ifndef BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_ConstructDestruct 1
#endif
#ifndef BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct 1
#endif
#ifndef BLL_set_CPP_nrsic
  #define BLL_set_CPP_nrsic 1
#endif
#ifndef BLL_set_Clear
  #define BLL_set_Clear 1
#endif

#ifndef bcontainer_set_alloc_open
	#define bcontainer_set_alloc_open(n) std::malloc(n)
#endif
#ifndef bcontainer_set_alloc_resize
	#define bcontainer_set_alloc_resize(ptr, n) std::realloc(ptr, n)
#endif
#ifndef bcontainer_set_alloc_close
	#define bcontainer_set_alloc_close(ptr) std::free(ptr)
#endif

#define _BLL_INCLUDE(_path) <FAN_INCLUDE_PATH/fan/_path>