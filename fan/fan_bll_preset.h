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

#ifndef BLL_set_iterator_type
  #define BLL_set_iterator_type(container_type) fan::bll_iterator_t<container_type>
#endif

#ifndef BLL_set_iterator
  #define BLL_set_iterator(container_type, nr) BLL_set_iterator_type(container_type){ this, nr }
#endif

#ifndef bcontainer_set_alloc_open
	#define bcontainer_set_alloc_open(n) fan::memory_profile_malloc_cb(n)
#endif
#ifndef bcontainer_set_alloc_resize
	#define bcontainer_set_alloc_resize(ptr, n) fan::memory_profile_realloc_cb(ptr, n)
#endif
#ifndef bcontainer_set_alloc_close
	#define bcontainer_set_alloc_close(ptr) fan::memory_profile_free_cb(ptr)
#endif
#define _BLL_INCLUDE(_path) <FAN_INCLUDE_PATH/fan/_path>