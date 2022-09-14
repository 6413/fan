struct stage {
	inline static struct stage0_t {
    #include "stages/stage0.h"
  }stage0;
  inline static std::vector<void*> stages{
	   &stage0,
  };
};
