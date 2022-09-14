#ifndef stage_loader_path
#define stage_loader_path
#endif

struct stage_loader_t {

	#include _FAN_PATH(CONCAT2(stage_loader_path, stage/stage.h))

	void open() {
		
	}
	void close() {

	}

};

#undef stage_loader_path