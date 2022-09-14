#ifndef stage_loader_path
#define stage_loader_path
#endif

struct stage_loader_t {

	#include _FAN_PATH(CONCAT2(stage_loader_path, stages/stage.h))

	void open() {
		
	}
	void close() {

	}

	template <typename T>
	auto* get_stage(uint32_t i) {
		return (T*)stage::stages[i];
	}

};

#undef stage_loader_path