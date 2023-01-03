#ifndef stage_loader_path
#define stage_loader_path
#endif

#include "common.h"

struct stage_loader_t {
	loco_t* get_loco() {
    return OFFSETLESS(this, loco_t, stage_loader);
	}

	#include _FAN_PATH(CONCAT2(stage_loader_path, stages/stage.h))

	void open(const char* compiled_tp_path) {
    texturepack.open_compiled(get_loco(), compiled_tp_path);
	}
	void close() {
    texturepack.close();
	}

	template <typename stage_t>
	void push_and_open_stage(const stage_common_t::open_properties_t& op) {
		stage_t stage;
		stage.lib_open(get_loco(), &stage.stage_common, op);
		stage.stage_common.open();
		auto& stages = loco_t::stage_loader_t::stage::stages;
		stages.push_back(new stage_common_t(stage.stage_common));
	}
	void erase_stage(uint32_t id) {
		auto& stages = loco_t::stage_loader_t::stage::stages;
		if (id >= stages.size()) {
			return;
		}

		auto loco = get_loco();
    fan::throw_error("todo");
		//auto it = stages[id]->instances.GetNodeFirst();
		//while (it != stages[id]->instances.dst) {
		//	auto node = stages[id]->instances[it];
		//	loco->button.erase(&node.cid);
		//	it = it.Next(&stages[id]->instances);
		//}
		delete stages[id];
		stages.erase(stages.begin() + id);
	}

  loco_t::texturepack texturepack;
};

#undef stage_loader_path