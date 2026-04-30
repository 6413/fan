module;

#include <fan/utility.h>

#include <cstdint>
#include <cstddef>

export module fan.physics.common_context;

import std;

#if defined(FAN_PHYSICS_2D)

import fan.utility;
import fan.memory;
import fan.types.vector;

export namespace fan::physics {
  struct physics_update_data_t {
    std::uint64_t shape_id;
    fan::vec2 draw_offset = 0;
    bool sync_visual_angle = true;
    std::uint64_t body_id;
    void* cb;
  };
  using shape_physics_update_cb = void(*)(const physics_update_data_t& data);
}

namespace bll_builds {
  #define BLL_set_SafeNext 1
  #define BLL_set_prefix physics_update_cbs
  #include <fan/fan_bll_preset.h>
  #define BLL_set_Link 1
  #define BLL_set_type_node uint16_t
  #define BLL_set_NodeDataType fan::physics::physics_update_data_t
  #include <BLL/BLL.h>
}

export namespace fan::physics{

  using shape_physics_update_cb = void(*)(const physics_update_data_t& data);
  using physics_update_cbs_t = bll_builds::physics_update_cbs_t;
}


#endif