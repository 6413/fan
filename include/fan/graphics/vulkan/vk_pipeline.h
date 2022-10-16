#pragma once

namespace fan {
	namespace vulkan {

    struct context_t;
    struct shader_t;

		struct pipelines_t {

			void open() {
			}
      void close(fan::vulkan::context_t* context);

      #include "pipeline_bll_settings.h"
    protected:
      #include _FAN_PATH(BLL/BLL.h)
    public:

      using nr_t = pipeline_list_NodeReference_t;

      struct properties_t {
        fan::vulkan::shader_t* shader;
      };
      nr_t push(fan::vulkan::context_t* context, const properties_t& p);

			pipeline_list_t pipeline_list;
		};
	}
}