#pragma once

namespace fan {
	namespace vulkan {

		struct shader_t;

		struct descriptor_sets_t {

			#include "descriptor_bll_settings.h"
			protected:
				#include _FAN_PATH(BLL/BLL.h)
			public:

			using nr_t = descriptor_list_NodeReference_t;

      void open(fan::vulkan::context_t* context);
			void close(fan::vulkan::context_t* context);
			
			descriptor_list_NodeData_t& get(nr_t nr) {
				return descriptor_list[nr];
			}

			template <uint16_t count>
			nr_t push(
				fan::vulkan::context_t* context,
				fan::vulkan::descriptor_set_layout_t<count> layout
			);

			descriptor_list_t descriptor_list;
      VkDescriptorPool descriptorPool;
		};
	}
}