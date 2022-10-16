#pragma once

namespace fan {
	namespace vulkan {

		struct descriptor_sets_t {

			#include "descriptor_bll_settings.h"
			protected:
				#include _FAN_PATH(BLL/BLL.h)
			public:

			using nr_t = descriptor_list_NodeReference_t;

      void open(fan::vulkan::context_t* context);
			void close(fan::vulkan::context_t* context);

			nr_t push(
				fan::vulkan::context_t* context,
				fan::vulkan::core::memory_t* memory,
				VkDescriptorSetLayout descriptor_set_layout,
				uint64_t buffer_size,
				uint32_t binding
			);

			descriptor_list_t descriptor_list;
      VkDescriptorPool descriptorPool;
		};

	}
}