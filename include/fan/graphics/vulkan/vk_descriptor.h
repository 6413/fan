#pragma once

namespace fan {
	namespace vulkan {

		struct shader_t;

		struct descriptor_sets_t {

			#include "descriptor_bll_settings.h"
			protected:
				#include _FAN_PATH(BLL/BLL.h)
			public:

			#include "descriptor_layout_bll_settings.h"
			protected:
				#include _FAN_PATH(BLL/BLL.h)
			public:

			using layout_nr_t = descriptor_layout_list_NodeReference_t;
			using nr_t = descriptor_list_NodeReference_t;

      void open(fan::vulkan::context_t* context);
			void close(fan::vulkan::context_t* context);
			
			descriptor_layout_list_NodeData_t& get(layout_nr_t nr) {
				return descriptor_layout_list[nr];
			}

			template <uint16_t count>
			layout_nr_t push_layout(
				fan::vulkan::context_t* context,
				fan::vulkan::descriptor_set_layout_t<count> properties
			);

			template <uint16_t count>
			nr_t push(
				fan::vulkan::context_t* context,
				fan::vulkan::descriptor_sets_t::layout_nr_t layout_nr,
				fan::vulkan::descriptor_set_layout_t<count> properties
			);

			descriptor_layout_list_t descriptor_layout_list;
			descriptor_list_t descriptor_list;
      VkDescriptorPool descriptorPool;
		};
	}
}