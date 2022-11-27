#pragma once

#include _FAN_PATH(graphics/vulkan/vk_core.h)

namespace fan {
	namespace vulkan {


		template <uint32_t descriptor_count>
		struct buffer_t {
			struct properties_t {
				fan::graphics::context_t* context;
				VkDescriptorPool descriptor_pool;
				std::array<fan::vulkan::write_descriptor_set_t, descriptor_count> ds_properties;
			};

			buffer_t(const properties_t& p) {
				m_descriptor.open(p.context, p.descriptor_pool, p.ds_properties);
			}

			void create(fan::graphics::context_t* context) {
				//fan::vulkan::
			}

			fan::vulkan::descriptor_t<descriptor_count> m_descriptor;
		};
	}
}