#pragma once

struct compute_shader_t {

	struct properties_t {
		struct {
			fan::string path;
		}shader;
	};

	compute_shader_t(loco_t* loco, const properties_t& p)
	{
		fan::string str;
		if (fan::io::file::read(p.shader.path, &str)) {
			fan::throw_error("file doesnt exist:" + p.shader.path);
		}
		m_shader_module = fan::vulkan::shader_t::createShaderModule(loco->get_context(),
			str
		);

		std::array<fan::vulkan::write_descriptor_set_t, 1> ds_properties{ 0 };

		uint32_t buffer_size = 10000;

		fan::vulkan::core::createBuffer(
			loco->get_context(),
			buffer_size,
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			//VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, // ?
			// faster ^? depends about buffer size maxMemoryAllocationCount maybe
			buffer,
			device_memory
		);

		ds_properties[0].binding = 0;
		ds_properties[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		ds_properties[0].flags = VK_SHADER_STAGE_COMPUTE_BIT;
		ds_properties[0].range = VK_WHOLE_SIZE;
		ds_properties[0].buffer = buffer;
		ds_properties[0].dst_binding = 0;

		m_descriptor.open(loco->get_context(), loco->descriptor_pool.m_descriptor_pool, ds_properties);
		m_descriptor.update(loco->get_context());

		VkComputePipelineCreateInfo info = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
		info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		info.stage.module = m_shader_module;
		info.stage.pName = "main";

		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;


		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &m_descriptor.m_layout;


		if (vkCreatePipelineLayout(loco->get_context()->device, &pipelineLayoutInfo, nullptr, &m_pipeline_layout) != VK_SUCCESS) {
			fan::throw_error("failed to create pipeline layout!");
		}

		info.layout = m_pipeline_layout;
		//                                                                          ?
		fan::vulkan::validate(vkCreateComputePipelines(loco->get_context()->device, 0, 1, &info, nullptr, &m_pipeline));
	}

	void execute(loco_t* loco, const fan::vec3ui& group_count) {
		auto context = loco->get_context();
		auto cmd = context->commandBuffers[context->currentFrame];
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);

		vkCmdBindDescriptorSets(
			cmd,
			VK_PIPELINE_BIND_POINT_COMPUTE,
			m_pipeline_layout,
			0,
			1,
			m_descriptor.m_descriptor_set,
			0,
			nullptr
		);

		vkCmdDispatch(cmd, group_count.x, group_count.y, group_count.z);
	}

	void wait_finish(loco_t* loco) {
		vkDeviceWaitIdle(loco->get_context()->device);
	}

	VkShaderModule m_shader_module;
	VkPipelineLayout m_pipeline_layout;

	VkPipeline m_pipeline;
	fan::vulkan::descriptor_t<1> m_descriptor;
	VkBuffer buffer = nullptr;
	VkDeviceMemory device_memory = nullptr;
};