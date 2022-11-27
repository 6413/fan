#pragma once

struct compute_shader_t {

	struct properties_t {
		struct {
			fan::string path;
		}shader;
		struct descriptor_t{
			VkDescriptorSet* sets;
			uint32_t count = 1;
			VkDescriptorSetLayout* layouts;
			uint32_t layout_count = 1;
		}descriptor;
	};

	compute_shader_t(loco_t* loco, const properties_t& p) : 
		m_descriptor{p.descriptor.sets, p.descriptor.count, p.descriptor.layouts, p.descriptor.layout_count}
	{
		fan::string str;
		if (fan::io::file::read(p.shader.path, &str)) {
			fan::throw_error("file doesnt exist:" + p.shader.path);
		}
		m_shader_module = fan::vulkan::shader_t::createShaderModule(loco->get_context(),
			str
		);

		VkComputePipelineCreateInfo info = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
		info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		info.stage.module = m_shader_module;
		info.stage.pName = "main";

		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;


		pipelineLayoutInfo.setLayoutCount = p.descriptor.layout_count;
		pipelineLayoutInfo.pSetLayouts = p.descriptor.layouts;

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
			m_descriptor.count,
			m_descriptor.sets,
			0,
			nullptr
		);

		vkCmdDispatch(cmd, group_count.x, group_count.y, group_count.z);
	}

	void wait_finish(loco_t* loco) {
		auto context = loco->get_context();
		vkWaitForFences(context->device, 1, &context->inFlightFences[context->currentFrame], VK_TRUE, UINT64_MAX);
	}

	VkShaderModule m_shader_module;
	VkPipelineLayout m_pipeline_layout;

	VkPipeline m_pipeline;
	properties_t::descriptor_t m_descriptor;
};