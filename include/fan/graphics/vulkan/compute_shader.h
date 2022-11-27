struct compute_shader_t {

  loco_t* get_loco() {
    return OFFSETLESS(this, loco_t, sb_shape_var_name);
  }

	compute_shader_t() 
  {
    fan::string str;
    fan::io::file::read("compute_shader.spv", &str);
    m_shader_module = fan::vulkan::shader_t::createShaderModule(get_loco()->get_context(), 
      str
    );

    auto loco = get_loco();
    VkComputePipelineCreateInfo info = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = m_shader_module;
    info.stage.pName = "main";

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;


    std::array<fan::vulkan::write_descriptor_set_t, 1> ds_properties{0};

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

		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &m_descriptor.m_layout;


    if (vkCreatePipelineLayout(loco->get_context()->device, &pipelineLayoutInfo, nullptr, &m_pipeline_layout) != VK_SUCCESS) {
			fan::throw_error("failed to create pipeline layout!");
		}

    info.layout = m_pipeline_layout;
    //                                                                          ?
    fan::vulkan::validate(vkCreateComputePipelines(loco->get_context()->device, 0, 1, &info, nullptr, &m_pipeline));
  }

  VkShaderModule m_shader_module;
  VkPipelineLayout m_pipeline_layout;

  VkPipeline m_pipeline;
  fan::vulkan::descriptor_t<1> m_descriptor;
  VkBuffer buffer = nullptr;
	VkDeviceMemory device_memory = nullptr;
};