struct shader_t {

  shader_list_NodeData_t& get_shader() const {
    return gloco->shader_list[shader_reference];
  }
  shader_list_NodeData_t& get_shader() {
    return gloco->shader_list[shader_reference];
  }

  shader_list_NodeReference_t shader_reference;

  static std::string preprocess_shader(const std::string& source_name,
    shaderc_shader_kind kind,
    const fan::string& source) {
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;

    // Like -DMY_DEFINE=1
    //options.AddMacroDefinition("MY_DEFINE", "1");

    shaderc::PreprocessedSourceCompilationResult result =
      compiler.PreprocessGlsl(source.c_str(), kind, source_name.c_str(), options);

    if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
      fan::throw_error(result.GetErrorMessage().c_str());
    }

    return { result.cbegin(), result.cend() };
  }

  static std::vector<uint32_t> compile_file(const fan::string& source_name,
    shaderc_shader_kind kind,
    const fan::string& source) {
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;
        
    // Like -DMY_DEFINE=1
    //options.AddMacroDefinition("MY_DEFINE", "1");
    #if fan_debug > 1
      options.SetOptimizationLevel(shaderc_optimization_level_zero);
    #else
      options.SetOptimizationLevel(shaderc_optimization_level_performance);
    #endif

    shaderc::SpvCompilationResult module =
      compiler.CompileGlslToSpv(source.c_str(), kind, source_name.c_str(), options);

    if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
      fan::throw_error(module.GetErrorMessage().c_str());
    }
        
    return { module.cbegin(), module.cend() };
  }

  void open(fan::vulkan::context_t& context, fan::vulkan::core::memory_write_queue_t* wq) {
    
    shader_reference = gloco->shader_list.NewNode();
    auto& shader = get_shader();
    shader.projection_view_block.open(context);
    for (uint32_t i = 0; i < fan::vulkan::max_camera; ++i) {
      shader.projection_view_block.push_ram_instance(context, wq, {});
    }
  }

  void close(fan::vulkan::context_t& context, fan::vulkan::core::memory_write_queue_t* write_queue) {
    auto& shader = get_shader();
    vkDestroyShaderModule(context.device, shader.shaderStages[0].module, nullptr);
    vkDestroyShaderModule(context.device, shader.shaderStages[1].module, nullptr);
    shader.projection_view_block.close(context, write_queue);
    gloco->shader_list.Recycle(shader_reference);
  }

  void set_vertex(fan::vulkan::context_t& context, const fan::string& shader_name, const fan::string& shader_code) {
    auto& shader = get_shader();
    // fan::print(
    //   "processed vertex shader:", path, "resulted in:",
    // preprocess_shader(shader_name.c_str(), shaderc_glsl_vertex_shader, shader_code);
    // );

    auto spirv =
      compile_file(shader_name.c_str(), shaderc_glsl_vertex_shader, shader_code);

    auto module_vertex = createShaderModule(context, spirv);

    VkPipelineShaderStageCreateInfo vert{};
    vert.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vert.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vert.module = module_vertex;
    vert.pName = "main";

    shader.shaderStages[0] = vert;
  }
  void set_fragment(fan::vulkan::context_t& context, const fan::string& shader_name, const fan::string& shader_code) {

    auto& shader = get_shader();
    //fan::print(
      // "processed vertex shader:", path, "resulted in:",
    //preprocess_shader(shader_name.c_str(), shaderc_glsl_fragment_shader, shader_code);
    //);

    auto spirv =
      compile_file(shader_name.c_str(), shaderc_glsl_fragment_shader, shader_code);

    auto module_fragment = createShaderModule(context, spirv);
        
    VkPipelineShaderStageCreateInfo frag{};
    frag.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    frag.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    frag.module = module_fragment;
    frag.pName = "main";

    shader.shaderStages[1] = frag;
  }

  static VkShaderModule createShaderModule(fan::vulkan::context_t& context, const std::vector<uint32_t>& code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size() * sizeof(typename std::remove_reference_t<decltype(code)>::value_type);
    createInfo.pCode = code.data();

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(context.device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
      fan::throw_error("failed to create shader module!");
    }

    return shaderModule;
  }

  void set_camera(auto* camera, uint32_t flags) {
    auto& shader = get_shader();
    auto& m = gloco->camera_list[camera->camera_reference];
    shader.projection_view_block.edit_instance(gloco->get_context(), &gloco->m_write_queue, flags, &viewprojection_t::projection, camera->m_projection);
    shader.projection_view_block.edit_instance(gloco->get_context(), &gloco->m_write_queue, flags, &viewprojection_t::view, camera->m_view);
  }
};