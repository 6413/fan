#pragma once

#include <shaderc/shaderc.hpp>

namespace fan {
	namespace vulkan {

		struct shader_t {

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

      void open(fan::vulkan::context_t* context, fan::vulkan::core::memory_write_queue_t* wq);

      void close(fan::vulkan::context_t* context, fan::vulkan::core::memory_write_queue_t* write_queue);

      void set_vertex(fan::vulkan::context_t* context, const fan::string& shader_name, const fan::string& shader_code) {
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

        shaderStages[0] = vert;
      }
      void set_fragment(fan::vulkan::context_t* context, const fan::string& shader_name, const fan::string& shader_code) {

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

        shaderStages[1] = frag;
      }

      static VkShaderModule createShaderModule(fan::vulkan::context_t* context, const std::vector<uint32_t>& code);

      void set_camera(auto* loco, auto* camera, uint32_t flags) {
        auto& m = loco->camera_list[camera->camera_reference];
        projection_view_block.edit_instance(loco->get_context(), &loco->m_write_queue, flags, &viewprojection_t::projection, camera->m_projection);
        projection_view_block.edit_instance(loco->get_context(), &loco->m_write_queue, flags, &viewprojection_t::view, camera->m_view);
      }

      struct viewprojection_t {
        fan::mat4 projection;
        fan::mat4 view;
      };

      VkPipelineShaderStageCreateInfo shaderStages[2];
      fan::vulkan::core::uniform_block_t<viewprojection_t, fan::vulkan::max_camera> projection_view_block;
		};
	}
}