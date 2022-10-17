#pragma once

namespace fan {
	namespace vulkan {

		struct shader_t {

      void open(fan::vulkan::context_t* context);

      void close(fan::vulkan::context_t* context, fan::vulkan::core::uniform_write_queue_t* write_queue);

      void set_vertex(fan::vulkan::context_t* context, const fan::string& path) {
        fan::string code;
        fan::io::file::read(path, &code);

        auto module_vertex = createShaderModule(context, code);

        VkPipelineShaderStageCreateInfo vert{};
        vert.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vert.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vert.module = module_vertex;
        vert.pName = "main";

        shaderStages[0] = vert;
      }
      void set_fragment(fan::vulkan::context_t* context, const fan::string& path) {
        fan::string code;
        fan::io::file::read(path, &code);

        auto module_fragment = createShaderModule(context, code);

        VkPipelineShaderStageCreateInfo frag{};
        frag.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        frag.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        frag.module = module_fragment;
        frag.pName = "main";

        shaderStages[1] = frag;
      }

      VkShaderModule createShaderModule(fan::vulkan::context_t* context, const fan::string& code);

      void set_matrices(fan::vulkan::context_t* context, fan::vulkan::matrices_t* matrices, core::uniform_write_queue_t* write_queue) {
        projection_view_block.edit_instance(context, 0, &viewprojection_t::projection, matrices->m_projection);
        projection_view_block.edit_instance(context, 0, &viewprojection_t::view, matrices->m_view);
        projection_view_block.common.edit(context, write_queue, 0, sizeof(viewprojection_t));
      }

      struct viewprojection_t {
        fan::mat4 projection;
        fan::mat4 view;
      };

      VkPipelineShaderStageCreateInfo shaderStages[2];
      fan::vulkan::core::uniform_block_t<viewprojection_t, 1> projection_view_block;
		};
	}
}