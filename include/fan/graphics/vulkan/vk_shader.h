#pragma once

namespace fan {
	namespace vulkan {

		struct shader_t {

      void open(fan::vulkan::context_t* context);

      void close(fan::vulkan::context_t* context, fan::vulkan::core::memory_write_queue_t* write_queue);

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

      void set_matrices(auto* loco, auto* matrices, core::memory_write_queue_t* write_queue, uint32_t flags) {
        auto& m = loco->matrices_list[matrices->matrices_reference];
        projection_view_block.edit_instance(loco->get_context(), flags, &viewprojection_t::projection, matrices->m_projection);
        projection_view_block.edit_instance(loco->get_context(), flags, &viewprojection_t::view, matrices->m_view);
        projection_view_block.common.edit(loco->get_context(), write_queue, 0, sizeof(viewprojection_t) * fan::vulkan::max_matrices);
      }

      struct viewprojection_t {
        fan::mat4 projection;
        fan::mat4 view;
      };

      VkPipelineShaderStageCreateInfo shaderStages[2];
      fan::vulkan::core::uniform_block_t<viewprojection_t, fan::vulkan::max_matrices> projection_view_block;
		};
	}
}