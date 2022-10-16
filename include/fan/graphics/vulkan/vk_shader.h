#pragma once

namespace fan {
	namespace vulkan {

		struct shader_t {

      void open(fan::vulkan::context_t* context, VkDescriptorSetLayout descriptor_set_layout) {
        uniform_buffer.open(context, descriptor_set_layout);
      }

      void close(fan::vulkan::context_t* context, fan::vulkan::core::uniform_write_queue_t* write_queue) {
        vkDestroyShaderModule(context->device, shaderStages[0].module, nullptr);
        vkDestroyShaderModule(context->device, shaderStages[1].module, nullptr);
        uniform_buffer.close(context, write_queue);
      }

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

      VkShaderModule createShaderModule(fan::vulkan::context_t* context, const fan::string& code) {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(context->device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
          throw std::runtime_error("failed to create shader module!");
        }

        return shaderModule;
      }

      void set_matrices(fan::vulkan::context_t* context, fan::vulkan::matrices_t* matrices) {
        uniform_buffer.edit_instance(context, 0, &viewprojection_t::projection, matrices->m_projection);
        uniform_buffer.edit_instance(context, 0, &viewprojection_t::view, matrices->m_view);
      }

      struct viewprojection_t {
        fan::mat4 projection;
        fan::mat4 view;
      };

      VkPipelineShaderStageCreateInfo shaderStages[2];
      // projection, view
      fan::vulkan::core::uniform_block_t<1, viewprojection_t, 1> uniform_buffer;
		};
	}
}