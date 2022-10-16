#pragma once

namespace fan {
	namespace vulkan {
		struct shader_t {

      void open(fan::vulkan::context_t* context) {

      }

      void close(fan::vulkan::context_t* context) {
        vkDestroyShaderModule(context->device, shaderStages[0].module, nullptr);
        vkDestroyShaderModule(context->device, shaderStages[1].module, nullptr);
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

      VkPipelineShaderStageCreateInfo shaderStages[2];
		};
	}
}