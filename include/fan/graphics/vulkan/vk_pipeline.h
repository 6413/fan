#pragma once

#include <fan/graphics/renderer.h>

#if fan_renderer == fan_renderer_vulkan

#include <fan/types/vector.h>
#include <fan/graphics/vulkan/vk_shader.h>

#include <fan/graphics/shared_core.h>

namespace fan {
	
	namespace vk {

		namespace opengl {

			struct pipeline {

				VkDevice* device = nullptr;

				VkRenderPass* render_pass = nullptr;

				VkDescriptorSetLayout* descriptor_set_layout = nullptr;
				VkPipelineLayout pipeline_layout = nullptr;

				struct pipeline_t {

					fan::vk::shader shader;
					VkPipeline pipeline = nullptr;

				};

				std::vector<pipeline_t> pipeline_info;

				struct pipelines_info_t {
					VkVertexInputBindingDescription binding_description;
					std::vector<VkVertexInputAttributeDescription> attribute_description;
					std::string vertex;
					std::string fragment;
				};

				std::vector<pipelines_info_t> old_data;

				struct flags_t {

					fan_2d::opengl::face_e face = fan_2d::opengl::face_e::back;
					fan_2d::opengl::fill_mode_e fill_mode = fan_2d::opengl::fill_mode_e::fill;

					fan_2d::opengl::shape topology;
					VkSampleCountFlagBits* msaa_samples = 0;
				};

				flags_t flags;

				pipeline(
					VkDevice* device,
					VkRenderPass* render_press,
					VkDescriptorSetLayout* descriptor_set_layout,
					flags_t flags_
				) : 
					device(device),
					render_pass(render_press),
					descriptor_set_layout(descriptor_set_layout),
					flags(flags_)
				{ }

				void push_back(const VkVertexInputBindingDescription& binding_description, const std::vector<VkVertexInputAttributeDescription>& attribute_description, const fan::vec2& window_size, const std::string& vertex, const std::string& fragment, VkExtent2D extent) {

					pipeline_info.emplace_back(pipeline_t{
						fan::vk::shader(device, vertex, fragment),
						nullptr
					});

					old_data.emplace_back(pipelines_info_t{ binding_description, attribute_description, vertex, fragment });
					recreate_pipeline(this->pipeline_info.size() - 1, window_size, extent);
				}

				~pipeline() {

					this->erase_pipes();
					this->erase_pipeline_layout();

				}

				void recreate_pipeline(uint32_t i, const fan::vec2& window_size, VkExtent2D extent) {

					this->erase_pipeline_layout();
					this->erase_pipes(i);

					pipeline_info[i].shader = fan::vk::shader(device, old_data[i].vertex, old_data[i].fragment);

					pipeline_t* instance = &pipeline_info[i];

					VkPipelineShaderStageCreateInfo vertex_shader_stage_info{};
					vertex_shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
					vertex_shader_stage_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
					vertex_shader_stage_info.module = instance->shader.get_vertex_module();
					vertex_shader_stage_info.pName = "main";

					VkPipelineShaderStageCreateInfo fragment_shader_stage_info{};
					fragment_shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
					fragment_shader_stage_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
					fragment_shader_stage_info.module = instance->shader.get_fragment_module();
					fragment_shader_stage_info.pName = "main";

					VkPipelineShaderStageCreateInfo shader_stages[] = { vertex_shader_stage_info, fragment_shader_stage_info };

					VkPipelineVertexInputStateCreateInfo vertex_input_info{};
					vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

					vertex_input_info.vertexBindingDescriptionCount = 1;
					vertex_input_info.vertexAttributeDescriptionCount = old_data[i].attribute_description.size();
					vertex_input_info.pVertexBindingDescriptions = &old_data[i].binding_description;
					vertex_input_info.pVertexAttributeDescriptions = old_data[i].attribute_description.data();

					VkPipelineInputAssemblyStateCreateInfo input_assembly{};
					input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;

					switch(flags.topology) {

						case fan_2d::opengl::shape::line: {
							input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
							break;
						}
						case fan_2d::opengl::shape::line_strip: {
							input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
							break;
						}
						case fan_2d::opengl::shape::triangle: {
							input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
							break;
						}
						case fan_2d::opengl::shape::triangle_strip: {
							input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
							break;
						}
						case fan_2d::opengl::shape::triangle_fan: {
							input_assembly.primitiveRestartEnable = true;
							input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
							break;
						}
						default: {
							input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
							fan::print("fan warning - unset input assembly topology in graphics pipeline");
							break;
						}

					}

					VkViewport viewport{};
					viewport.x = 0.0f;
					viewport.y = 0.0f;
					viewport.width = window_size.x;
					viewport.height = window_size.y;
					viewport.minDepth = 0.0f;
					viewport.maxDepth = 1.0f;
					
					VkRect2D scissor{};
					scissor.offset = { 0, 0 };
					scissor.extent = extent;

					VkPipelineViewportStateCreateInfo viewportState{};
					viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
					viewportState.viewportCount = 1;
					viewportState.pViewports = &viewport;
					viewportState.scissorCount = 1;
					viewportState.pScissors = &scissor;

					VkPipelineRasterizationStateCreateInfo rasterizer{};
					rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
					rasterizer.depthClampEnable = VK_FALSE;
					rasterizer.rasterizerDiscardEnable = VK_FALSE;
					rasterizer.polygonMode = (VkPolygonMode)flags.fill_mode;
					rasterizer.lineWidth = 1.0f;
					rasterizer.cullMode = (VkCullModeFlags) flags.face;
					rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
					rasterizer.depthBiasEnable = VK_FALSE;

					VkPipelineMultisampleStateCreateInfo multisampling{};
					multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
					multisampling.sampleShadingEnable = VK_FALSE;
					multisampling.rasterizationSamples = *flags.msaa_samples;

					VkPipelineDepthStencilStateCreateInfo depthStencil{};
					depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
					depthStencil.depthTestEnable = VK_TRUE;
					depthStencil.depthWriteEnable = VK_TRUE;
					depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
					depthStencil.depthBoundsTestEnable = VK_FALSE;
					depthStencil.stencilTestEnable = VK_FALSE;

					VkPipelineColorBlendAttachmentState color_blend_attachment{};
					color_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
					color_blend_attachment.blendEnable = VK_TRUE;
					color_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
					color_blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
					color_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
					color_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
					color_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
					color_blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;

					VkPipelineColorBlendStateCreateInfo color_blending{};
					color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
					color_blending.logicOpEnable = VK_FALSE;
					color_blending.logicOp = VK_LOGIC_OP_COPY;
					color_blending.attachmentCount = 1;
					color_blending.pAttachments = &color_blend_attachment;
					color_blending.blendConstants[0] = 0.0f;
					color_blending.blendConstants[1] = 0.0f;
					color_blending.blendConstants[2] = 0.0f;
					color_blending.blendConstants[3] = 0.0f;


					VkPipelineLayoutCreateInfo pipeline_layout_info{};
					pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
					pipeline_layout_info.setLayoutCount = 1;
					pipeline_layout_info.pSetLayouts = descriptor_set_layout;

					if (vkCreatePipelineLayout(*device, &pipeline_layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
						throw std::runtime_error("failed to create pipeline layout!");
					}

					VkGraphicsPipelineCreateInfo pipeline_create_info{};
					pipeline_create_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
					pipeline_create_info.stageCount = 2;
					pipeline_create_info.pStages = shader_stages;
					pipeline_create_info.pVertexInputState = &vertex_input_info;
					pipeline_create_info.pInputAssemblyState = &input_assembly;
					pipeline_create_info.pViewportState = &viewportState;
					pipeline_create_info.pRasterizationState = &rasterizer;
					pipeline_create_info.pMultisampleState = &multisampling;
					pipeline_create_info.pDepthStencilState = &depthStencil;
					pipeline_create_info.pColorBlendState = &color_blending;
					pipeline_create_info.layout = pipeline_layout;
					pipeline_create_info.renderPass = *render_pass;
					pipeline_create_info.subpass = 0;
					pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;

					if (vkCreateGraphicsPipelines(*device, VK_NULL_HANDLE, 1, &pipeline_create_info, nullptr, &instance->pipeline) != VK_SUCCESS) {
						throw std::runtime_error("failed to create graphics pipeline!");
					}

					instance->shader.erase_modules();
				}

				void erase_pipeline_layout() {

					if (pipeline_layout) {
						vkDestroyPipelineLayout(*device, pipeline_layout, nullptr);
						pipeline_layout = nullptr;
					}

				}

				void erase_pipes() {

					for (int i = 0; i < pipeline_info.size(); i++) {
						erase_pipes(i);
					}

				}

				void erase_pipes(uint32_t i) {

					if (pipeline_info[i].pipeline) {
						vkDestroyPipeline(*device, pipeline_info[i].pipeline, nullptr);
						pipeline_info[i].pipeline = nullptr;
					}

				}

			};

		}

	}

}

#endif