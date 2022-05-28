#pragma once

#include _FAN_PATH(graphics/renderer.h)

#if fan_renderer == fan_renderer_vulkan

#include _FAN_PATH(io/file.h)

#ifdef fan_platform_windows
#define VK_USE_PLATFORM_WIN32_KHR
#elif defined(fan_platform_unix)
#define VK_USE_PLATFORM_XLIB_KHR
#endif

#include <vulkan/vulkan.h)

#include <string>

#include _FAN_PATH(time/time.h)

namespace fan {

	namespace vk {

		class shader {
		public:

			inline static bool recompile_shaders = false;

			shader(VkDevice* device, const std::string& vertex, const std::string& fragment) 
			: m_device(device), m_vertex_path(vertex), m_fragment_path(fragment) {

				auto shader_code = get_shader_code(vertex, fragment);

				m_vertex_code = shader_code.first;
				m_fragment_code = shader_code.second;

				m_vertex = create_shader(true);
				m_fragment = create_shader(false);

			}

			~shader() {
				erase_modules();
			}

			shader(const shader& s) {
				this->operator=(s);
			}
			shader(shader&& s) {
				this->operator=(std::move(s));
			}

			shader& operator=(const shader& s) {

				this->erase_modules();

				this->m_device = s.m_device;
				this->m_vertex_path = s.m_vertex_path;
				this->m_fragment_path = s.m_fragment_path;

				auto shader_code = get_shader_code(s.m_vertex_path, s.m_fragment_path);

				m_vertex_code = shader_code.first;
				m_fragment_code = shader_code.second;

				this->m_vertex = this->create_shader(true);
				this->m_fragment = this->create_shader(false);

				return *this;
			}

			shader& operator=(shader&& s) {

				this->erase_modules();

				this->m_device = s.m_device;
				this->m_vertex = s.m_vertex;
				this->m_fragment = s.m_fragment;
				this->m_vertex_path = s.m_vertex_path;
				this->m_fragment_path = s.m_fragment_path;
				
				s.m_vertex = nullptr;
				s.m_fragment = nullptr;

				return *this;
			}

			std::pair<std::string, std::string> get_shader_code(const std::string& vertex, const std::string& fragment) {
					std::string compiler_path;

			#ifdef fan_platform_windows

				compiler_path = "glslc.exe";

			#elif defined(fan_platform_unix)

				compiler_path = "/usr/bin/glslc";

			#endif

				if (!fan::io::file::exists(vertex)) {
					throw std::runtime_error("failed to find file" + vertex);
				}
				if (!fan::io::file::exists(fragment)) {
					throw std::runtime_error("failed to find file" + fragment);
				}

				auto vspv = vertex + ".spv";
				auto fsvp = fragment + ".spv";

				auto v = (compiler_path + ' ' + vertex + " -o " + vspv);
				auto f = (compiler_path + ' ' + fragment + " -o " + fsvp);

				//if (fan::vk::shader::recompile_shaders) {
				if (false) {
					// temp fix

					std::system(v.c_str());
					std::system(f.c_str());
				}


				auto vcode = fan::io::file::read(vspv);
				auto fcode = fan::io::file::read(fsvp);

				return std::make_pair(vcode, fcode);
			}

			VkShaderModule get_vertex_module() {
				return m_vertex;
			}

			VkShaderModule get_fragment_module() {
				return m_fragment;
			}

			void erase_modules() {

				if (m_vertex) {
					vkDestroyShaderModule(*m_device, m_vertex, nullptr);
					m_vertex = nullptr;
				}
				if (m_fragment) {
					vkDestroyShaderModule(*m_device, m_fragment, nullptr);
					m_fragment = nullptr;
				}
			}

			VkShaderModule create_shader(bool vertex) {
				VkShaderModuleCreateInfo createInfo{};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;

				createInfo.codeSize = vertex ? m_vertex_code.size() : m_fragment_code.size();
				createInfo.pCode = (const uint32_t*)(vertex ?  m_vertex_code.data() : m_fragment_code.data());

				VkShaderModule shaderModule;

				if (vkCreateShaderModule(*m_device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
					throw std::runtime_error("failed to create shader module!");
				}

				return shaderModule;
			}

			std::string m_vertex_path;
			std::string m_fragment_path;

			VkShaderModule m_vertex = nullptr;
			VkShaderModule m_fragment = nullptr;

			std::string m_vertex_code;
			std::string m_fragment_code;

			VkDevice* m_device = nullptr;

		};

	}
}

#endif