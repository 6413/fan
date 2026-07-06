module;

#if defined(fan_platform_windows)
#define VK_USE_PLATFORM_WIN32_KHR
#elif defined(fan_platform_unix)
#define VK_USE_PLATFORM_XLIB_KHR
#endif
#if defined(FAN_GUI)
#include <fan/imgui/imgui_impl_vulkan.h>
#endif
#define loco_window
#include <vulkan/vulkan.h>
#include <shaderc/shaderc.hpp>
#if defined(fan_platform_windows)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_EXPOSE_NATIVE_WGL
#define GLFW_NATIVE_INCLUDE_NONE
#endif
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

module fan.graphics.vulkan.core;

import std;

import fan.types.fstring;
import fan.types.color;

#if defined(loco_window)
import fan.window;
#endif

import fan.utility;
import fan.print;
import fan.print.error;
import fan.graphics.image_load;
import fan.graphics.common_context;

import fan.math;
import fan.math.intersection;

import fan.io.file;

#define __fan_internal_camera_list (*fan::graphics::ctx().camera_list)
#define __fan_internal_shader_list (*fan::graphics::ctx().shader_list)
#define __fan_internal_image_list (*fan::graphics::ctx().image_list)
#define __fan_internal_viewport_list (*fan::graphics::ctx().viewport_list)

#if defined(fan_compiler_msvc)
#pragma comment(lib, "vulkan-1.lib")
#pragma comment(lib, "shaderc_combined_mt.lib")
#endif

#define ENABLE_RAYTRACING_DEPENDENCIES

#define VK_CTX ((fan::vulkan::context_t*)context)

fan::vulkan::context_t::shader_t& fan::vulkan::context_t::shader_get(fan::graphics::shader_nr_t nr) {
  return *(fan::vulkan::context_t::shader_t*)__fan_internal_shader_list[nr].internal;
}
std::vector<std::uint32_t> fan::vulkan::context_t::compile_file(const std::string& source_name,
  shaderc_shader_kind kind,
  const std::string& source) {
  shaderc::Compiler compiler;
  shaderc::CompileOptions options;

  options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);

  // Like -DMY_DEFINE=1
  //options.AddMacroDefinition("MY_DEFINE", "1");
#if FAN_DEBUG > 1
  options.SetOptimizationLevel(shaderc_optimization_level_zero);
#else
  options.SetOptimizationLevel(shaderc_optimization_level_performance);
#endif

  shaderc::SpvCompilationResult module =
    compiler.CompileGlslToSpv(source.c_str(), kind, source_name.c_str(), options);

  if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
    fan::throw_error(module.GetErrorMessage().c_str());
  }

  return {module.cbegin(), module.cend()};
}

std::vector<std::uint32_t> fan::vulkan::context_t::load_or_compile(const std::string& source_name,
  shaderc_shader_kind kind,
  const std::string& source) {

  if (source_name.empty()) {
    return compile_file(source_name, kind, source);
  }

  std::string flat = source_name;
  std::replace(flat.begin(), flat.end(), '/', '_');
  std::replace(flat.begin(), flat.end(), '\\', '_');
  std::string cache_path = ".shader_cache/" + flat + ".spv";

  std::error_code ec;
  if (std::filesystem::exists(cache_path, ec) &&
      std::filesystem::last_write_time(source_name, ec) <= std::filesystem::last_write_time(cache_path, ec)) {
    auto spv_size = std::filesystem::file_size(cache_path, ec);
    if (!ec && spv_size > 0) {
      std::ifstream file(cache_path, std::ios::binary);
      if (file) {
        std::vector<std::uint32_t> spv(spv_size / sizeof(std::uint32_t));
        file.read(reinterpret_cast<char*>(spv.data()), spv_size);
        if (file) {
          return spv;
        }
      }
    }
  }

  auto spv = compile_file(source_name, kind, source);
  std::filesystem::create_directories(".shader_cache", ec);
  std::string tmp_path = cache_path + ".tmp";
  fan::io::file::write(tmp_path, std::string(reinterpret_cast<const char*>(spv.data()), spv.size() * sizeof(std::uint32_t)), std::ios_base::binary);
  std::filesystem::remove(cache_path, ec);
  std::filesystem::rename(tmp_path, cache_path, ec);
  return spv;
}
fan::graphics::shader_nr_t fan::vulkan::context_t::shader_create() {
  fan::graphics::shader_nr_t nr = __fan_internal_shader_list.NewNode();
  __fan_internal_shader_list[nr].internal = new fan::vulkan::context_t::shader_t;
  auto& shader = shader_get(nr);
  shader.projection_view_block = new std::remove_pointer_t<decltype(shader.projection_view_block)>;
  //TODO
  shader.projection_view_block->open(*this);
  for (std::uint32_t i = 0; i < fan::vulkan::max_camera; ++i) {
    shader.projection_view_block->push_ram_instance(*this, {});
  }
  return nr;
}
// .cpp
void fan::vulkan::context_t::shader_erase(fan::graphics::shader_nr_t nr, int recycle) {
  auto& shader = shader_get(nr);
  for (auto& stage : shader.shader_stages) {
    if (stage.module) {
      vkDestroyShaderModule(device, stage.module, nullptr);
    }
  }
  // TODO
  shader.projection_view_block->close(*this);
  delete shader.projection_view_block;
  delete static_cast<fan::vulkan::context_t::shader_t*>(__fan_internal_shader_list[nr].internal);
  if (recycle) {
    __fan_internal_shader_list.Recycle(nr);
  }
}
void fan::vulkan::context_t::shader_use(fan::graphics::shader_nr_t nr) {
  // TODO - required?
}
VkShaderModule fan::vulkan::context_t::create_shader_module(const std::vector<std::uint32_t>& code) {
  VkShaderModuleCreateInfo createInfo {};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = code.size() * sizeof(typename std::remove_reference_t<decltype(code)>::value_type);
  createInfo.pCode = code.data();

  VkShaderModule shaderModule;
  if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
    fan::throw_error("failed to create shader module!");
  }

  return shaderModule;
}
void fan::vulkan::context_t::shader_set_vertex(fan::graphics::shader_nr_t nr, const std::string_view file_path, const std::string& vertex_code) {
  __fan_internal_shader_list[nr].path_vertex = file_path;
  __fan_internal_shader_list[nr].svertex = vertex_code;
  // fan::print_impl(
  //   "processed vertex shader:", path, "resulted in:",
  // preprocess_shader(shader_name.c_str(), shaderc_glsl_vertex_shader, shader_code);
  // );
}
void fan::vulkan::context_t::shader_set_vertex(fan::graphics::shader_nr_t nr, const std::string& vertex_code) {
  shader_set_vertex(nr, {}, vertex_code);
}
void fan::vulkan::context_t::shader_set_fragment(fan::graphics::shader_nr_t nr, const std::string_view file_path, const std::string& fragment_code) {
  auto& shader = shader_get(nr);
  __fan_internal_shader_list[nr].path_fragment = file_path;
  __fan_internal_shader_list[nr].sfragment = fragment_code;
  //fan::print_impl(
    // "processed vertex shader:", path, "resulted in:",
  //preprocess_shader(shader_name.c_str(), shaderc_glsl_fragment_shader, shader_code);
  //);
}
void fan::vulkan::context_t::shader_set_fragment(fan::graphics::shader_nr_t nr, const std::string& fragment_code) {
  shader_set_fragment(nr, {}, fragment_code);
}
void fan::vulkan::context_t::shader_set_compute(
  fan::graphics::shader_nr_t nr,
  const std::string_view file_path,
  const std::string& compute_code
) {
  __fan_internal_shader_list[nr].path_compute = file_path;
  __fan_internal_shader_list[nr].scompute = compute_code;
}
void fan::vulkan::context_t::shader_set_camera(fan::graphics::shader_nr_t nr, fan::graphics::camera_nr_t camera_nr) {
  auto& shader = shader_get(nr);
  auto& camera = camera_get(camera_nr);

  std::uint32_t camera_index = camera_nr.gint();

#if FAN_DEBUG >= fan_debug_medium
  if (camera_index >= fan::vulkan::max_camera) {
    fan::throw_error("vulkan camera index exceeds max_camera");
  }
#endif

  shader.projection_view_block->edit_instance(
    *this,
    camera_index,
    &fan::vulkan::view_projection_t::projection,
    camera.projection
  );

  shader.projection_view_block->edit_instance(
    *this,
    camera_index,
    &fan::vulkan::view_projection_t::view,
    camera.view
  );
}
void fan::vulkan::context_t::shader_dispatch_compute(
  fan::graphics::shader_nr_t nr,
  std::uint32_t x,
  std::uint32_t y,
  std::uint32_t z
) {
  fan::throw_error("vulkan compute dispatch is not implemented");
}

bool fan::vulkan::context_t::shader_compile(fan::graphics::shader_nr_t nr) {
  auto& shader = shader_get(nr);
  auto& list_item = __fan_internal_shader_list[nr];

  bool has_vertex = !list_item.svertex.empty();
  bool has_fragment = !list_item.sfragment.empty();
  bool has_compute = !list_item.scompute.empty();

  if (has_compute && (has_vertex || has_fragment)) {
    fan::print_impl("compute shader cannot be linked with graphics shaders");
    return false;
  }

  auto compile_stage = [&](const std::string& path, std::string& code, shaderc_shader_kind kind, VkShaderStageFlagBits stage, int index) {
    if (code.empty()) { return; }
    auto spirv = load_or_compile(path, kind, code);
    if (shader.shader_stages[index].module != VK_NULL_HANDLE) {
      VkShaderModule old_module = shader.shader_stages[index].module;
      get_current_deletion_queue().push_function([=, device = this->device]() {
        vkDestroyShaderModule(device, old_module, nullptr);
      });
    }
    shader.shader_stages[index] = {
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
      stage, create_shader_module(spirv), "main", nullptr
    };
  };

  compile_stage(list_item.path_vertex.c_str(), list_item.svertex, shaderc_glsl_vertex_shader, VK_SHADER_STAGE_VERTEX_BIT, 0);
  compile_stage(list_item.path_fragment.c_str(), list_item.sfragment, shaderc_glsl_fragment_shader, VK_SHADER_STAGE_FRAGMENT_BIT, 1);
  compile_stage(list_item.path_compute.c_str(), list_item.scompute, shaderc_glsl_compute_shader, VK_SHADER_STAGE_COMPUTE_BIT, 0);

  ++shader.compile_generation;
  return true;
}