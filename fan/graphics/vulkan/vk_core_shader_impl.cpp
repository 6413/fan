module;

#if defined(FAN_2D)


#include <fan/utility.h>

#define USE_SHADERC

#if defined(fan_platform_windows)
  #if defined(USE_SHADERC)
    #pragma comment (lib, "shaderc_combined_mt.lib")
  #endif
  #define VK_USE_PLATFORM_WIN32_KHR
#elif defined(fan_platform_unix)
  #define VK_USE_PLATFORM_XLIB_KHR
#endif
#include <vulkan/vulkan.h>
#include <shaderc/shaderc.hpp>

module fan.graphics.vulkan.core;

import std;

import fan.types.fstring;
import fan.types.color;

import fan.window;

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

#define VK_CTX ((fan::vulkan::context_t*)context)

fan::vulkan::shader_t& fan::vulkan::shader_subsystem_t::shader_get(fan::graphics::shader_nr_t nr) {
  return *(fan::vulkan::shader_t*)__fan_internal_shader_list[nr].internal;
}
std::vector<std::uint32_t> fan::vulkan::shader_subsystem_t::compile_file(const std::string& source_name,
  int kind,
  const std::string& source) 
{
#if defined(USE_SHADERC)
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
    compiler.CompileGlslToSpv(source.c_str(), static_cast<shaderc_shader_kind>(kind), source_name.c_str(), options);

  if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
    fan::throw_error(module.GetErrorMessage().c_str());
  }

  return {module.cbegin(), module.cend()};
#else
  fan::throw_error("shaderc runtime compilation not available - build with USE_SHADERC");
  return {};
#endif
}

std::vector<std::uint32_t> fan::vulkan::shader_subsystem_t::load_or_compile(const std::string& source_name, int kind, const std::string& source) {
  if (source_name.empty()) {
    return compile_file(source_name, (shaderc_shader_kind)kind, source);
  }

  auto read_cache = [](const std::string& path) {
    auto size = fan::io::file::file_size(path);
    std::vector<std::uint32_t> spv(size / sizeof(std::uint32_t));
    fan::io::file::read_bytes(path, spv.data(), size);
    return spv;
  };

  auto write_cache = [](const std::string& path, const std::vector<std::uint32_t>& spv) {
    std::error_code ec;
    std::filesystem::create_directories(".shader_cache", ec);
    std::string tmp = path + ".tmp";
    fan::io::file::try_write(tmp, std::string(reinterpret_cast<const char*>(spv.data()), spv.size() * sizeof(std::uint32_t)), std::ios_base::binary);
    std::filesystem::remove(path, ec);
    std::filesystem::rename(tmp, path, ec);
  };

  std::string flat = source_name;
  std::replace(flat.begin(), flat.end(), '/', '_');
  std::replace(flat.begin(), flat.end(), '\\', '_');

  std::string cache_path = ".shader_cache/" + flat + ".spv";
  std::filesystem::path resolved_source = fan::io::file::find_relative_path(source_name);

  if (!resolved_source.empty()) {
    if (fan::io::file::is_up_to_date(resolved_source.string(), cache_path)) {
      return read_cache(cache_path);
    }
  } 
  else {
    if (!source.empty()) {
      cache_path = ".shader_cache/" + flat + "_" + std::to_string(std::hash<std::string>{}(source)) + ".spv";
    }
    if (std::filesystem::exists(cache_path)) {
      return read_cache(cache_path);
    }
  }

  auto spv = compile_file(source_name, (shaderc_shader_kind)kind, source);
  if (!spv.empty()) {
    write_cache(cache_path, spv);
  }
  
  return spv;
}

fan::graphics::shader_nr_t fan::vulkan::shader_subsystem_t::shader_create() {
  fan::graphics::shader_nr_t nr = __fan_internal_shader_list.NewNode();
  __fan_internal_shader_list[nr].internal = new fan::vulkan::shader_t;
  auto& shader = shader_get(nr);
  shader.projection_view_block = new std::remove_pointer_t<decltype(shader.projection_view_block)>;
  shader.projection_view_block->open(*ctx);
  for (std::uint32_t i = 0; i < fan::vulkan::max_camera; ++i) {
    shader.projection_view_block->push_ram_instance(*ctx, {});
  }
  return nr;
}
void fan::vulkan::shader_subsystem_t::shader_erase(fan::graphics::shader_nr_t nr, int recycle) {
  auto& shader = shader_get(nr);
  for (auto& stage : shader.shader_stages) {
    if (stage.module) {
      vkDestroyShaderModule(ctx->device, stage.module, nullptr);
    }
  }
  shader.projection_view_block->close(*ctx);
  delete shader.projection_view_block;
  delete static_cast<fan::vulkan::shader_t*>(__fan_internal_shader_list[nr].internal);
  if (recycle) {
    __fan_internal_shader_list.Recycle(nr);
  }
}
void fan::vulkan::shader_subsystem_t::shader_use(fan::graphics::shader_nr_t nr) {
}
VkShaderModule fan::vulkan::shader_subsystem_t::create_shader_module(const std::vector<std::uint32_t>& code) {
  VkShaderModuleCreateInfo createInfo {};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = code.size() * sizeof(typename std::remove_reference_t<decltype(code)>::value_type);
  createInfo.pCode = code.data();

  VkShaderModule shaderModule;
  if (vkCreateShaderModule(ctx->device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
    fan::throw_error("failed to create shader module!");
  }

  return shaderModule;
}
void fan::vulkan::shader_subsystem_t::shader_set_vertex(fan::graphics::shader_nr_t nr, const std::string_view file_path, const std::string& vertex_code) {
  __fan_internal_shader_list[nr].path_vertex = file_path;
  __fan_internal_shader_list[nr].svertex = vertex_code;
  // fan::print_impl(
  //   "processed vertex shader:", path, "resulted in:",
  // preprocess_shader(shader_name.c_str(), shaderc_glsl_vertex_shader, shader_code);
  // );
}
void fan::vulkan::shader_subsystem_t::shader_set_vertex(fan::graphics::shader_nr_t nr, const std::string& vertex_code) {
  shader_set_vertex(nr, {}, vertex_code);
}
void fan::vulkan::shader_subsystem_t::shader_set_fragment(fan::graphics::shader_nr_t nr, const std::string_view file_path, const std::string& fragment_code) {
  auto& shader = __fan_internal_shader_list[nr];
  shader.path_fragment = file_path;
  shader.sfragment = fragment_code;
  //fan::print_impl(
    // "processed vertex shader:", path, "resulted in:",
  //preprocess_shader(shader_name.c_str(), shaderc_glsl_fragment_shader, shader_code);
  //);
}
void fan::vulkan::shader_subsystem_t::shader_set_fragment(fan::graphics::shader_nr_t nr, const std::string& fragment_code) {
  shader_set_fragment(nr, {}, fragment_code);
}
void fan::vulkan::shader_subsystem_t::shader_set_compute(
  fan::graphics::shader_nr_t nr,
  const std::string_view file_path,
  const std::string& compute_code
) {
  __fan_internal_shader_list[nr].path_compute = file_path;
  __fan_internal_shader_list[nr].scompute = compute_code;
}
void fan::vulkan::shader_subsystem_t::shader_set_camera(fan::graphics::shader_nr_t nr, fan::graphics::camera_nr_t camera_nr) {
  auto& shader = shader_get(nr);
  auto& camera = ctx->cameras.camera_get(camera_nr);

  std::uint32_t camera_index = camera_nr.gint();

#if FAN_DEBUG >= fan_debug_medium
  if (camera_index >= fan::vulkan::max_camera) {
    fan::throw_error("vulkan camera index exceeds max_camera");
  }
#endif

  shader.projection_view_block->edit_instance(
    *ctx,
    camera_index,
    &fan::vulkan::view_projection_t::projection,
    camera.projection
  );

  shader.projection_view_block->edit_instance(
    *ctx,
    camera_index,
    &fan::vulkan::view_projection_t::view,
    camera.view
  );
}
void fan::vulkan::shader_subsystem_t::shader_dispatch_compute(
  fan::graphics::shader_nr_t nr,
  std::uint32_t x,
  std::uint32_t y,
  std::uint32_t z
) {
  fan::throw_error("vulkan compute dispatch is not implemented");
}

bool fan::vulkan::shader_subsystem_t::shader_compile(fan::graphics::shader_nr_t nr) {
  auto& shader = shader_get(nr);
  auto& list_item = __fan_internal_shader_list[nr];

  bool has_vertex = !list_item.svertex.empty();
  bool has_fragment = !list_item.sfragment.empty();
  bool has_compute = !list_item.scompute.empty();

  if (has_compute && (has_vertex || has_fragment)) {
    fan::print_impl("compute shader cannot be linked with graphics shaders");
    return false;
  }

  auto compile_stage = [&](const std::string& path, std::string& code, std::vector<std::uint32_t>& preloaded_spv, shaderc_shader_kind kind, VkShaderStageFlagBits stage, int index) {
    if (code.empty()) { return; }
    auto spirv = preloaded_spv.empty() ? load_or_compile(path, kind, code) : std::move(preloaded_spv);
    if (shader.shader_stages[index].module != VK_NULL_HANDLE) {
      VkShaderModule old_module = shader.shader_stages[index].module;
      ctx->get_current_deletion_queue().push_function([=, device = ctx->device]() {
        vkDestroyShaderModule(device, old_module, nullptr);
      });
    }
    shader.shader_stages[index] = {
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
      stage, create_shader_module(spirv), "main", nullptr
    };
  };

  compile_stage(list_item.path_vertex.c_str(), list_item.svertex, list_item.spv_vertex, shaderc_glsl_vertex_shader, VK_SHADER_STAGE_VERTEX_BIT, 0);
  compile_stage(list_item.path_fragment.c_str(), list_item.sfragment, list_item.spv_fragment, shaderc_glsl_fragment_shader, VK_SHADER_STAGE_FRAGMENT_BIT, 1);
  compile_stage(list_item.path_compute.c_str(), list_item.scompute, list_item.spv_compute, shaderc_glsl_compute_shader, VK_SHADER_STAGE_COMPUTE_BIT, 0);

  ++shader.compile_generation;
  return true;
}

#endif