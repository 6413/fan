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

namespace {

// Minimal SPIR-V reflector: locates a uniform block at (set 0, binding 3) and
// extracts its members (name, type, offset, size) using the Offset/MemberName
// decorations emitted by glslang. Used to auto-generate shader uniform UI.
struct spirv_parser_t {
  struct type_info_t {
    std::uint32_t op = 0;
    std::uint32_t vcount = 0;          // OpTypeVector length
    std::uint32_t mcol = 0, mrow = 0;  // OpTypeMatrix dims
    std::uint32_t arr_elem = 0;        // OpTypeArray element type id
    std::uint32_t arr_len = 1;         // OpTypeArray length (from constant)
    std::uint32_t elem = 0;            // base component type id (vector/matrix)
    std::vector<std::uint32_t> members;
  };

  std::unordered_map<std::uint32_t, type_info_t> types;
  std::unordered_map<std::uint32_t, std::uint32_t> pointer_elem;                       // pointer id -> element id
  std::unordered_map<std::uint32_t, std::pair<std::uint32_t, std::uint32_t>> var_binding; // var id -> {set, binding}
  std::unordered_map<std::uint32_t, std::uint32_t> var_type;                           // var id -> pointer type id
  std::unordered_map<std::uint32_t, std::vector<std::pair<std::uint32_t, std::uint32_t>>> member_offsets;  // struct id -> {member, offset}
  std::unordered_map<std::uint32_t, std::vector<std::pair<std::uint32_t, std::uint32_t>>> member_strides;  // struct id -> {member, array stride}
  std::unordered_map<std::uint32_t, std::vector<std::string>> member_names;            // struct id -> member names
  std::vector<std::uint32_t> uniform_vars;

  void parse(const std::vector<std::uint32_t>& data) {
    if (data.size() < 5 || data[0] != 0x07230203) { return; }

    std::unordered_map<std::uint32_t, std::uint32_t> constants;
    std::size_t pos = 5;
    while (pos + 1 <= data.size()) {
      std::uint32_t w = data[pos];
      std::uint32_t count = w >> 16;
      std::uint32_t op = w & 0xFFFFu;
      if (count == 0 || pos + count > data.size()) { break; }
      const std::uint32_t* o = data.data() + pos + 1;

      switch (op) {
      case 20: { // OpTypeBool
        if (count >= 1) { types[o[0]].op = op; }
        break; }
      case 21: { // OpTypeInt: result, width, signedness
        if (count >= 2) { types[o[0]].op = op; }
        break; }
      case 22: { // OpTypeFloat
        if (count >= 1) { types[o[0]].op = op; }
        break; }
      case 23: { // OpTypeVector: result, component type, count
        if (count >= 3) {
          auto& t = types[o[0]];
          t.op = op;
          t.elem = o[1];
          t.vcount = o[2];
        }
        break; }
      case 24: { // OpTypeMatrix: result, column type, count, rows
        if (count >= 3) {
          auto& t = types[o[0]];
          t.op = op;
          t.elem = o[1];
          t.mcol = o[2];
          t.mrow = count >= 4 ? o[3] : 0;
        }
        break; }
      case 28: { // OpTypeArray: result, element type, length constant
        if (count >= 3) {
          auto& t = types[o[0]];
          t.op = op;
          t.arr_elem = o[1];
          auto it = constants.find(o[2]);
          if (it != constants.end()) { t.arr_len = it->second; }
        }
        break; }
      case 30: { // OpTypeStruct: result, member type ids...
        auto& t = types[o[0]];
        t.op = op;
        for (std::uint32_t k = 1; k + 1 < count; ++k) { t.members.push_back(o[k]); }
        break; }
      case 32: { // OpTypePointer: result, storage class, element type
        if (count >= 3) { pointer_elem[o[0]] = o[2]; }
        break; }
      case 43: { // OpConstant: result type, result id, value
        if (count >= 3) { constants[o[1]] = o[2]; }
        break; }
      case 59: { // OpVariable: result type, result id, storage class
        if (count >= 3 && o[2] == 2) {
          uniform_vars.push_back(o[1]);
          var_type[o[1]] = o[0];
        }
        break; }
      case 71: { // OpDecorate: target, decoration, literal
        if (count >= 3) {
          if (o[1] == 33) { var_binding[o[0]].second = o[2]; }        // Binding
          else if (o[1] == 34) { var_binding[o[0]].first = o[2]; }    // DescriptorSet
        }
        break; }
      case 72: { // OpMemberDecorate: target, member, decoration, literal
        if (count >= 4) {
          if (o[2] == 35) { member_offsets[o[0]].push_back({o[1], o[3]}); }   // Offset
          else if (o[2] == 6) { member_strides[o[0]].push_back({o[1], o[3]}); } // ArrayStride
        }
        break; }
      case 6: { // OpMemberName: struct id, member, name
        if (count >= 3) {
          std::string name((const char*)(o + 2));
          member_names[o[0]].resize(std::max<std::size_t>(member_names[o[0]].size(), (std::size_t)o[1] + 1));
          member_names[o[0]][o[1]] = std::move(name);
        }
        break; }
      default:
        break;
      }
      pos += count;
    }
  }
  std::uint32_t type_size(std::uint32_t id) const {
    auto it = types.find(id);
    if (it == types.end()) { return 0; }
    const auto& t = it->second;
    switch (t.op) {
    case 20: case 21: case 22: return 4;
    case 23: return 4 * t.vcount;
    case 24: return 4 * t.mcol * (t.mrow ? t.mrow : 4);
    case 28: return type_size(t.arr_elem) * t.arr_len;
    case 30: {
      std::uint32_t size = 0;
      for (auto& m : t.members) { size += type_size(m); }
      return size;
    }
    default: return 0;
    }
  }
};

fan::graphics::shader_uniform_t::type_e spv_type_to_enum(const spirv_parser_t& p, std::uint32_t id, std::uint32_t& array_size) {
  array_size = 1;
  const auto& types = p.types;
  auto it = types.find(id);
  if (it == types.end()) { return fan::graphics::shader_uniform_t::type_e::unknown; }
  const auto& t = it->second;
  std::uint32_t base = id;
  if (t.op == 28) { // array
    array_size = t.arr_len;
    base = t.arr_elem;
  }
  auto bit = types.find(base);
  if (bit == types.end()) { return fan::graphics::shader_uniform_t::type_e::unknown; }
  const auto& b = bit->second;
  switch (b.op) {
  case 22: return fan::graphics::shader_uniform_t::type_e::f32;
  case 21: return fan::graphics::shader_uniform_t::type_e::u32;
  case 23:
    switch (b.vcount) {
    case 2: return fan::graphics::shader_uniform_t::type_e::vec2;
    case 3: return fan::graphics::shader_uniform_t::type_e::vec3;
    case 4: return fan::graphics::shader_uniform_t::type_e::vec4;
    default: return fan::graphics::shader_uniform_t::type_e::unknown;
    }
  case 24: return fan::graphics::shader_uniform_t::type_e::f32_mat4;
  default: return fan::graphics::shader_uniform_t::type_e::unknown;
  }
}

void spirv_reflect_uniform_block(const std::vector<std::uint32_t>& spirv, std::vector<fan::graphics::shader_uniform_t>& out, std::uint32_t& block_size) {
  if (spirv.empty()) { return; }

  spirv_parser_t parser;
  parser.parse(spirv);

  std::uint32_t struct_id = 0;
  for (auto var : parser.uniform_vars) {
    auto vb = parser.var_binding.find(var);
    if (vb == parser.var_binding.end()) { continue; }
    if (vb->second.first == 0 && vb->second.second == 3) {
      auto vt = parser.var_type.find(var);
      if (vt == parser.var_type.end()) { continue; }
      auto pe = parser.pointer_elem.find(vt->second);
      if (pe == parser.pointer_elem.end()) { continue; }
      struct_id = pe->second;
      break;
    }
  }
  if (struct_id == 0) { return; }

  auto tit = parser.types.find(struct_id);
  if (tit == parser.types.end()) { return; }
  const auto& members = tit->second.members;

  auto offsets = parser.member_offsets.find(struct_id);
  auto strides = parser.member_strides.find(struct_id);
  auto names = parser.member_names.find(struct_id);

  std::uint32_t max_end = 0;
  for (std::uint32_t i = 0; i < members.size(); ++i) {
    fan::graphics::shader_uniform_t u;
    if (names != parser.member_names.end() && i < names->second.size() && !names->second[i].empty()) {
      u.name = names->second[i];
    }
    else {
      u.name = "member_" + std::to_string(i);
    }

    std::uint32_t array_size = 1;
    u.type = spv_type_to_enum(parser, members[i], array_size);
    u.array_size = array_size;

    if (offsets != parser.member_offsets.end()) {
      for (auto& mo : offsets->second) {
        if (mo.first == i) { u.offset = mo.second; }
      }
    }
    u.size = parser.type_size(members[i]);
    if (strides != parser.member_strides.end() && u.array_size > 1) {
      for (auto& ms : strides->second) {
        if (ms.first == i) { u.size = ms.second * u.array_size; }
      }
    }
    max_end = std::max<std::uint32_t>(max_end, u.offset + u.size);
    out.push_back(std::move(u));
  }
  block_size = (max_end + 15) & ~15u;
}

void apply_source_member_names(const std::string_view source, std::vector<fan::graphics::shader_uniform_t>& uniforms, std::vector<std::uint8_t>& blob) {
  if (uniforms.empty() || source.empty()) { return; }
  std::size_t pos = source.rfind("uniform");
  if (pos == std::string::npos) { return; }
  std::size_t open = source.find('{', pos);
  if (open == std::string::npos) { return; }
  std::size_t close = source.find('}', open);
  if (close == std::string::npos) { return; }
  std::string_view body(source.data() + open + 1, close - open - 1);
  std::uint32_t i = 0;
  std::size_t p = 0;
  while (p < body.size() && i < uniforms.size()) {
    std::size_t semi = body.find(';', p);
    if (semi == std::string::npos) { break; }
    std::string_view decl = body.substr(p, semi - p);
    std::size_t nl = body.find('\n', semi + 1);
    std::string_view tail = nl == std::string::npos ? body.substr(semi + 1) : body.substr(semi + 1, nl - semi - 1);
    p = nl == std::string::npos ? body.size() : nl + 1;
    std::size_t e = decl.find('=');
    if (e != std::string::npos) { decl = decl.substr(0, e); }
    std::size_t ns = decl.find_last_not_of(" \t\r\n");
    if (ns == std::string::npos) { continue; }
    std::size_t nf = decl.substr(0, ns).find_last_of(" \t\r\n");
    std::string name = nf == std::string::npos
      ? std::string(decl.substr(0, ns + 1))
      : std::string(decl.substr(nf + 1, ns - nf));
    if (name.empty()) { continue; }
    uniforms[i].name = std::move(name);

    std::size_t cm = tail.find("//");
    if (cm != std::string::npos) {
      std::string_view comment = tail.substr(cm + 2);
      auto& u = uniforms[i];
      if (comment.find("color") != std::string_view::npos) {
        u.is_color = true;
      }
      if (!blob.empty()) {
        std::vector<f32_t> vals;
        std::string_view pending; // last keyword (min/max/step) awaiting its value
        std::size_t p = 0;
        while (p < comment.size()) {
          while (p < comment.size() && (comment[p] == ' ' || comment[p] == '\t' || comment[p] == ',')) { ++p; }
          if (p >= comment.size()) { break; }
          std::size_t start = p;
          while (p < comment.size() && comment[p] != ' ' && comment[p] != '\t' && comment[p] != ',') { ++p; }
          std::string_view token = comment.substr(start, p - start);
          if (token == "color" || token == "min" || token == "max" || token == "step") {
            if (token != "color") { pending = token; }
            continue;
          }
          double v;
          auto res = std::from_chars(token.data(), token.data() + token.size(), v);
          if (res.ec != std::errc() || res.ptr != token.data() + token.size()) { continue; }
          if (pending == "min") { u.has_min = true; u.min = (f32_t)v; pending = {}; }
          else if (pending == "max") { u.has_max = true; u.max = (f32_t)v; pending = {}; }
          else if (pending == "step") { u.step = (f32_t)v; pending = {}; }
          else { vals.push_back((f32_t)v); }
        }
        std::uint8_t* dst = blob.data() + u.offset;
        switch (u.type) {
        case fan::graphics::shader_uniform_t::type_e::f32:
          if (!vals.empty()) { std::memcpy(dst, &vals[0], 4); }
          break;
        case fan::graphics::shader_uniform_t::type_e::i32:
        case fan::graphics::shader_uniform_t::type_e::u32:
          if (!vals.empty()) { std::uint32_t iv = (std::uint32_t)vals[0]; std::memcpy(dst, &iv, 4); }
          break;
        case fan::graphics::shader_uniform_t::type_e::vec2:
        case fan::graphics::shader_uniform_t::type_e::vec3:
        case fan::graphics::shader_uniform_t::type_e::vec4: {
          std::uint32_t count = u.type == fan::graphics::shader_uniform_t::type_e::vec2 ? 2 :
            u.type == fan::graphics::shader_uniform_t::type_e::vec3 ? 3 : 4;
          for (std::uint32_t k = 0; k < count && k < vals.size(); ++k) { std::memcpy(dst + k * 4, &vals[k], 4); }
          break;
        }
        default:
          break;
        }
      }
    }

    ++i;
  }
}

}

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
  if (shader.uniform_block_valid) {
    for (std::uint32_t i = 0; i < fan::vulkan::max_frames_in_flight; ++i) {
      ctx->destroy_buffer(shader.uniform_block[i]);
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
    if (code.empty() && !path.empty()) {
      auto resolved = fan::io::file::find_relative_path(path);
      if (!resolved.empty()) {
        fan::io::file::read(resolved.string(), &code);
      }
    }
    if (code.empty()) { return; }
    auto spirv = preloaded_spv.empty() ? load_or_compile(path, kind, code) : std::move(preloaded_spv);
    if (shader.shader_stages[index].module != VK_NULL_HANDLE) {
      VkShaderModule old_module = shader.shader_stages[index].module;
      ctx->get_current_deletion_queue().push_function([=, device = ctx->device]() {
        vkDestroyShaderModule(device, old_module, nullptr);
      });
    }
    shader.spirv_stages[index] = spirv;
    shader.shader_stages[index] = {
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
      stage, create_shader_module(spirv), "main", nullptr
    };
  };

  compile_stage(list_item.path_vertex.c_str(), list_item.svertex, list_item.spv_vertex, shaderc_glsl_vertex_shader, VK_SHADER_STAGE_VERTEX_BIT, 0);
  compile_stage(list_item.path_fragment.c_str(), list_item.sfragment, list_item.spv_fragment, shaderc_glsl_fragment_shader, VK_SHADER_STAGE_FRAGMENT_BIT, 1);
  compile_stage(list_item.path_compute.c_str(), list_item.scompute, list_item.spv_compute, shaderc_glsl_compute_shader, VK_SHADER_STAGE_COMPUTE_BIT, 0);

  ++shader.compile_generation;

  list_item.uniforms.clear();
  list_item.uniform_block_size = 0;
  list_item.uniform_blob.clear();

  spirv_reflect_uniform_block(shader.spirv_stages[1], list_item.uniforms, list_item.uniform_block_size);
  std::string_view src = list_item.sfragment;
  if (list_item.uniform_block_size == 0) {
    list_item.uniforms.clear();
    spirv_reflect_uniform_block(shader.spirv_stages[0], list_item.uniforms, list_item.uniform_block_size);
    src = list_item.svertex;
  }
  if (list_item.uniform_block_size > 0) {
    list_item.uniform_blob.assign(list_item.uniform_block_size, 0);
  }
  apply_source_member_names(src, list_item.uniforms, list_item.uniform_blob);

  if (list_item.uniform_block_size > 0) {
    if (shader.uniform_block_valid) {
      for (std::uint32_t i = 0; i < fan::vulkan::max_frames_in_flight; ++i) {
        ctx->destroy_buffer(shader.uniform_block[i]);
        shader.uniform_block[i] = {};
        shader.uniform_block_mapped[i] = nullptr;
      }
      shader.uniform_block_valid = false;
    }
    for (std::uint32_t i = 0; i < fan::vulkan::max_frames_in_flight; ++i) {
      ctx->create_buffer(
        list_item.uniform_block_size,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        shader.uniform_block[i]
      );
      void* mapped = nullptr;
      ctx->map_buffer(shader.uniform_block[i], &mapped);
      shader.uniform_block_mapped[i] = mapped;
      if (mapped != nullptr) {
        std::memcpy(mapped, list_item.uniform_blob.data(), list_item.uniform_block_size);
      }
    }
    shader.uniform_block_valid = true;
  }

  return true;
}

#endif