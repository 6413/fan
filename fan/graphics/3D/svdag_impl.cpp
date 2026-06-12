module;

#if defined (FAN_WINDOW)

#if defined(FAN_3D)

#include <fan/graphics/gl_api.h>
#include <fan/utility.h>

#endif

#endif

module fan.graphics.svdag;

#if defined (FAN_WINDOW)

#if defined(FAN_3D)

import std;

import fan.time;
import fan.io.file;

import fan.graphics.loco;
import fan.graphics.opengl3D.objects.model;

import fan.print;

namespace fm = fan::math;
namespace file = fan::io::file;

#define gl (gloco()->get_context().gl)

namespace fan::graphics {
  bool svdag_t::sample_alpha(const build_tri_t& t, const std::vector<fan::model::cpu_texture_t>& textures, fan::vec3 p, f32_t alpha_threshold) {
    if (t.alpha_tex < 0 || t.alpha_tex >= int(textures.size())) { return true; }

    const auto& tex = textures[t.alpha_tex];
    if (tex.data == nullptr || tex.data_size == 0 || tex.channels < 4 || tex.size.x == 0 || tex.size.y == 0) { return true; }

    fan::vec3 b = fm::d3::closest_barycentric(p, t.v0, t.v1, t.v2);
    fan::vec2 uv = t.uv0 * b.x + t.uv1 * b.y + t.uv2 * b.z;
    f32_t u = uv.x - std::floor(uv.x);
    f32_t v = uv.y - std::floor(uv.y);
    std::uint32_t x = std::min<std::uint32_t>(std::uint32_t(u * tex.size.x), tex.size.x - 1);
    std::uint32_t y = std::min<std::uint32_t>(std::uint32_t(v * tex.size.y), tex.size.y - 1);
    std::uint32_t i = (y * tex.size.x + x) * tex.channels + 3;
    return tex.data.get()[i] >= std::uint8_t(alpha_threshold * 255.f);
  }

  bool svdag_t::alpha_allows_cell(const build_tri_t& t, const std::vector<fan::model::cpu_texture_t>& textures, fan::vec3 child_min, int size, f32_t alpha_threshold) {
    if (t.alpha_tex < 0) { return true; }

    if (sample_alpha(t, textures, child_min + fan::vec3(size * 0.5f), alpha_threshold)) { return true; }

    for (int i = 0; i < 8; ++i) {
      fan::vec3 p = child_min + fan::vec3((i & 1) ? size : 0, (i >> 1 & 1) ? size : 0, (i >> 2 & 1) ? size : 0);
      if (sample_alpha(t, textures, p, alpha_threshold)) { return true; }
    }

    return false;
  }

  fan::vec4 svdag_t::sample_diffuse(const build_tri_t& t, const std::vector<fan::model::cpu_texture_t>& textures, fan::vec3 p) {
    if (t.alpha_tex < 0 || t.alpha_tex >= int(textures.size())) { return fan::vec4(1.f); }

    const auto& tex = textures[t.alpha_tex];
    if (tex.data == nullptr || tex.data_size == 0 || tex.channels < 3 || tex.size.x == 0 || tex.size.y == 0) { return fan::vec4(1.f); }

    fan::vec3 b = fm::d3::closest_barycentric(p, t.v0, t.v1, t.v2);
    fan::vec2 uv = t.uv0 * b.x + t.uv1 * b.y + t.uv2 * b.z;
    f32_t u = uv.x - std::floor(uv.x);
    f32_t v = uv.y - std::floor(uv.y);
    std::uint32_t x = std::min<std::uint32_t>(std::uint32_t(u * tex.size.x), tex.size.x - 1);
    std::uint32_t y = std::min<std::uint32_t>(std::uint32_t(v * tex.size.y), tex.size.y - 1);
    std::uint32_t i = (y * tex.size.x + x) * tex.channels;
    constexpr f32_t inv_255 = 1.f / 255.f;
    return fan::vec4(
      tex.data.get()[i + 0] * inv_255,
      tex.data.get()[i + 1] * inv_255,
      tex.data.get()[i + 2] * inv_255,
      tex.channels >= 4 ? tex.data.get()[i + 3] * inv_255 : 1.f
    );
  }

  gpu_lod_data_t svdag_t::make_leaf_lod_data(
    const build_tri_t& t,
    const std::vector<gpu_mat_t>& mats,
    const std::vector<fan::model::cpu_texture_t>& textures,
    fan::vec3 child_min, int size
  ) {
    fan::vec3 p = child_min + fan::vec3(size * 0.5f);
    fan::vec3 b = fm::d3::closest_barycentric(p, t.v0, t.v1, t.v2);
    fan::vec4 albedo = t.c0 * b.x + t.c1 * b.y + t.c2 * b.z;
    albedo *= sample_diffuse(t, textures, p);
    if (t.mat_id < mats.size()) {
      albedo *= mats[t.mat_id].base_color;
    }
    albedo.w = 1.f;

    fan::vec3 n = (t.v1 - t.v0).cross(t.v2 - t.v0);
    f32_t nl = std::sqrt(n.dot(n));
    if (nl > 1e-8f) { n /= nl; }
    else { n = fan::vec3(0.f, 1.f, 0.f); }

    return {albedo, fan::vec4(n, 1.f)};
  }

  gpu_lod_data_t svdag_t::make_cell_lod_data(
    const std::vector<build_tri_t>& tris,
    const std::vector<std::uint32_t>& tri_idx,
    const std::vector<gpu_mat_t>& mats,
    const std::vector<fan::model::cpu_texture_t>& textures,
    fan::vec3 child_min, int size
  ) {
    fan::vec4 albedo(0.f);
    fan::vec3 normal(0.f);
    f32_t weight_sum = 0.f;
    fan::vec3 p = child_min + fan::vec3(size * 0.5f);

    for (std::uint32_t ti : tri_idx) {
      const auto& t = tris[ti];

      if (!alpha_allows_cell(t, textures, child_min, size, 0.2f)) {
        continue;
      }

      fan::vec3 e0 = t.v1 - t.v0;
      fan::vec3 e1 = t.v2 - t.v0;
      fan::vec3 n = e0.cross(e1);
      f32_t area = std::sqrt(n.dot(n));
      if (area <= 1e-8f) {
        continue;
      }

      n /= area;

      fan::vec3 b = fm::d3::closest_barycentric(p, t.v0, t.v1, t.v2);
      fan::vec4 c = t.c0 * b.x + t.c1 * b.y + t.c2 * b.z;
      c *= sample_diffuse(t, textures, p);

      if (t.mat_id < mats.size()) {
        c *= mats[t.mat_id].base_color;
      }

      if (c.w <= 0.2f) {
        continue;
      }

      albedo += fan::vec4(c.x, c.y, c.z, 1.f) * area;
      normal += n * area;
      weight_sum += area;
    }

    if (weight_sum <= 0.f) {
      return {fan::vec4(0.f, 0.f, 0.f, 0.f), fan::vec4(0.f, 1.f, 0.f, 0.f)};
    }

    albedo /= weight_sum;
    albedo.w = 1.f;

    f32_t nl = std::sqrt(normal.dot(normal));
    if (nl > 1e-8f) {
      normal /= nl;
    }
    else {
      normal = fan::vec3(0.f, 1.f, 0.f);
    }

    return {albedo, fan::vec4(normal, 1.f)};
  }

  gpu_lod_data_t svdag_t::merge_lod_data(const gpu_lod_data_t child_lod[8], const bool child_lod_valid[8]) {
    fan::vec4 albedo(0.f);
    fan::vec3 normal(0.f);
    f32_t count = 0.f;

    for (int i = 0; i < 8; ++i) {
      if (!child_lod_valid[i]) { continue; }
      albedo += child_lod[i].albedo;
      normal += fan::vec3(child_lod[i].normal.x, child_lod[i].normal.y, child_lod[i].normal.z);
      count += 1.f;
    }

    if (count == 0.f) {
      return {fan::vec4(0.f, 0.f, 0.f, 0.f), fan::vec4(0.f, 1.f, 0.f, 0.f)};
    }

    albedo /= count;
    albedo.w = 1.f;

    f32_t nl = std::sqrt(normal.dot(normal));
    if (nl > 1e-8f) { normal /= nl; }
    else { normal = fan::vec3(0.f, 1.f, 0.f); }

    return {albedo, fan::vec4(normal, 1.f)};
  }

  std::uint32_t svdag_t::new_node() {
    std::uint32_t idx = nodes.size();
    nodes.push_back(0);
    child_ptrs.push_back(0);
    leaf_base.push_back(0);
    lod_data.push_back({});
    return idx;
  }

  void svdag_t::build_node(
    const std::vector<build_tri_t>& tris,
    const std::vector<fm::d3::aabb_t>& tri_bounds,
    const std::vector<std::uint32_t>& tri_idx,
    const std::vector<gpu_mat_t>& mats,
    const std::vector<fan::model::cpu_texture_t>& textures,
    fan::vec3i o, int size, std::uint32_t node_idx) 
{
    leaf_base[node_idx] = leaf_data.size();
    int hs = size / 2;
    std::uint32_t vm = 0, lm = 0;
    child_info_t children[8];
    gpu_lod_data_t child_lod[8];
    bool child_lod_valid[8] {};
    fan::vec3 hsv3(hs * 0.5f + 0.001f);
    bool mat_set = false;
    std::uint16_t leaf_mat = 0;
    int non_leaf_count = 0;

    for (int i = 0; i < 8; ++i) {
      auto& c = children[i];
      c.c = o + fan::vec3i(i & 1, i >> 1 & 1, i >> 2 & 1) * hs;
      fan::vec3 child_min(c.c), child_max = child_min + fan::vec3(hs);
      fan::vec3 center = child_min + fan::vec3(hs * 0.5f);

      for (std::uint32_t ti : tri_idx) {
        if (!fm::d3::aabb_intersects_aabb(tri_bounds[ti], child_min, child_max)) { continue; }
        if (fm::d3::triangle_intersects_aabb(tris[ti].v0, tris[ti].v1, tris[ti].v2, center, hsv3)) {
          if (hs <= 2 && !alpha_allows_cell(tris[ti], textures, child_min, hs, 0.2f)) { continue; }
          c.tris.push_back(ti);
        }
      }

      if (c.tris.empty()) { continue; }

      if (hs == 1) {
        std::uint32_t best_ti = c.tris[0];
        f32_t best_area = -1.f;
        for (std::uint32_t ti : c.tris) {
          const auto& bt = tris[ti];
          fan::vec3 e0 = bt.v1 - bt.v0;
          fan::vec3 e1 = bt.v2 - bt.v0;
          fan::vec3 n = e0.cross(e1);
          if (f32_t area = n.dot(n); area > best_area) {
            best_area = area;
            best_ti = ti;
          }
        }

        child_lod[i] = make_cell_lod_data(tris, c.tris, mats, textures, child_min, hs);
        child_lod_valid[i] = child_lod[i].albedo.w > 0.5f && child_lod[i].normal.w > 0.5f;

        if (!child_lod_valid[i]) {
          continue;
        }

        if (!mat_set) {
          leaf_mat = tris[c.tris[0]].mat_id;
          mat_set = true;
        }

        vm |= 1u << i;
        lm |= 1u << i;

        const auto& t = tris[best_ti];
        leaf_data.push_back({fan::vec4(t.v0, 1.f), fan::vec4(t.v1, 1.f), fan::vec4(t.v2, 1.f), t.c0, t.c1, t.c2, t.uv0, t.uv1, t.uv2, t.mat_id, {}});
      }
      else {
        ++non_leaf_count;
      }
    }

    if (non_leaf_count) {
      std::uint32_t first_child = nodes.size();
      child_ptrs[node_idx] = first_child;
      for (int i = 0; i < non_leaf_count; ++i) { new_node(); }

      int slot = 0;
      int packed_slot = 0;
      for (int i = 0; i < 8; ++i) {
        if (children[i].tris.empty() || (lm & (1u << i))) { continue; }

        std::uint32_t child_node_idx = first_child + slot;
        build_node(tris, tri_bounds, children[i].tris, mats, textures, children[i].c, hs, child_node_idx);

        if ((nodes[child_node_idx] & 0xFFu) != 0) {
          std::uint32_t packed_idx = first_child + packed_slot;
          if (packed_idx != child_node_idx) {
            nodes[packed_idx] = nodes[child_node_idx];
            child_ptrs[packed_idx] = child_ptrs[child_node_idx];
            leaf_base[packed_idx] = leaf_base[child_node_idx];
            lod_data[packed_idx] = lod_data[child_node_idx];
          }

          vm |= 1u << i;
          child_lod[i] = lod_data[packed_idx];
          child_lod_valid[i] = child_lod[i].albedo.w > 0.5f && child_lod[i].normal.w > 0.5f;
          ++packed_slot;
        }

        ++slot;
      }
    }

    nodes[node_idx] = vm | (lm << 8) | (std::uint32_t(leaf_mat) << 16);
    lod_data[node_idx] = merge_lod_data(child_lod, child_lod_valid);
  }

  void svdag_t::build_from_mesh(const std::vector<build_tri_t>& tris, int res, const std::vector<gpu_mat_t>& mats, const std::vector<fan::model::cpu_texture_t>& textures) {
    nodes.clear(); child_ptrs.clear(); leaf_base.clear(); leaf_data.clear(); lod_data.clear();
    if (tris.empty()) { return; }
    nodes.reserve(1 << 20); child_ptrs.reserve(1 << 20);
    leaf_base.reserve(1 << 20); leaf_data.reserve(1 << 20); lod_data.reserve(1 << 20);
    std::vector<fm::d3::aabb_t> tri_bounds(tris.size());
    for (std::size_t i = 0; i < tris.size(); ++i) { tri_bounds[i] = fm::d3::triangle_bounds(tris[i].v0, tris[i].v1, tris[i].v2); }
    std::vector<std::uint32_t> root_idx(tris.size());
    std::iota(root_idx.begin(), root_idx.end(), 0);
    new_node();
    build_node(tris, tri_bounds, root_idx, mats, textures, {0,0,0}, res, 0);
  }

  build_data_t extract_build_tris_cpu(const std::string& path, int voxel_res) {
    fan::time::scope_timer_print sp{};
    fan::model::fms_t fms({
      .path = path,
      .load_skeleton = false,
      .load_animations = false,
      .fix_uv_diagonals = false,
      .texture_loading = fan::model::fms_t::properties_t::texture_loading_e::wait,
      .load_texture_types = fan::model::make_texture_filter({
        fan::texture_type::base_color,
        fan::texture_type::diffuse,
        fan::texture_type::ambient,
        fan::texture_type::unknown,
        fan::texture_type::normals
      })
    });

    build_data_t out;
    out.mats.reserve(fms.meshes.size());
    out.tris.reserve(fms.get_triangle_count());

    std::vector<const fan::model::pm_texture_data_t*> active_textures;
    std::unordered_map<std::string, int> texture_layer_map;
    texture_layer_map.reserve(fms.meshes.size() * 2);

    for (std::size_t mi = 0; mi < fms.meshes.size(); ++mi) {
      const auto& m = fms.meshes[mi];
      gpu_mat_t mat;
      mat.base_color = fms.get_material_base_color(mi);

      mat.diffuse_tex = fan::model::get_first_texture_index(
        m,
        {
          fan::texture_type::base_color,
          fan::texture_type::diffuse,
          fan::texture_type::ambient,
          fan::texture_type::unknown
        },
        active_textures,
        texture_layer_map
      );

      mat.normal_tex = fan::model::get_texture_index_fast(
        m.texture_names[fan::texture_type::normals],
        active_textures,
        texture_layer_map
      );

      out.mats.push_back(mat);
    }

    out.textures = fan::model::copy_cpu_textures(active_textures);

    fan::print("textures loaded:", out.textures.size());

    fan::vec3 bmin = fms.aabbmin;
    fan::vec3 bmax = fms.aabbmax;
    out.sf = (voxel_res * 0.85f) / fan::vec4(bmax - bmin, 1.f).max();
    out.go = fan::vec3(voxel_res * 0.075f);

    fms.for_each_triangle([&](const fan::model::fms_t::triangle_ref_t& t) {
      const auto& v0 = *t.vertices[0];
      const auto& v1 = *t.vertices[1];
      const auto& v2 = *t.vertices[2];
      auto xf = [&](const auto& v) { return (v.position - bmin) * out.sf + out.go; };
      out.tris.push_back({
        xf(v0), xf(v1), xf(v2),
        v0.color, v1.color, v2.color,
        v0.uv, v1.uv, v2.uv,
        std::uint16_t(t.mesh_id),
        out.mats[t.mesh_id].diffuse_tex
      });
    });

    out.bmin = bmin;
    out.bmax = bmax;
    return out;
  }

  bool load_svdag_cache(svdag_t& dag, const std::string& path, int voxel_res) {
    if (path.empty()) { return false; }
    std::ifstream f(path, std::ios::binary);
    if (!f) { return false; }
    svdag_cache_header_t h;
    f.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!f || h.magic != 0x47414453 || h.version != 8
         || h.voxel_res != std::uint32_t(voxel_res)
         || h.leaf_size != sizeof(gpu_leaf_data_t)
         || h.lod_size != sizeof(gpu_lod_data_t)) { return false; }
    return file::read(f, dag.nodes,      h.nodes)
        && file::read(f, dag.child_ptrs, h.child_ptrs)
        && file::read(f, dag.leaf_base,  h.leaf_base)
        && file::read(f, dag.leaf_data,  h.leaf_data)
        && file::read(f, dag.lod_data,   h.lod_data);
  }

  void save_svdag_cache(const svdag_t& dag, const std::string& path, int voxel_res) {
    if (path.empty()) { return; }
    auto parent_path = std::filesystem::path(path).parent_path();
    if (!parent_path.empty()) { std::filesystem::create_directories(parent_path); }
    std::ofstream f(path, std::ios::binary);
    if (!f) { return; }
    svdag_cache_header_t h;
    h.voxel_res   = voxel_res;
    h.nodes       = dag.nodes.size();
    h.child_ptrs  = dag.child_ptrs.size();
    h.leaf_base   = dag.leaf_base.size();
    h.leaf_data   = dag.leaf_data.size();
    h.lod_data    = dag.lod_data.size();
    f.write(reinterpret_cast<const char*>(&h), sizeof(h));
    file::write(f, dag.nodes);
    file::write(f, dag.child_ptrs);
    file::write(f, dag.leaf_base);
    file::write(f, dag.leaf_data);
    file::write(f, dag.lod_data);
  }

  svdag_load_result_t load_svdag_cpu(const std::string& path, int voxel_res, const std::string& cache_path) {
    svdag_load_result_t out;
    out.build_data = extract_build_tris_cpu(path, voxel_res);
    if (!load_svdag_cache(out.dag, cache_path, voxel_res)) {
      fan::print("Building SVDAG...");
      fan::time::scope_timer_print sp{};
      out.dag.build_from_mesh(out.build_data.tris, voxel_res, out.build_data.mats, out.build_data.textures);
      save_svdag_cache(out.dag, cache_path, voxel_res);
    }
    else {
      fan::print("Loaded SVDAG from cache.");
    }
    fan::print("SVDAG tris:", out.build_data.tris.size(), "nodes:", out.dag.nodes.size(), "leaves:", out.dag.leaf_data.size());
    return out;
  }

  void svdag_loader_t::start(const std::string& path, int voxel_res, const std::string& cache_path) {
    job = std::async(std::launch::async, [=] {
      return load_svdag_cpu(path, voxel_res, cache_path);
    });
  }

  bool svdag_loader_t::ready() const {
    if (!job.valid()) { return false; }
    return job.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
  }

  svdag_renderer_t::svdag_renderer_t(build_data_t&& bdata, svdag_t&& in_dag, int res)
    : dag(std::move(in_dag)), bmin(bdata.bmin), bmax(bdata.bmax),
      go(bdata.go), sf(bdata.sf), voxel_res(res)
  {
    trace_nr = gloco()->shader_make_compute("shaders/opengl/3D/compute/svdag_tracer.comp");
    tex_array = fan::model::upload_texture_array(bdata.textures);
    fan::print("uploaded texture layers:", bdata.textures.size());
    max_depth = fm::max(1, int(std::round(std::log2(voxel_res))));

    ssbo_nodes.open(gl, GL_SHADER_STORAGE_BUFFER);
    ssbo_nodes.write_buffer(gl, dag.nodes.data(), dag.nodes.size() * sizeof(std::uint32_t));
    ssbo_ptrs.open(gl, GL_SHADER_STORAGE_BUFFER);
    ssbo_ptrs.write_buffer(gl, dag.child_ptrs.data(), dag.child_ptrs.size() * sizeof(std::uint32_t));
    ssbo_mats.open(gl, GL_SHADER_STORAGE_BUFFER);
    ssbo_mats.write_buffer(gl, bdata.mats.data(), bdata.mats.size() * sizeof(gpu_mat_t));
    ssbo_leaf_base.open(gl, GL_SHADER_STORAGE_BUFFER);
    ssbo_leaf_base.write_buffer(gl, dag.leaf_base.data(), dag.leaf_base.size() * sizeof(std::uint32_t));
    ssbo_leaf_data.open(gl, GL_SHADER_STORAGE_BUFFER);
    ssbo_leaf_data.write_buffer(gl, dag.leaf_data.data(), dag.leaf_data.size() * sizeof(gpu_leaf_data_t));
    ssbo_lod_data.open(gl, GL_SHADER_STORAGE_BUFFER);
    ssbo_lod_data.write_buffer(gl, dag.lod_data.data(), dag.lod_data.size() * sizeof(gpu_lod_data_t));

    vao.open(gl);

    if (tex_array) {
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D_ARRAY, tex_array);
      trace_nr.set_value(gl, "u_tex_array", 0);
    }
  }

  svdag_renderer_t::~svdag_renderer_t() {
    ssbo_nodes.close(gl); ssbo_ptrs.close(gl); ssbo_mats.close(gl);
    ssbo_leaf_base.close(gl); ssbo_leaf_data.close(gl); ssbo_lod_data.close(gl);
    vao.close(gl);
    if (tex_array) { glDeleteTextures(1, &tex_array); }
  }

  void svdag_renderer_t::render(fan::vec3 cam_pos, const fan::mat4& inv_view_proj, const fan::vec3& sun_dir) {
    fan::vec2i window_res = gloco()->window.get_size();
    f32_t scale = std::clamp(render_scale, 0.25f, 1.f);
    fan::vec2i res(
      std::max(1, int(std::round(window_res.x * scale))),
      std::max(1, int(std::round(window_res.y * scale)))
    );

    if (!img_screen || res != img_screen.get_size()) {
      img_screen.remove();
      img_screen = image_t(
        {.data = nullptr, .size = res},
        {.internal_format = image_format_e::rgba8, .format = image_format_e::rgba, .type = data_type_e::fan_unsigned_byte}
      );
    }

    img_screen.bind(0, GL_WRITE_ONLY, GL_RGBA8);
    ssbo_nodes.bind_base(gl, 1); ssbo_ptrs.bind_base(gl, 2); ssbo_mats.bind_base(gl, 3);
    ssbo_leaf_base.bind_base(gl, 4); ssbo_leaf_data.bind_base(gl, 5); ssbo_lod_data.bind_base(gl, 6);

    if (tex_array) {
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D_ARRAY, tex_array);
    }

    trace_nr.set_value(gl, "u_tex_array", 0);
    trace_nr.set_value(gl, "u_inv_view_proj", inv_view_proj);
    trace_nr.set_value(gl, "u_cam_pos", (cam_pos - bmin) * sf + go);
    trace_nr.set_value(gl, "u_resolution", fan::vec2(res.x, res.y));
    trace_nr.set_value(gl, "u_voxel_res", voxel_res);
    trace_nr.set_value(gl, "u_sun_dir", sun_dir.normalize());
    trace_nr.set_value(gl, "u_max_depth", max_depth);
    trace_nr.set_value(gl, "u_lod_bias", lod_bias);
    trace_nr.set_value(gl, "u_ao_quality", ao_quality);
    trace_nr.set_value(gl, "u_debug_heatmap", debug_heatmap);

    glDispatchCompute((res.x + 7) / 8, (res.y + 7) / 8, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

    auto set_blend = fan::make_restore_flag(false, [&](bool f) { gl.set_depth_test(f); gl.set_blending(f); });
    vao.bind(gl);
    glActiveTexture(GL_TEXTURE0);
    img_screen.bind(0);
    gloco()->shaders.blit.set_value(*gloco(), "u_tex", 0);
    glDrawArrays(GL_TRIANGLES, 0, 3);
  }

  std::unique_ptr<svdag_renderer_t> svdag_loader_t::finish(int voxel_res) {
    auto result = job.get();
    return std::make_unique<svdag_renderer_t>(std::move(result.build_data), std::move(result.dag), voxel_res);
  }
}

#undef gl

#endif

#endif