#include <fan/graphics/gl_api.h>
#include <fan/utility.h>

import std;
import fan;

using namespace fan::graphics;
namespace fm = fan::math;

#define gl (gloco()->get_context().gl)

struct gpu_mat_t {
  fan::color base_color{1.f};
  int diffuse_tex = -1;
  int normal_tex = -1;
  f32_t tex_scale = 1.f;
  std::uint32_t pad;
};

struct gpu_leaf_data_t {
  fan::vec4 v0{}, v1{}, v2{};
  fan::vec2 uv0{}, uv1{}, uv2{};
  std::uint32_t mat_id{};
  std::uint32_t pad[5]{};
};
static_assert(sizeof(gpu_leaf_data_t) == 96);

struct svdag_t {
  struct build_tri_t { fan::vec3 v0, v1, v2; fan::vec2 uv0, uv1, uv2; std::uint16_t mat_id; };
  struct child_info_t { fan::vec3i c; std::vector<std::uint32_t> tris; };

  std::uint32_t build_node(const std::vector<build_tri_t>& tris, const std::vector<std::uint32_t>& tri_idx, fan::vec3i o, int size) {
    std::uint32_t node_idx = nodes.size();
    nodes.push_back(0); child_ptrs.push_back(0); leaf_base.push_back(0);
    leaf_base[node_idx] = leaf_data.size();

    int hs = size / 2;
    std::uint32_t vm = 0, lm = 0;
    child_info_t children[8];
    fan::vec3 hsv3(hs * 0.5f + 0.001f);

    bool mat_set = false;
    std::uint16_t leaf_mat = 0;

    int non_leaf_count = 0;

    for (int i = 0; i < 8; ++i) {
      auto& c = children[i];
      c.c[0] = o[0] + ((i & 1) ? hs : 0);
      c.c[1] = o[1] + ((i >> 1 & 1) ? hs : 0);
      c.c[2] = o[2] + ((i >> 2 & 1) ? hs : 0);
      fan::vec3 center(c.c + hs * 0.5f);
      for (std::uint32_t ti : tri_idx) {
        if (fm::d3::triangle_intersects_aabb(tris[ti].v0, tris[ti].v1, tris[ti].v2, center, hsv3)) {
          c.tris.push_back(ti);
        }
      }
      if (c.tris.empty()) { continue; }
      vm |= (1u << i);
      if (hs == 1) {
        lm |= (1u << i);
        if (!mat_set) { leaf_mat = tris[c.tris[0]].mat_id; mat_set = true; }
        auto& t = tris[c.tris[0]];
        leaf_data.push_back({fan::vec4(t.v0, 1.f), fan::vec4(t.v1, 1.f), fan::vec4(t.v2, 1.f), t.uv0, t.uv1, t.uv2, t.mat_id, {}});
      }
      else {
        ++non_leaf_count;
      }
    }

    nodes[node_idx] = vm | (lm << 8) | (std::uint32_t(leaf_mat) << 16);
    if (!non_leaf_count) { return node_idx; }

    std::uint32_t first_child = nodes.size();
    child_ptrs[node_idx] = first_child;
    nodes.resize(nodes.size() + non_leaf_count, 0);
    child_ptrs.resize(child_ptrs.size() + non_leaf_count, 0);
    leaf_base.resize(leaf_base.size() + non_leaf_count, 0);

    int slot = 0;
    for (int i = 0; i < 8; ++i) {
      if (!children[i].tris.empty() && !(lm & (1u << i))) {
        std::uint32_t cr = build_node(tris, children[i].tris, children[i].c, hs);
        nodes[first_child + slot] = nodes[cr];
        child_ptrs[first_child + slot] = child_ptrs[cr];
        leaf_base[first_child + slot] = leaf_base[cr];
        ++slot;
      }
    }
    return node_idx;
  }

  void build_from_mesh(const std::vector<build_tri_t>& tris, int res) {
    nodes.clear(); child_ptrs.clear(); leaf_base.clear(); leaf_data.clear();
    if (tris.empty()) { return; }
    nodes.reserve(1 << 20); child_ptrs.reserve(1 << 20);
    leaf_base.reserve(1 << 20); leaf_data.reserve(1 << 20);
    std::vector<std::uint32_t> root_idx(tris.size());
    std::iota(root_idx.begin(), root_idx.end(), 0);
    build_node(tris, root_idx, 0, res);
  }

  bool save(const std::string& path) const {
    std::uint32_t n = nodes.size(), ld = leaf_data.size();
    std::string buf(8 + n * 12 + ld * sizeof(gpu_leaf_data_t), '\0');
    char* p = buf.data();
    auto write = [&](const void* src, std::size_t sz) { std::memcpy(p, src, sz); p += sz; };
    write(&n, 4); write(&ld, 4);
    write(nodes.data(), n * 4); write(child_ptrs.data(), n * 4);
    write(leaf_base.data(), n * 4); write(leaf_data.data(), ld * sizeof(gpu_leaf_data_t));
    return fan::io::file::write(path, buf, std::ios::binary);
  }

  bool load(const std::string& path) {
    std::string buf;
    if (fan::io::file::read(path, &buf) || buf.size() < 8) { return false; }
    std::uint32_t n = 0, ld = 0;
    std::memcpy(&n, buf.data(), 4); std::memcpy(&ld, buf.data() + 4, 4);
    if (buf.size() < 8 + n * 12 + ld * sizeof(gpu_leaf_data_t)) { return false; }
    nodes.resize(n); child_ptrs.resize(n); leaf_base.resize(n); leaf_data.resize(ld);
    const char* p = buf.data() + 8;
    auto read = [&](void* dst, std::size_t sz) { std::memcpy(dst, p, sz); p += sz; };
    read(nodes.data(), n * 4); read(child_ptrs.data(), n * 4);
    read(leaf_base.data(), n * 4); read(leaf_data.data(), ld * sizeof(gpu_leaf_data_t));
    return true;
  }

  std::vector<std::uint32_t> nodes, child_ptrs, leaf_base;
  std::vector<gpu_leaf_data_t> leaf_data;
};

struct build_data_t {
  std::vector<svdag_t::build_tri_t> tris;
  std::vector<gpu_mat_t> mats;
  GLuint tex_array = 0;
  fan::vec3 bmin, bmax, go;
  f32_t sf;
};

static build_data_t extract_build_tris(const std::string& path, int voxel_res) {
  fan::model::fms_t fms({.path = path});
  build_data_t out_data({.mats{!fms.meshes.size()}});
  std::vector<const fan::model::pm_texture_data_t*> active_textures;
  for (std::size_t mi = 0; mi < fms.meshes.size(); ++mi) {
    const auto& m = fms.meshes[mi];
    gpu_mat_t mat;
    if (mi < fms.material_data_vector.size()) {
      const auto& bc = fms.material_data_vector[mi].color[12];
      const auto& dc = fms.material_data_vector[mi].color[1];
      auto not_white = [](const auto& c) { return c != fan::colors::white; };
      mat.base_color = not_white(bc) ? bc : (not_white(dc) ? dc : mat.base_color);
    }
    mat.diffuse_tex = fan::model::get_texture_index(m.texture_names[fan::texture_type::diffuse], active_textures);
    mat.normal_tex = fan::model::get_texture_index(m.texture_names[fan::texture_type::normals], active_textures);
    out_data.mats.push_back(std::move(mat));
  }

  if (!active_textures.empty()) {
    glGenTextures(1, &out_data.tex_array);
    glBindTexture(GL_TEXTURE_2D_ARRAY, out_data.tex_array);
    int tsz = 1024;
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8, tsz, tsz, active_textures.size(), 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    std::vector<std::uint8_t> resized(tsz * tsz * 4, 255);
    for (int i = 0; i < (int)active_textures.size(); ++i) {
      auto td = active_textures[i];
      for (int y = 0; y < tsz; ++y) {
        for (int x = 0; x < tsz; ++x) {
          fan::vec2 s = (fan::vec2(x, y) / (tsz - 1) * td->size).clamp({0}, td->size - 1);
          int sidx = (s.y * td->size.x + s.x) * td->channels, didx = (y * tsz + x) * 4;
          for(int c = 0; c < 4; ++c) { resized[didx+c] = (c < td->channels) ? td->data[sidx+c] : 255; }
        }
      }
      glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, i, tsz, tsz, 1, GL_RGBA, GL_UNSIGNED_BYTE, resized.data());
    }
    glGenerateMipmap(GL_TEXTURE_2D_ARRAY);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_REPEAT);
  }

  fan::vec3 bmin = fan::vec3::vmax(), bmax = -bmin;
  for (const auto& m : fms.meshes) {
    for (const auto& v : m.vertices) {
      bmin = bmin.min(v.position);
      bmax = bmax.max(v.position);
    }
  }
  fan::vec3 sz = bmax - bmin;
  out_data.sf = (voxel_res * 0.85f) / fm::max(sz.x, sz.y, sz.z, 1.f);
  out_data.go = fan::vec3(voxel_res * 0.075f);

  for (std::size_t mi = 0; mi < fms.meshes.size(); ++mi) {
    const auto& m = fms.meshes[mi];
    std::uint16_t mat_id = mi;
    for (std::size_t i = 0; i + 2 < m.indices.size(); i += 3) {
      auto& v0 = m.vertices[m.indices[i]], &v1 = m.vertices[m.indices[i+1]], &v2 = m.vertices[m.indices[i+2]];
      auto xf = [&](const auto& v) { return (v.position - bmin) * out_data.sf + out_data.go; };
      out_data.tris.push_back({xf(v0), xf(v1), xf(v2), v0.uv, v1.uv, v2.uv, mat_id});
    }
  }
  out_data.bmin = bmin; out_data.bmax = bmax;
  return out_data;
}

struct svdag_renderer_t {
  svdag_renderer_t(const build_data_t& bdata, int res, const std::string& cache_path)
    : bmin(bdata.bmin), bmax(bdata.bmax), go(bdata.go), sf(bdata.sf), tex_array(bdata.tex_array), voxel_res(res)
  {
    trace_nr = gloco()->shader_make_compute("shaders/opengl/3D/compute/svdag_tracer.comp");

    if (cache_path.empty() || !dag.load(cache_path)) {
      fan::print("Building SVDAG...");
      fan::time::scope_timer_print sp{};
      dag.build_from_mesh(bdata.tris, voxel_res);
      if (!cache_path.empty()) {
        dag.save(cache_path);
      }
    } else {
      fan::print("Loaded SVDAG from cache.");
    }

    ssbo_nodes.open(gl, GL_SHADER_STORAGE_BUFFER);
    ssbo_nodes.write_buffer(gl, dag.nodes.data(), dag.nodes.size() * sizeof(decltype(dag.nodes)::value_type));
    ssbo_ptrs.open(gl, GL_SHADER_STORAGE_BUFFER);
    ssbo_ptrs.write_buffer(gl, dag.child_ptrs.data(), dag.child_ptrs.size() * sizeof(decltype(dag.child_ptrs)::value_type));
    ssbo_mats.open(gl, GL_SHADER_STORAGE_BUFFER);
    ssbo_mats.write_buffer(gl, bdata.mats.data(), bdata.mats.size() * sizeof(gpu_mat_t));
    ssbo_leaf_base.open(gl, GL_SHADER_STORAGE_BUFFER);
    ssbo_leaf_base.write_buffer(gl, dag.leaf_base.data(), dag.leaf_base.size() * sizeof(decltype(dag.leaf_base)::value_type));
    ssbo_leaf_data.open(gl, GL_SHADER_STORAGE_BUFFER);
    ssbo_leaf_data.write_buffer(gl, dag.leaf_data.data(), dag.leaf_data.size() * sizeof(gpu_leaf_data_t));
    vao.open(gl);
  }

  ~svdag_renderer_t() {
    ssbo_nodes.close(gl); ssbo_ptrs.close(gl); ssbo_mats.close(gl);
    ssbo_leaf_base.close(gl); ssbo_leaf_data.close(gl);
    vao.close(gl);
    if (tex_array) { glDeleteTextures(1, &tex_array); }
  }

  void render(fan::vec3 cam_pos, const fan::mat4& inv_view_proj, const fan::vec3& sun_dir) {
    fan::vec2i res = fan::window::get_size();
    if (!img_screen || (img_screen && res != img_screen.get_size())) {
      img_screen.remove();
      img_screen = image_t(
        {.data = nullptr, .size = res}, 
        {.internal_format = image_format_e::rgba8, .format = image_format_e::rgba, .type = data_type_e::fan_unsigned_byte}
      );
    }
    img_screen.bind(0, GL_WRITE_ONLY, GL_RGBA8);
    ssbo_nodes.bind_base(gl, 1); ssbo_ptrs.bind_base(gl, 2); ssbo_mats.bind_base(gl, 3);
    ssbo_leaf_base.bind_base(gl, 4); ssbo_leaf_data.bind_base(gl, 5);

    if (tex_array) {
      glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D_ARRAY, tex_array);
      gloco()->shader_set_value(trace_nr, "u_tex_array", 1);
    }

    trace_nr.set_value(gl, "u_inv_view_proj", inv_view_proj);
    trace_nr.set_value(gl, "u_cam_pos", (cam_pos - bmin) * sf + go);
    trace_nr.set_value(gl, "u_resolution", fan::vec2(res.x, res.y));
    trace_nr.set_value(gl, "u_voxel_res", voxel_res);
    trace_nr.set_value(gl, "u_sun_dir", sun_dir.normalize());
    trace_nr.set_value(gl, "u_max_depth", fm::max(1, (int)std::round(std::log2(voxel_res))));

    glDispatchCompute((res.x + 7) / 8, (res.y + 7) / 8, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

    vao.bind(gl); img_screen.bind(0);
    gloco()->shaders.blit.set_value(*gloco(), "u_tex", 0);
    glDrawArrays(GL_TRIANGLES, 0, 3);
  }

  shader_t trace_nr;
  fan::opengl::core::gpu_buffer_t ssbo_nodes, ssbo_ptrs, ssbo_mats, ssbo_leaf_base, ssbo_leaf_data;
  fan::opengl::core::vao_t vao;
  image_t img_screen;
  svdag_t dag;
  fan::vec3 bmin, bmax, go;
  f32_t sf;
  GLuint tex_array = 0;
  int voxel_res;
};

struct game_t {
  game_t() : renderer(extract_build_tris("models/oldman.gltf", voxel_res), voxel_res, "") {
    engine.camera_set_position(engine.perspective_render_view, {209.4, -9.7, 122.5});
    auto& cam = engine.camera_get(engine.perspective_render_view);
    cam.yaw = -118.1; cam.pitch = 0.3;

    static auto mouse_motion_handle = engine.window.add_mouse_motion_callback([&](const auto& d) {
      if (gui::is_any_item_active() || !engine.is_mouse_down(fan::mouse_right)) { return; }
      engine.camera_rotate(engine.perspective_render_view, d.motion);
      engine.camera_get(engine.perspective_render_view).update_view();
    });

    gloco()->add_custom_draw([&] {
      auto& c = engine.camera_get(engine.perspective_render_view);
      renderer.render(c.position, (c.projection * c.view).inverse(), sun_direction);
    });

    engine.loop([&](f32_t dt) {
      engine.camera_move(move_speed);
      if (auto h = gui::hud_interactive("##ctrl", 0.f)) {
        gui::text(engine.perspective_render_view.get_camera());
        gui::drag("sun", &sun_direction);
        gui::drag("speed", &move_speed);
      }
    });
  }

  engine_t engine;
  int voxel_res = 64;
  svdag_renderer_t renderer;
  fan::vec3 sun_direction{0.6f, 0.9f, 0.4f};
  f32_t move_speed = 1000.f;
};

int main() {
  game_t game;
}