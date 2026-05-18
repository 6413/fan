#include <fan/graphics/gl_api.h>
#include <fan/utility.h>

import std;
import fan;

using namespace fan::graphics;
namespace fm = fan::math;

static constexpr f32_t world_size = 512.f;

#define gl (gloco()->get_context().gl)

struct svdag_t {
  struct build_tri_t { fan::vec3 v0, v1, v2, color; };

  std::uint32_t build_node(const std::vector<build_tri_t>& tris,
    const std::vector<std::uint32_t>& tri_idx, int ox, int oy, int oz, int size)
  {
    std::uint32_t node_idx = nodes.size();
    nodes.push_back(0);
    child_ptrs.push_back(0);

    int hs = size / 2;
    std::uint32_t vm = 0, lm = 0;
    std::uint16_t leaf_color = 0;

    struct child_info_t { int cx, cy, cz; std::vector<std::uint32_t> tris; };
    child_info_t children[8];

    f32_t hsv = hs * 0.5f + 0.001f;
    fan::vec3 hsv3(hsv);

    for (int i = 0; i < 8; ++i) {
      int cx = ox + ((i & 1) ? hs : 0), cy = oy + ((i >> 1 & 1) ? hs : 0), cz = oz + ((i >> 2 & 1) ? hs : 0);
      fan::vec3 center(cx + hs * 0.5f, cy + hs * 0.5f, cz + hs * 0.5f);
      for (std::uint32_t ti : tri_idx) {
        if (fm::d3::triangle_intersects_aabb(tris[ti].v0, tris[ti].v1, tris[ti].v2, center, hsv3)) {
          children[i].tris.push_back(ti);
        }
      }
      if (children[i].tris.empty()) { continue; }
      vm |= (1u << i);
      children[i].cx = cx; children[i].cy = cy; children[i].cz = cz;
      if (hs == 1) {
        lm |= (1u << i);
        if (!leaf_color) { leaf_color = fan::pack_rgb565(tris[children[i].tris[0]].color); }
      }
    }

    nodes[node_idx] = vm | (lm << 8) | (std::uint32_t(leaf_color) << 16);

    int non_leaf_count = 0;
    for (int i = 0; i < 8; ++i) {
      if (!children[i].tris.empty() && !(lm & (1u << i))) { ++non_leaf_count; }
    }
    if (!non_leaf_count) { return node_idx; }

    std::uint32_t first_child = nodes.size();
    child_ptrs[node_idx] = first_child;
    nodes.resize(nodes.size() + non_leaf_count, 0);
    child_ptrs.resize(child_ptrs.size() + non_leaf_count, 0);

    int slot = 0;
    for (int i = 0; i < 8; ++i) {
      if (!children[i].tris.empty() && !(lm & (1u << i))) {
        std::uint32_t cr = build_node(tris, children[i].tris, children[i].cx, children[i].cy, children[i].cz, hs);
        nodes[first_child + slot] = nodes[cr];
        child_ptrs[first_child + slot] = child_ptrs[cr];
        ++slot;
      }
    }
    return node_idx;
  }

  void build_from_mesh(const std::vector<build_tri_t>& tris, int res) {
    nodes.clear(); child_ptrs.clear();
    if (tris.empty()) { return; }
    nodes.reserve(1 << 20); child_ptrs.reserve(1 << 20);
    std::vector<std::uint32_t> root_idx(tris.size());
    std::iota(root_idx.begin(), root_idx.end(), 0);
    build_node(tris, root_idx, 0, 0, 0, res);
  }

  bool save(const std::string& path) const {
    std::uint32_t n = nodes.size();
    std::string buf(4 + n * 8, '\0');
    std::memcpy(buf.data(), &n, 4);
    std::memcpy(buf.data() + 4, nodes.data(), n * 4);
    std::memcpy(buf.data() + 4 + n * 4, child_ptrs.data(), n * 4);
    return fan::io::file::write(path, buf, std::ios::binary);
  }

  bool load(const std::string& path) {
    std::string buf;
    if (fan::io::file::read(path, &buf) || buf.size() < 4) { return false; }
    std::uint32_t n = 0;
    std::memcpy(&n, buf.data(), 4);
    if (buf.size() < 4 + n * 8) { return false; }
    nodes.resize(n); child_ptrs.resize(n);
    std::memcpy(nodes.data(), buf.data() + 4, n * 4);
    std::memcpy(child_ptrs.data(), buf.data() + 4 + n * 4, n * 4);
    return true;
  }

  std::uint32_t count_voxels() const {
    if (nodes.empty()) { return 0; }
    std::function<std::uint32_t(std::uint32_t)> count = [&](std::uint32_t ni) -> std::uint32_t {
      std::uint32_t hdr = nodes[ni], vm = hdr & 0xFF, lm = (hdr >> 8) & 0xFF;
      std::uint32_t total = std::popcount(lm), ptr = child_ptrs[ni], slot = 0;
      for (int i = 0; i < 8; ++i) {
        if ((vm & (1u << i)) && !(lm & (1u << i))) { total += count(ptr + slot++); }
      }
      return total;
    };
    return count(0);
  }

  std::vector<std::uint32_t> nodes, child_ptrs;
};

struct svdag_renderer_t {
  svdag_renderer_t(const std::vector<svdag_t::build_tri_t>& geo, int res, const std::string& cache_path)
    : voxel_res(res)
  {
    blit_nr = gloco()->shader_create();
    gloco()->shader_set_vertex(blit_nr, "", R"(#version 430 core
out vec2 v_uv; void main() { v_uv = vec2((gl_VertexID<<1)&2, gl_VertexID&2); gl_Position = vec4(v_uv*2.0-1.0, 0.0, 1.0); })");
    gloco()->shader_set_fragment(blit_nr, "", R"(#version 430 core
uniform sampler2D u_tex; in vec2 v_uv; out vec4 out_color; void main() { out_color = texture(u_tex, v_uv); })");
    gloco()->shader_compile(blit_nr);

    std::string_view comp_path = "shaders/opengl/3D/compute/svdag_tracer.comp";
    std::string comp_src; fan::io::file::read(comp_path, &comp_src);
    trace_nr = gloco()->shader_create();
    gloco()->shader_set_compute(trace_nr, comp_path, comp_src);
    gloco()->shader_compile(trace_nr);

    if (!dag.load(cache_path)) {
      fan::print("Building SVDAG...");
      fan::time::scope_timer_print sp{};
      dag.build_from_mesh(geo, voxel_res);
      dag.save(cache_path);
    } else {
      fan::print("Loaded SVDAG from cache.");
    }
    fan::print("SVDAG nodes:", dag.nodes.size(), "voxels:", dag.count_voxels());

    ssbo_nodes.open(gl, GL_SHADER_STORAGE_BUFFER);
    ssbo_nodes.write_buffer(gl, dag.nodes.data(), dag.nodes.size() * 4);
    ssbo_ptrs.open(gl, GL_SHADER_STORAGE_BUFFER);
    ssbo_ptrs.write_buffer(gl, dag.child_ptrs.data(), dag.child_ptrs.size() * 4);
    vao.open(gl);
  }

  ~svdag_renderer_t() { ssbo_nodes.close(gl); ssbo_ptrs.close(gl); vao.close(gl); }

  void rebuild_texture() {
    fan::vec2i s = fan::window::get_size();
    if (s == res) { return; }
    res = s;
    img_screen.remove();
    img_screen = image_t({.data=nullptr, .size=res}, {
      .internal_format = image_format_e::rgba8,
      .format          = image_format_e::rgba,
      .type            = data_type_e::fan_unsigned_byte,
    });
  }

  void render(fan::vec3 cam_pos, const fan::mat4& inv_view_proj, const fan::vec3& sun_dir) {
    rebuild_texture();
    gl.image_bind(img_screen, 0, GL_WRITE_ONLY, GL_RGBA8);
    ssbo_nodes.bind_base(gl, 1);
    ssbo_ptrs.bind_base(gl, 2);

    gloco()->shader_set_value(trace_nr, "u_inv_view_proj", inv_view_proj);
    gloco()->shader_set_value(trace_nr, "u_cam_pos", cam_pos);
    gloco()->shader_set_value(trace_nr, "u_resolution", fan::vec2(res.x, res.y));
    gloco()->shader_set_value(trace_nr, "u_voxel_scale", world_size);
    gloco()->shader_set_value(trace_nr, "u_voxel_res", voxel_res);
    gloco()->shader_set_value(trace_nr, "u_sun_dir", sun_dir.normalize());
    gloco()->shader_set_value(trace_nr, "u_max_depth", fm::max(1, (int)std::round(std::log2(voxel_res))));

    glDispatchCompute((res.x + 7) / 8, (res.y + 7) / 8, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

    auto set_blend = fan::make_restore_flag(false, [&](bool f) { gl.set_depth_test(f); gl.set_blending(f); });
    vao.bind(gl);
    glActiveTexture(GL_TEXTURE0);
    img_screen.bind();
    gloco()->shader_set_value(blit_nr, "u_tex", 0);
    glDrawArrays(GL_TRIANGLES, 0, 3);
  }

  shader_t trace_nr, blit_nr;
  fan::opengl::core::gpu_buffer_t ssbo_nodes, ssbo_ptrs;
  fan::opengl::core::vao_t vao;
  image_t img_screen;
  fan::vec2i res{0, 0};
  svdag_t dag;
  int voxel_res;
};

static std::vector<svdag_t::build_tri_t> extract_build_tris(const std::string& path, int voxel_res) {
  auto meshes = load_meshes({.path = path});
  fan::vec3 bmin(std::numeric_limits<f32_t>::max()), bmax(-std::numeric_limits<f32_t>::max());
  for (const auto& m : meshes) {
    for (const auto& v : m.vertices) {
      bmin = fan::vec3(fm::min(bmin.x, v.position.x), fm::min(bmin.y, v.position.y), fm::min(bmin.z, v.position.z));
      bmax = fan::vec3(fm::max(bmax.x, v.position.x), fm::max(bmax.y, v.position.y), fm::max(bmax.z, v.position.z));
    }
  }
  fan::vec3 sz = bmax - bmin;
  f32_t gv = voxel_res * 0.075f, sf = (voxel_res * 0.85f) / fm::max(sz.x, sz.y, sz.z, 1.f);
  fan::vec3 go(gv);

  std::vector<svdag_t::build_tri_t> out;
  for (const auto& m : meshes) {
    for (std::size_t i = 0; i + 2 < m.indices.size(); i += 3) {
      auto xf = [&](const fan::model::vertex_t& v) { return (v.position - bmin) * sf + go; };
      auto& v0 = m.vertices[m.indices[i]], &v1 = m.vertices[m.indices[i+1]], &v2 = m.vertices[m.indices[i+2]];
      fan::vec3 p0 = xf(v0), p1 = xf(v1), p2 = xf(v2), c = (p0+p1+p2)/3.f;
      fan::vec3 col = fm::centroid(
        fan::vec3(v0.color.x, v0.color.y, v0.color.z),
        fan::vec3(v1.color.x, v1.color.y, v1.color.z),
        fan::vec3(v2.color.x, v2.color.y, v2.color.z)
      );
      out.push_back({p0, p1, p2, col});
    }
  }
  return out;
}

struct game_t {
  game_t() : engine({.window_size = {1280, 720}}) {
    renderer = std::make_unique<svdag_renderer_t>(
      extract_build_tris("models/objworld.obj", voxel_res), voxel_res, "svdag_cache.bin");

    engine.camera_set_position(engine.perspective_render_view, {564.9f, 438.4f, 597.9f});
    auto& cam = engine.camera_get(engine.perspective_render_view);
    cam.yaw = -139.10f; cam.pitch = -18.00f;

    engine.window.add_mouse_motion_callback([&](const auto& d) {
      if (gui::is_any_item_active() || !engine.is_mouse_down(fan::mouse_right)) { return; }
      engine.camera_rotate(engine.perspective_render_view, d.motion);
      engine.camera_get(engine.perspective_render_view).update_view();
    });

    gloco()->add_custom_draw([&] {
      auto& c = engine.camera_get(engine.perspective_render_view);
      renderer->render(c.position, (c.projection * c.view).inverse(), sun_direction);
    });

    engine.loop([&](f32_t dt) {
      engine.camera_move(move_speed);
      auto& c = engine.camera_get(engine.perspective_render_view);
      if (auto h = gui::hud_interactive("##ctrl", 0.f)) {
        gui::text(std::format("pos: {:.1f} {:.1f} {:.1f}\nyaw: {:.2f}  pitch: {:.2f}",
          c.position.x, c.position.y, c.position.z, c.yaw, c.pitch));
        gui::drag("sun", &sun_direction);
        gui::drag("speed", &move_speed);
      }
    });
  }

  engine_t engine;
  std::unique_ptr<svdag_renderer_t> renderer;
  f32_t move_speed = 4000.f;
  fan::vec3 sun_direction{0.6f, 0.9f, 0.4f};
  int voxel_res = 4096;
};

int main() {
  game_t game;
}