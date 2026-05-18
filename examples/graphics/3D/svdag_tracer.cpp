#include <fan/graphics/gl_api.h>
#include <fan/utility.h>

import std;
import fan;

using namespace fan::graphics;
namespace fm = fan::math;

static constexpr f32_t world_size = 512.f;

// for shorter usage
#define gl (gloco()->get_context().gl)

struct svdag_t {
  struct child_t { int x, y, z; std::vector<std::size_t> tri_indices; };

  void build_from_mesh(const std::vector<fan::triangle_t>& tris, int res) {
    nodes.clear();
    if (tris.empty()) { return; }
    std::vector<std::size_t> root_indices(tris.size());
    std::iota(root_indices.begin(), root_indices.end(), 0);
    build_node(tris, root_indices, 0, 0, 0, res);
  }

  std::uint32_t build_node(const std::vector<fan::triangle_t>& tris, const std::vector<std::size_t>& tri_indices, int ox, int oy, int oz, int size) {
    std::uint32_t idx = nodes.size();
    nodes.push_back(0);
    int hs = size / 2;
    std::uint8_t vm = 0, lm = 0;
    std::vector<child_t> non_leaves;
    non_leaves.reserve(8);
    f32_t hs_val = hs * 0.5f + 0.001f;
    fan::vec3 hsv(hs_val, hs_val, hs_val);
    for (int i = 0; i < 8; ++i) {
      int cx = ox + ((i & 1) ? hs : 0), cy = oy + ((i >> 1 & 1) ? hs : 0), cz = oz + ((i >> 2 & 1) ? hs : 0);
      std::vector<std::size_t> over;
      fan::vec3 center(cx + hs * 0.5f, cy + hs * 0.5f, cz + hs * 0.5f);
      for (std::size_t idx : tri_indices) {
        if (fm::d3::triangle_intersects_aabb(tris[idx].v0, tris[idx].v1, tris[idx].v2, center, hsv)) {
          over.push_back(idx);
        }
      }
      if (over.empty()) { continue; }
      vm |= (1 << i);
      if (hs == 1) { lm |= (1 << i); }
      else { non_leaves.push_back({cx, cy, cz, std::move(over)}); }
    }
    nodes[idx] = vm | (lm << 8);
    std::uint32_t ptr_idx = nodes.size();
    nodes.resize(nodes.size() + non_leaves.size(), 0);
    for (std::size_t i = 0; i < non_leaves.size(); ++i) {
      nodes[ptr_idx + i] = build_node(tris, non_leaves[i].tri_indices, non_leaves[i].x, non_leaves[i].y, non_leaves[i].z, hs);
    }
    return idx;
  }

  std::uint32_t count_voxels() const { return nodes.empty() ? 0 : count_node_voxels(0); }

  std::uint32_t count_node_voxels(std::uint32_t n_idx) const {
    std::uint32_t n = nodes[n_idx], total = std::popcount((std::uint32_t)((n >> 8) & 0xFF)), ptr = n_idx + 1;
    for (int i = 0; i < 8; ++i) {
      if ((n & (1 << i)) && !((n >> 8) & (1 << i))) {
        total += count_node_voxels(nodes[ptr++]);
      }
    }
    return total;
  }

  std::vector<std::uint32_t> nodes;
};

struct svdag_renderer_t {
  svdag_renderer_t(const std::vector<fan::triangle_t>& geo, int current_res) : voxel_res(current_res) {
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

    dag.build_from_mesh(geo, voxel_res);
    fan::print("SVDAG Nodes:", dag.nodes.size(), "Voxels:", dag.count_voxels());

    ssbo.open(gl, GL_SHADER_STORAGE_BUFFER);
    ssbo.write_buffer(gl, dag.nodes.data(), dag.nodes.size() * sizeof(std::uint32_t));
    vao.open(gl);
  }

  ~svdag_renderer_t() {
    ssbo.close(gl);
    vao.close(gl);
  }

  void rebuild_texture() {
    fan::vec2i s = fan::window::get_size();
    if (s == res) { return; }
    res = s;
    img_screen.remove();
    img_screen = image_t({.data=nullptr, .size= res }, {
      .internal_format = image_format_e::rgba8,
      .format          = image_format_e::rgba,
      .type            = data_type_e::fan_unsigned_byte,
    });
  }

  void render(fan::vec3 cam_pos, const fan::mat4& inv_view_proj, const fan::vec3& sun_dir) {
    rebuild_texture();
    gl.image_bind(img_screen, 0, GL_WRITE_ONLY, GL_RGBA8);
    ssbo.bind_base(gl, 1);

    int max_layers = std::max(1, static_cast<int>(std::round(std::log2(voxel_res))));
    int dynamic_loop_guard = max_layers * 32;

    gloco()->shader_set_value(trace_nr, "u_inv_view_proj", inv_view_proj);
    gloco()->shader_set_value(trace_nr, "u_cam_pos", cam_pos);
    gloco()->shader_set_value(trace_nr, "u_resolution", fan::vec2(res.x, res.y));
    gloco()->shader_set_value(trace_nr, "u_voxel_scale", world_size);
    gloco()->shader_set_value(trace_nr, "u_voxel_res", voxel_res);
    gloco()->shader_set_value(trace_nr, "u_sun_dir", sun_dir.normalize());
    gloco()->shader_set_value(trace_nr, "u_max_loop_guard", dynamic_loop_guard);

    glDispatchCompute((res.x + 15) / 16, (res.y + 15) / 16, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

    auto set_blend = fan::make_restore_flag(false, [&] (bool f) {
      gl.set_depth_test(f); gl.set_blending(f);
    });
    vao.bind(gl);
    glActiveTexture(GL_TEXTURE0);
    img_screen.bind();
    gloco()->shader_set_value(blit_nr, "u_tex", 0);
    glDrawArrays(GL_TRIANGLES, 0, 3);
  }

  shader_t trace_nr, blit_nr;
  fan::opengl::core::gpu_buffer_t ssbo;
  fan::opengl::core::vao_t vao;
  image_t img_screen;
  fan::vec2i res {0, 0};
  svdag_t dag;
  int voxel_res;
};

std::vector<fan::triangle_t> extract_and_scale_mesh_triangles(const std::string& path, int voxel_res) {
  auto meshes = fan::graphics::load_meshes({.path = path});
  std::vector<fan::triangle_t> out;
  f32_t fmax = std::numeric_limits<f32_t>::max(), fmin = -fmax;
  fan::vec3 bmin(fmax, fmax, fmax), bmax(fmin, fmin, fmin);
  for (const auto& m : meshes) {
    for (const auto& v : m.vertices) {
      bmin = fan::vec3(std::min(bmin.x, v.position.x), std::min(bmin.y, v.position.y), std::min(bmin.z, v.position.z));
      bmax = fan::vec3(std::max(bmax.x, v.position.x), std::max(bmax.y, v.position.y), std::max(bmax.z, v.position.z));
    }
  }
  f32_t gv = voxel_res * 0.075f;
  fan::vec3 sz = bmax - bmin, go(gv, gv, gv);
  f32_t sf = (voxel_res * 0.85f) / std::max({sz.x, sz.y, sz.z, 1.0f});
  for (const auto& m : meshes) {
    for (std::size_t i = 0; i < m.indices.size(); i += 3) {
      out.push_back({(m.vertices[m.indices[i]].position - bmin) * sf + go,
                     (m.vertices[m.indices[i + 1]].position - bmin) * sf + go,
                     (m.vertices[m.indices[i + 2]].position - bmin) * sf + go});
    }
  }
  return out;
}

struct game_t {
  game_t() : engine({.window_size = {1280, 720}}) {
    {
      fan::time::scope_timer_print sp{};
      renderer = std::make_unique<svdag_renderer_t>(extract_and_scale_mesh_triangles("models/monkey_head.fbx", voxel_res), voxel_res);
    }
    engine.camera_set_position(engine.perspective_render_view, {564.9f, 438.4f, 597.9f});
    auto& cam = engine.camera_get(engine.perspective_render_view);
    cam.yaw = -139.10f; cam.pitch = -18.00f;

    auto cb = engine.window.add_mouse_motion_callback([&](const auto& d) {
      if (gui::is_any_item_active() || !engine.is_mouse_down(fan::mouse_right)) { return; }
      engine.camera_rotate(engine.perspective_render_view, d.motion);
      engine.camera_get(engine.perspective_render_view).update_view();
    });

    dump_svdag_occupancy(renderer->dag, voxel_res);

    gloco()->add_custom_draw([&] {
      auto& c = engine.camera_get(engine.perspective_render_view);
      renderer->render(c.position, (c.projection * c.view).inverse(), sun_direction);
    });

    engine.loop([&](f32_t dt) {
      if (fan::time::every(5000)) { gui::print(fan::format_thousands(renderer->dag.count_voxels())); }
      engine.camera_move(move_speed);
      auto& c = engine.camera_get(engine.perspective_render_view);
      if (auto h = gui::hud_interactive("##ctrl", 0.f)) {
        gui::text(std::format("pos: {:.1f} {:.1f} {:.1f}\nyaw: {:.2f}  pitch: {:.2f}", c.position.x, c.position.y, c.position.z, c.yaw,   c.pitch));
        gui::drag("sun", &sun_direction);
        gui::drag("speed", &move_speed);
      }
    });
  }

  void dump_svdag_occupancy(svdag_t& dag, int res) {
    int target_z = res / 2;
    std::vector<std::uint8_t> occ(res * res, 0);
    auto mark = [&](this auto& self, std::uint32_t ptr, int ox, int oy, int oz, int sz) -> void {
      std::uint32_t v = dag.nodes[ptr] & 0xFF, l = (dag.nodes[ptr] >> 8) & 0xFF, csz = std::max(1, sz / 2), cptr = ptr + 1;
      for (int b = 0; b < 8; ++b) {
        if (v & (1 << b)) {
          int cx = ox + (b & 1) * csz, cy = oy + (b >> 1 & 1) * csz, cz = oz + (b >> 2 & 1) * csz;
          bool is_leaf = (sz == 1 || (l & (1 << b)));
          std::uint32_t child_node = is_leaf ? 0u : dag.nodes[cptr++];

          if (target_z >= cz && target_z < cz + csz) {
            if (is_leaf) {
              for (int y = cy; y < cy + csz; ++y) {
                for (int x = cx; x < cx + csz; ++x) {
                  if (x >= 0 && x < res && y >= 0 && y < res) {
                    occ[(res - 1 - y) * res + x] = 255;
                  }
                }
              }
            }
            else {
              self(child_node, cx, cy, cz, csz);
            }
          }
        }
      }
    };
    if (!dag.nodes.empty()) { mark(0, 0, 0, 0, res); }

    std::vector<uint8_t> dat(res * res);
    fan::image::write("svdag_slice_z_mid.png", 
      {.data = occ.data(), .size = fan::vec2ui(res, res), .channels = 1}
    );
  }

  engine_t engine;
  std::unique_ptr<svdag_renderer_t> renderer;
  f32_t move_speed = 4000.f;
  fan::vec3 sun_direction{0.6f, 0.9f, 0.4f};
  int voxel_res = 2048;
};

int main() {
  game_t game;
}