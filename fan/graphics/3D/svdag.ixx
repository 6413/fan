module;

#if defined (FAN_WINDOW)
#if defined(FAN_3D)

#endif
#endif

export module fan.graphics.svdag;

#if defined (FAN_WINDOW)

#if defined(FAN_3D)

import std;
import fan.types;
import fan.types.vector;
import fan.types.color;
import fan.types.matrix;

import fan.math.intersection;

import fan.graphics.common_context;
import fan.graphics.opengl.core;
import fan.graphics.fms;

namespace fm = fan::math;

export namespace fan::graphics {
  struct gpu_mat_t {
    fan::color base_color{1.f};
    int diffuse_tex = -1;
    int normal_tex = -1;
    f32_t tex_scale = 1.f;
    std::uint32_t pad;
  };

  struct gpu_leaf_data_t {
    fan::vec4 v0{}, v1{}, v2{};
    fan::vec4 c0{1.f}, c1{1.f}, c2{1.f};
    fan::vec2 uv0{}, uv1{}, uv2{};
    std::uint32_t mat_id{};
    std::uint32_t pad[5]{};
  };
  static_assert(sizeof(gpu_leaf_data_t) == 144);

  struct gpu_lod_data_t {
    fan::vec4 albedo{0.f, 0.f, 0.f, 0.f};
    fan::vec4 normal{0.f, 1.f, 0.f, 0.f};
  };
  static_assert(sizeof(gpu_lod_data_t) == 32);

  struct svdag_t {
    struct build_tri_t {
      fan::vec3 v0, v1, v2;
      fan::vec4 c0, c1, c2;
      fan::vec2 uv0, uv1, uv2;
      std::uint16_t mat_id;
      int alpha_tex = -1;
    };

    struct child_info_t {
      fan::vec3i c;
      std::vector<std::uint32_t> tris;
    };

    static bool sample_alpha(const build_tri_t& t, const std::vector<fan::model::cpu_texture_t>& textures, fan::vec3 p, f32_t alpha_threshold);

    static bool alpha_allows_cell(const build_tri_t& t, const std::vector<fan::model::cpu_texture_t>& textures, fan::vec3 child_min, int size, f32_t alpha_threshold);

    static fan::vec4 sample_diffuse(const build_tri_t& t, const std::vector<fan::model::cpu_texture_t>& textures, fan::vec3 p);

    static gpu_lod_data_t make_leaf_lod_data(
      const build_tri_t& t,
      const std::vector<gpu_mat_t>& mats,
      const std::vector<fan::model::cpu_texture_t>& textures,
      fan::vec3 child_min, int size
    );

    gpu_lod_data_t make_cell_lod_data(
      const std::vector<build_tri_t>& tris,
      const std::vector<std::uint32_t>& tri_idx,
      const std::vector<gpu_mat_t>& mats,
      const std::vector<fan::model::cpu_texture_t>& textures,
      fan::vec3 child_min, int size
    );

    static gpu_lod_data_t merge_lod_data(const gpu_lod_data_t child_lod[8], const bool child_lod_valid[8]);

    std::uint32_t new_node();

    void build_node(
      const std::vector<build_tri_t>& tris,
      const std::vector<fm::d3::aabb_t>& tri_bounds,
      const std::vector<std::uint32_t>& tri_idx,
      const std::vector<gpu_mat_t>& mats,
      const std::vector<fan::model::cpu_texture_t>& textures,
      fan::vec3i o, int size, std::uint32_t node_idx
    );

    void build_from_mesh(const std::vector<build_tri_t>& tris, int res, const std::vector<gpu_mat_t>& mats = {}, const std::vector<fan::model::cpu_texture_t>& textures = {});

    std::vector<std::uint32_t> nodes, child_ptrs, leaf_base;
    std::vector<gpu_leaf_data_t> leaf_data;
    std::vector<gpu_lod_data_t> lod_data;
  };

  struct build_data_t {
    std::vector<svdag_t::build_tri_t> tris;
    std::vector<gpu_mat_t> mats;
    std::vector<fan::model::cpu_texture_t> textures;
    fan::vec3 bmin{}, bmax{}, go{};
    f32_t sf = 1.f;
  };

  build_data_t extract_build_tris_cpu(const std::string& path, int voxel_res);

  struct svdag_cache_header_t {
    std::uint32_t magic     = 0x47414453;
    std::uint32_t version   = 8;
    std::uint32_t voxel_res = 0;
    std::uint32_t leaf_size = sizeof(gpu_leaf_data_t);
    std::uint32_t lod_size  = sizeof(gpu_lod_data_t);
    std::uint32_t pad       = 0;
    std::uint64_t nodes      = 0;
    std::uint64_t child_ptrs = 0;
    std::uint64_t leaf_base  = 0;
    std::uint64_t leaf_data  = 0;
    std::uint64_t lod_data   = 0;
  };

  bool load_svdag_cache(svdag_t& dag, const std::string& path, int voxel_res);

  void save_svdag_cache(const svdag_t& dag, const std::string& path, int voxel_res);

  struct svdag_renderer_t;

  struct svdag_load_result_t {
    build_data_t build_data;
    svdag_t dag;
  };

  svdag_load_result_t load_svdag_cpu(const std::string& path, int voxel_res, const std::string& cache_path);

  struct svdag_loader_t {
    void start(const std::string& path, int voxel_res, const std::string& cache_path);

    bool ready() const;

    std::unique_ptr<svdag_renderer_t> finish(int voxel_res);

    std::future<svdag_load_result_t> job;
  };

  struct svdag_renderer_t {
    svdag_renderer_t(build_data_t&& bdata, svdag_t&& in_dag, int res);

    ~svdag_renderer_t();

    void render(fan::vec3 cam_pos, const fan::mat4& inv_view_proj, const fan::vec3& sun_dir);

    shader_t trace_nr;
    fan::opengl::core::gpu_buffer_t ssbo_nodes, ssbo_ptrs, ssbo_mats, ssbo_leaf_base, ssbo_leaf_data, ssbo_lod_data;
    fan::opengl::core::vao_t vao;
    image_t img_screen;
    svdag_t dag;
    fan::vec3 bmin{}, bmax{}, go{};
    f32_t sf = 1.f;
    f32_t render_scale = 1.f;
    f32_t lod_bias = 0.f;
    GLuint tex_array = 0;
    int voxel_res = 0;
    int max_depth = 1;
    int ao_quality = 0;
    int debug_heatmap = 0;
  };
}

#endif

#endif