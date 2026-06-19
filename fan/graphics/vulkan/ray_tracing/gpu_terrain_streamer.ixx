module;

#if defined(FAN_3D) && defined(FAN_VULKAN)
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>
#endif

export module fan.graphics.vulkan.ray_tracing.gpu_terrain_streamer;

import std;

#if defined(FAN_3D) && defined(FAN_VULKAN)
import fan;
import fan.graphics.vulkan.ray_tracing.hardware_renderer;

namespace gui = fan::graphics::gui;

export namespace fan::graphics::vulkan::ray_tracing {
  using chunk_coord_t = fan::vec2_wrap_t<std::int32_t>;
  struct block_pos_t {
    std::int32_t x = 0;
    std::int32_t y = 0;
    std::int32_t z = 0;

    bool operator==(const block_pos_t& o) const {
      return x == o.x && y == o.y && z == o.z;
    }
  };
  struct block_pos_hash_t {
    std::size_t operator()(const block_pos_t& p) const {
      std::uint32_t h =
        (std::uint32_t)p.x * 0x8da6b343u ^
        (std::uint32_t)p.y * 0xd8163841u ^
        (std::uint32_t)p.z * 0xcb1ab31fu;
      h ^= h >> 16;
      h *= 0x7feb352du;
      h ^= h >> 15;
      h *= 0x846ca68bu;
      h ^= h >> 16;
      return h;
    }
  };
  struct block_edit_t {
    std::int32_t x = 0;
    std::int32_t y = 0;
    std::int32_t z = 0;
    std::int32_t id = -1;
  };
  struct chunk_coord_hash_t {
    std::size_t operator()(const chunk_coord_t& c) const {
    return (std::size_t)c[0] | ((std::size_t)c[1] << 32); }
  };
  struct terrain_chunk_t {
    context_t::object_handle_t handle;
    std::uint32_t vertex_capacity = 0;
    std::uint32_t index_capacity = 0;
    std::uint32_t tree_cell = 0;
    std::uint32_t tree_density_u8 = 0;
    bool uploaded = false;
    bool pending = false;
    bool dirty = false;
  };
  struct free_chunk_handle_t {
    context_t::object_handle_t handle;
    std::uint32_t vertex_capacity = 0;
    std::uint32_t index_capacity = 0;
    std::uint32_t age = 0;
  };

  using chunk_map_t = std::unordered_map<chunk_coord_t, terrain_chunk_t, chunk_coord_hash_t>;

  struct gpu_chunk_generator_t {
    static constexpr std::uint32_t chunk_size = 32;
    static constexpr std::uint32_t max_faces_per_column = 16;
    static constexpr std::uint32_t max_vertices = chunk_size * chunk_size * max_faces_per_column * 4;
    static constexpr std::uint32_t max_indices = chunk_size * chunk_size * max_faces_per_column * 6;
    static constexpr std::uint32_t slot_count = 4;
    static constexpr std::uint32_t max_edits_per_chunk = 2048;
    struct counters_t { std::uint32_t vertex_count = 0, index_count = 0, pad0 = 0, pad1 = 0; };
    struct edit_header_t { std::uint32_t count = 0, pad0 = 0, pad1 = 0, pad2 = 0; };
    static constexpr std::uint32_t edit_buffer_size = sizeof(edit_header_t) + max_edits_per_chunk * sizeof(block_edit_t);
    struct push_t {
      std::int32_t chunk_x = 0, chunk_z = 0;
      std::uint32_t chunk_size = 32, sy = 96, sea_level = 28, snow_line = 74, rock_line = 62, tree_cell = 10;
      f32_t voxel_size = 1.5f, tree_density = 1.f, pad2 = 0.f, pad3 = 0.f;
    };
    struct state_t { chunk_coord_t coord {}; push_t push {}; };
    using vk_context_t = fan::vulkan::context_t;

    void open(context_t& renderer) {
      ctx = renderer.ctx;
      pipeline.open(*ctx, "shaders/vulkan/ray_tracing/chunk_gen.comp", sizeof(push_t), {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}
      });
      constexpr VkMemoryPropertyFlags read_mem = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
      constexpr VkMemoryPropertyFlags write_mem = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
      slots.open(*ctx, slot_count, pipeline.descriptor_layout, {
        {max_vertices * sizeof(context_t::vertex_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, read_mem},
        {max_indices * sizeof(std::uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, read_mem},
        {sizeof(counters_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, read_mem},
        {edit_buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, write_mem}
      });
    }
    void close() {
      if (!ctx) { return; }
      slots.close(*ctx);
      pipeline.close(*ctx);
      ctx = nullptr;
    }
    bool ready() const { return ctx && pipeline.pipeline != VK_NULL_HANDLE; }
    std::uint32_t free_slot_count() const { return slots.free_slot_count(); }
    std::uint32_t busy_slot_count() const { return slot_count - slots.free_slot_count(); }
    bool submit(const push_t& push, chunk_coord_t coord, const std::vector<block_edit_t>& edits) {
      if (!ready()) { return false; }
      std::uint32_t si = slots.acquire();
      if (si == vk_context_t::compute_slot_ring_t::invalid_slot) { return false; }
      auto& slot = slots.get(si);
      std::uint32_t edit_count = std::min<std::uint32_t>((std::uint32_t)edits.size(), max_edits_per_chunk);
      auto* header = static_cast<edit_header_t*>(slot.buffers[3].mapped);
      header->count = edit_count;
      header->pad0 = header->pad1 = header->pad2 = 0;
      if (edit_count) {
        std::memcpy((std::uint8_t*)slot.buffers[3].mapped + sizeof(edit_header_t), edits.data(), (std::size_t)edit_count * sizeof(block_edit_t));
      }
      auto cmd = slots.begin(*ctx, si);
      ctx->fill_buffer_cmd(cmd, slot.buffers[2], 0, sizeof(counters_t), 0);
      ctx->buffer_barrier_cmd(cmd, slot.buffers[3], VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, edit_buffer_size);
      ctx->buffer_barrier_cmd(cmd, slot.buffers[2], VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, sizeof(counters_t));
      pipeline.dispatch(*ctx, cmd, slot.descriptor_set, &push, (chunk_size + 7) / 8, (chunk_size + 7) / 8, 1);
      ctx->buffer_barriers_cmd(cmd, {
        {&slot.buffers[2], VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT, 0, sizeof(counters_t)},
        {&slot.buffers[0], VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT},
        {&slot.buffers[1], VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT}
      }, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT);
      states[si] = {coord, push};
      slots.submit(*ctx, si);
      return true;
    }
    bool fetch_completed(chunk_coord_t& coord, context_t::voxel_mesh_input_t& mesh) {
      if (!ready()) { return false; }
      for (std::uint32_t i = 0; i < slot_count; ++i) {
        if (!slots.done(*ctx, i)) { continue; }
        coord = states[i].coord;
        read_mesh(slots.get(i), states[i], mesh);
        slots.set_idle(i);
        return true;
      }
      return false;
    }
    bool read_mesh(vk_context_t::compute_slot_ring_t::slot_t& slot, const state_t& state, context_t::voxel_mesh_input_t& mesh) {
      ctx->invalidate_buffer(slot.buffers[2], 0, sizeof(counters_t));
      auto* c = static_cast<counters_t*>(slot.buffers[2].mapped);
      debug_last_vertex_count = c->vertex_count;
      debug_last_index_count = c->index_count;
      if (!c->vertex_count || !c->index_count || c->vertex_count > max_vertices || c->index_count > max_indices) {
        if (c->vertex_count > max_vertices || c->index_count > max_indices) {
          ++debug_overflow_count;
        }
        mesh.vertices.clear();
        mesh.indices.clear();
        return false;
      }
      mesh.vertices.resize(c->vertex_count);
      mesh.indices.resize(c->index_count);
      auto vb = mesh.vertices.size() * sizeof(context_t::vertex_t), ib = mesh.indices.size() * sizeof(std::uint32_t);
      ctx->invalidate_buffer(slot.buffers[0], 0, vb);
      ctx->invalidate_buffer(slot.buffers[1], 0, ib);
      std::memcpy(mesh.vertices.data(), slot.buffers[0].mapped, (std::size_t)vb);
      std::memcpy(mesh.indices.data(), slot.buffers[1].mapped, (std::size_t)ib);
      mesh.materials.clear(); mesh.primitive_material_indices.clear();
      mesh.transform = fan::translate(fan::vec3(
        (f32_t)(state.coord.x * (std::int32_t)state.push.chunk_size) * state.push.voxel_size,
        -(f32_t)state.push.sea_level * state.push.voxel_size,
        (f32_t)(state.coord.y * (std::int32_t)state.push.chunk_size) * state.push.voxel_size
      ));
      return true;
    }

    fan::vulkan::context_t* ctx = nullptr;
    vk_context_t::compute_pipeline_t pipeline;
    vk_context_t::compute_slot_ring_t slots;
    std::array<state_t, slot_count> states {};
    std::uint32_t debug_last_vertex_count = 0;
    std::uint32_t debug_last_index_count = 0;
    std::uint32_t debug_overflow_count = 0;
  };

  context_t::voxel_mesh_input_t make_bootstrap_mesh() {
    context_t::voxel_mesh_input_t mesh;
    mesh.vertices.resize(3);
    mesh.indices = {0, 1, 2};
    return mesh;
  }

  struct gpu_terrain_streamer_t {
    static constexpr std::uint32_t chunk_size = 32, sy = 96, sea_level = 28, snow_line = 74, rock_line = 62;
    static constexpr f32_t voxel_size = 1.5f;

    struct debug_stats_t {
      std::size_t chunks = 0;
      std::size_t uploaded_chunks = 0;
      std::size_t pending_chunks = 0;
      std::size_t dirty_chunks = 0;
      std::size_t queued_chunks = 0;
      std::size_t queued_missing_chunks = 0;
      std::size_t queued_dirty_chunks = 0;
      std::size_t free_handles = 0;
      std::size_t removed_blocks = 0;
      std::size_t placed_blocks = 0;
      std::size_t placed_vertices = 0;
      std::size_t placed_indices = 0;
      std::uint32_t gpu_slots_busy = 0;
      std::uint32_t gpu_slots_free = 0;
      std::uint32_t last_uploads = 0;
      std::uint32_t last_submits = 0;
      std::uint32_t last_unloads = 0;
      std::uint32_t last_queue_pruned = 0;
      std::uint32_t last_free_pool_destroys = 0;
      std::uint32_t effective_uploads = 0;
      std::uint32_t effective_submits = 0;
      f32_t effective_upload_budget_ms = 0.f;
      f32_t frame_ms = 0.f;
      f32_t fps = 0.f;
      std::uint32_t last_generated_vertices = 0;
      std::uint32_t last_generated_indices = 0;
      std::uint32_t mesh_overflows = 0;
      std::int32_t current_chunk_x = 0;
      std::int32_t current_chunk_z = 0;
      f32_t estimated_terrain_mesh_mb = 0.f;
      f32_t estimated_free_pool_mesh_mb = 0.f;
      f32_t free_pool_budget_mb = 0.f;
      f32_t terrain_budget_mb = 0.f;
      std::uint32_t max_loaded_chunks = 0;
      std::int32_t unload_margin = 0;
      std::int32_t tree_detail_distance = 0;
      std::int32_t tree_cell = 0;
      f32_t tree_density = 0.f;
      std::size_t tree_enabled_chunks = 0;
      f32_t estimated_compute_slots_mb = 0.f;
      f32_t estimated_placed_mesh_mb = 0.f;
      f32_t estimated_cpu_cached_mb = 0.f;
    };
    void init() {}
    void open(context_t& renderer) {
      if (gpu.ready()) { return; }
      gpu.open(renderer);
      reserve_terrain_scene(renderer);
      work_queue.reserve(max_pending_chunks);
      chunks.reserve((render_distance * 2 + 5) * (render_distance * 2 + 5));
      free_handles.reserve((render_distance * 2 + 5) * (render_distance * 2 + 5));
      atlas = fan::graphics::image_t {"images/mctp.webp"};
      atlas_slot = register_rt_image(renderer, atlas);
      terrain_mat.albedo_texture_id = atlas_slot;
      terrain_mat.base_color = fan::vec3(1.f, 1.f, 1.f);
    }
    void reserve_terrain_scene(context_t& renderer) const {
      std::uint32_t r = (std::uint32_t)(render_distance + preload_extra + 1);
      std::uint32_t chunk_count = (r * 2 + 1) * (r * 2 + 1);
      constexpr std::uint32_t reserve_vertices_per_chunk = 10'240;
      constexpr std::uint32_t reserve_indices_per_chunk = 15'360;
      renderer.reserve_scene_buffers(
        chunk_count * reserve_vertices_per_chunk,
        chunk_count * reserve_indices_per_chunk,
        64'000,
        chunk_count * reserve_indices_per_chunk / 3
      );
    }
    static std::uint32_t round_up_capacity(std::uint32_t v, std::uint32_t step) {
      return ((v + step - 1) / step) * step;
    }
    static std::uint32_t terrain_vertex_capacity(std::uint32_t used) {
      return std::min<std::uint32_t>(gpu_chunk_generator_t::max_vertices, round_up_capacity(std::max<std::uint32_t>(used + 256, 4096), 1024));
    }
    static std::uint32_t terrain_index_capacity(std::uint32_t used) {
      return std::min<std::uint32_t>(gpu_chunk_generator_t::max_indices, round_up_capacity(std::max<std::uint32_t>(used + 384, 6144), 1536));
    }
    static std::size_t free_handle_mesh_bytes(const free_chunk_handle_t& h) {
      return
        (std::size_t)h.vertex_capacity * sizeof(context_t::vertex_t) +
        (std::size_t)h.index_capacity * sizeof(std::uint32_t);
    }
    std::size_t free_pool_mesh_bytes() const {
      std::size_t bytes = 0;
      for (const auto& h : free_handles) {
        bytes += free_handle_mesh_bytes(h);
      }
      return bytes;
    }
    static std::size_t chunk_mesh_bytes(const terrain_chunk_t& ch) {
      return
        (std::size_t)ch.vertex_capacity * sizeof(context_t::vertex_t) +
        (std::size_t)ch.index_capacity * sizeof(std::uint32_t);
    }
    std::size_t uploaded_chunk_count() const {
      std::size_t n = 0;
      for (const auto& [coord, ch] : chunks) {
        (void)coord;
        if (ch.uploaded) {
          ++n;
        }
      }
      return n;
    }
    std::size_t terrain_mesh_bytes() const {
      std::size_t bytes = 0;
      for (const auto& [coord, ch] : chunks) {
        (void)coord;
        if (ch.uploaded) {
          bytes += chunk_mesh_bytes(ch);
        }
      }
      return bytes;
    }
    void sanitize_settings() {
      render_distance = std::clamp(render_distance, 1, 16);
      initial_load_radius = std::clamp(initial_load_radius, 1, render_distance);
      max_uploads_per_frame = std::clamp<std::uint32_t>(max_uploads_per_frame, 1, 8);
      max_submits_per_frame = std::clamp<std::uint32_t>(max_submits_per_frame, 1, 8);
      max_pending_chunks = std::clamp<std::uint32_t>(max_pending_chunks, max_submits_per_frame + gpu_chunk_generator_t::slot_count, 4096);
      max_free_pool_mesh_mb = std::clamp(max_free_pool_mesh_mb, 0, 512);
      max_terrain_mesh_mb = std::clamp(max_terrain_mesh_mb, 0, 4096);
      max_loaded_chunks = std::clamp(max_loaded_chunks, 0, 4096);
      preload_extra = std::clamp(preload_extra, 0, 4);
      unload_margin = std::clamp(unload_margin, 0, 4);
      tree_cell_size = std::clamp(tree_cell_size, 4, 64);
      tree_detail_distance = std::clamp(tree_detail_distance, 0, render_distance + preload_extra);
      tree_density = std::clamp(tree_density, 0.f, 1.f);
      upload_budget_ms = std::clamp(upload_budget_ms, 0.05f, 10.f);
      max_adaptive_uploads_per_frame = std::clamp<std::uint32_t>(max_adaptive_uploads_per_frame, max_uploads_per_frame, 8);
      max_adaptive_submits_per_frame = std::clamp<std::uint32_t>(max_adaptive_submits_per_frame, max_submits_per_frame, 8);
    }
    void age_free_handles() {
      for (auto& h : free_handles) {
        if (h.age < 0xffffffffu) {
          ++h.age;
        }
      }
    }
    bool can_recycle_geometry(const context_t& renderer) const {
      return gpu.busy_slot_count() == 0 && renderer.pending_mesh_upload_count() == 0;
    }
    bool free_pool_over_hard_budget() const {
      std::size_t budget = (std::size_t)std::max(0, max_free_pool_mesh_mb) * 1024ull * 1024ull;
      return budget != 0 && free_pool_mesh_bytes() > budget * 2;
    }
    void trim_free_pool(context_t& renderer) {
      std::size_t budget = (std::size_t)std::max(0, max_free_pool_mesh_mb) * 1024ull * 1024ull;
      std::size_t bytes = free_pool_mesh_bytes();
      if (!can_recycle_geometry(renderer)) {
        if (budget == 0 || bytes <= budget * 2) { return; }
        renderer.wait_idle();
      }

      while (bytes > budget && !free_handles.empty()) {
        auto best = free_handles.end();
        bool hard = budget != 0 && bytes > budget * 2;
        for (auto it = free_handles.begin(); it != free_handles.end(); ++it) {
          if (!hard && it->age < free_handle_safe_age) {
            continue;
          }
          if (best == free_handles.end() || free_handle_mesh_bytes(*it) > free_handle_mesh_bytes(*best)) {
            best = it;
          }
        }
        if (best == free_handles.end()) {
          break;
        }

        bytes -= std::min(bytes, free_handle_mesh_bytes(*best));
        renderer.destroy_object_geometry(best->handle);
        *best = free_handles.back();
        free_handles.pop_back();
        ++debug_last_free_pool_destroys;
      }
    }
    void prepare_chunk_mesh(context_t::voxel_mesh_input_t& mesh) const {
      mesh.materials = {terrain_mat};
      mesh.primitive_material_indices.assign(mesh.indices.size() / 3, 0);
      mesh.vertex_capacity = terrain_vertex_capacity((std::uint32_t)mesh.vertices.size());
      mesh.index_capacity = terrain_index_capacity((std::uint32_t)mesh.indices.size());
      mesh.material_capacity = 1;
      mesh.keep_cpu_copy = false;
    }
    void destroy() { gpu.close(); }
    bool in_range(chunk_coord_t c, chunk_coord_t center, std::int32_t extra = 0) const {
      chunk_coord_t d = c - center;
      std::int32_t r = render_distance + extra;
      return d.length_squared() <= r * r;
    }
    void prune_queue(chunk_coord_t center) {
      for (std::size_t i = 0; i < work_queue.size();) {
        if (in_range(work_queue[i], center, 1)) { ++i; continue; }
        auto it = chunks.find(work_queue[i]);
        if (it != chunks.end()) { it->second.pending = false; if (!it->second.uploaded) { chunks.erase(it); } }
        work_queue.erase(work_queue.begin() + i);
        ++debug_last_queue_pruned;
      }
    }
    context_t::voxel_mesh_input_t make_hidden_chunk_mesh() const {
      context_t::voxel_mesh_input_t mesh;
      mesh.vertices.resize(3);
      mesh.vertices[0].position = fan::vec4(0.f, 0.f, 0.f, 0.f);
      mesh.vertices[1].position = fan::vec4(0.01f, 0.f, 0.f, 0.f);
      mesh.vertices[2].position = fan::vec4(0.f, 0.f, 0.01f, 0.f);
      for (auto& vertex : mesh.vertices) {
        vertex.normal = fan::vec4(0.f, 1.f, 0.f, 0.f);
        vertex.texcoord = fan::vec2(0.f);
        vertex.color = 0;
      }
      mesh.indices = {0, 1, 2};
      prepare_chunk_mesh(mesh);
      mesh.transform = fan::translate(fan::vec3(0.f, -1000000.f, 0.f));
      return mesh;
    }
    struct block_tiles_t {
      fan::vec2 top;
      fan::vec2 side;
    };
    static block_tiles_t block_tiles(std::uint8_t id) {
      if (id == 2) { return {fan::vec2(1.f, 0.f), fan::vec2(1.f, 0.f)}; }
      if (id == 3) { return {fan::vec2(2.f, 1.f), fan::vec2(2.f, 1.f)}; }
      if (id == 4) { return {fan::vec2(2.f, 4.f), fan::vec2(4.f, 4.f)}; }
      if (id == 5) { return {fan::vec2(4.f, 1.f), fan::vec2(4.f, 1.f)}; }
      if (id == 6) { return {fan::vec2(3.f, 10.f), fan::vec2(3.f, 10.f)}; }
      return {fan::vec2(4.f, 9.f), fan::vec2(3.f, 0.f)};
    }
    static fan::vec2 atlas_uv(fan::vec2 tile_id, fan::vec2 uv) {
      constexpr f32_t atlas_size = 2048.f;
      constexpr f32_t tile_size = 128.f;
      constexpr f32_t pad = 0.5f;
      uv.y = 1.f - uv.y;
      return (tile_id * tile_size + fan::vec2(pad) + uv * (tile_size - pad * 2.f)) / atlas_size;
    }
    static fan::vec2 block_uv(block_tiles_t b, fan::vec2 uv, bool top) {
      return atlas_uv(top ? b.top : b.side, uv);
    }
    static void push_vertex(context_t::voxel_mesh_input_t& mesh, fan::vec3 p, fan::vec3 n, fan::vec2 uv, block_tiles_t b, bool top) {
      context_t::vertex_t v {};
      v.position = fan::vec4(p.x, p.y, p.z, 0.f);
      v.normal = fan::vec4(n.x, n.y, n.z, 0.f);
      v.texcoord = block_uv(b, uv, top);
      v.color = 0xffffffffu;
      mesh.vertices.push_back(v);
    }
    static void push_quad(context_t::voxel_mesh_input_t& mesh, fan::vec3 p0, fan::vec3 p1, fan::vec3 p2, fan::vec3 p3, fan::vec3 n, block_tiles_t b, bool top) {
      std::uint32_t vi = (std::uint32_t)mesh.vertices.size();
      push_vertex(mesh, p0, n, fan::vec2(0.f, 0.f), b, top);
      push_vertex(mesh, p1, n, fan::vec2(1.f, 0.f), b, top);
      push_vertex(mesh, p2, n, fan::vec2(1.f, 1.f), b, top);
      push_vertex(mesh, p3, n, fan::vec2(0.f, 1.f), b, top);
      mesh.indices.insert(mesh.indices.end(), {vi + 0, vi + 1, vi + 2, vi + 0, vi + 2, vi + 3});
    }
    void push_block_face(context_t::voxel_mesh_input_t& mesh, block_pos_t p, std::int32_t face, block_tiles_t b) const {
      f32_t s = voxel_size;
      f32_t x0 = (f32_t)p.x * s;
      f32_t x1 = (f32_t)(p.x + 1) * s;
      f32_t y0 = ((f32_t)p.y - (f32_t)sea_level) * s;
      f32_t y1 = ((f32_t)(p.y + 1) - (f32_t)sea_level) * s;
      f32_t z0 = (f32_t)p.z * s;
      f32_t z1 = (f32_t)(p.z + 1) * s;

      if (face == 0) { push_quad(mesh, fan::vec3(x1, y0, z1), fan::vec3(x1, y0, z0), fan::vec3(x1, y1, z0), fan::vec3(x1, y1, z1), fan::vec3(1.f, 0.f, 0.f), b, false); }
      else if (face == 1) { push_quad(mesh, fan::vec3(x0, y0, z0), fan::vec3(x0, y0, z1), fan::vec3(x0, y1, z1), fan::vec3(x0, y1, z0), fan::vec3(-1.f, 0.f, 0.f), b, false); }
      else if (face == 2) { push_quad(mesh, fan::vec3(x0, y1, z0), fan::vec3(x1, y1, z0), fan::vec3(x1, y1, z1), fan::vec3(x0, y1, z1), fan::vec3(0.f, 1.f, 0.f), b, true); }
      else if (face == 3) { push_quad(mesh, fan::vec3(x0, y0, z0), fan::vec3(x0, y0, z1), fan::vec3(x1, y0, z1), fan::vec3(x1, y0, z0), fan::vec3(0.f, -1.f, 0.f), b, false); }
      else if (face == 4) { push_quad(mesh, fan::vec3(x0, y0, z1), fan::vec3(x1, y0, z1), fan::vec3(x1, y1, z1), fan::vec3(x0, y1, z1), fan::vec3(0.f, 0.f, 1.f), b, false); }
      else { push_quad(mesh, fan::vec3(x1, y0, z0), fan::vec3(x0, y0, z0), fan::vec3(x0, y1, z0), fan::vec3(x1, y1, z0), fan::vec3(0.f, 0.f, -1.f), b, false); }
    }
    context_t::voxel_mesh_input_t make_placed_blocks_mesh() const {
      if (placed_blocks.empty()) {
        return make_hidden_chunk_mesh();
      }

      context_t::voxel_mesh_input_t mesh;
      mesh.transform = fan::mat4(1);
      constexpr std::array<block_pos_t, 6> dirs {{
        {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}
      }};
      for (auto& [p, id] : placed_blocks) {
        block_tiles_t tiles = block_tiles(id);
        for (std::int32_t face = 0; face < 6; ++face) {
          block_pos_t n {p.x + dirs[face].x, p.y + dirs[face].y, p.z + dirs[face].z};
          if (!solid_block(n)) {
            push_block_face(mesh, p, face, tiles);
          }
        }
      }
      if (mesh.vertices.empty() || mesh.indices.empty()) {
        return make_hidden_chunk_mesh();
      }
      mesh.materials = {terrain_mat};
      mesh.primitive_material_indices.assign(mesh.indices.size() / 3, 0);
      mesh.vertex_capacity = terrain_vertex_capacity((std::uint32_t)mesh.vertices.size());
      mesh.index_capacity = terrain_index_capacity((std::uint32_t)mesh.indices.size());
      mesh.material_capacity = 1;
      mesh.keep_cpu_copy = false;
      return mesh;
    }
    void rebuild_placed_blocks(context_t& renderer) {
      context_t::voxel_mesh_input_t mesh = make_placed_blocks_mesh();
      debug_placed_vertices = mesh.vertices.size();
      debug_placed_indices = mesh.indices.size();
      if (!placed_overlay_uploaded) {
        placed_overlay = renderer.add_mesh_incremental(mesh);
        placed_overlay_uploaded = true;
        return;
      }
      if (!renderer.update_mesh(placed_overlay, mesh)) {
        renderer.set_transform_deferred(placed_overlay, fan::translate(fan::vec3(0.f, -1000000.f, 0.f)));
        placed_overlay = renderer.add_mesh_incremental(mesh);
      }
    }
    struct tree_lod_t {
      std::uint32_t cell = 10;
      std::uint32_t density_u8 = 255;
      f32_t density = 1.f;
    };
    tree_lod_t tree_lod_for_chunk(chunk_coord_t c, chunk_coord_t center) const {
      chunk_coord_t d = c - center;
      if (tree_detail_distance > 0 && d.length_squared() > tree_detail_distance * tree_detail_distance) {
        return {(std::uint32_t)tree_cell_size, 0, 0.f};
      }

      f32_t density = std::clamp(tree_density, 0.f, 1.f);
      return {
        (std::uint32_t)std::max(1, tree_cell_size),
        (std::uint32_t)std::round(density * 255.f),
        density
      };
    }
    void discard_uploaded_chunk(context_t& renderer, chunk_map_t::iterator& it) {
      renderer.set_transform_deferred(it->second.handle, fan::translate(fan::vec3(0.f, -1000000.f, 0.f)));
      free_handles.push_back({it->second.handle, it->second.vertex_capacity, it->second.index_capacity, 0});
      it = chunks.erase(it);
      ++debug_last_unloads;
    }
    bool over_chunk_budget() const {
      bool over_count = max_loaded_chunks > 0 && uploaded_chunk_count() > (std::size_t)max_loaded_chunks;
      bool over_memory = max_terrain_mesh_mb > 0 && terrain_mesh_bytes() > (std::size_t)max_terrain_mesh_mb * 1024ull * 1024ull;
      return over_count || over_memory;
    }
    void enforce_loaded_budget(context_t& renderer, chunk_coord_t center) {
      while (over_chunk_budget()) {
        auto best = chunks.end();
        std::int64_t best_d2 = -1;

        for (auto it = chunks.begin(); it != chunks.end(); ++it) {
          if (!it->second.uploaded || it->second.pending || it->second.dirty) {
            continue;
          }
          if (in_range(it->first, center, 0)) {
            continue;
          }

          chunk_coord_t d = it->first - center;
          std::int64_t d2 = (std::int64_t)d.length_squared();
          if (d2 > best_d2) {
            best_d2 = d2;
            best = it;
          }
        }

        if (best == chunks.end()) {
          break;
        }

        discard_uploaded_chunk(renderer, best);
      }
    }
    std::size_t pending_chunk_count() const {
      std::size_t n = 0;
      for (const auto& [coord, ch] : chunks) {
        (void)coord;
        if (ch.pending) {
          ++n;
        }
      }
      return n;
    }
    std::uint32_t effective_upload_limit() const {
      if (!adaptive_streaming) {
        return max_uploads_per_frame;
      }
      std::size_t backlog = work_queue.size() + pending_chunk_count();
      std::uint32_t limit = max_uploads_per_frame;
      if (backlog > 24) {
        limit = std::max<std::uint32_t>(limit, std::min<std::uint32_t>(max_adaptive_uploads_per_frame, max_uploads_per_frame + 1));
      }
      if (backlog > 48 && debug_frame_ms < 22.f) {
        limit = std::max<std::uint32_t>(limit, max_adaptive_uploads_per_frame);
      }
      if (debug_frame_ms > 28.f) {
        limit = max_uploads_per_frame;
      }
      return std::clamp<std::uint32_t>(limit, 1, 8);
    }
    std::uint32_t effective_submit_limit() const {
      if (!adaptive_streaming) {
        return max_submits_per_frame;
      }
      std::size_t backlog = work_queue.size() + pending_chunk_count();
      std::uint32_t limit = max_submits_per_frame;
      if (backlog > 16) {
        limit = std::max<std::uint32_t>(limit, std::min<std::uint32_t>(max_adaptive_submits_per_frame, max_submits_per_frame + 1));
      }
      if (backlog > 40 && debug_frame_ms < 22.f) {
        limit = std::max<std::uint32_t>(limit, max_adaptive_submits_per_frame);
      }
      if (debug_frame_ms > 28.f) {
        limit = max_submits_per_frame;
      }
      return std::clamp<std::uint32_t>(limit, 1, 8);
    }
    f32_t effective_upload_budget_ms() const {
      if (!adaptive_streaming) {
        return upload_budget_ms;
      }
      std::size_t backlog = work_queue.size() + pending_chunk_count();
      f32_t budget = upload_budget_ms;
      if (backlog > 24) {
        budget = std::max(budget, upload_budget_ms * 2.f);
      }
      if (backlog > 48 && debug_frame_ms < 22.f) {
        budget = std::max(budget, upload_budget_ms * 4.f);
      }
      if (debug_frame_ms > 28.f) {
        budget = upload_budget_ms;
      }
      return std::clamp(budget, 0.05f, 4.f);
    }
    void update_frame_timing() {
      using clock_t = std::chrono::steady_clock;
      auto now = clock_t::now();
      if (!timing_initialized) {
        last_update_time = now;
        timing_initialized = true;
        return;
      }
      f32_t ms = std::chrono::duration<f32_t, std::milli>(now - last_update_time).count();
      last_update_time = now;
      if (debug_frame_ms <= 0.f) {
        debug_frame_ms = ms;
      }
      else {
        debug_frame_ms = debug_frame_ms * 0.9f + ms * 0.1f;
      }
    }
    std::int32_t work_category(chunk_coord_t c) const {
      auto it = chunks.find(c);
      if (it == chunks.end()) {
        return 3;
      }
      const auto& ch = it->second;
      if (!ch.uploaded) {
        return 0;
      }
      if (ch.dirty) {
        return 1;
      }
      return 2;
    }
    f32_t work_priority(chunk_coord_t c, chunk_coord_t center, const fan::vec3& cam) const {
      chunk_coord_t d = c - center;
      return (f32_t)work_category(c) * 1000000000.f + chunk_distance2(c, cam) + (f32_t)d.length_squared() * 0.001f;
    }
    void queue_visible_dirty(chunk_coord_t center) {
      std::uint32_t pending_cap = std::max<std::uint32_t>(max_pending_chunks, effective_submit_limit() + gpu_chunk_generator_t::slot_count);
      if (work_queue.size() >= pending_cap) {
        return;
      }
      for (auto& [coord, ch] : chunks) {
        if (work_queue.size() >= pending_cap) {
          break;
        }
        if (!ch.uploaded || !ch.dirty || ch.pending || !in_range(coord, center, 0)) {
          continue;
        }
        ch.pending = true;
        work_queue.push_back(coord);
      }
    }
    void refresh_tree_lod(chunk_coord_t center) {
      std::uint32_t queued = 0;
      constexpr std::uint32_t max_tree_lod_rebuilds_per_frame = 1;

      for (auto& [coord, ch] : chunks) {
        if (queued >= max_tree_lod_rebuilds_per_frame) {
          break;
        }
        if (!ch.uploaded || ch.pending || ch.dirty || !in_range(coord, center, 0)) {
          continue;
        }

        tree_lod_t lod = tree_lod_for_chunk(coord, center);
        if (ch.tree_cell == lod.cell && ch.tree_density_u8 == lod.density_u8) {
          continue;
        }

        ch.dirty = true;
        ch.pending = true;
        work_queue.push_back(coord);
        ++queued;
      }
    }
    void unload_far(context_t& renderer, chunk_coord_t center) {
      for (auto it = chunks.begin(); it != chunks.end();) {
        if (!it->second.uploaded || in_range(it->first, center, preload_extra + unload_margin)) { ++it; continue; }
        discard_uploaded_chunk(renderer, it);
      }
      enforce_loaded_budget(renderer, center);
      trim_free_pool(renderer);
    }
    bool ready() const { return gpu.ready(); }
    void update(context_t& renderer, const fan::vec3& cam) {
      sanitize_settings();
      debug_last_uploads = 0;
      debug_last_submits = 0;
      debug_last_unloads = 0;
      debug_last_queue_pruned = 0;
      debug_last_free_pool_destroys = 0;
      update_frame_timing();

      age_free_handles();

      if (!bootstrap_added) { renderer.add_mesh(make_bootstrap_mesh()); bootstrap_added = true; }
      if (!gpu.ready()) {
        if (!renderer.ready) { return; }
        open(renderer);
        load_initial(renderer, to_chunk(cam), initial_load_radius, cam);
      }
      chunk_coord_t center = to_chunk(cam);
      debug_center = center;
      prune_queue(center); unload_far(renderer, center); queue_missing(center, cam); queue_visible_dirty(center); refresh_tree_lod(center); poll_completed(renderer, center); submit_pending(center, cam);
    }
    debug_stats_t debug_stats() const {
      debug_stats_t s;
      s.chunks = chunks.size();
      s.queued_chunks = work_queue.size();
      for (chunk_coord_t c : work_queue) {
        auto it = chunks.find(c);
        if (it == chunks.end()) { continue; }
        if (!it->second.uploaded) { ++s.queued_missing_chunks; }
        else if (it->second.dirty) { ++s.queued_dirty_chunks; }
      }
      s.free_handles = free_handles.size();
      s.removed_blocks = block_edits.size();
      s.placed_blocks = placed_blocks.size();
      s.gpu_slots_free = gpu.ready() ? gpu.free_slot_count() : 0;
      s.gpu_slots_busy = gpu.ready() ? gpu.busy_slot_count() : 0;
      s.last_uploads = debug_last_uploads;
      s.last_submits = debug_last_submits;
      s.last_unloads = debug_last_unloads;
      s.last_queue_pruned = debug_last_queue_pruned;
      s.last_free_pool_destroys = debug_last_free_pool_destroys;
      s.effective_uploads = effective_upload_limit();
      s.effective_submits = effective_submit_limit();
      s.effective_upload_budget_ms = effective_upload_budget_ms();
      s.frame_ms = debug_frame_ms;
      s.fps = debug_frame_ms > 0.f ? 1000.f / debug_frame_ms : 0.f;
      s.last_generated_vertices = gpu.debug_last_vertex_count;
      s.last_generated_indices = gpu.debug_last_index_count;
      s.mesh_overflows = gpu.debug_overflow_count;
      s.current_chunk_x = debug_center.x;
      s.current_chunk_z = debug_center.y;
      s.placed_vertices = debug_placed_vertices;
      s.placed_indices = debug_placed_indices;

      for (auto& [coord, chunk] : chunks) {
        (void)coord;
        if (chunk.uploaded) { ++s.uploaded_chunks; }
        if (chunk.pending) { ++s.pending_chunks; }
        if (chunk.dirty) { ++s.dirty_chunks; }
      }

      std::size_t terrain_chunk_bytes = 0;
      std::size_t free_pool_bytes = 0;
      for (auto& [coord, chunk] : chunks) {
        (void)coord;
        if (!chunk.uploaded) { continue; }
        terrain_chunk_bytes +=
          (std::size_t)chunk.vertex_capacity * sizeof(context_t::vertex_t) +
          (std::size_t)chunk.index_capacity * sizeof(std::uint32_t);
      }
      for (auto& h : free_handles) {
        free_pool_bytes +=
          (std::size_t)h.vertex_capacity * sizeof(context_t::vertex_t) +
          (std::size_t)h.index_capacity * sizeof(std::uint32_t);
      }
      std::size_t compute_slot_bytes =
        (std::size_t)gpu_chunk_generator_t::max_vertices * sizeof(context_t::vertex_t) +
        (std::size_t)gpu_chunk_generator_t::max_indices * sizeof(std::uint32_t) +
        sizeof(gpu_chunk_generator_t::counters_t) +
        gpu_chunk_generator_t::edit_buffer_size;
      std::size_t placed_bytes =
        s.placed_vertices * sizeof(context_t::vertex_t) +
        s.placed_indices * sizeof(std::uint32_t);
      std::size_t cpu_cached_bytes = 0;
      cpu_cached_bytes += work_queue.capacity() * sizeof(chunk_coord_t);
      cpu_cached_bytes += free_handles.capacity() * sizeof(free_chunk_handle_t);
      cpu_cached_bytes += submit_edits.capacity() * sizeof(block_edit_t);
      cpu_cached_bytes += chunks.bucket_count() * sizeof(void*);
      cpu_cached_bytes += chunks.size() * (sizeof(chunk_coord_t) + sizeof(terrain_chunk_t) + sizeof(void*) * 2);
      cpu_cached_bytes += block_edits.bucket_count() * sizeof(void*);
      cpu_cached_bytes += block_edits.size() * (sizeof(block_pos_t) + sizeof(std::uint8_t) + sizeof(void*) * 2);
      cpu_cached_bytes += placed_blocks.bucket_count() * sizeof(void*);
      cpu_cached_bytes += placed_blocks.size() * (sizeof(block_pos_t) + sizeof(std::uint8_t) + sizeof(void*) * 2);

      constexpr f32_t mb = 1024.f * 1024.f;
      s.estimated_terrain_mesh_mb = (f32_t)terrain_chunk_bytes / mb;
      s.estimated_free_pool_mesh_mb = (f32_t)free_pool_bytes / mb;
      s.free_pool_budget_mb = (f32_t)std::max(0, max_free_pool_mesh_mb);
      s.terrain_budget_mb = (f32_t)std::max(0, max_terrain_mesh_mb);
      s.max_loaded_chunks = (std::uint32_t)std::max(0, max_loaded_chunks);
      s.unload_margin = unload_margin;
      s.tree_detail_distance = tree_detail_distance;
      s.tree_cell = tree_cell_size;
      s.tree_density = tree_density;
      for (auto& [coord, chunk] : chunks) {
        (void)coord;
        if (chunk.uploaded && chunk.tree_density_u8 != 0) {
          ++s.tree_enabled_chunks;
        }
      }
      s.estimated_compute_slots_mb = (f32_t)(gpu_chunk_generator_t::slot_count * compute_slot_bytes) / mb;
      s.estimated_placed_mesh_mb = (f32_t)placed_bytes / mb;
      s.estimated_cpu_cached_mb = (f32_t)cpu_cached_bytes / mb;
      return s;
    }
    void render_gui_controls(context_t& renderer) {
      gui::drag("render distance", &render_distance);
      gui::drag("max loaded chunks", &max_loaded_chunks);
      gui::drag("terrain mesh budget MB", &max_terrain_mesh_mb);
      gui::drag("unload margin", &unload_margin);
      gui::drag("tree detail distance", &tree_detail_distance);
      gui::drag("tree cell size", &tree_cell_size);
      gui::drag("tree density", &tree_density);
      gui::drag("max uploads", &max_uploads_per_frame);
      gui::drag("max submits", &max_submits_per_frame);
      gui::drag("max adaptive uploads", &max_adaptive_uploads_per_frame);
      gui::drag("max adaptive submits", &max_adaptive_submits_per_frame);
      gui::drag("max pending", &max_pending_chunks);
      gui::drag("max free pool MB", &max_free_pool_mesh_mb);
      gui::drag("preload extra", &preload_extra);
      gui::drag("upload budget ms", &upload_budget_ms);

      auto s = debug_stats();
      auto debug_text = [](std::string_view text) {
        gui::text_box(text, fan::vec2(0.f), fan::colors::white, fan::colors::black.set_alpha(0.72f));
      };

      debug_text("terrain debug");
      debug_text(std::format("center chunk: {}, {}", s.current_chunk_x, s.current_chunk_z));
      debug_text(std::format("chunks: {} uploaded:{} pending:{} dirty:{} queued:{}", s.chunks, s.uploaded_chunks, s.pending_chunks, s.dirty_chunks, s.queued_chunks));
      debug_text(std::format("queue split: missing:{} dirty:{}", s.queued_missing_chunks, s.queued_dirty_chunks));
      debug_text(std::format("gpu slots: busy:{} free:{} / {}", s.gpu_slots_busy, s.gpu_slots_free, gpu_chunk_generator_t::slot_count));
      debug_text(std::format("frame: {:.2f} ms {:.0f} fps", s.frame_ms, s.fps));
      debug_text(std::format("frame stream: uploads:{} submits:{} unloads:{} queue pruned:{}", s.last_uploads, s.last_submits, s.last_unloads, s.last_queue_pruned));
      debug_text(std::format("stream limits: uploads:{} submits:{} budget:{:.2f} ms", s.effective_uploads, s.effective_submits, s.effective_upload_budget_ms));
      debug_text(std::format("edits: removed:{} placed:{}", s.removed_blocks, s.placed_blocks));
      debug_text(std::format("last chunk mesh: vertices:{} indices:{} overflows:{}", s.last_generated_vertices, s.last_generated_indices, s.mesh_overflows));
      debug_text(std::format("placed mesh: vertices:{} indices:{}", s.placed_vertices, s.placed_indices));
      debug_text(std::format("terrain vertex/index memory: {:.2f} MB", s.estimated_terrain_mesh_mb));
      debug_text(std::format("free pool vertex/index memory: {:.2f} MB", s.estimated_free_pool_mesh_mb));
      debug_text(std::format("free pool handles:{} budget:{:.0f} MB destroyed:{}", s.free_handles, s.free_pool_budget_mb, s.last_free_pool_destroys));
      debug_text(std::format("chunk budget: max:{} mesh budget:{:.0f} MB unload margin:{}", s.max_loaded_chunks, s.terrain_budget_mb, s.unload_margin));
      debug_text(std::format("trees: detail:{} cell:{} density:{:.2f} enabled chunks:{}", s.tree_detail_distance, s.tree_cell, s.tree_density, s.tree_enabled_chunks));
      debug_text(std::format("compute memory: {:.2f} MB, placed mesh: {:.2f} MB", s.estimated_compute_slots_mb, s.estimated_placed_mesh_mb));
      debug_text(std::format("cpu cached chunk data: {:.2f} MB", s.estimated_cpu_cached_mb));

      auto rt = renderer.memory_debug();
      debug_text(std::format("rt blas memory: {:.2f} MB count:{}", rt.rt_blas_mb, rt.blas_count));
      debug_text(std::format("rt blas scratch: {:.2f} MB peak:{:.2f} MB", rt.rt_blas_scratch_mb, rt.rt_blas_scratch_peak_mb));
      debug_text(std::format("rt tlas memory: {:.2f} MB scratch:{:.2f} MB peak:{:.2f} MB", rt.rt_tlas_mb, rt.rt_tlas_scratch_mb, rt.rt_tlas_scratch_peak_mb));
      debug_text(std::format("rt scene buffers: {:.2f} MB cpu cached:{:.2f} MB", rt.rt_scene_buffers_mb, rt.rt_cpu_cached_mb));
      debug_text(std::format("rt CPU source vertices: {:.2f} MB", rt.rt_cpu_source_vertices_mb));
      debug_text(std::format("rt CPU vertices: {:.2f} MB indices:{:.2f} MB", rt.rt_cpu_vertices_mb, rt.rt_cpu_indices_mb));
      debug_text(std::format("rt CPU materials: {:.2f} MB", rt.rt_cpu_materials_mb));
      debug_text(std::format("rt GPU source buffer: {:.2f} MB", rt.rt_gpu_source_buffer_mb));
      debug_text(std::format("rt GPU vertex buffer: {:.2f} MB index:{:.2f} MB", rt.rt_gpu_vertex_buffer_mb, rt.rt_gpu_index_buffer_mb));
      debug_text(std::format("rt global used: vertex:{:.2f} MB index:{:.2f} MB", rt.rt_global_vertex_used_mb, rt.rt_global_index_used_mb));
      debug_text(std::format("rt global free: vertex:{:.2f} MB/{} ranges index:{:.2f} MB/{} ranges", rt.rt_global_vertex_free_mb, rt.rt_global_vertex_free_ranges, rt.rt_global_index_free_mb, rt.rt_global_index_free_ranges));
      debug_text(std::format("rt global wasted/capacity slack: {:.2f} MB", rt.rt_global_wasted_mb));
      debug_text(std::format("textures/images: {:.2f} MB count:{}", rt.textures_images_mb, rt.texture_count));
      debug_text(std::format("compute capacity: vertices:{} indices:{}", gpu_chunk_generator_t::max_vertices, gpu_chunk_generator_t::max_indices));

      renderer.render_gui_controls();
    }
    bool render_gui(context_t& renderer, const char* window_name = "##rt") {
      if (auto h = gui::hud_interactive {window_name}; h) {
        render_gui_controls(renderer);
        return true;
      }
      return false;
    }
    void load_initial(context_t& renderer, chunk_coord_t center, std::int32_t radius, const fan::vec3& cam) {
      if (!gpu.ready()) { return; }

      for (std::int32_t r = 0; r <= radius; ++r) {
        (center - r).rect(center + r, [&](std::int32_t x, std::int32_t z) {
          chunk_coord_t c(x, z);
          chunk_coord_t d = c - center;
          if (d.length_squared() <= r * r) {
            queue_coord(c);
          }
        });
      }

      sort_and_trim_work_queue(center, cam);
      submit_pending(center, cam);
    }
    chunk_coord_t to_chunk(const fan::vec3& p) const {
      f32_t s = (f32_t)chunk_size * voxel_size;
      return {(std::int32_t)std::floor(p.x / s), (std::int32_t)std::floor(p.z / s)};
    }
    static std::int32_t floor_div(std::int32_t a, std::int32_t b) {
      std::int32_t q = a / b;
      std::int32_t r = a % b;
      return q - ((r != 0) && ((r < 0) != (b < 0)));
    }
    static std::int32_t floor_mod(std::int32_t a, std::int32_t b) {
      std::int32_t r = a % b;
      return r < 0 ? r + b : r;
    }
    block_pos_t world_to_block(const fan::vec3& p) const {
      return {
        (std::int32_t)std::floor(p.x / voxel_size),
        (std::int32_t)std::floor(p.y / voxel_size + (f32_t)sea_level),
        (std::int32_t)std::floor(p.z / voxel_size)
      };
    }
    fan::vec3 block_to_world(block_pos_t b) const {
      return fan::vec3(
        (f32_t)b.x * voxel_size,
        ((f32_t)b.y - (f32_t)sea_level) * voxel_size,
        (f32_t)b.z * voxel_size
      );
    }
    chunk_coord_t block_to_chunk(block_pos_t b) const {
      return {
        floor_div(b.x, (std::int32_t)chunk_size),
        floor_div(b.z, (std::int32_t)chunk_size)
      };
    }
    void collect_chunk_edits(chunk_coord_t coord, std::vector<block_edit_t>& out) const {
      out.clear();
      std::int32_t ox = coord.x * (std::int32_t)chunk_size;
      std::int32_t oz = coord.y * (std::int32_t)chunk_size;
      std::int32_t min_x = ox - 1;
      std::int32_t max_x = ox + (std::int32_t)chunk_size;
      std::int32_t min_z = oz - 1;
      std::int32_t max_z = oz + (std::int32_t)chunk_size;

      for (auto& [p, id] : block_edits) {
        if (p.x < min_x || p.x > max_x || p.z < min_z || p.z > max_z) {
          continue;
        }
        out.push_back({p.x, p.y, p.z, (std::int32_t)id});
        if (out.size() >= gpu_chunk_generator_t::max_edits_per_chunk) {
          break;
        }
      }
    }
    void queue_dirty_chunk(chunk_coord_t c) {
      auto& ch = chunks[c];
      ch.dirty = true;
      if (!ch.pending) {
        ch.pending = true;
        work_queue.push_back(c);
      }
    }
    void queue_dirty_block(block_pos_t b) {
      chunk_coord_t c = block_to_chunk(b);
      queue_dirty_chunk(c);
      std::int32_t lx = floor_mod(b.x, (std::int32_t)chunk_size);
      std::int32_t lz = floor_mod(b.z, (std::int32_t)chunk_size);
      if (lx == 0) { queue_dirty_chunk({c.x - 1, c.y}); }
      if (lx == (std::int32_t)chunk_size - 1) { queue_dirty_chunk({c.x + 1, c.y}); }
      if (lz == 0) { queue_dirty_chunk({c.x, c.y - 1}); }
      if (lz == (std::int32_t)chunk_size - 1) { queue_dirty_chunk({c.x, c.y + 1}); }
    }
    void set_block(context_t& renderer, block_pos_t b, std::uint8_t id) {
      auto it = placed_blocks.find(b);
      if (it != placed_blocks.end() && it->second == id) {
        return;
      }
      placed_blocks[b] = id;
      rebuild_placed_blocks(renderer);
    }
    void remove_block(context_t& renderer, block_pos_t b) {
      auto placed = placed_blocks.find(b);
      if (placed != placed_blocks.end()) {
        placed_blocks.erase(placed);
        rebuild_placed_blocks(renderer);
        return;
      }

      if (!solid_block(b)) {
        return;
      }

      block_edits[b] = 0;
      queue_dirty_block(b);
    }
    static std::uint32_t hash_u(std::uint32_t x) {
      x ^= x >> 16;
      x *= 0x7feb352du;
      x ^= x >> 15;
      x *= 0x846ca68bu;
      x ^= x >> 16;
      return x;
    }
    static f32_t hash21(std::int32_t x, std::int32_t z) {
      std::uint32_t h = hash_u((std::uint32_t)x * 0x8da6b343u ^ (std::uint32_t)z * 0xd8163841u);
      return (f32_t)(h & 0x00ffffffu) / (f32_t)0x01000000u;
    }
    static f32_t fade(f32_t t) {
      return t * t * t * (t * (t * 6.f - 15.f) + 10.f);
    }
    static f32_t lerp(f32_t a, f32_t b, f32_t t) {
      return a + (b - a) * t;
    }
    static f32_t value_noise(f32_t x, f32_t z) {
      std::int32_t ix = (std::int32_t)std::floor(x);
      std::int32_t iz = (std::int32_t)std::floor(z);
      f32_t fx = x - (f32_t)ix;
      f32_t fz = z - (f32_t)iz;
      f32_t ux = fade(fx);
      f32_t uz = fade(fz);

      f32_t a = hash21(ix, iz);
      f32_t b = hash21(ix + 1, iz);
      f32_t c = hash21(ix, iz + 1);
      f32_t d = hash21(ix + 1, iz + 1);

      return lerp(lerp(a, b, ux), lerp(c, d, ux), uz);
    }
    static f32_t fbm(f32_t x, f32_t z, std::int32_t o, f32_t l, f32_t g) {
      f32_t a = 0.5f;
      f32_t s = 0.f;
      f32_t n = 0.f;

      for (std::int32_t i = 0; i < o; ++i) {
        s += value_noise(x, z) * a;
        n += a;
        x *= l;
        z *= l;
        a *= g;
      }

      return s / std::max(n, 0.0001f);
    }
    static std::uint32_t height_at(std::int32_t x, std::int32_t z) {
      f32_t h =
        fbm((f32_t)x * 0.0028f, (f32_t)z * 0.0028f, 5, 2.f, 0.55f) * 0.7f +
        fbm((f32_t)x * 0.011f, (f32_t)z * 0.011f, 5, 2.f, 0.5f) * 0.25f +
        fbm((f32_t)x * 0.045f, (f32_t)z * 0.045f, 3, 2.f, 0.45f) * 0.05f;

      h = std::clamp((h - 0.25f) / 0.72f, 0.f, 1.f);
      h = h * h * (3.f - 2.f * h);

      return (std::uint32_t)std::clamp(8.f + h * ((f32_t)sy - 14.f), 1.f, (f32_t)sy - 2.f);
    }
    std::int32_t edit_at(block_pos_t b) const {
      auto it = block_edits.find(b);
      return it == block_edits.end() ? -1 : (std::int32_t)it->second;
    }
    bool solid_block(block_pos_t b) const {
      if (b.y < 0) {
        return false;
      }
      if (placed_blocks.contains(b)) {
        return true;
      }
      std::int32_t edit = edit_at(b);
      if (edit == 0) {
        return false;
      }
      if (edit > 0) {
        return true;
      }
      return b.y <= (std::int32_t)height_at(b.x, b.z);
    }
    std::int32_t column_top_y(std::int32_t bx, std::int32_t bz) const {
      std::int32_t max_y = (std::int32_t)sy - 1;
      for (auto& [p, id] : placed_blocks) {
        if (p.x == bx && p.z == bz) {
          max_y = std::max(max_y, p.y);
        }
      }
      for (std::int32_t y = max_y; y >= 0; --y) {
        if (solid_block({bx, y, bz})) {
          return y;
        }
      }
      return -1;
    }
    f32_t ground_y(f32_t x, f32_t z) const {
      std::int32_t bx = (std::int32_t)std::floor(x / voxel_size);
      std::int32_t bz = (std::int32_t)std::floor(z / voxel_size);
      std::int32_t y = column_top_y(bx, bz);
      return ((f32_t)y + 1.f - (f32_t)sea_level) * voxel_size;
    }
    f32_t ground_y(f32_t x, f32_t z, f32_t radius) const {
      f32_t y = ground_y(x, z);
      y = std::max(y, ground_y(x + radius, z + radius));
      y = std::max(y, ground_y(x + radius, z - radius));
      y = std::max(y, ground_y(x - radius, z + radius));
      y = std::max(y, ground_y(x - radius, z - radius));
      return y;
    }
    gpu_chunk_generator_t::push_t make_push(chunk_coord_t c, chunk_coord_t center) const {
      tree_lod_t lod = tree_lod_for_chunk(c, center);
      return {
        .chunk_x = c.x,
        .chunk_z = c.y,
        .chunk_size = chunk_size,
        .sy = sy,
        .sea_level = sea_level,
        .snow_line = snow_line,
        .rock_line = rock_line,
        .tree_cell = lod.cell,
        .voxel_size = voxel_size,
        .tree_density = lod.density,
      };
    }
    std::size_t find_free_handle(const context_t& renderer, std::uint32_t vertex_capacity, std::uint32_t index_capacity) const {
      bool hard = free_pool_over_hard_budget();
      if (!hard && !can_recycle_geometry(renderer)) { return std::size_t(-1); }
      for (std::size_t i = 0; i < free_handles.size(); ++i) {
        if ((hard || free_handles[i].age >= free_handle_safe_age) && free_handles[i].vertex_capacity >= vertex_capacity && free_handles[i].index_capacity >= index_capacity) {
          return i;
        }
      }
      return std::size_t(-1);
    }
    context_t::object_handle_t upload_or_reuse_handle(context_t& renderer, context_t::voxel_mesh_input_t& mesh, std::uint32_t mesh_vertex_capacity, std::uint32_t mesh_index_capacity) {
      std::size_t free_index = find_free_handle(renderer, mesh_vertex_capacity, mesh_index_capacity);
      if (free_index == std::size_t(-1)) {
        return renderer.add_mesh_incremental(mesh);
      }

      renderer.wait_idle();
      context_t::object_handle_t handle = free_handles[free_index].handle;
      free_handles[free_index] = free_handles.back();
      free_handles.pop_back();
      if (!renderer.update_mesh(handle, mesh)) {
        renderer.destroy_object_geometry(handle);
        return renderer.add_mesh_incremental(mesh);
      }

      renderer.set_transform_deferred(handle, mesh.transform);
      return handle;
    }
    void poll_completed(context_t& renderer, chunk_coord_t center) {
      fan::time::timer t{true};
      std::uint32_t uploaded = 0;
      bool batch = false;
      std::uint32_t upload_limit = effective_upload_limit();
      f32_t upload_budget = effective_upload_budget_ms();
      while (uploaded < upload_limit) {
        if (uploaded && t.millis() > upload_budget) { break; }
        chunk_coord_t coord {};
        context_t::voxel_mesh_input_t mesh;
        if (!gpu.fetch_completed(coord, mesh)) { break; }
        auto it = chunks.find(coord);
        if (it == chunks.end()) { continue; }
        auto& ch = it->second;
        if (!in_range(coord, center, 1)) { ch.pending = false; ch.dirty = false; if (!ch.uploaded) { chunks.erase(it); } continue; }
        if (mesh.vertices.empty() || mesh.indices.empty()) { ch.pending = false; ch.dirty = false; continue; }

        if (ch.dirty) {
          ch.pending = false;
          ch.dirty = false;
          queue_dirty_chunk(coord);
          continue;
        }

        prepare_chunk_mesh(mesh);
        if (!batch) { renderer.begin_incremental_upload(); batch = true; }
        std::uint32_t mesh_vertex_capacity = mesh.vertex_capacity;
        std::uint32_t mesh_index_capacity = mesh.index_capacity;
        if (ch.uploaded) {
          bool updated_in_place = false;
          if (mesh.vertices.size() <= ch.vertex_capacity && mesh.indices.size() <= ch.index_capacity) {
            mesh.vertex_capacity = ch.vertex_capacity;
            mesh.index_capacity = ch.index_capacity;
            updated_in_place = renderer.update_mesh(ch.handle, mesh);
            mesh_vertex_capacity = ch.vertex_capacity;
            mesh_index_capacity = ch.index_capacity;
          }

          if (!updated_in_place) {
            context_t::object_handle_t old_handle = ch.handle;
            std::uint32_t old_vertex_capacity = ch.vertex_capacity;
            std::uint32_t old_index_capacity = ch.index_capacity;

            ch.handle = upload_or_reuse_handle(renderer, mesh, mesh_vertex_capacity, mesh_index_capacity);

            renderer.set_transform_deferred(old_handle, fan::translate(fan::vec3(0.f, -1000000.f, 0.f)));
            free_handles.push_back({old_handle, old_vertex_capacity, old_index_capacity, 0});
          }
        }
        else {
          ch.handle = upload_or_reuse_handle(renderer, mesh, mesh_vertex_capacity, mesh_index_capacity);
        }
        ch.vertex_capacity = mesh_vertex_capacity;
        ch.index_capacity = mesh_index_capacity;
        tree_lod_t applied_lod = tree_lod_for_chunk(coord, center);
        ch.tree_cell = applied_lod.cell;
        ch.tree_density_u8 = applied_lod.density_u8;
        ch.uploaded = true; ch.pending = false; ch.dirty = false; ++uploaded; ++debug_last_uploads;
      }
      if (batch) { renderer.end_incremental_upload(); }
      trim_free_pool(renderer);
    }
    f32_t chunk_distance2(chunk_coord_t c, const fan::vec3& cam) const {
      f32_t s = (f32_t)chunk_size * voxel_size;
      f32_t x0 = (f32_t)c.x * s;
      f32_t x1 = (f32_t)(c.x + 1) * s;
      f32_t z0 = (f32_t)c.y * s;
      f32_t z1 = (f32_t)(c.y + 1) * s;

      f32_t dx = cam.x < x0 ? x0 - cam.x : (cam.x > x1 ? cam.x - x1 : 0.f);
      f32_t dz = cam.z < z0 ? z0 - cam.z : (cam.z > z1 ? cam.z - z1 : 0.f);
      return dx * dx + dz * dz;
    }
    void sort_and_trim_work_queue(chunk_coord_t center, const fan::vec3& cam) {
      std::ranges::sort(work_queue, {}, [&](const chunk_coord_t& p) {
        return work_priority(p, center, cam);
      });

      std::uint32_t pending_cap = std::max<std::uint32_t>(max_pending_chunks, effective_submit_limit() + gpu_chunk_generator_t::slot_count);
      std::vector<chunk_coord_t> kept;
      kept.reserve(std::min<std::size_t>(work_queue.size(), pending_cap));

      for (chunk_coord_t c : work_queue) {
        auto it = chunks.find(c);
        if (it == chunks.end()) {
          continue;
        }

        if (kept.size() < pending_cap) {
          kept.push_back(c);
          continue;
        }

        it->second.pending = false;
        if (!it->second.uploaded) {
          chunks.erase(it);
        }
      }

      work_queue = std::move(kept);
    }
    void submit_pending(chunk_coord_t center, const fan::vec3& cam) {
      sort_and_trim_work_queue(center, cam);

      std::uint32_t submit_limit = effective_submit_limit();
      std::uint32_t submitted = 0;
      while (!work_queue.empty() && submitted < submit_limit && gpu.free_slot_count()) {
        std::size_t bi = closest_work_index(cam);
        chunk_coord_t coord = work_queue[bi];
        if (!in_range(coord, center, 1)) {
          auto it = chunks.find(coord);
          if (it != chunks.end()) { it->second.pending = false; if (!it->second.uploaded) { chunks.erase(it); } }
          work_queue.erase(work_queue.begin() + bi);
          continue;
        }
        collect_chunk_edits(coord, submit_edits);
        if (!gpu.submit(make_push(coord, center), coord, submit_edits)) { break; }
        chunks[coord].dirty = false;
        work_queue.erase(work_queue.begin() + bi);
        ++submitted;
        ++debug_last_submits;
      }
    }
    std::size_t closest_work_index(const fan::vec3& cam) const {
      const auto it = std::ranges::min_element(work_queue, {}, [&](const chunk_coord_t& p) {
        return work_priority(p, debug_center, cam);
      });
      return std::distance(work_queue.begin(), it);
    }
    void queue_coord(chunk_coord_t c) {
      auto& ch = chunks[c];
      if (!ch.uploaded && !ch.pending) {
        ch.pending = true;
        work_queue.push_back(c);
      }
    }
    void queue_missing(chunk_coord_t center, const fan::vec3& cam) {
      std::int32_t r = render_distance + preload_extra;
      std::int32_t r2 = r * r;
      (center - r).rect(center + r, [&](std::int32_t x, std::int32_t z) {
        chunk_coord_t c(x, z);
        if ((c - center).length_squared() <= r2) {
          queue_coord(c);
        }
      });

      sort_and_trim_work_queue(center, cam);
    }
    std::int32_t register_rt_image(context_t& renderer, fan::graphics::image_t& img) {
      auto& vk_img = renderer.ctx->image_get(img);
      renderer.texture_ids.push_back(img);
      renderer.rt_texture_infos.push_back({vk_img.sampler, vk_img.image_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
      std::int32_t slot = (std::int32_t)renderer.rt_texture_infos.size() - 1;
      renderer.update_rt_textures_descriptor();
      return slot;
    }
    fan::graphics::image_t atlas;
    std::int32_t atlas_slot = -1;
    ray_tracing::material_info_t terrain_mat {};
    gpu_chunk_generator_t gpu;
    bool bootstrap_added = false;
    std::vector<chunk_coord_t> work_queue;
    std::vector<free_chunk_handle_t> free_handles;
    chunk_map_t chunks;
    std::unordered_map<block_pos_t, std::uint8_t, block_pos_hash_t> block_edits;
    std::unordered_map<block_pos_t, std::uint8_t, block_pos_hash_t> placed_blocks;
    context_t::object_handle_t placed_overlay;
    bool placed_overlay_uploaded = false;
    std::vector<block_edit_t> submit_edits;
    chunk_coord_t debug_center {};
    std::uint32_t debug_last_uploads = 0;
    std::uint32_t debug_last_submits = 0;
    std::uint32_t debug_last_unloads = 0;
    std::uint32_t debug_last_queue_pruned = 0;
    std::uint32_t debug_last_free_pool_destroys = 0;
    std::size_t debug_placed_vertices = 0;
    std::size_t debug_placed_indices = 0;
    std::int32_t render_distance = 8, initial_load_radius = 3;
    static constexpr std::uint32_t free_handle_safe_age = 32;
    std::uint32_t max_uploads_per_frame = 2, max_submits_per_frame = 4, max_pending_chunks = 384;
    bool adaptive_streaming = false;
    std::uint32_t max_adaptive_uploads_per_frame = 1;
    std::uint32_t max_adaptive_submits_per_frame = 1;
    std::int32_t max_free_pool_mesh_mb = 24;
    std::int32_t max_loaded_chunks = 220;
    std::int32_t max_terrain_mesh_mb = 180;
    std::int32_t preload_extra = 0;
    std::int32_t unload_margin = 0;
    std::int32_t tree_detail_distance = 7;
    std::int32_t tree_cell_size = 10;
    f32_t tree_density = 1.f;
    f32_t upload_budget_ms = 0.25f;
    bool timing_initialized = false;
    std::chrono::steady_clock::time_point last_update_time {};
    f32_t debug_frame_ms = 0.f;
  };

}

#endif