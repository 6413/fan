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

export namespace fan::graphics::vulkan::ray_tracing {
  struct chunk_coord_t {
    bool operator==(const chunk_coord_t& o) const { return x == o.x && z == o.z; }
    std::int32_t x = 0;
    std::int32_t z = 0;
  };
  struct chunk_coord_hash_t {
    std::size_t operator()(const chunk_coord_t& c) const {
    return (std::size_t)c.x | ((std::size_t)c.z << 32);    }
  };
  struct terrain_chunk_t {
    context_t::object_handle_t handle;
    bool uploaded = false;
    bool pending = false;
  };

  struct gpu_chunk_generator_t {
    static constexpr std::uint32_t chunk_size = 32;
    static constexpr std::uint32_t max_faces_per_column = 6;
    static constexpr std::uint32_t max_vertices = chunk_size * chunk_size * max_faces_per_column * 4;
    static constexpr std::uint32_t max_indices = chunk_size * chunk_size * max_faces_per_column * 6;
    static constexpr std::uint32_t slot_count = 4;
    struct counters_t { std::uint32_t vertex_count = 0, index_count = 0, pad0 = 0, pad1 = 0; };
    struct push_t {
      std::int32_t chunk_x = 0, chunk_z = 0;
      std::uint32_t chunk_size = 32, sy = 96, sea_level = 28, snow_line = 74, rock_line = 62, pad0 = 0;
      f32_t voxel_size = 1.5f, pad1 = 0.f, pad2 = 0.f, pad3 = 0.f;
    };
    struct state_t { chunk_coord_t coord {}; push_t push {}; };
    using vk_context_t = fan::vulkan::context_t;

    void open(context_t& renderer) {
      ctx = renderer.ctx;
      pipeline.open(*ctx, "shaders/vulkan/ray_tracing/chunk_gen.comp", sizeof(push_t), {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}
      });
      constexpr VkMemoryPropertyFlags mem = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
      slots.open(*ctx, slot_count, pipeline.descriptor_layout, {
        {max_vertices * sizeof(context_t::vertex_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mem},
        {max_indices * sizeof(std::uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mem},
        {sizeof(counters_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, mem}
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
    bool submit(const push_t& push, chunk_coord_t coord) {
      if (!ready()) { return false; }
      std::uint32_t si = slots.acquire();
      if (si == vk_context_t::compute_slot_ring_t::invalid_slot) { return false; }
      auto& slot = slots.get(si);
      auto cmd = slots.begin(*ctx, si);
      ctx->fill_buffer_cmd(cmd, slot.buffers[2], 0, sizeof(counters_t), 0);
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
      if (!c->vertex_count || !c->index_count || c->vertex_count > max_vertices || c->index_count > max_indices) { mesh.vertices.clear(); mesh.indices.clear(); return false; }
      mesh.vertices.resize(c->vertex_count);
      mesh.indices.resize(c->index_count);
      auto vb = mesh.vertices.size() * sizeof(context_t::vertex_t), ib = mesh.indices.size() * sizeof(std::uint32_t);
      ctx->invalidate_buffer(slot.buffers[0], 0, vb);
      ctx->invalidate_buffer(slot.buffers[1], 0, ib);
      std::memcpy(mesh.vertices.data(), slot.buffers[0].mapped, (std::size_t)vb);
      std::memcpy(mesh.indices.data(), slot.buffers[1].mapped, (std::size_t)ib);
      mesh.materials.clear(); mesh.primitive_material_indices.clear();
      mesh.transform = fan::translate(fan::vec3((f32_t)(state.coord.x * (std::int32_t)state.push.chunk_size) * state.push.voxel_size, -(f32_t)state.push.sea_level * state.push.voxel_size, (f32_t)(state.coord.z * (std::int32_t)state.push.chunk_size) * state.push.voxel_size));
      return true;
    }

    fan::vulkan::context_t* ctx = nullptr;
    vk_context_t::compute_pipeline_t pipeline;
    vk_context_t::compute_slot_ring_t slots;
    std::array<state_t, slot_count> states {};
  };

  context_t::voxel_mesh_input_t make_bootstrap_mesh() {
    context_t::voxel_mesh_input_t mesh;
    mesh.vertices.resize(3);
    mesh.indices = {0, 1, 2};
    for (auto& v : mesh.vertices) { v.position = fan::vec3(0); v.normal = fan::vec3(0, 1, 0); v.texcoord = fan::vec2(0); v.color = fan::vec3(0); }
    return mesh;
  }

  struct gpu_terrain_streamer_t {
    static constexpr std::uint32_t chunk_size = 32, sy = 96, sea_level = 28, snow_line = 74, rock_line = 62;
    static constexpr f32_t voxel_size = 1.5f;
    void init() {}
    void open(context_t& renderer) {
      if (gpu.ready()) { return; }
      gpu.open(renderer);
      renderer.reserve_scene_buffers(2'000'000, 6'000'000, 64'000, 2'000'000);
      atlas = fan::graphics::image_t {"images/mctp.webp"};
      atlas_slot = register_rt_image(renderer, atlas);
      terrain_mat.albedo_texture_id = atlas_slot;
      terrain_mat.base_color = fan::vec3(1.f, 1.f, 1.f);
    }
    void destroy() { gpu.close(); }
    bool in_range(chunk_coord_t c, chunk_coord_t center, std::int32_t extra = 0) const {
      std::int32_t r = render_distance + extra, dx = c.x - center.x, dz = c.z - center.z;
      return dx * dx + dz * dz <= r * r;
    }
    void prune_queue(chunk_coord_t center) {
      for (std::size_t i = 0; i < work_queue.size();) {
        if (in_range(work_queue[i], center, 1)) { ++i; continue; }
        auto it = chunks.find(work_queue[i]);
        if (it != chunks.end()) { it->second.pending = false; if (!it->second.uploaded) { chunks.erase(it); } }
        work_queue.erase(work_queue.begin() + i);
      }
    }
    void unload_far(context_t& renderer, chunk_coord_t center) {
      for (auto it = chunks.begin(); it != chunks.end();) {
        if (!it->second.uploaded || in_range(it->first, center, 2)) { ++it; continue; }
        renderer.remove_object(it->second.handle);
        it = chunks.erase(it);
      }
    }
    bool ready() const { return gpu.ready(); }
    void update(context_t& renderer, const fan::vec3& cam) {
      if (!bootstrap_added) { renderer.add_mesh(make_bootstrap_mesh()); bootstrap_added = true; }
      if (!gpu.ready()) {
        if (!renderer.ready) { return; }
        open(renderer);
        load_initial(renderer, to_chunk(cam), initial_load_radius);
      }
      chunk_coord_t center = to_chunk(cam);
      prune_queue(center); unload_far(renderer, center); queue_missing(center); poll_completed(renderer, center); submit_pending(center);
    }
    void render_gui(context_t& renderer) {
      fan::graphics::gui::drag("render distance", &render_distance);
      fan::graphics::gui::drag("max uploads", &max_uploads_per_frame);
      fan::graphics::gui::drag("max submits", &max_submits_per_frame);
      fan::graphics::gui::drag("upload budget ms", &upload_budget_ms);
      renderer.render_gui();
    }
    void load_initial(context_t& renderer, chunk_coord_t center, std::int32_t radius) {
      if (!gpu.ready()) { return; }
      for (std::int32_t z = center.z - radius; z <= center.z + radius; ++z) {
        for (std::int32_t x = center.x - radius; x <= center.x + radius; ++x) {
          std::int32_t dx = x - center.x, dz = z - center.z;
          if (dx * dx + dz * dz <= radius * radius) { queue_coord({x, z}); }
        }
      }
      submit_pending(center);
    }
    chunk_coord_t to_chunk(const fan::vec3& p) const {
      f32_t s = (f32_t)chunk_size * voxel_size;
      return {(std::int32_t)std::floor(p.x / s), (std::int32_t)std::floor(p.z / s)};
    }
    gpu_chunk_generator_t::push_t make_push(chunk_coord_t c) const {
      gpu_chunk_generator_t::push_t p;
      p.chunk_x = c.x; p.chunk_z = c.z; p.chunk_size = chunk_size; p.sy = sy; p.sea_level = sea_level; p.snow_line = snow_line; p.rock_line = rock_line; p.voxel_size = voxel_size;
      return p;
    }
    void poll_completed(context_t& renderer, chunk_coord_t center) {
      fan::time::timer t{true};
      std::uint32_t uploaded = 0;
      bool batch = false;
      while (uploaded < max_uploads_per_frame) {
        if (uploaded && t.millis() > upload_budget_ms) { break; }
        chunk_coord_t coord {};
        context_t::voxel_mesh_input_t mesh;
        if (!gpu.fetch_completed(coord, mesh)) { break; }
        auto it = chunks.find(coord);
        if (it == chunks.end()) { continue; }
        auto& ch = it->second;
        if (!in_range(coord, center, 1)) { ch.pending = false; if (!ch.uploaded) { chunks.erase(it); } continue; }
        if (ch.uploaded) { ch.pending = false; continue; }
        if (mesh.vertices.empty() || mesh.indices.empty()) { ch.pending = false; continue; }
        mesh.materials = {terrain_mat};
        mesh.primitive_material_indices.assign(mesh.indices.size() / 3, 0);
        if (!batch) { renderer.begin_incremental_upload(); batch = true; }
        ch.handle = renderer.add_mesh_incremental(mesh); ch.uploaded = true; ch.pending = false; ++uploaded;
      }
      if (batch) { renderer.end_incremental_upload(); }
    }
    void submit_pending(chunk_coord_t center) {
      std::uint32_t submitted = 0;
      while (!work_queue.empty() && submitted < max_submits_per_frame && gpu.free_slot_count()) {
        std::size_t bi = closest_work_index(center);
        chunk_coord_t coord = work_queue[bi];
        if (!in_range(coord, center, 1)) {
          auto it = chunks.find(coord);
          if (it != chunks.end()) { it->second.pending = false; if (!it->second.uploaded) { chunks.erase(it); } }
          work_queue.erase(work_queue.begin() + bi);
          continue;
        }
        if (!gpu.submit(make_push(coord), coord)) { break; }
        work_queue.erase(work_queue.begin() + bi);
        ++submitted;
      }
    }
    std::size_t closest_work_index(chunk_coord_t center) const {
      std::size_t best = 0;
      std::int32_t best_d = std::numeric_limits<std::int32_t>::max();
      for (std::size_t i = 0; i < work_queue.size(); ++i) {
        std::int32_t dx = work_queue[i].x - center.x, dz = work_queue[i].z - center.z, d = dx * dx + dz * dz;
        if (d < best_d) { best_d = d; best = i; }
      }
      return best;
    }
    void queue_coord(chunk_coord_t c) {
      auto& ch = chunks[c];
      if (!ch.uploaded && !ch.pending && work_queue.size() < max_pending_chunks) { ch.pending = true; work_queue.push_back(c); }
    }
    void queue_missing(chunk_coord_t center) {
      for (std::int32_t z = center.z - render_distance; z <= center.z + render_distance; ++z) {
        for (std::int32_t x = center.x - render_distance; x <= center.x + render_distance; ++x) {
          std::int32_t dx = x - center.x, dz = z - center.z;
          if (dx * dx + dz * dz <= render_distance * render_distance) { queue_coord({x, z}); }
        }
      }
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
    std::unordered_map<chunk_coord_t, terrain_chunk_t, chunk_coord_hash_t> chunks;
    std::int32_t render_distance = 12, initial_load_radius = 3;
    std::uint32_t max_uploads_per_frame = 2, max_submits_per_frame = 4, max_pending_chunks = 256;
    f32_t upload_budget_ms = 2.0f;
  };

}

#endif