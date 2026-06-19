module;

#if defined(FAN_3D) && defined(FAN_VULKAN)
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <shaderc/shaderc.hpp>
#endif

export module fan.graphics.vulkan.ray_tracing.hardware_renderer;

import std;

#if defined(FAN_3D) && defined(FAN_VULKAN)

import fan.types.matrix;
import fan.graphics.vulkan.core;
import fan.graphics;
import fan.graphics.vulkan.ray_tracing.shapes;
import fan.random;
import fan.graphics.fms;
import fan.graphics.gui.base;
import fan.graphics.loco;
import fan.math.intersection;
import fan.print.error;

export namespace fan::graphics::vulkan::ray_tracing {
  struct acceleration_structure_t {
    void destroy(fan::vulkan::context_t& ctx) {
      if (handle) {
        auto destroy_as = (PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(ctx.device, "vkDestroyAccelerationStructureKHR");
        destroy_as(ctx.device, handle, nullptr);
      }
      ctx.destroy_buffer(buffer);
      handle = VK_NULL_HANDLE;
      device_address = 0;
    }
    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
    fan::vulkan::context_t::buffer_t buffer;
    VkDeviceAddress device_address = 0;
  };

#pragma pack(push, 1)
  struct material_info_t {
    std::int32_t albedo_texture_id = -1;
    std::int32_t normal_texture_id = -1;
    std::int32_t metallic_texture_id = -1;
    std::int32_t roughness_texture_id = -1;
    fan::vec3 base_color = fan::vec3(1.0f, 1.0f, 1.0f);
    std::uint32_t source_material_id = 0;
    fan::vec4 uv_transform = fan::vec4(0, 0, 0, 0);
  };
#pragma pack(pop)

  struct light_t {
    fan::vec3 position;
    f32_t radius;
    fan::vec3 color;
    f32_t intensity;
  };

  struct light_ubo_t {
    fan::vec3 position;
    f32_t pad0;
    fan::vec3 color;
    f32_t intensity;
  };

  struct context_t {
    struct submesh_t {
      std::uint32_t first_vertex;
      std::uint32_t vertex_count;
      std::uint32_t first_index;
      std::uint32_t index_count;
    };
    struct model_t {
      std::uint32_t first_index;
      std::uint32_t index_count;
      std::uint32_t material_index;
      std::uint32_t first_primitive;
      std::uint32_t first_vertex;
      std::uint32_t vertex_count;
      std::uint32_t vertex_capacity = 0;
      std::uint32_t index_capacity = 0;
      std::uint32_t blas_primitive_count = 0;
      std::uint32_t blas_primitive_capacity = 0;
      std::uint32_t material_count = 1;
      std::uint32_t material_capacity = 1;
      std::uint32_t first_bone = 0;
      std::uint32_t bone_count = 0;
      fan::vec3 aabb_min = fan::vec3(0);
      fan::vec3 aabb_max = fan::vec3(0);
      bool has_bounds = false;
      bool animated = false;
      bool keep_cpu_geometry = true;
    };
    struct instance_t {
      std::uint32_t model_index;
      fan::mat4 transform;
      std::uint32_t mask = 0x01;
    };
    struct object_handle_t {
      bool valid() const {
        return index != std::uint32_t(-1);
      }
      std::uint32_t index = std::uint32_t(-1);
      std::uint32_t generation = 0;
    };
    struct scene_model_t {
      std::string path;
      fan::mat4 transform = fan::mat4(1);
      std::string texture_path = "models/textures";
      std::string animation_name;
      f32_t animation_time_offset = 0.f;
      f32_t animation_speed = 1.f;
      std::source_location callers_path;
      std::uint32_t procedural_mesh_index = std::uint32_t(-1);
      bool fix_uv_diagonals = false;
      bool animated = false;
    };
    struct model_cache_entry_t {
      std::uint32_t first_model = 0;
      std::uint32_t model_count = 0;
    };
    struct animation_frame_t {
      std::uint32_t first_model = 0;
      std::uint32_t model_count = 0;
      f32_t time_ms = 0.f;
    };
    struct animation_clip_cache_t {
      std::string name;
      f32_t duration_ms = 0.f;
      std::vector<animation_frame_t> frames;
    };
    struct animation_cache_entry_t {
      std::string key;
      std::string default_clip;
      std::uint32_t bone_count = 0;
      f32_t sample_rate = 12.f;
      std::vector<animation_clip_cache_t> clips;
      std::unordered_map<std::string, std::uint32_t> clip_indices;
    };
    struct memory_debug_t {
      std::size_t model_count = 0;
      std::size_t instance_count = 0;
      std::size_t blas_count = 0;
      std::size_t texture_count = 0;
      f32_t rt_blas_mb = 0.f;
      f32_t rt_tlas_mb = 0.f;
      f32_t rt_blas_scratch_mb = 0.f;
      f32_t rt_blas_scratch_peak_mb = 0.f;
      f32_t rt_tlas_scratch_mb = 0.f;
      f32_t rt_tlas_scratch_peak_mb = 0.f;
      f32_t rt_scene_buffers_mb = 0.f;
      f32_t rt_cpu_cached_mb = 0.f;
      f32_t rt_cpu_source_vertices_mb = 0.f;
      f32_t rt_cpu_vertices_mb = 0.f;
      f32_t rt_cpu_indices_mb = 0.f;
      f32_t rt_cpu_materials_mb = 0.f;
      f32_t rt_gpu_source_buffer_mb = 0.f;
      f32_t rt_gpu_vertex_buffer_mb = 0.f;
      f32_t rt_gpu_index_buffer_mb = 0.f;
      f32_t rt_global_vertex_used_mb = 0.f;
      f32_t rt_global_vertex_free_mb = 0.f;
      f32_t rt_global_index_used_mb = 0.f;
      f32_t rt_global_index_free_mb = 0.f;
      f32_t rt_global_wasted_mb = 0.f;
      std::size_t rt_global_vertex_free_ranges = 0;
      std::size_t rt_global_index_free_ranges = 0;
      f32_t textures_images_mb = 0.f;
    };
    struct scene_range_t {
      std::uint32_t first = 0;
      std::uint32_t count = 0;
    };
    struct object_t {
      std::uint32_t generation = 0;
      std::uint32_t first_instance = 0;
      std::uint32_t instance_count = 0;
      std::uint32_t animated_model_index = std::uint32_t(-1);
      std::uint32_t animation_cache_index = std::uint32_t(-1);
      std::string animation_name;
      f32_t animation_time_offset = 0.f;
      f32_t animation_speed = 1.f;
      std::uint32_t animation_frame = std::uint32_t(-1);
      std::uint32_t ray_mask = 0x01;
    };
    struct engine_open_properties_t {
      fan::vec2ui size {};
      bool create_output_sprite = true;
    };
    struct animated_model_t {
      std::unique_ptr<fan::model::fms_t> fms;
      std::uint32_t first_model = 0;
      std::uint32_t model_count = 0;
      std::uint32_t first_bone = 0;
      std::uint32_t bone_count = 0;
      bool dirty = false;
    };
  #pragma pack(push, 1)
    struct vertex_t {
      fan::vec4 position;
      fan::vec4 normal;
      fan::vec2 texcoord;
      std::uint32_t color;
      std::uint32_t pad0;
    };

    static_assert(sizeof(vertex_t) == 48);
    struct source_vertex_t {
      fan::vec3 position; f32_t pad0;
      fan::vec3 normal; f32_t pad1;
      fan::vec2 texcoord; fan::vec2 pad2;
      fan::vec3 color; f32_t pad3;
      fan::vec4i bone_ids;
      fan::vec4 bone_weights;
    };
    struct skinning_push_constants_t {
      std::uint32_t first_vertex;
      std::uint32_t vertex_count;
      std::uint32_t first_bone;
      std::uint32_t bone_count;
    };
  #pragma pack(pop)
    struct voxel_t {
      std::uint8_t id = 0;
      fan::vec4 color = fan::vec4(1, 1, 1, 1);
    };
    struct voxel_surface_t {
      bool visible = false;
      std::uint8_t id = 0;
      std::uint32_t color_key = 0;
      std::uint32_t material_index = 0;
      fan::vec4 color = fan::vec4(1, 1, 1, 1);
    };
    struct voxel_grid_t {
      void resize(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
        sx = x; sy = y; sz = z;
        std::size_t req = (std::size_t)x * y * z;
        if (data.size() < req) {
          data.resize(req);
        }
        std::memset(data.data(), 0, req * sizeof(voxel_t));
      }
      voxel_t& at(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
        return data[(std::size_t)z * sy * sx + (std::size_t)y * sx + x];
      }
      const voxel_t& at(std::uint32_t x, std::uint32_t y, std::uint32_t z) const {
        return data[(std::size_t)z * sy * sx + (std::size_t)y * sx + x];
      }
      std::uint32_t sx = 0;
      std::uint32_t sy = 0;
      std::uint32_t sz = 0;
      std::vector<voxel_t> data;
    };
    struct voxelizer_workspace_t {
      voxel_grid_t grid;
      std::vector<f32_t> cell_area;
      std::vector<voxel_surface_t> mask;
    };
    struct voxel_mesh_input_t {
      std::vector<vertex_t> vertices;
      std::vector<std::uint32_t> indices;
      std::vector<material_info_t> materials;
      std::vector<std::uint32_t> primitive_material_indices;
      fan::mat4 transform = fan::mat4(1);
      fan::vec3 base_color = fan::vec3(1, 1, 1);
      std::int32_t albedo_texture_id = -1;
      std::uint32_t vertex_capacity = 0;
      std::uint32_t index_capacity = 0;
      std::uint32_t material_capacity = 0;
      bool keep_cpu_copy = true;
    };
    struct atlas_t {
      fan::vec4 tile_uv(std::uint32_t tile) const {
        std::uint32_t columns = std::max<std::uint32_t>(tile_count.x, 1);
        std::uint32_t rows = std::max<std::uint32_t>(tile_count.y, 1);
        std::uint32_t x = tile % columns;
        std::uint32_t y = tile / columns;
        fan::vec2 tile_size(1.f / (f32_t)columns, 1.f / (f32_t)rows);
        fan::vec2 min_uv((f32_t)x * tile_size.x, (f32_t)y * tile_size.y);
        fan::vec2 pad = padding.min(tile_size * 0.45f);
        fan::vec2 size = (tile_size - pad * 2.f).max(fan::vec2(0));
        return fan::vec4(min_uv.x + pad.x, min_uv.y + pad.y, size.x, size.y);
      }
      std::int32_t texture_id = -1;
      fan::vec2ui tile_count = fan::vec2ui(1, 1);
      fan::vec2 padding = fan::vec2(0);
    };
    struct block_face_tiles_t {
      enum face_e : std::uint32_t {
        positive_x, negative_x, positive_y, negative_y, positive_z, negative_z, count
      };
      std::uint32_t tile(std::uint32_t face) const {
        return face < count ? tiles[face] : 0;
      }
      std::uint32_t tiles[count] {};
    };
    struct voxelized_model_properties_t {
      std::uint32_t voxel_resolution = 64;
      std::string texture_path;
      fan::vec3 tint = fan::vec3(1, 1, 1);
      f32_t alpha_threshold = 0.2f;
      f32_t occupancy_padding = 0.001f;
      bool fix_uv_diagonals = false;
    };
    struct time_ubo_t {
      f32_t time = 0;
      std::uint32_t frame_index = 0;
    };
    struct rt_camera_t {
      fan::mat4 projection;
      fan::mat4 view;
      fan::mat4 inv_projection;
      fan::mat4 inv_view;
      fan::vec4 ray;
    };
    struct exposure_ubo_t {
      f32_t exposure = 1.f;
      f32_t enable_gi = 0.f;
      f32_t enable_reflections = 0.f;
      f32_t enable_shadows = 1.f;
      f32_t ambient_strength = 0.18f;
      f32_t shadow_strength = 0.35f;
      f32_t wrap_strength = 0.35f;
      f32_t show_light_indicator = 1.f;
      f32_t light_indicator_radius = 6.f;
      f32_t pad0 = 0.f;
    };
    struct pick_result_t {
      fan::vec4 position = fan::vec4(0.f);
      fan::vec4 normal = fan::vec4(0.f);
    };

    context_t() = default;
    context_t(fan::graphics::engine_t& engine, const engine_open_properties_t& properties = {}) {
      attached_engine = &engine;
      pending_open_properties = properties;
      engine.single_queue.push_back([this, &engine]() {
        auto sz = pending_open_properties.size;
        if (sz.x == 0 || sz.y == 0) { sz = engine.window.get_size(); }
        output_sprite_enabled = pending_open_properties.create_output_sprite;
        open(engine.context.vk, sz);
        ready = true;
        sync_output_sprite();
        attach_engine_callbacks(engine);
      });
    }
    ~context_t() {
      detach_engine();
      close();
    }
    VkDeviceAddress get_buffer_address(VkBuffer buffer) const {
      VkBufferDeviceAddressInfoKHR info {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = buffer};
      return vkGetBufferDeviceAddressKHR(ctx->device, &info);
    }
    void load_functions() {
    #define L(n) n = (PFN_##n)vkGetDeviceProcAddr(ctx->device, #n)
      L(vkGetBufferDeviceAddressKHR); L(vkCmdBuildAccelerationStructuresKHR);
      L(vkGetAccelerationStructureBuildSizesKHR); L(vkCreateAccelerationStructureKHR);
      L(vkGetAccelerationStructureDeviceAddressKHR); L(vkCreateRayTracingPipelinesKHR);
      L(vkGetRayTracingShaderGroupHandlesKHR); L(vkCmdTraceRaysKHR);
    #undef L
    }
    void fill_blas_build_info(
      std::uint32_t model_index,
      VkAccelerationStructureGeometryTrianglesDataKHR& triangles,
      VkAccelerationStructureGeometryKHR& geometry,
      VkAccelerationStructureBuildGeometryInfoKHR& build_info
    ) const {
      const model_t& model = models[model_index];
      triangles = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
        .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
        .vertexData = {.deviceAddress = get_buffer_address(vertex_buffer)},
        .vertexStride = sizeof(vertex_t),
        .maxVertex = model.vertex_capacity ? model.first_vertex + model.vertex_capacity - 1 : 0,
        .indexType = VK_INDEX_TYPE_UINT32,
        .indexData = {.deviceAddress = get_buffer_address(index_buffer)}
      };
      geometry = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
        .geometry = {.triangles = triangles},
        .flags = VK_GEOMETRY_OPAQUE_BIT_KHR
      };
      VkBuildAccelerationStructureFlagsKHR build_flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
      if (model.animated) {
        build_flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
      }
      build_info = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
        .flags = build_flags,
        .geometryCount = 1,
        .pGeometries = &geometry
      };
    }
    VkDeviceSize get_scratch_alignment() const {
      VkPhysicalDeviceAccelerationStructurePropertiesKHR as_props {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR
      };
      VkPhysicalDeviceProperties2 props2 {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &as_props
      };
      vkGetPhysicalDeviceProperties2(ctx->physical_device, &props2);
      return as_props.minAccelerationStructureScratchOffsetAlignment;
    }

    void create_scratch_buffer(VkDeviceSize size, fan::vulkan::context_t::buffer_t& buffer) {
      VkDeviceSize align = get_scratch_alignment();
      VkDeviceSize aligned_size = (size + align - 1) & ~(align - 1);
      VmaAllocationCreateInfo alloc_ci {.usage = VMA_MEMORY_USAGE_GPU_ONLY};
      VkBufferCreateInfo buf_ci {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = aligned_size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
      };
      vmaCreateBufferWithAlignment(ctx->allocator, &buf_ci, &alloc_ci, align,
        &buffer.buffer, &buffer.allocation, nullptr);
      buffer.size = aligned_size;
    }
    std::uint32_t model_blas_primitive_count(const model_t& model) const {
      return model.blas_primitive_count ? model.blas_primitive_count : model.index_count / 3;
    }
    std::uint32_t model_blas_primitive_capacity(const model_t& model) const {
      return model.blas_primitive_capacity ? model.blas_primitive_capacity : model_blas_primitive_count(model);
    }
    bool model_has_blas_geometry(const model_t& model) const {
      return model.vertex_capacity != 0 && model.index_capacity != 0 && model_blas_primitive_count(model) != 0 && model_blas_primitive_capacity(model) != 0;
    }
    bool has_gpu_only_models() const {
      for (const model_t& model : models) {
        if (model_has_blas_geometry(model) && !model.keep_cpu_geometry) { return true; }
      }
      return false;
    }
    void destroy_acceleration_structures_only() {
      destroy_mesh_upload_jobs();
      for (auto& blas : blas_list) { blas.destroy(*ctx); }
      blas_list.clear();
      destroy_tlas_resources();
      destroy_buffer(blas_scratch_buffer);
      blas_scratch_size = 0;
      blas_scratch_peak_size = 0;
    }
    void create_blas_for_models() {
      if (models.empty()) { return; }
      std::vector<VkAccelerationStructureBuildSizesInfoKHR> sizes(models.size());
      std::vector<std::uint32_t> primitive_counts(models.size());
      VkDeviceSize scratch_size = 0;
      blas_list.resize(models.size());

      for (std::uint32_t i = 0; i < models.size(); ++i) {
        const model_t& model = models[i];
        if (!model_has_blas_geometry(model)) {
          blas_list[i].destroy(*ctx);
          continue;
        }

        primitive_counts[i] = model_blas_primitive_capacity(model);
        VkAccelerationStructureGeometryTrianglesDataKHR triangles {};
        VkAccelerationStructureGeometryKHR geometry {};
        VkAccelerationStructureBuildGeometryInfoKHR build_info {};
        fill_blas_build_info(i, triangles, geometry, build_info);
        sizes[i].sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        vkGetAccelerationStructureBuildSizesKHR(ctx->device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build_info, &primitive_counts[i], &sizes[i]);
        if (sizes[i].accelerationStructureSize == 0) {
          blas_list[i].destroy(*ctx);
          primitive_counts[i] = 0;
          continue;
        }

        scratch_size = std::max(scratch_size, std::max(sizes[i].buildScratchSize, sizes[i].updateScratchSize));
        acceleration_structure_t& blas = blas_list[i];
        blas.destroy(*ctx);
        ctx->create_buffer(sizes[i].accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, blas.buffer);
        VkAccelerationStructureCreateInfoKHR create_info {
          .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
          .buffer = blas.buffer,
          .size = sizes[i].accelerationStructureSize,
          .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR
        };
        fan::vulkan::validate(vkCreateAccelerationStructureKHR(ctx->device, &create_info, nullptr, &blas.handle));
      }

      if (scratch_size == 0) { return; }
      destroy_buffer(blas_scratch_buffer);
      blas_scratch_size = scratch_size;
      create_scratch_buffer(blas_scratch_size, blas_scratch_buffer);
      blas_scratch_peak_size = std::max(blas_scratch_peak_size, blas_scratch_buffer.size);
      VkDeviceAddress scratch_address = get_buffer_address(blas_scratch_buffer);
      VkCommandBuffer cmd = ctx->begin_single_time_commands();
      bool recorded = false;
      for (std::uint32_t i = 0; i < models.size(); ++i) {
        const model_t& model = models[i];
        std::uint32_t prim_count = model_blas_primitive_count(model);
        if (prim_count == 0 || i >= blas_list.size() || blas_list[i].handle == VK_NULL_HANDLE) { continue; }

        VkAccelerationStructureGeometryTrianglesDataKHR triangles {};
        VkAccelerationStructureGeometryKHR geometry {};
        VkAccelerationStructureBuildGeometryInfoKHR build_info {};
        fill_blas_build_info(i, triangles, geometry, build_info);
        build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        build_info.dstAccelerationStructure = blas_list[i].handle;
        build_info.scratchData.deviceAddress = scratch_address;
        VkAccelerationStructureBuildRangeInfoKHR range_info {
          .primitiveCount = prim_count,
          .primitiveOffset = model.first_index * sizeof(std::uint32_t),
          .firstVertex = 0,
          .transformOffset = 0
        };
        const VkAccelerationStructureBuildRangeInfoKHR* range_infos = &range_info;
        vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build_info, &range_infos);
        VkMemoryBarrier barrier {
          .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
          .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
          .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR
        };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        recorded = true;
      }
      ctx->end_single_time_commands(cmd);
      if (!recorded) { return; }
      for (auto& blas : blas_list) {
        if (blas.handle == VK_NULL_HANDLE) { blas.device_address = 0; continue; }
        VkAccelerationStructureDeviceAddressInfoKHR addr_info {
          .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
          .accelerationStructure = blas.handle
        };
        blas.device_address = vkGetAccelerationStructureDeviceAddressKHR(ctx->device, &addr_info);
      }
    }
    void record_blas_update(VkCommandBuffer cmd, std::uint32_t model_index) {
      if (model_index >= models.size() || model_index >= blas_list.size() || !blas_scratch_buffer || blas_list[model_index].handle == VK_NULL_HANDLE) { return; }
      const model_t& model = models[model_index];
      if (!model_has_blas_geometry(model)) { return; }
      VkAccelerationStructureGeometryTrianglesDataKHR triangles {};
      VkAccelerationStructureGeometryKHR geometry {};
      VkAccelerationStructureBuildGeometryInfoKHR build_info {};
      fill_blas_build_info(model_index, triangles, geometry, build_info);
      build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
      build_info.srcAccelerationStructure = blas_list[model_index].handle;
      build_info.dstAccelerationStructure = blas_list[model_index].handle;
      build_info.scratchData.deviceAddress = get_buffer_address(blas_scratch_buffer);
      VkAccelerationStructureBuildRangeInfoKHR range_info {
        .primitiveCount = model_blas_primitive_count(model),
        .primitiveOffset = model.first_index * sizeof(std::uint32_t)
      };
      const VkAccelerationStructureBuildRangeInfoKHR* range_infos = &range_info;
      vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build_info, &range_infos);
    }
    void record_blas_rebuild(VkCommandBuffer cmd, std::uint32_t model_index) {
      if (model_index >= models.size() || model_index >= blas_list.size() || !blas_scratch_buffer || blas_list[model_index].handle == VK_NULL_HANDLE) { return; }
      const model_t& model = models[model_index];
      std::uint32_t prim_count = model_blas_primitive_count(model);
      if (!prim_count) { return; }
      VkAccelerationStructureGeometryTrianglesDataKHR triangles {};
      VkAccelerationStructureGeometryKHR geometry {};
      VkAccelerationStructureBuildGeometryInfoKHR build_info {};
      fill_blas_build_info(model_index, triangles, geometry, build_info);
      build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
      build_info.srcAccelerationStructure = VK_NULL_HANDLE;
      build_info.dstAccelerationStructure = blas_list[model_index].handle;
      build_info.scratchData.deviceAddress = get_buffer_address(blas_scratch_buffer);
      VkAccelerationStructureBuildRangeInfoKHR range_info {
        .primitiveCount = prim_count,
        .primitiveOffset = model.first_index * sizeof(std::uint32_t),
        .firstVertex = 0,
        .transformOffset = 0
      };
      const VkAccelerationStructureBuildRangeInfoKHR* range_infos = &range_info;
      vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build_info, &range_infos);
    }
    bool recreate_blas(std::uint32_t model_index) {
      if (!ctx || model_index >= models.size()) { return false; }
      const model_t& model = models[model_index];
      std::uint32_t prim_count = model_blas_primitive_count(model);
      std::uint32_t prim_capacity = model_blas_primitive_capacity(model);
      if (!prim_count || !prim_capacity) { return false; }
      blas_list.resize(models.size());
      acceleration_structure_t& blas = blas_list[model_index];
      blas.destroy(*ctx);

      VkAccelerationStructureGeometryTrianglesDataKHR triangles {};
      VkAccelerationStructureGeometryKHR geometry {};
      VkAccelerationStructureBuildGeometryInfoKHR build_info {};
      fill_blas_build_info(model_index, triangles, geometry, build_info);

      VkAccelerationStructureBuildSizesInfoKHR size_info { .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
      vkGetAccelerationStructureBuildSizesKHR(ctx->device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build_info, &prim_capacity, &size_info);

      ctx->create_buffer(size_info.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, blas.buffer);
      VkAccelerationStructureCreateInfoKHR as_ci {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
        .buffer = blas.buffer,
        .size = size_info.accelerationStructureSize,
        .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR
      };
      fan::vulkan::validate(vkCreateAccelerationStructureKHR(ctx->device, &as_ci, nullptr, &blas.handle));

      VkDeviceSize needed_scratch = std::max(size_info.buildScratchSize, size_info.updateScratchSize);
      if (!blas_scratch_buffer || blas_scratch_size < needed_scratch) {
        destroy_buffer(blas_scratch_buffer);
        blas_scratch_size = needed_scratch;
        create_scratch_buffer(blas_scratch_size, blas_scratch_buffer);
        blas_scratch_peak_size = std::max(blas_scratch_peak_size, blas_scratch_buffer.size);
      }

      build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
      build_info.dstAccelerationStructure = blas.handle;
      build_info.scratchData.deviceAddress = get_buffer_address(blas_scratch_buffer);
      VkAccelerationStructureBuildRangeInfoKHR range_info {
        .primitiveCount = prim_count,
        .primitiveOffset = model.first_index * sizeof(std::uint32_t),
        .firstVertex = 0,
        .transformOffset = 0
      };
      const VkAccelerationStructureBuildRangeInfoKHR* range_infos = &range_info;
      VkCommandBuffer cmd = ctx->begin_single_time_commands();
      vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build_info, &range_infos);
      ctx->end_single_time_commands(cmd);

      VkAccelerationStructureDeviceAddressInfoKHR addr_info {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
        .accelerationStructure = blas.handle
      };
      blas.device_address = vkGetAccelerationStructureDeviceAddressKHR(ctx->device, &addr_info);
      return true;
    }
    std::vector<VkAccelerationStructureInstanceKHR> make_tlas_instances() const {
      std::vector<VkAccelerationStructureInstanceKHR> vk_instances;
      vk_instances.reserve(instances.size());
      for (const instance_t& inst : instances) {
        if (inst.model_index >= models.size() || inst.model_index >= blas_list.size()) { continue; }
        if (blas_list[inst.model_index].device_address == 0) { continue; }
        const model_t& model = models[inst.model_index];
        if (!model_has_blas_geometry(model)) { continue; }
        VkTransformMatrixKHR t {};
        t.matrix[0][0] = inst.transform[0][0]; t.matrix[0][1] = inst.transform[1][0]; t.matrix[0][2] = inst.transform[2][0]; t.matrix[0][3] = inst.transform[3][0];
        t.matrix[1][0] = inst.transform[0][1]; t.matrix[1][1] = inst.transform[1][1]; t.matrix[1][2] = inst.transform[2][1]; t.matrix[1][3] = inst.transform[3][1];
        t.matrix[2][0] = inst.transform[0][2]; t.matrix[2][1] = inst.transform[1][2]; t.matrix[2][2] = inst.transform[2][2]; t.matrix[2][3] = inst.transform[3][2];
        vk_instances.push_back({
          .transform = t,
          .instanceCustomIndex = model.first_primitive,
          .mask = inst.mask,
          .instanceShaderBindingTableRecordOffset = 0,
          .flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR,
          .accelerationStructureReference = blas_list[inst.model_index].device_address
        });
      }
      return vk_instances;
    }
    void ensure_tlas_instance_staging(VkDeviceSize size) {
      if (tlas_instance_staging_buffer && tlas_instance_staging_size >= size) { return; }
      if (tlas_instance_staging_buffer.mapped) {
        ctx->unmap_buffer(tlas_instance_staging_buffer);
      }
      destroy_buffer(tlas_instance_staging_buffer);
      ctx->create_buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, tlas_instance_staging_buffer);
      fan::vulkan::validate(ctx->map_buffer(tlas_instance_staging_buffer, &tlas_instance_staging_buffer.mapped));
      tlas_instance_staging_size = size;
    }
    void upload_tlas_instances_to_staging(const std::vector<VkAccelerationStructureInstanceKHR>& vk_instances, VkDeviceSize size) {
      ensure_tlas_instance_staging(size);
      std::memcpy(tlas_instance_staging_buffer.mapped, vk_instances.data(), (std::size_t)size);
    }
    bool can_update_tlas_transforms() const {
      return ctx && tlas.handle && tlas_instance_buffer && tlas_scratch_buffer && tlas_instance_count != 0 && tlas_instance_count == (std::uint32_t)instances.size();
    }
    void reset_accumulation() {
      frame_index = 0;
      accumulation_reset_pending = true;
    }
    void record_tlas_instance_build(VkCommandBuffer cmd, VkBuildAccelerationStructureModeKHR mode) {
      std::vector<VkAccelerationStructureInstanceKHR> vk_instances = make_tlas_instances();
      tlas_instance_count = (std::uint32_t)vk_instances.size();
      if (tlas_instance_count == 0) { return; }
      VkDeviceSize instance_size = sizeof(VkAccelerationStructureInstanceKHR) * tlas_instance_count;
      upload_tlas_instances_to_staging(vk_instances, instance_size);

      VkAccelerationStructureGeometryInstancesDataKHR instances_data {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
        .arrayOfPointers = VK_FALSE,
        .data = {.deviceAddress = get_buffer_address(tlas_instance_buffer)}
      };
      VkAccelerationStructureGeometryKHR geometry {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
        .geometry = {.instances = instances_data}
      };
      VkAccelerationStructureBuildGeometryInfoKHR build_info {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
        .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
        .mode = mode,
        .srcAccelerationStructure = mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR ? tlas.handle : VK_NULL_HANDLE,
        .dstAccelerationStructure = tlas.handle,
        .geometryCount = 1,
        .pGeometries = &geometry,
        .scratchData = {.deviceAddress = get_buffer_address(tlas_scratch_buffer)}
      };
      VkAccelerationStructureBuildRangeInfoKHR range_info {.primitiveCount = tlas_instance_count};
      const VkAccelerationStructureBuildRangeInfoKHR* range_infos = &range_info;
      VkBufferCopy copy_region {.size = instance_size};
      vkCmdCopyBuffer(cmd, tlas_instance_staging_buffer, tlas_instance_buffer, 1, &copy_region);
      VkBufferMemoryBarrier instance_barrier {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = tlas_instance_buffer,
        .offset = 0,
        .size = instance_size
      };
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 0, nullptr, 1, &instance_barrier, 0, nullptr);
      vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build_info, &range_infos);
      VkMemoryBarrier build_barrier {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR
      };
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 1, &build_barrier, 0, nullptr, 0, nullptr);
      reset_accumulation();
    }
    void record_tlas_transform_update(VkCommandBuffer cmd) {
      record_tlas_instance_build(cmd, VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR);
    }
    void record_tlas_rebuild(VkCommandBuffer cmd) {
      record_tlas_instance_build(cmd, VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR);
    }
    void create_tlas() {
      std::vector<VkAccelerationStructureInstanceKHR> vk_instances = make_tlas_instances();
      tlas_instance_count = (std::uint32_t)vk_instances.size();
      if (tlas_instance_count == 0) { return; }
      VkDeviceSize instance_size = sizeof(VkAccelerationStructureInstanceKHR) * tlas_instance_count;
      upload_tlas_instances_to_staging(vk_instances, instance_size);

      if (tlas_instance_count > tlas_instance_capacity || !tlas_instance_buffer) {
        destroy_buffer(tlas_instance_buffer);
        tlas_instance_capacity = std::max<VkDeviceSize>(tlas_instance_capacity * 2, tlas_instance_count + 1024);
        VkDeviceSize capacity_size = sizeof(VkAccelerationStructureInstanceKHR) * tlas_instance_capacity;
        ctx->create_buffer(capacity_size, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, tlas_instance_buffer);
      }
      ctx->copy_buffer(tlas_instance_staging_buffer, tlas_instance_buffer, instance_size);

      VkAccelerationStructureGeometryInstancesDataKHR instances_data {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
        .arrayOfPointers = VK_FALSE,
        .data = {.deviceAddress = get_buffer_address(tlas_instance_buffer)}
      };
      VkAccelerationStructureGeometryKHR geometry {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
        .geometry = {.instances = instances_data},
        .flags = 0
      };
      VkAccelerationStructureBuildGeometryInfoKHR build_info {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
        .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
        .geometryCount = 1,
        .pGeometries = &geometry
      };

      VkAccelerationStructureBuildSizesInfoKHR size_info {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
      vkGetAccelerationStructureBuildSizesKHR(ctx->device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build_info, &tlas_instance_count, &size_info);

      if (size_info.accelerationStructureSize > tlas_capacity || !tlas.buffer) {
        tlas.destroy(*ctx);
        tlas_capacity = std::max<VkDeviceSize>(tlas_capacity * 2, size_info.accelerationStructureSize + 65536);
        ctx->create_buffer(tlas_capacity, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, tlas.buffer);
      }

      if (!tlas.handle) {
        VkAccelerationStructureCreateInfoKHR create_info {
          .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
          .buffer = tlas.buffer,
          .size = tlas_capacity,
          .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR
        };
        fan::vulkan::validate(vkCreateAccelerationStructureKHR(ctx->device, &create_info, nullptr, &tlas.handle));
      }

      VkDeviceSize scratch_size = std::max(size_info.buildScratchSize, size_info.updateScratchSize);
      if (scratch_size > tlas_scratch_capacity || !tlas_scratch_buffer) {
        destroy_buffer(tlas_scratch_buffer);
        tlas_scratch_capacity = std::max<VkDeviceSize>(tlas_scratch_capacity * 2, scratch_size + 65536);
        create_scratch_buffer(tlas_scratch_capacity, tlas_scratch_buffer);
        tlas_scratch_peak_size = std::max(tlas_scratch_peak_size, tlas_scratch_buffer.size);
      }

      build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
      build_info.dstAccelerationStructure = tlas.handle;
      build_info.scratchData.deviceAddress = get_buffer_address(tlas_scratch_buffer);
      VkAccelerationStructureBuildRangeInfoKHR range_info {.primitiveCount = tlas_instance_count};
      const VkAccelerationStructureBuildRangeInfoKHR* range_infos = &range_info;
      VkCommandBuffer cmd = ctx->begin_single_time_commands();
      vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build_info, &range_infos);
      ctx->end_single_time_commands(cmd);

      VkAccelerationStructureDeviceAddressInfoKHR addr_info {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
        .accelerationStructure = tlas.handle
      };
      tlas.device_address = vkGetAccelerationStructureDeviceAddressKHR(ctx->device, &addr_info);
    }
    bool update_tlas_transforms() {
      if (!can_update_tlas_transforms()) { return false; }
      VkCommandBuffer cmd = ctx->begin_single_time_commands();
      if (tlas_rebuild_dirty) { record_tlas_rebuild(cmd); }
      else { record_tlas_transform_update(cmd); }
      ctx->end_single_time_commands(cmd);
      return true;
    }
    fan::graphics::image_t create_rt_image(bool& valid, VkImageLayout& tracked_layout, VkPipelineStageFlags dst_stage, bool clear) {
      fan::graphics::image_t image = ctx->image_create();
      valid = true;
      auto& img = ctx->image_get(image);
      VkImageCreateInfo image_info {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_R16G16B16A16_SFLOAT,
        .extent = { size.x, size.y, 1 },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
      };
      fan::vulkan::image_create(*ctx, size, image_info.format, VK_IMAGE_TILING_OPTIMAL, image_info.usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, img.image_index, img.image_allocation);
      VkImageViewCreateInfo view_info {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = img.image_index,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = image_info.format,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
      };
      fan::vulkan::validate(vkCreateImageView(ctx->device, &view_info, nullptr, &img.image_view));
      ctx->create_texture_sampler(img.sampler, {});
      VkCommandBuffer cmd = ctx->begin_single_time_commands();
      ctx->insert_image_barrier(cmd, img.image_index, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, dst_stage);
      if (clear) {
        VkClearColorValue clear_color {};
        clear_color.float32[3] = 1.0f;
        VkImageSubresourceRange range {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, img.image_index, VK_IMAGE_LAYOUT_GENERAL, &clear_color, 1, &range);
      }
      ctx->end_single_time_commands(cmd);
      tracked_layout = VK_IMAGE_LAYOUT_GENERAL;
      return image;
    }
    void create_output_image() {
      output_image = create_rt_image(output_image_valid, current_layout, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, false);
    }
    VkShaderModule load_shader(const char* path, shaderc_shader_kind kind) {
      std::string code = fan::graphics::read_shader(path);
      auto spirv = fan::vulkan::context_t::compile_file(path, kind, code);
      return ctx->create_shader_module(spirv);
    }
    void create_pipeline() {
      VkDescriptorSetLayoutBinding bindings[] = {
        { 0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr },
        { 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr },
        { 3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr },
        { 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, max_textures, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, nullptr },
        { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, nullptr },
        { 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, nullptr },
        { 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, nullptr },
        { 8, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, nullptr },
        { 9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, nullptr },
        { 10, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, nullptr },
        { 11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr }
      };
      VkDescriptorBindingFlags binding_flags[std::size(bindings)] {};
      for (std::uint32_t i = 0; i < (std::uint32_t)std::size(binding_flags); ++i) {
        binding_flags[i] =
          VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
          VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT |
          VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
      }
      VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags_info {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .bindingCount = (std::uint32_t)std::size(binding_flags),
        .pBindingFlags = binding_flags
      };
      VkDescriptorSetLayoutCreateInfo layout_info {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = &binding_flags_info,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        .bindingCount = (std::uint32_t)std::size(bindings),
        .pBindings = bindings
      };
      fan::vulkan::validate(vkCreateDescriptorSetLayout(ctx->device, &layout_info, nullptr, &descriptor_layout));
      VkPipelineLayoutCreateInfo pipeline_layout_info {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptor_layout
      };
      fan::vulkan::validate(vkCreatePipelineLayout(ctx->device, &pipeline_layout_info, nullptr, &pipeline_layout));
      VkShaderModule rgen = load_shader("shaders/vulkan/ray_tracing/raygen.rgen", shaderc_glsl_raygen_shader);
      VkShaderModule miss = load_shader("shaders/vulkan/ray_tracing/miss.rmiss", shaderc_glsl_miss_shader);
      VkShaderModule chit = load_shader("shaders/vulkan/ray_tracing/closesthit.rchit", shaderc_glsl_closesthit_shader);
      VkShaderModule shadow = load_shader("shaders/vulkan/ray_tracing/shadow.rmiss", shaderc_glsl_miss_shader);
      VkPipelineShaderStageCreateInfo stages[] = {
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_RAYGEN_BIT_KHR, rgen, "main", nullptr },
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_MISS_BIT_KHR, miss, "main", nullptr },
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, chit, "main", nullptr },
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_MISS_BIT_KHR, shadow, "main", nullptr }
      };
      VkRayTracingShaderGroupCreateInfoKHR groups[] = {
        { VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR, nullptr, VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 0, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, nullptr },
        { VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR, nullptr, VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 1, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, nullptr },
        { VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR, nullptr, VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR, VK_SHADER_UNUSED_KHR, 2, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, nullptr },
        { VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR, nullptr, VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 3, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, nullptr }
      };
      VkRayTracingPipelineCreateInfoKHR pipeline_info {
        .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
        .stageCount = 4,
        .pStages = stages,
        .groupCount = 4,
        .pGroups = groups,
        .maxPipelineRayRecursionDepth = 3,
        .layout = pipeline_layout
      };
      fan::vulkan::validate(vkCreateRayTracingPipelinesKHR(ctx->device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline));
      vkDestroyShaderModule(ctx->device, rgen, nullptr);
      vkDestroyShaderModule(ctx->device, miss, nullptr);
      vkDestroyShaderModule(ctx->device, chit, nullptr);
      vkDestroyShaderModule(ctx->device, shadow, nullptr);
    }
    void create_sbt() {
      VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt_props {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
      VkPhysicalDeviceProperties2 props {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &rt_props};
      vkGetPhysicalDeviceProperties2(ctx->physical_device, &props);
      const std::uint32_t handle_size = rt_props.shaderGroupHandleSize;
      const std::uint32_t handle_align = rt_props.shaderGroupHandleAlignment;
      const std::uint32_t base_align = rt_props.shaderGroupBaseAlignment;
      handle_size_aligned = (handle_size + handle_align - 1) & ~(handle_align - 1);
      const std::uint32_t group_count = 4;
      const std::uint32_t aligned_group_sz = (handle_size_aligned + base_align - 1) & ~(base_align - 1);
      const std::uint32_t sbt_size = group_count * aligned_group_sz;
      std::vector<std::uint8_t> handles(group_count * handle_size);
      fan::vulkan::validate(vkGetRayTracingShaderGroupHandlesKHR(ctx->device, pipeline, 0, group_count, group_count * handle_size, handles.data()));
      ctx->create_buffer(sbt_size, VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, shader_binding_table);
      ctx->create_buffer(sbt_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, sbt_staging);
      void* data;
      fan::vulkan::validate(ctx->map_buffer(sbt_staging, &data));
      std::uint8_t* sbt_ptr = static_cast<std::uint8_t*>(data);
      std::memset(sbt_ptr, 0, sbt_size);
      group_stride = aligned_group_sz;
      rgen_offset = 0 * aligned_group_sz;
      miss_offset = 1 * aligned_group_sz;
      shadow_miss_offset = 2 * aligned_group_sz;
      hit_offset = 3 * aligned_group_sz;
      std::memcpy(sbt_ptr + rgen_offset, handles.data() + 0 * handle_size, handle_size);
      std::memcpy(sbt_ptr + miss_offset, handles.data() + 1 * handle_size, handle_size);
      std::memcpy(sbt_ptr + shadow_miss_offset, handles.data() + 3 * handle_size, handle_size);
      std::memcpy(sbt_ptr + hit_offset, handles.data() + 2 * handle_size, handle_size);
      ctx->unmap_buffer(sbt_staging);
      ctx->copy_buffer(sbt_staging, shader_binding_table, sbt_size);
    }
    void set_descriptor_image_write(VkWriteDescriptorSet& write, std::uint32_t binding, VkDescriptorType type, const VkDescriptorImageInfo* info) const {
      write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set, binding, 0, 1, type, info, nullptr, nullptr};
    }
    void set_descriptor_buffer_write(VkWriteDescriptorSet& write, std::uint32_t binding, VkDescriptorType type, const VkDescriptorBufferInfo* info) const {
      write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set, binding, 0, 1, type, nullptr, info, nullptr};
    }
    void create_descriptor_set() {
      VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, max_textures },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6 }
      };
      VkDescriptorPoolCreateInfo pool_info {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
        .maxSets = 1,
        .poolSizeCount = (std::uint32_t)std::size(pool_sizes),
        .pPoolSizes = pool_sizes
      };
      fan::vulkan::validate(vkCreateDescriptorPool(ctx->device, &pool_info, nullptr, &descriptor_pool));
      VkDescriptorSetAllocateInfo alloc_info {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &descriptor_layout
      };
      fan::vulkan::validate(vkAllocateDescriptorSets(ctx->device, &alloc_info, &descriptor_set));
      VkDescriptorImageInfo image_info {VK_NULL_HANDLE, ctx->image_get(output_image).image_view, VK_IMAGE_LAYOUT_GENERAL};
      VkDescriptorBufferInfo cam_info {camera_buffer, 0, sizeof(rt_camera_t) * 16};
      VkDescriptorBufferInfo time_info {time_buffer, 0, sizeof(time_ubo_t)};
      VkDescriptorBufferInfo light_info {light_buffer, 0, sizeof(light_ubo_t)};
      VkDescriptorBufferInfo exposure_info {exposure_ubo, 0, sizeof(exposure_ubo_t)};
      VkDescriptorBufferInfo pick_info {pick_buffer, 0, sizeof(pick_result_t)};
      VkWriteDescriptorSet writes[6] {};
      set_descriptor_image_write(writes[0], 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &image_info);
      set_descriptor_buffer_write(writes[1], 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &cam_info);
      set_descriptor_buffer_write(writes[2], 3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &time_info);
      set_descriptor_buffer_write(writes[3], 8, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &light_info);
      set_descriptor_buffer_write(writes[4], 10, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &exposure_info);
      set_descriptor_buffer_write(writes[5], 11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &pick_info);
      vkUpdateDescriptorSets(ctx->device, 6, writes, 0, nullptr);
      update_tlas_descriptor();
      update_scene_buffers_descriptor();
      update_rt_textures_descriptor();
    }
    void update_rt_textures_descriptor() {
      std::vector<VkDescriptorImageInfo> infos(max_textures);
      auto& dummy = ctx->image_get(fan::graphics::ctx().default_texture);
      for (std::uint32_t i = 0; i < max_textures; i++) {
        if (i < rt_texture_infos.size()) { infos[i] = rt_texture_infos[i]; }
        else {
          infos[i].sampler = dummy.sampler;
          infos[i].imageView = dummy.image_view;
          infos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
      }
      VkWriteDescriptorSet write {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = descriptor_set,
        .dstBinding = 4,
        .descriptorCount = max_textures,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = infos.data()
      };
      vkUpdateDescriptorSets(ctx->device, 1, &write, 0, nullptr);
    }
    void create_source_vertex_buffer() {
      if (source_vertex_data.empty()) {
        source_vertex_data.push_back({});
      }
      ctx->upload_buffer(source_vertex_data, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, source_vertex_buffer);
      source_vertex_capacity = source_vertex_data.size();
      if (source_vertex_data.size() == 1 && scene_vertex_count == 0) {
        source_vertex_data.clear();
      }
    }
    void create_vertex_buffer() { 
      ctx->upload_buffer(vertex_data, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_TRANSFER_DST_BIT, vertex_buffer);
      vertex_capacity = vertex_data.size();
    }
    void create_index_buffer() { 
      ctx->upload_buffer(index_data, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_TRANSFER_DST_BIT, index_buffer);
      index_capacity = index_data.size();
    }
    void create_material_buffer() { 
      ctx->upload_buffer(materials, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, material_buffer);
      material_capacity = materials.size();
    }
    void create_material_index_buffer() {
      if (material_indices_per_primitive.empty()) { return; }
      destroy_buffer(material_index_buffer);
      ctx->upload_buffer(material_indices_per_primitive, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, material_index_buffer);
      material_index_capacity = material_indices_per_primitive.size();
    }
    void upload_bone_buffer() {
      if (!bone_buffer.mapped) { return; }
      if (bone_matrices.empty()) {
        fan::mat4 identity(1);
        std::memcpy(bone_buffer.mapped, &identity, sizeof(identity));
      }
      else {
        std::memcpy(bone_buffer.mapped, bone_matrices.data(), sizeof(fan::mat4) * bone_matrices.size());
      }
      bone_buffer_dirty = false;
    }
    void upload_bone_buffer_if_dirty() { if (bone_buffer_dirty) { upload_bone_buffer(); } }
    void create_bone_buffer() {
      if (bone_buffer.mapped) {
        ctx->unmap_buffer(bone_buffer);
      }
      destroy_buffer(bone_buffer);
      std::uint32_t count = std::max<std::uint32_t>((std::uint32_t)bone_matrices.size(), 1);
      VkDeviceSize size = sizeof(fan::mat4) * count;
      ctx->create_buffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, bone_buffer);
      fan::vulkan::validate(ctx->map_buffer(bone_buffer, &bone_buffer.mapped));
      upload_bone_buffer();
    }
    fan::vec3 transform_direction(const fan::mat4& m, const fan::vec3& v) const {
      return fan::vec3(
        m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z,
        m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z,
        m[0][2] * v.x + m[1][2] * v.y + m[2][2] * v.z
      );
    }
    template <typename source_vertex_type_t, typename vertex_type_t>
    void bake_cached_animation_vertex(const source_vertex_type_t& src, vertex_type_t& dst) const {
      fan::vec4 position(0);
      fan::vec3 normal(0);
      f32_t total_weight = 0;
      for (std::uint32_t i = 0; i < 4; ++i) {
        int bone_id = src.bone_ids[i];
        f32_t weight = src.bone_weights[i];
        if (bone_id < 0 || weight <= 0 || (std::uint32_t)bone_id >= bone_matrices.size()) { continue; }
        const fan::mat4& bone = bone_matrices[bone_id];
        fan::vec4 local_position = bone * fan::vec4(src.position, 1.f);
        fan::vec3 local_normal = transform_direction(bone, src.normal);
        position += local_position * weight;
        normal += local_normal * weight;
        total_weight += weight;
      }
      if (total_weight > 0) {
        dst.position = fan::vec3(position);
        dst.normal = normal.length_squared() > 0 ? normal.normalize() : src.normal;
      }
    }
    model_cache_entry_t load_model_from_fms(fan::model::fms_t& fms, std::uint32_t first_bone = 0, std::uint32_t bone_count = 0, bool animated = false) {
      std::uint32_t first_model = (std::uint32_t)models.size();
      std::uint32_t added_index_count = 0;
      for (const auto& mesh : fms.meshes) { added_index_count += (std::uint32_t)mesh.indices.size(); }
      std::uint32_t needed_primitive_count = ((std::uint32_t)index_data.size() + added_index_count) / 3;
      if (material_indices_per_primitive.size() < needed_primitive_count) { material_indices_per_primitive.resize(needed_primitive_count); }
      for (std::uint32_t mesh_idx = 0; mesh_idx < fms.meshes.size(); mesh_idx++) {
        const auto& src_mesh = fms.meshes[mesh_idx];
        model_t model {};
        model.first_index = (std::uint32_t)index_data.size();
        model.index_count = (std::uint32_t)src_mesh.indices.size();
        model.first_vertex = (std::uint32_t)vertex_data.size();
        model.vertex_count = (std::uint32_t)src_mesh.vertices.size();
        model.first_bone = first_bone;
        model.bone_count = bone_count;
        model.animated = animated;
        std::uint32_t first_vertex = model.first_vertex;
        for (const auto& v : src_mesh.vertices) {
          vertex_t out {};
          out.position = v.position;
          out.normal = v.normal;
          out.texcoord = v.uv;
          out.color = pack_vertex_color(fan::vec3(v.color.x, v.color.y, v.color.z));
          source_vertex_t src {};
          src.position = fan::vec3(out.position);
          src.normal = fan::vec3(out.normal);
          src.texcoord = out.texcoord;
          src.color = unpack_vertex_color(out.color);
          src.bone_ids = v.bone_ids;
          src.bone_weights = v.bone_weights;
          for (int i = 0; i < 4; ++i) { if (src.bone_ids[i] >= 0) { src.bone_ids[i] += first_bone; } }
          if (animated && bone_count != 0) { bake_cached_animation_vertex(src, out); }
          if (!model.has_bounds) {
            model.aabb_min = fan::vec3(out.position);
            model.aabb_max = fan::vec3(out.position);
            model.has_bounds = true;
          }
          else {
            model.aabb_min = model.aabb_min.min(fan::vec3(out.position));
            model.aabb_max = model.aabb_max.max(fan::vec3(out.position));
          }
          source_vertex_data.push_back(src);
          vertex_data.push_back(out);
        }
        for (std::uint32_t idx : src_mesh.indices) { index_data.push_back(idx + first_vertex); }
        material_info_t mat;
        mat.base_color = fan::vec3(1, 1, 1);
        mat.source_material_id = mesh_idx;
        if (mesh_idx < fms.material_data_vector.size()) {
          const auto& md = fms.material_data_vector[mesh_idx];
          const fan::color* c = &md.color[fan::texture_type::base_color];
          if ((*c)[0] == 1 && (*c)[1] == 1 && (*c)[2] == 1 && (*c)[3] == 1) { c = &md.color[fan::texture_type::diffuse]; }
          mat.base_color = fan::vec3((*c)[0], (*c)[1], (*c)[2]);
        }
        auto load_rt_texture = [&](const std::string& name) -> std::int32_t {
          if (name.empty()) { return -1; }
          auto cached = rt_texture_cache.find(name);
          if (cached != rt_texture_cache.end()) { return cached->second; }
          auto it = fan::model::cached_texture_data.find(name);
          if (it == fan::model::cached_texture_data.end()) { return -1; }
          const auto& td = it->second;
          if (!td.valid()) { return -1; }
          fan::image::info_t ii {.data = (void*)td.data.get(), .size = td.size, .channels = td.channels};
          auto tex = ctx->image_load(ii);
          texture_ids.push_back(tex);
          auto& img = ctx->image_get(tex);
          VkDescriptorImageInfo di {.sampler = img.sampler, .imageView = img.image_view, .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
          rt_texture_infos.push_back(di);
          std::int32_t slot = (std::int32_t)rt_texture_infos.size() - 1;
          rt_texture_cache[name] = slot;
          return slot;
        };
        auto load_first_rt_texture = [&](std::initializer_list<std::uint32_t> types) -> std::int32_t {
          for (auto type : types) {
            const std::string& tn = src_mesh.texture_names[type];
            if (tn.empty()) { continue; }
            std::int32_t slot = load_rt_texture(tn);
            if (slot >= 0) { return slot; }
          }
          return -1;
        };
        { std::int32_t slot = load_first_rt_texture({fan::texture_type::base_color, fan::texture_type::diffuse}); if (slot >= 0) mat.albedo_texture_id = slot; }
        { std::int32_t slot = load_first_rt_texture({fan::texture_type::normals, fan::texture_type::normal_camera}); if (slot >= 0) mat.normal_texture_id = slot; }
        { const std::string& tn = src_mesh.texture_names[fan::texture_type::metalness]; if (!tn.empty()) { std::int32_t slot = load_rt_texture(tn); if (slot >= 0) mat.metallic_texture_id = slot; } }
        { const std::string& tn = src_mesh.texture_names[fan::texture_type::diffuse_roughness]; if (!tn.empty()) { std::int32_t slot = load_rt_texture(tn); if (slot >= 0) mat.roughness_texture_id = slot; } }
        model.material_index = (std::uint32_t)materials.size();
        materials.push_back(mat);
        std::uint32_t mesh_first_index = model.first_index;
        std::uint32_t mesh_index_count = model.index_count;
        std::uint32_t mesh_primitive_cnt = mesh_index_count / 3;
        model.first_primitive = mesh_first_index / 3;
        for (std::uint32_t p = 0; p < mesh_primitive_cnt; ++p) { material_indices_per_primitive[model.first_primitive + p] = model.material_index; }
        models.push_back(model);
      }
      return {first_model, (std::uint32_t)models.size() - first_model};
    }
    void add_instance(std::uint32_t model_index, const fan::mat4& transform) {
      instance_t inst {};
      inst.model_index = model_index;
      inst.transform = transform;
      instances.push_back(inst);
    }
    void apply_object_ray_mask(std::uint32_t object_index) {
      if (object_index >= objects.size()) { return; }
      object_t& object = objects[object_index];
      for (std::uint32_t i = 0; i < object.instance_count; ++i) {
        std::uint32_t instance_index = object.first_instance + i;
        if (instance_index < instances.size()) { instances[instance_index].mask = object.ray_mask; }
      }
    }
    std::string make_model_cache_key(const scene_model_t& model) const {
      std::string key = model.path;
      key.push_back('\x1f');
      key += model.texture_path;
      key.push_back('\x1f');
      key += model.fix_uv_diagonals ? "1" : "0";
      return key;
    }
    void add_cached_model_instances(const scene_model_t& model, const model_cache_entry_t& entry) {
      for (std::uint32_t i = 0; i < entry.model_count; ++i) { add_instance(entry.first_model + i, model.transform); }
    }
    static std::uint32_t pack_voxel_color(const fan::vec4& color) {
      auto pack_channel = [](f32_t value) { return (std::uint32_t)std::clamp<int>((int)std::round(std::clamp(value, 0.f, 1.f) * 255.f), 0, 255); };
      return pack_channel(color.x) | (pack_channel(color.y) << 8) | (pack_channel(color.z) << 16) | (pack_channel(color.w) << 24);
    }
    static bool same_voxel_surface(const voxel_surface_t& a, const voxel_surface_t& b) {
      return a.visible == b.visible && a.id == b.id && a.color_key == b.color_key && a.material_index == b.material_index;
    }
    static std::uint32_t voxel_face_index(int axis, int sign) {
      return (std::uint32_t)axis * 2u + (sign > 0 ? 0u : 1u);
    }
    static std::uint32_t get_or_add_voxel_material(voxel_mesh_input_t& mesh, std::unordered_map<std::uint64_t, std::uint32_t>& material_map, const atlas_t& atlas, std::uint32_t tile) {
      std::uint64_t key = (std::uint64_t)(std::uint32_t)atlas.texture_id | ((std::uint64_t)tile << 32);
      auto found = material_map.find(key);
      if (found != material_map.end()) { return found->second; }
      material_info_t material {};
      material.base_color = fan::vec3(1, 1, 1);
      material.albedo_texture_id = atlas.texture_id;
      material.uv_transform = atlas.tile_uv(tile);
      std::uint32_t index = (std::uint32_t)mesh.materials.size();
      mesh.materials.push_back(material);
      material_map[key] = index;
      return index;
    }
    static voxel_mesh_input_t greedy_mesh_grid_impl(const voxel_grid_t& grid, const atlas_t* atlas, std::span<const block_face_tiles_t> block_tiles, f32_t voxel_size, std::vector<voxel_surface_t>* mask_buffer = nullptr) {
      voxel_mesh_input_t out;
      std::unordered_map<std::uint64_t, std::uint32_t> material_map;
      int dims[3] = {(int)grid.sx, (int)grid.sy, (int)grid.sz};
      int strides[3] = {1, (int)grid.sx, (int)(grid.sx * grid.sy)};
      std::uint32_t max_dim = std::max({grid.sx, grid.sy, grid.sz});
      std::size_t req_mask_size = (std::size_t)max_dim * max_dim;
      std::vector<voxel_surface_t> local_mask;
      std::vector<voxel_surface_t>& mask = mask_buffer ? *mask_buffer : local_mask;
      if (mask.size() < req_mask_size) {
        mask.resize(req_mask_size);
      }

      auto make_position = [&](int d, int u, int v, f32_t plane, f32_t a, f32_t b) {
        f32_t p[3] {};
        p[d] = plane; p[u] = a; p[v] = b;
        return fan::vec3(p[0] * voxel_size, p[1] * voxel_size, p[2] * voxel_size);
      };

      for (int d = 0; d < 3; ++d) {
        int u = (d + 1) % 3;
        int v = (d + 2) % 3;
        int stride_d = strides[d];
        int stride_u = strides[u];
        int stride_v = strides[v];

        for (int sign : { 1, -1 }) {
          fan::vec3 normal(0);
          normal[d] = (f32_t)sign;

          for (int layer = 0; layer < dims[d]; ++layer) {
            bool check_neighbor_bounds = (sign == 1 && layer == dims[d] - 1) || (sign == -1 && layer == 0);
            int n_offset = sign * stride_d;
            int layer_offset = layer * stride_d;

            for (int j = 0; j < dims[v]; ++j) {
              int j_offset = j * stride_v;
              for (int i = 0; i < dims[u]; ++i) {
                int current_idx = layer_offset + (i * stride_u) + j_offset;
                const voxel_t& voxel = grid.data[current_idx];
                voxel_surface_t& cell = mask[j * dims[u] + i];

                if (voxel.id != 0) {
                  bool neighbor_solid = !check_neighbor_bounds && (grid.data[current_idx + n_offset].id != 0);
                  if (!neighbor_solid) {
                    cell.visible = true;
                    cell.id = voxel.id;
                    cell.color = voxel.color;
                    cell.color_key = pack_voxel_color(voxel.color);
                    if (atlas != nullptr && !block_tiles.empty()) {
                      std::uint32_t tile = 0;
                      if (voxel.id < block_tiles.size()) { tile = block_tiles[voxel.id].tile(voxel_face_index(d, sign)); }
                      else { tile = block_tiles.front().tile(voxel_face_index(d, sign)); }
                      cell.material_index = get_or_add_voxel_material(out, material_map, *atlas, tile);
                    }
                  }
                  else {
                    cell.visible = false;
                  }
                }
                else {
                  cell.visible = false;
                }
              }
            }

            for (int j = 0; j < dims[v]; ++j) {
              for (int i = 0; i < dims[u];) {
                voxel_surface_t cell = mask[j * dims[u] + i];
                if (!cell.visible) { ++i; continue; }
                int w = 1;
                while (i + w < dims[u] && same_voxel_surface(mask[j * dims[u] + i + w], cell)) { ++w; }
                int h = 1;
                bool done = false;
                while (!done && j + h < dims[v]) {
                  for (int k = 0; k < w; ++k) {
                    if (!same_voxel_surface(mask[(j + h) * dims[u] + i + k], cell)) { done = true; break; }
                  }
                  if (!done) { ++h; }
                }

                f32_t plane = sign > 0 ? (f32_t)(layer + 1) : (f32_t)layer;
                fan::vec3 p0 = make_position(d, u, v, plane, (f32_t)i, (f32_t)j);
                fan::vec3 p1 = make_position(d, u, v, plane, (f32_t)(i + w), (f32_t)j);
                fan::vec3 p2 = make_position(d, u, v, plane, (f32_t)(i + w), (f32_t)(j + h));
                fan::vec3 p3 = make_position(d, u, v, plane, (f32_t)i, (f32_t)(j + h));
                std::uint32_t vi = (std::uint32_t)out.vertices.size();

                vertex_t v0 {}, v1 {}, v2 {}, v3 {};
                fan::vec3 clr(cell.color.x, cell.color.y, cell.color.z);
                v0.position = p0; v0.normal = normal; v0.texcoord = fan::vec2(0.f, 0.f); v0.color = pack_vertex_color(clr);
                v1.position = p1; v1.normal = normal; v1.texcoord = fan::vec2((f32_t)w, 0.f); v1.color = pack_vertex_color(clr);
                v2.position = p2; v2.normal = normal; v2.texcoord = fan::vec2((f32_t)w, (f32_t)h); v2.color = pack_vertex_color(clr);
                v3.position = p3; v3.normal = normal; v3.texcoord = fan::vec2(0.f, (f32_t)h); v3.color = pack_vertex_color(clr);

                out.vertices.insert(out.vertices.end(), {v0, v1, v2, v3});

                if (sign > 0) { out.indices.insert(out.indices.end(), {vi, vi + 1, vi + 2, vi, vi + 2, vi + 3}); }
                else { out.indices.insert(out.indices.end(), {vi, vi + 2, vi + 1, vi, vi + 3, vi + 2}); }
                if (atlas != nullptr && !block_tiles.empty()) {
                  out.primitive_material_indices.push_back(cell.material_index);
                  out.primitive_material_indices.push_back(cell.material_index);
                }

                for (int y = j; y < j + h; ++y) {
                  for (int x = i; x < i + w; ++x) { mask[y * dims[u] + x].visible = false; }
                }
                i += w;
              }
            }
          }
        }
      }
      return out;
    }
    static voxel_mesh_input_t greedy_mesh_grid(const voxel_grid_t& grid, f32_t voxel_size = 1.f) {
      return greedy_mesh_grid_impl(grid, nullptr, {}, voxel_size, nullptr);
    }
    static voxel_mesh_input_t greedy_mesh_grid(const voxel_grid_t& grid, const atlas_t& atlas, std::span<const block_face_tiles_t> block_tiles, f32_t voxel_size = 1.f) {
      return greedy_mesh_grid_impl(grid, &atlas, block_tiles, voxel_size, nullptr);
    }
    static material_info_t make_default_mesh_material(const voxel_mesh_input_t& mesh_input) {
      material_info_t material {};
      material.base_color = mesh_input.base_color;
      material.albedo_texture_id = mesh_input.albedo_texture_id;
      return material;
    }
    static std::uint32_t pack_vertex_color(const fan::vec3& color) {
      auto pack = [](f32_t v) { return (std::uint32_t)std::clamp(v * 255.f + 0.5f, 0.f, 255.f); };
      return pack(color.x) | (pack(color.y) << 8) | (pack(color.z) << 16) | 0xff000000u;
    }
    static fan::vec3 unpack_vertex_color(std::uint32_t color) {
      return fan::vec3(
        (f32_t)(color & 0xffu) / 255.f,
        (f32_t)((color >> 8) & 0xffu) / 255.f,
        (f32_t)((color >> 16) & 0xffu) / 255.f
      );
    }
    static source_vertex_t make_source_vertex(const vertex_t& vertex) {
      source_vertex_t source {};
      source.position = fan::vec3(vertex.position); source.normal = fan::vec3(vertex.normal); source.texcoord = vertex.texcoord; source.color = unpack_vertex_color(vertex.color);
      source.bone_ids = fan::vec4i(-1); source.bone_weights = fan::vec4(0);
      return source;
    }
    static void reset_model_bounds(model_t& model) {
      model.aabb_min = fan::vec3(0); model.aabb_max = fan::vec3(0); model.has_bounds = false;
    }
    static void include_model_vertex(model_t& model, const fan::vec3& position) {
      if (!model.has_bounds) {
        model.aabb_min = position; model.aabb_max = position; model.has_bounds = true;
      }
      else {
        model.aabb_min = model.aabb_min.min(position); model.aabb_max = model.aabb_max.max(position);
      }
    }
    std::vector<source_vertex_t> make_source_vertices(std::span<const vertex_t> vertices) const {
      std::vector<source_vertex_t> out;
      out.reserve(vertices.size());
      for (const auto& vertex : vertices) {
        out.push_back(make_source_vertex(vertex));
      }
      return out;
    }
    std::vector<std::uint32_t> make_gpu_indices(const voxel_mesh_input_t& mesh_input, std::uint32_t first_vertex, std::uint32_t count) const {
      std::vector<std::uint32_t> out;
      out.resize(count);
      for (std::uint32_t i = 0; i < count; ++i) {
        out[i] = i < mesh_input.indices.size() ? mesh_input.indices[i] + first_vertex : first_vertex;
      }
      return out;
    }
    void write_material_indices_for_model(const model_t& model, const voxel_mesh_input_t& mesh_input) {
      std::uint32_t primitive_count = model.index_count / 3;
      std::uint32_t needed_primitive_count = model.first_primitive + model.blas_primitive_capacity;
      if (material_indices_per_primitive.size() < needed_primitive_count) { material_indices_per_primitive.resize(needed_primitive_count); }
      for (std::uint32_t primitive = 0; primitive < model.blas_primitive_capacity; ++primitive) {
        std::uint32_t relative_material = 0;
        if (primitive < primitive_count && primitive < mesh_input.primitive_material_indices.size()) { relative_material = mesh_input.primitive_material_indices[primitive]; }
        if (relative_material >= model.material_count) { relative_material = 0; }
        material_indices_per_primitive[model.first_primitive + primitive] = model.material_index + relative_material;
      }
    }
    static void merge_scene_ranges(std::vector<scene_range_t>& ranges) {
      if (ranges.empty()) { return; }
      std::sort(ranges.begin(), ranges.end(), [](const scene_range_t& a, const scene_range_t& b) {
        return a.first < b.first;
      });
      std::size_t out = 0;
      for (std::size_t i = 1; i < ranges.size(); ++i) {
        scene_range_t& back = ranges[out];
        scene_range_t& next = ranges[i];
        std::uint64_t back_end = (std::uint64_t)back.first + back.count;
        if (back_end >= next.first) {
          std::uint64_t next_end = (std::uint64_t)next.first + next.count;
          back.count = (std::uint32_t)(std::max(back_end, next_end) - back.first);
        }
        else {
          ++out;
          if (out != i) { ranges[out] = next; }
        }
      }
      ranges.resize(out + 1);
    }
    static void release_scene_range(std::vector<scene_range_t>& ranges, std::uint32_t first, std::uint32_t count) {
      if (count == 0) { return; }
      ranges.push_back({first, count});
      merge_scene_ranges(ranges);
    }
    static std::uint64_t scene_range_free_count(const std::vector<scene_range_t>& ranges) {
      std::uint64_t n = 0;
      for (const auto& range : ranges) { n += range.count; }
      return n;
    }
    static std::uint32_t alloc_scene_range(std::vector<scene_range_t>& ranges, VkDeviceSize& high_water, std::uint32_t count) {
      if (count == 0) { return 0; }
      for (std::size_t i = 0; i < ranges.size(); ++i) {
        scene_range_t& range = ranges[i];
        if (range.count < count) { continue; }
        std::uint32_t first = range.first;
        range.first += count;
        range.count -= count;
        if (range.count == 0) { ranges.erase(ranges.begin() + i); }
        return first;
      }
      std::uint32_t first = (std::uint32_t)high_water;
      high_water += count;
      return first;
    }
    std::uint32_t append_mesh_model(const voxel_mesh_input_t& mesh_input) {
      model_t model {};
      model.index_count = (std::uint32_t)mesh_input.indices.size();
      model.vertex_count = (std::uint32_t)mesh_input.vertices.size();
      model.vertex_capacity = std::max(model.vertex_count, mesh_input.vertex_capacity);
      model.index_capacity = std::max(model.index_count, mesh_input.index_capacity);
      model.blas_primitive_count = model.index_count / 3;
      model.blas_primitive_capacity = model.index_capacity / 3;
      model.first_vertex = alloc_scene_range(vertex_free_ranges, scene_vertex_count, model.vertex_capacity);
      model.first_index = alloc_scene_range(index_free_ranges, scene_index_count, model.index_capacity);
      model.first_primitive = alloc_scene_range(primitive_free_ranges, scene_primitive_count, model.blas_primitive_capacity);
      model.keep_cpu_geometry = mesh_input.keep_cpu_copy;

      reset_model_bounds(model);
      for (const auto& vertex : mesh_input.vertices) {
        include_model_vertex(model, fan::vec3(vertex.position));
      }

      if (model.keep_cpu_geometry) {
        if (vertex_data.size() < (std::size_t)model.first_vertex + model.vertex_capacity) { vertex_data.resize((std::size_t)model.first_vertex + model.vertex_capacity); }
        if (source_vertex_data.size() < (std::size_t)model.first_vertex + model.vertex_capacity) { source_vertex_data.resize((std::size_t)model.first_vertex + model.vertex_capacity); }
        if (index_data.size() < (std::size_t)model.first_index + model.index_capacity) { index_data.resize((std::size_t)model.first_index + model.index_capacity); }
        for (std::uint32_t i = 0; i < model.vertex_capacity; ++i) {
          vertex_t vertex {};
          if (i < mesh_input.vertices.size()) { vertex = mesh_input.vertices[i]; }
          vertex_data[model.first_vertex + i] = vertex;
          source_vertex_data[model.first_vertex + i] = make_source_vertex(vertex);
        }
        for (std::uint32_t i = 0; i < model.index_capacity; ++i) {
          std::uint32_t index = i < mesh_input.indices.size() ? mesh_input.indices[i] : 0;
          index_data[model.first_index + i] = index + model.first_vertex;
        }
      }

      model.material_index = (std::uint32_t)materials.size();
      model.material_count = std::max<std::uint32_t>(1, (std::uint32_t)mesh_input.materials.size());
      model.material_capacity = std::max(model.material_count, mesh_input.material_capacity);
      material_info_t default_material = make_default_mesh_material(mesh_input);
      for (std::uint32_t i = 0; i < model.material_capacity; ++i) {
        materials.push_back(i < mesh_input.materials.size() ? mesh_input.materials[i] : default_material);
      }

      write_material_indices_for_model(model, mesh_input);

      scene_material_count = materials.size();

      std::uint32_t model_index = (std::uint32_t)models.size();
      models.push_back(model);
      return model_index;
    }
    bool append_mesh_geometry(const voxel_mesh_input_t& mesh_input, std::uint32_t object_index) {
      if (mesh_input.vertices.empty() || mesh_input.indices.empty()) { fan::throw_error("ray tracing mesh has no geometry"); }
      if (object_index >= objects.size() || object_index >= scene_models.size()) { fan::throw_error("ray tracing mesh object index is invalid"); }
      object_t& object = objects[object_index];
      object.first_instance = (std::uint32_t)instances.size();
      std::uint32_t model_index = append_mesh_model(mesh_input);
      add_instance(model_index, scene_models[object_index].transform);
      object.instance_count = (std::uint32_t)instances.size() - object.first_instance;
      apply_object_ray_mask(object_index);
      return true;
    }
    object_handle_t add_mesh(const voxel_mesh_input_t& mesh_input) {
      object_handle_t handle {(std::uint32_t)scene_models.size(), object_generation_counter++};
      std::uint32_t mesh_index = (std::uint32_t)procedural_meshes.size();
      procedural_meshes.push_back(mesh_input);
      scene_model_t scene_model {};
      scene_model.transform = mesh_input.transform;
      scene_model.procedural_mesh_index = mesh_index;
      scene_models.push_back(scene_model);
      objects.push_back({.generation = handle.generation, .first_instance = 0, .instance_count = 0});
      if (ctx && ready) {
        bool geometry_changed = load_scene_model(handle.index);
        scene_geometry_dirty = scene_geometry_dirty || geometry_changed;
        tlas_dirty = true;
        reset_accumulation();
      }
      return handle;
    }
    void begin_incremental_upload() {
      incremental_upload_batch = true;
      incremental_upload_had_changes = false;
      incremental_upload_needs_tlas_rebuild = false;
    }
    void end_incremental_upload() {
      if (!incremental_upload_batch) { return; }
      incremental_upload_batch = false;
      if (!incremental_upload_had_changes) { return; }
      if (incremental_upload_needs_tlas_rebuild) { rebuild_tlas(); }
      reset_accumulation();
    }
    template <typename T>
    void reserve_gpu_buffer(
      fan::vulkan::context_t::buffer_t& dst_buffer,
      VkDeviceSize& capacity,
      VkBufferUsageFlags usage,
      VkDeviceSize required_capacity,
      VkDeviceSize old_count
    ) {
      if (required_capacity <= capacity && dst_buffer) { return; }
      VkDeviceSize new_capacity = std::max<VkDeviceSize>(capacity * 2, required_capacity + 1024);
      VkDeviceSize old_bytes = std::min<VkDeviceSize>(sizeof(T) * old_count, dst_buffer.size);
      VkDeviceSize new_bytes = sizeof(T) * new_capacity;
      fan::vulkan::context_t::buffer_t new_buffer;
      ctx->create_buffer(new_bytes, usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, new_buffer);
      if (dst_buffer && old_bytes > 0) {
        VkCommandBuffer cmd = ctx->begin_single_time_commands();
        VkBufferCopy r { .srcOffset = 0, .dstOffset = 0, .size = old_bytes };
        vkCmdCopyBuffer(cmd, dst_buffer, new_buffer, 1, &r);
        ctx->end_single_time_commands(cmd);
      }
      destroy_buffer(dst_buffer);
      dst_buffer = new_buffer;
      capacity = new_capacity;
    }
    void reserve_scene_buffers(
      VkDeviceSize vertex_count,
      VkDeviceSize index_count,
      VkDeviceSize material_count,
      VkDeviceSize primitive_count
    ) {
      if (!ctx || !ready) { return; }
      constexpr VkBufferUsageFlags vert_usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
      constexpr VkBufferUsageFlags stor_usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
      if (!source_vertex_buffer && !source_vertex_data.empty()) {
        create_source_vertex_buffer();
      }
      reserve_gpu_buffer<vertex_t>(vertex_buffer, vertex_capacity, vert_usage, vertex_count, scene_vertex_count);
      reserve_gpu_buffer<std::uint32_t>(index_buffer, index_capacity, vert_usage, index_count, scene_index_count);
      reserve_gpu_buffer<material_info_t>(material_buffer, material_capacity, stor_usage, material_count, scene_material_count);
      reserve_gpu_buffer<std::uint32_t>(material_index_buffer, material_index_capacity, stor_usage, primitive_count, scene_primitive_count);
      update_scene_buffers_descriptor();
    }
    template <typename T>
    void grow_gpu_buffer(
      fan::vulkan::context_t::buffer_t& dst_buffer,
      VkDeviceSize& capacity,
      VkBufferUsageFlags usage,
      const T* new_data,
      VkDeviceSize new_count,
      VkDeviceSize old_count
    ) {
      if (new_count == 0) { return; }

      VkDeviceSize old_bytes = std::min<VkDeviceSize>(sizeof(T) * old_count, dst_buffer.size);
      VkDeviceSize new_bytes = sizeof(T) * new_count;
      VkDeviceSize required_count = old_count + new_count;
      std::vector<fan::vulkan::context_t::buffer_t> deferred_destroy;

      VkCommandBuffer cmd = ctx->begin_single_time_commands();
      if (required_count > capacity || !dst_buffer) {
        VkDeviceSize new_capacity = std::max<VkDeviceSize>(capacity * 2, required_count + 1024);
        fan::vulkan::context_t::buffer_t old_buffer = dst_buffer;
        fan::vulkan::context_t::buffer_t new_buffer;
        ctx->create_buffer(
          sizeof(T) * new_capacity,
          usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
          new_buffer
        );
        if (old_buffer && old_bytes > 0) {
          ctx->copy_buffer_cmd(cmd, old_buffer, new_buffer, 0, 0, old_bytes);
          deferred_destroy.push_back(old_buffer);
        }
        dst_buffer = new_buffer;
        capacity = new_capacity;
      }

      fan::vulkan::context_t::buffer_t staging;
      ctx->create_buffer(new_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging);
      void* mapped = nullptr;
      fan::vulkan::validate(ctx->map_buffer(staging, &mapped));
      std::memcpy(mapped, new_data, (std::size_t)new_bytes);
      ctx->unmap_buffer(staging);
      ctx->copy_buffer_cmd(cmd, staging, dst_buffer, 0, sizeof(T) * old_count, new_bytes);
      deferred_destroy.push_back(staging);
      ctx->end_single_time_commands(cmd);

      for (auto& b : deferred_destroy) { destroy_buffer(b); }
    }
    void upload_slice_direct(
      VkCommandBuffer cmd,
      std::vector<fan::vulkan::context_t::buffer_t>& deferred_destroy,
      fan::vulkan::context_t::buffer_t& dst_buffer,
      VkDeviceSize& cap,
      VkBufferUsageFlags usage,
      const void* data,
      VkDeviceSize elem_size,
      VkDeviceSize dst_first,
      VkDeviceSize count,
      VkDeviceSize old_count
    ) {
      if (!count) { return; }
      VkDeviceSize old_bytes = std::min<VkDeviceSize>(elem_size * old_count, dst_buffer.size);
      VkDeviceSize bytes = elem_size * count;
      VkDeviceSize required_count = dst_first + count;
      if (required_count > cap || !dst_buffer) {
        VkDeviceSize new_cap = std::max<VkDeviceSize>(cap * 2, required_count + 1024);
        fan::vulkan::context_t::buffer_t old_buffer = dst_buffer;
        fan::vulkan::context_t::buffer_t new_buffer;
        ctx->create_buffer(
          elem_size * new_cap,
          usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
          new_buffer
        );
        if (old_buffer && old_bytes > 0) {
          ctx->copy_buffer_cmd(cmd, old_buffer, new_buffer, 0, 0, old_bytes);
          deferred_destroy.push_back(old_buffer);
        }
        dst_buffer = new_buffer;
        cap = new_cap;
      }
      fan::vulkan::context_t::buffer_t staging;
      ctx->create_buffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging);
      void* mapped = nullptr;
      fan::vulkan::validate(ctx->map_buffer(staging, &mapped));
      std::memcpy(mapped, data, (std::size_t)bytes);
      ctx->unmap_buffer(staging);
      ctx->copy_buffer_cmd(cmd, staging, dst_buffer, 0, elem_size * dst_first, bytes);
      deferred_destroy.push_back(staging);
    }
    object_handle_t add_mesh_incremental(const voxel_mesh_input_t& mesh_input) {
      if (!ctx || !ready) { return add_mesh(mesh_input); }

      object_handle_t handle {(std::uint32_t)scene_models.size(), object_generation_counter++};
      scene_model_t scene_model {};
      scene_model.transform = mesh_input.transform;
      if (mesh_input.keep_cpu_copy) {
        scene_model.procedural_mesh_index = (std::uint32_t)procedural_meshes.size();
        procedural_meshes.push_back(mesh_input);
      }
      scene_models.push_back(scene_model);
      objects.push_back({.generation = handle.generation});

      VkDeviceSize old_vertex_count    = scene_vertex_count;
      VkDeviceSize old_index_count     = scene_index_count;
      VkDeviceSize old_material_count  = scene_material_count;
      VkDeviceSize old_prim_count      = scene_primitive_count;

      object_t& object = objects[handle.index];
      object.first_instance = (std::uint32_t)instances.size();
      std::uint32_t model_index = append_mesh_model(mesh_input);
      add_instance(model_index, mesh_input.transform);
      object.instance_count = 1;
      apply_object_ray_mask(handle.index);

      model_t& model = models[model_index];

      constexpr VkBufferUsageFlags vert_usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
      constexpr VkBufferUsageFlags stor_usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

      std::vector<fan::vulkan::context_t::buffer_t> deferred_destroy;
      deferred_destroy.reserve(10);

      std::vector<std::uint32_t> temp_indices;
      const source_vertex_t* source_upload = nullptr;
      const vertex_t* vertex_upload = nullptr;
      const std::uint32_t* index_upload = nullptr;

      if (model.keep_cpu_geometry) {
        source_upload = source_vertex_data.data() + model.first_vertex;
        vertex_upload = vertex_data.data() + model.first_vertex;
        index_upload = index_data.data() + model.first_index;
      }
      else {
        temp_indices = make_gpu_indices(mesh_input, model.first_vertex, model.index_count);
        vertex_upload = mesh_input.vertices.data();
        index_upload = temp_indices.data();
      }

      VkCommandBuffer cmd = ctx->begin_single_time_commands();
      if (model.keep_cpu_geometry) {
        upload_slice_direct(cmd, deferred_destroy, source_vertex_buffer, source_vertex_capacity, vert_usage, source_upload, sizeof(source_vertex_t), model.first_vertex, model.vertex_count, old_vertex_count);
      }
      upload_slice_direct(cmd, deferred_destroy, vertex_buffer, vertex_capacity, vert_usage, vertex_upload, sizeof(vertex_t), model.first_vertex, model.vertex_count, old_vertex_count);
      upload_slice_direct(cmd, deferred_destroy, index_buffer, index_capacity, vert_usage, index_upload, sizeof(std::uint32_t), model.first_index, model.index_count, old_index_count);
      upload_slice_direct(cmd, deferred_destroy, material_buffer, material_capacity, stor_usage, materials.data() + model.material_index, sizeof(material_info_t), model.material_index, model.material_count, old_material_count);
      upload_slice_direct(cmd, deferred_destroy, material_index_buffer, material_index_capacity, stor_usage, material_indices_per_primitive.data() + model.first_primitive, sizeof(std::uint32_t), model.first_primitive, model.blas_primitive_count, old_prim_count);

      ctx->buffer_barriers_cmd(cmd, {
        {&source_vertex_buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR},
        {&vertex_buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR},
        {&index_buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR},
        {&material_buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT},
        {&material_index_buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT}
      }, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR);
      ctx->end_single_time_commands(cmd);
      for (auto& b : deferred_destroy) { destroy_buffer(b); }
      update_scene_buffers_descriptor();

      blas_list.resize(models.size());
      acceleration_structure_t& new_blas = blas_list[model_index];

      VkAccelerationStructureGeometryTrianglesDataKHR triangles {};
      VkAccelerationStructureGeometryKHR geometry {};
      VkAccelerationStructureBuildGeometryInfoKHR build_info {};
      fill_blas_build_info(model_index, triangles, geometry, build_info);

      std::uint32_t prim_count = models[model_index].blas_primitive_count ? models[model_index].blas_primitive_count : models[model_index].index_count / 3;
      std::uint32_t prim_capacity = models[model_index].blas_primitive_capacity ? models[model_index].blas_primitive_capacity : prim_count;
      VkAccelerationStructureBuildSizesInfoKHR size_info { .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
      vkGetAccelerationStructureBuildSizesKHR(ctx->device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build_info, &prim_capacity, &size_info);

      ctx->create_buffer(size_info.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, new_blas.buffer);
      VkAccelerationStructureCreateInfoKHR as_ci {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
        .buffer = new_blas.buffer,
        .size = size_info.accelerationStructureSize,
        .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR
      };
      fan::vulkan::validate(vkCreateAccelerationStructureKHR(ctx->device, &as_ci, nullptr, &new_blas.handle));

      VkDeviceSize needed_scratch = std::max(size_info.buildScratchSize, size_info.updateScratchSize);
      if (!blas_scratch_buffer || blas_scratch_size < needed_scratch) {
        destroy_buffer(blas_scratch_buffer);
        blas_scratch_size = needed_scratch;
        create_scratch_buffer(blas_scratch_size, blas_scratch_buffer);
        blas_scratch_peak_size = std::max(blas_scratch_peak_size, blas_scratch_buffer.size);
      }

      build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
      build_info.dstAccelerationStructure = new_blas.handle;
      build_info.scratchData.deviceAddress = get_buffer_address(blas_scratch_buffer);
      VkAccelerationStructureBuildRangeInfoKHR range_info {
        .primitiveCount = prim_count,
        .primitiveOffset = models[model_index].first_index * sizeof(std::uint32_t),
        .firstVertex = 0,
        .transformOffset = 0
      };
      const VkAccelerationStructureBuildRangeInfoKHR* range_infos = &range_info;
      cmd = ctx->begin_single_time_commands();
      vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build_info, &range_infos);
      ctx->end_single_time_commands(cmd);

      VkAccelerationStructureDeviceAddressInfoKHR addr_info {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
        .accelerationStructure = new_blas.handle
      };
      new_blas.device_address = vkGetAccelerationStructureDeviceAddressKHR(ctx->device, &addr_info);

      if (incremental_upload_batch) {
        incremental_upload_had_changes = true;
        incremental_upload_needs_tlas_rebuild = true;
        tlas_dirty = true;
      }
      else {
        rebuild_tlas();
        reset_accumulation();
      }
      return handle;
    }
    static voxel_mesh_input_t make_degenerate_mesh(const voxel_mesh_input_t& mesh_input) {
      voxel_mesh_input_t mesh = mesh_input;
      mesh.vertices.resize(3); mesh.indices = {0, 1, 2};
      for (vertex_t& vertex : mesh.vertices) {
        vertex.position = fan::vec3(0); vertex.normal = fan::vec3(0, 1, 0); vertex.texcoord = fan::vec2(0); vertex.color = pack_vertex_color(fan::vec3(0));
      }
      mesh.primitive_material_indices.clear();
      return mesh;
    }
    bool rewrite_mesh_model(model_t& model, const voxel_mesh_input_t& mesh_input) {
      std::uint32_t new_vertex_count = (std::uint32_t)mesh_input.vertices.size();
      std::uint32_t new_index_count = (std::uint32_t)mesh_input.indices.size();
      std::uint32_t new_material_count = std::max<std::uint32_t>(1, (std::uint32_t)mesh_input.materials.size());
      if (new_vertex_count > model.vertex_capacity || new_index_count > model.index_capacity || new_material_count > model.material_capacity) { return false; }
      model.vertex_count = new_vertex_count;
      model.index_count = new_index_count;
      model.blas_primitive_count = model.index_count / 3;
      model.material_count = new_material_count;
      model.keep_cpu_geometry = mesh_input.keep_cpu_copy;
      reset_model_bounds(model);
      for (const auto& vertex : mesh_input.vertices) {
        include_model_vertex(model, fan::vec3(vertex.position));
      }

      if (model.keep_cpu_geometry) {
        if (vertex_data.size() < (std::size_t)model.first_vertex + model.vertex_capacity) { vertex_data.resize((std::size_t)model.first_vertex + model.vertex_capacity); }
        if (source_vertex_data.size() < (std::size_t)model.first_vertex + model.vertex_capacity) { source_vertex_data.resize((std::size_t)model.first_vertex + model.vertex_capacity); }
        if (index_data.size() < (std::size_t)model.first_index + model.index_capacity) { index_data.resize((std::size_t)model.first_index + model.index_capacity); }
        for (std::uint32_t i = 0; i < model.vertex_capacity; ++i) {
          vertex_t vertex {};
          if (i < mesh_input.vertices.size()) { vertex = mesh_input.vertices[i]; }
          vertex_data[model.first_vertex + i] = vertex;
          source_vertex_data[model.first_vertex + i] = make_source_vertex(vertex);
        }
        for (std::uint32_t i = 0; i < model.index_capacity; ++i) {
          std::uint32_t index = i < mesh_input.indices.size() ? mesh_input.indices[i] : 0;
          index_data[model.first_index + i] = index + model.first_vertex;
        }
      }

      material_info_t default_material = make_default_mesh_material(mesh_input);
      for (std::uint32_t i = 0; i < model.material_capacity; ++i) {
        materials[model.material_index + i] = i < mesh_input.materials.size() ? mesh_input.materials[i] : default_material;
      }
      write_material_indices_for_model(model, mesh_input);
      return true;
    }
    struct pending_buffer_copy_t {
      fan::vulkan::context_t::buffer_t staging;
      fan::vulkan::context_t::buffer_t* dst_buffer = nullptr;
      VkDeviceSize dst_offset = 0;
      VkDeviceSize size = 0;
    };
    struct pending_mesh_upload_t {
      std::uint32_t model_index = 0;
      std::vector<pending_buffer_copy_t> copies;
    };
    struct retired_mesh_upload_t {
      std::vector<fan::vulkan::context_t::buffer_t> staging_buffers;
      std::uint32_t frames_left = 0;
    };
    void upload_model_buffer_slice(VkCommandBuffer cmd, std::vector<fan::vulkan::context_t::buffer_t>& deferred_destroy, fan::vulkan::context_t::buffer_t& dst_buffer, const void* data, VkDeviceSize elem_size, VkDeviceSize first, VkDeviceSize count) {
      if (!count) { return; }
      VkDeviceSize bytes = elem_size * count;
      fan::vulkan::context_t::buffer_t staging;
      ctx->create_buffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging);
      void* mapped = nullptr;
      fan::vulkan::validate(ctx->map_buffer(staging, &mapped));
      std::memcpy(mapped, data, (std::size_t)bytes);
      ctx->unmap_buffer(staging);
      ctx->copy_buffer_cmd(cmd, staging, dst_buffer, 0, elem_size * first, bytes);
      deferred_destroy.push_back(staging);
    }
    bool upload_mesh_model_gpu(const model_t& model) {
      if (!ctx || !ready) { return false; }
      std::vector<fan::vulkan::context_t::buffer_t> deferred_destroy;
      deferred_destroy.reserve(5);
      VkCommandBuffer cmd = ctx->begin_single_time_commands();

      if (model.keep_cpu_geometry) {
        upload_model_buffer_slice(cmd, deferred_destroy, source_vertex_buffer, source_vertex_data.data() + model.first_vertex, sizeof(source_vertex_t), model.first_vertex, model.vertex_capacity);
      }
      upload_model_buffer_slice(cmd, deferred_destroy, vertex_buffer, vertex_data.data() + model.first_vertex, sizeof(vertex_t), model.first_vertex, model.vertex_capacity);
      upload_model_buffer_slice(cmd, deferred_destroy, index_buffer, index_data.data() + model.first_index, sizeof(std::uint32_t), model.first_index, model.index_capacity);
      upload_model_buffer_slice(cmd, deferred_destroy, material_buffer, materials.data() + model.material_index, sizeof(material_info_t), model.material_index, model.material_capacity);
      upload_model_buffer_slice(cmd, deferred_destroy, material_index_buffer, material_indices_per_primitive.data() + model.first_primitive, sizeof(std::uint32_t), model.first_primitive, model.index_count / 3);

      ctx->buffer_barriers_cmd(cmd, {
        {&source_vertex_buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR},
        {&vertex_buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR},
        {&index_buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR},
        {&material_buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT},
        {&material_index_buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT}
      }, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR);
      ctx->end_single_time_commands(cmd);
      for (auto& b : deferred_destroy) { destroy_buffer(b); }
      return true;
    }
    void make_pending_model_buffer_copy(pending_mesh_upload_t& job, fan::vulkan::context_t::buffer_t& dst_buffer, const void* data, VkDeviceSize elem_size, VkDeviceSize first, VkDeviceSize count) {
      if (!count) { return; }
      VkDeviceSize bytes = elem_size * count;
      fan::vulkan::context_t::buffer_t staging;
      ctx->create_buffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging);
      void* mapped = nullptr;
      fan::vulkan::validate(ctx->map_buffer(staging, &mapped));
      std::memcpy(mapped, data, (std::size_t)bytes);
      ctx->unmap_buffer(staging);
      job.copies.push_back({.staging = staging, .dst_buffer = &dst_buffer, .dst_offset = elem_size * first, .size = bytes});
    }
    bool enqueue_mesh_model_gpu_upload(std::uint32_t model_index, const voxel_mesh_input_t* mesh_input = nullptr) {
      if (!ctx || !ready || model_index >= models.size()) { return false; }
      const model_t& model = models[model_index];
      pending_mesh_upload_t job;
      job.model_index = model_index;
      job.copies.reserve(5);

      std::vector<std::uint32_t> temp_indices;
      const source_vertex_t* source_upload = nullptr;
      const vertex_t* vertex_upload = nullptr;
      const std::uint32_t* index_upload = nullptr;

      if (model.keep_cpu_geometry) {
        source_upload = source_vertex_data.data() + model.first_vertex;
        vertex_upload = vertex_data.data() + model.first_vertex;
        index_upload = index_data.data() + model.first_index;
      }
      else {
        if (mesh_input == nullptr) { return false; }
        temp_indices = make_gpu_indices(*mesh_input, model.first_vertex, model.index_count);
        vertex_upload = mesh_input->vertices.data();
        index_upload = temp_indices.data();
      }

      if (model.keep_cpu_geometry) {
        make_pending_model_buffer_copy(job, source_vertex_buffer, source_upload, sizeof(source_vertex_t), model.first_vertex, model.vertex_count);
      }
      make_pending_model_buffer_copy(job, vertex_buffer, vertex_upload, sizeof(vertex_t), model.first_vertex, model.vertex_count);
      make_pending_model_buffer_copy(job, index_buffer, index_upload, sizeof(std::uint32_t), model.first_index, model.index_count);
      make_pending_model_buffer_copy(job, material_buffer, materials.data() + model.material_index, sizeof(material_info_t), model.material_index, model.material_count);
      make_pending_model_buffer_copy(job, material_index_buffer, material_indices_per_primitive.data() + model.first_primitive, sizeof(std::uint32_t), model.first_primitive, model.blas_primitive_count);
      pending_mesh_uploads.push_back(std::move(job));
      return true;
    }
    void retire_mesh_upload(pending_mesh_upload_t& job) {
      retired_mesh_upload_t retired;
      retired.frames_left = 4;
      retired.staging_buffers.reserve(job.copies.size());
      for (auto& copy : job.copies) { retired.staging_buffers.push_back(copy.staging); copy.staging = {}; }
      retired_mesh_uploads.push_back(std::move(retired));
    }
    void gc_retired_mesh_uploads() {
      for (std::size_t i = 0; i < retired_mesh_uploads.size();) {
        auto& retired = retired_mesh_uploads[i];
        if (retired.frames_left) { --retired.frames_left; ++i; continue; }
        for (auto& staging : retired.staging_buffers) { destroy_buffer(staging); }
        retired_mesh_uploads.erase(retired_mesh_uploads.begin() + i);
      }
    }
    void destroy_mesh_upload_jobs() {
      for (auto& job : pending_mesh_uploads) {
        for (auto& copy : job.copies) { destroy_buffer(copy.staging); }
      }
      pending_mesh_uploads.clear();
      for (auto& retired : retired_mesh_uploads) {
        for (auto& staging : retired.staging_buffers) { destroy_buffer(staging); }
      }
      retired_mesh_uploads.clear();
    }
    void record_pending_mesh_uploads(VkCommandBuffer cmd) {
      if (pending_mesh_uploads.empty()) { return; }
      pending_mesh_upload_t job = std::move(pending_mesh_uploads.front());
      pending_mesh_uploads.erase(pending_mesh_uploads.begin());
      for (auto& copy : job.copies) { ctx->copy_buffer_cmd(cmd, copy.staging, *copy.dst_buffer, 0, copy.dst_offset, copy.size); }
      ctx->buffer_barriers_cmd(cmd, {
        {&source_vertex_buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR},
        {&vertex_buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR},
        {&index_buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR},
        {&material_buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT},
        {&material_index_buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT}
      }, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR);
      record_blas_rebuild(cmd, job.model_index);
      VkMemoryBarrier barrier {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_SHADER_READ_BIT
      };
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &barrier, 0, nullptr, 0, nullptr);
      retire_mesh_upload(job);
    }
    static void release_mesh_input_storage(voxel_mesh_input_t& mesh) {
      std::vector<vertex_t>().swap(mesh.vertices);
      std::vector<std::uint32_t>().swap(mesh.indices);
      std::vector<material_info_t>().swap(mesh.materials);
      std::vector<std::uint32_t>().swap(mesh.primitive_material_indices);
      mesh.vertex_capacity = 0;
      mesh.index_capacity = 0;
      mesh.material_capacity = 0;
    }
    void clear_procedural_mesh_copy(scene_model_t& scene_model) {
      if (scene_model.procedural_mesh_index == std::uint32_t(-1) || scene_model.procedural_mesh_index >= procedural_meshes.size()) { return; }
      release_mesh_input_storage(procedural_meshes[scene_model.procedural_mesh_index]);
      scene_model.procedural_mesh_index = std::uint32_t(-1);
    }
    void store_procedural_mesh_copy(scene_model_t& scene_model, const voxel_mesh_input_t& mesh) {
      if (!mesh.keep_cpu_copy) {
        clear_procedural_mesh_copy(scene_model);
        return;
      }
      if (scene_model.procedural_mesh_index == std::uint32_t(-1)) {
        scene_model.procedural_mesh_index = (std::uint32_t)procedural_meshes.size();
        procedural_meshes.push_back(mesh);
      }
      else {
        procedural_meshes[scene_model.procedural_mesh_index] = mesh;
      }
    }
    bool update_mesh(object_handle_t handle, const voxel_mesh_input_t& mesh_input) {
      if (!is_object_valid(handle)) { return false; }
      voxel_mesh_input_t effective_mesh = mesh_input.vertices.empty() || mesh_input.indices.empty() ? make_degenerate_mesh(mesh_input) : mesh_input;
      scene_model_t& scene_model = scene_models[handle.index];
      scene_model.transform = effective_mesh.transform;
      scene_model.path.clear();
      scene_model.animated = false;
      if (!ctx || !ready) { store_procedural_mesh_copy(scene_model, effective_mesh); return true; }
      store_procedural_mesh_copy(scene_model, effective_mesh);
      object_t& object = objects[handle.index];
      if (object.instance_count == 0 || object.first_instance >= instances.size()) { return false; }
      instance_t& instance = instances[object.first_instance];
      instance.transform = effective_mesh.transform;
      if (instance.model_index >= models.size()) { return false; }
      model_t& model = models[instance.model_index];
      if (!rewrite_mesh_model(model, effective_mesh)) { return false; }
      if (!enqueue_mesh_model_gpu_upload(instance.model_index, &effective_mesh)) { return false; }
      tlas_dirty = true;
      tlas_rebuild_dirty = false;
      if (incremental_upload_batch) {
        incremental_upload_had_changes = true;
      }
      reset_accumulation();
      return true;
    }
    object_handle_t add_voxel_mesh(const voxel_grid_t& grid, const fan::mat4& transform = fan::mat4(1), const fan::vec3& base_color = fan::vec3(1, 1, 1), std::int32_t albedo_texture_id = -1, f32_t voxel_size = 1.f) {
      voxel_mesh_input_t mesh = greedy_mesh_grid(grid, voxel_size);
      mesh.transform = transform;
      mesh.base_color = base_color;
      mesh.albedo_texture_id = albedo_texture_id;
      return add_mesh(mesh);
    }
    static fan::vec4 sample_cpu_texture(const fan::model::cpu_texture_t& texture, fan::vec2 uv, const fan::vec4& fallback = fan::vec4(1, 1, 1, 1)) {
      if (!texture.valid() || texture.channels < 3 || texture.size.x == 0 || texture.size.y == 0) { return fallback; }
      f32_t u = uv.x - std::floor(uv.x);
      f32_t v = uv.y - std::floor(uv.y);
      std::uint32_t x = std::min<std::uint32_t>((std::uint32_t)(u * texture.size.x), texture.size.x - 1);
      std::uint32_t y = std::min<std::uint32_t>((std::uint32_t)(v * texture.size.y), texture.size.y - 1);
      std::uint32_t index = (y * texture.size.x + x) * texture.channels;
      constexpr f32_t inv_255 = 1.f / 255.f;
      return fan::vec4(
        texture.data.get()[index + 0] * inv_255,
        texture.data.get()[index + 1] * inv_255,
        texture.data.get()[index + 2] * inv_255,
        texture.channels >= 4 ? texture.data.get()[index + 3] * inv_255 : 1.f
      );
    }
    voxel_mesh_input_t voxelize_model_to_mesh(
      const scene_model_t& model,
      const voxelized_model_properties_t& properties = {},
      std::source_location callers_path = std::source_location::current(),
      const fan::model::fms_t* source_fms = nullptr,
      const std::vector<fan::model::cpu_texture_t>* preloaded_textures = nullptr,
      const std::vector<int>* preloaded_mesh_albedo_textures = nullptr,
      voxelizer_workspace_t* workspace = nullptr
    ) const {
      if (properties.voxel_resolution == 0) { fan::throw_error("voxelized model resolution must be greater than zero"); }
      std::unique_ptr<fan::model::fms_t> loaded_fms;
      if (source_fms == nullptr) {
        fan::model::fms_t::properties_t fms_properties;
        fms_properties.path = model.path;
        fms_properties.texture_path = properties.texture_path.empty() ? model.texture_path : properties.texture_path;
        fms_properties.load_skeleton = false;
        fms_properties.load_animations = false;
        fms_properties.fix_uv_diagonals = properties.fix_uv_diagonals || model.fix_uv_diagonals;
        fms_properties.texture_loading = fan::model::fms_t::properties_t::texture_loading_e::wait;
        fms_properties.load_texture_types = fan::model::make_texture_filter({fan::texture_type::base_color, fan::texture_type::diffuse, fan::texture_type::ambient, fan::texture_type::unknown});
        loaded_fms = std::make_unique<fan::model::fms_t>(fms_properties, callers_path);
        source_fms = loaded_fms.get();
      }
      const fan::model::fms_t& fms = *source_fms;
      struct voxel_tri_t {
        fan::vec3 v0, v1, v2;
        fan::vec4 c0, c1, c2;
        fan::vec2 uv0, uv1, uv2;
        fan::vec4 base_color = fan::vec4(1, 1, 1, 1);
        int albedo_texture = -1;
        f32_t area = 0;
      };

      std::vector<const fan::model::pm_texture_data_t*> active_textures;
      std::unordered_map<std::string, int> texture_layer_map;
      std::vector<int> local_mesh_albedo_textures;
      const std::vector<int>* mesh_albedo_textures = preloaded_mesh_albedo_textures;
      if (!mesh_albedo_textures) {
        local_mesh_albedo_textures.resize(fms.meshes.size(), -1);
        active_textures.reserve(fms.meshes.size());
        texture_layer_map.reserve(fms.meshes.size() * 2);
        for (std::uint32_t mesh_index = 0; mesh_index < fms.meshes.size(); ++mesh_index) {
          local_mesh_albedo_textures[mesh_index] = fan::model::get_first_texture_index(
            fms.meshes[mesh_index],
            {fan::texture_type::base_color, fan::texture_type::diffuse, fan::texture_type::ambient, fan::texture_type::unknown},
            active_textures,
            texture_layer_map
          );
        }
        mesh_albedo_textures = &local_mesh_albedo_textures;
      }

      std::vector<fan::model::cpu_texture_t> local_textures;
      const std::vector<fan::model::cpu_texture_t>* textures = preloaded_textures;
      if (!textures) {
        local_textures = fan::model::copy_cpu_textures(active_textures);
        textures = &local_textures;
      }
      fan::vec3 bmin = fms.aabbmin;
      fan::vec3 bmax = fms.aabbmax;
      f32_t max_extent = fan::vec4(bmax - bmin, 1.f).max();
      if (max_extent <= 0.f) { fan::throw_error("voxelized model has invalid bounds"); }
      f32_t voxel_res = (f32_t)properties.voxel_resolution;
      f32_t scale = (voxel_res * 0.85f) / max_extent;
      fan::vec3 grid_offset(voxel_res * 0.075f);

      std::vector<voxel_tri_t> tris;
      tris.reserve(fms.get_triangle_count());
      fms.for_each_triangle([&](const fan::model::fms_t::triangle_ref_t& triangle) {
        const auto& v0 = *triangle.vertices[0];
        const auto& v1 = *triangle.vertices[1];
        const auto& v2 = *triangle.vertices[2];
        fan::color material_color = fms.get_material_base_color(triangle.mesh_id);
        auto to_grid = [&](const fan::vec3& position) { return (position - bmin) * scale + grid_offset; };
        voxel_tri_t out;
        out.v0 = to_grid(v0.position); out.v1 = to_grid(v1.position); out.v2 = to_grid(v2.position);
        out.c0 = fan::vec4(v0.color.x, v0.color.y, v0.color.z, v0.color.w); out.c1 = fan::vec4(v1.color.x, v1.color.y, v1.color.z, v1.color.w); out.c2 = fan::vec4(v2.color.x, v2.color.y, v2.color.z, v2.color.w);
        out.uv0 = v0.uv; out.uv1 = v1.uv; out.uv2 = v2.uv;
        out.base_color = material_color;
        out.albedo_texture = triangle.mesh_id < mesh_albedo_textures->size() ? (*mesh_albedo_textures)[triangle.mesh_id] : -1;
        fan::vec3 normal = (out.v1 - out.v0).cross(out.v2 - out.v0);
        out.area = normal.length_squared();
        if (out.area > 1e-10f) { tris.push_back(out); }
      });

      voxelizer_workspace_t local_workspace;
      voxelizer_workspace_t& ws = workspace ? *workspace : local_workspace;

      ws.grid.resize(properties.voxel_resolution, properties.voxel_resolution, properties.voxel_resolution);
      std::size_t grid_size = (std::size_t)properties.voxel_resolution * properties.voxel_resolution * properties.voxel_resolution;
      if (ws.cell_area.size() < grid_size) { ws.cell_area.resize(grid_size); }
      std::fill_n(ws.cell_area.begin(), grid_size, -1.f);

      auto clamp_cell = [&](int value) { return std::clamp(value, 0, (int)properties.voxel_resolution - 1); };
      auto sample_tri_color = [&](const voxel_tri_t& tri, const fan::vec3& p) {
        fan::vec3 bary = fan::math::d3::closest_barycentric(p, tri.v0, tri.v1, tri.v2);
        fan::vec4 color = tri.c0 * bary.x + tri.c1 * bary.y + tri.c2 * bary.z;
        fan::vec2 uv = tri.uv0 * bary.x + tri.uv1 * bary.y + tri.uv2 * bary.z;
        if (tri.albedo_texture >= 0 && tri.albedo_texture < (int)textures->size()) { color *= sample_cpu_texture((*textures)[tri.albedo_texture], uv); }
        color *= tri.base_color;
        color.x *= properties.tint.x; color.y *= properties.tint.y; color.z *= properties.tint.z;
        return color;
      };

      fan::vec3 half_size(0.5f + std::max(properties.occupancy_padding, 0.f));
      for (const voxel_tri_t& tri : tris) {
        fan::math::d3::aabb_t bounds = fan::math::d3::triangle_bounds(tri.v0, tri.v1, tri.v2);
        int min_x = clamp_cell((int)std::floor(bounds.min.x));
        int min_y = clamp_cell((int)std::floor(bounds.min.y));
        int min_z = clamp_cell((int)std::floor(bounds.min.z));
        int max_x = clamp_cell((int)std::ceil(bounds.max.x));
        int max_y = clamp_cell((int)std::ceil(bounds.max.y));
        int max_z = clamp_cell((int)std::ceil(bounds.max.z));
        for (int z = min_z; z <= max_z; ++z) {
          f32_t cz = (f32_t)z + 0.5f;
          for (int y = min_y; y <= max_y; ++y) {
            f32_t cy = (f32_t)y + 0.5f;
            for (int x = min_x; x <= max_x; ++x) {
              std::size_t cell_index = (std::size_t)z * ws.grid.sy * ws.grid.sx + (std::size_t)y * ws.grid.sx + (std::size_t)x;
              if (tri.area <= ws.cell_area[cell_index]) { continue; }
              fan::vec3 center((f32_t)x + 0.5f, cy, cz);
              if (!fan::math::d3::triangle_intersects_aabb(tri.v0, tri.v1, tri.v2, center, half_size)) { continue; }
              fan::vec4 color = sample_tri_color(tri, center);
              if (color.w <= properties.alpha_threshold) { continue; }
              voxel_t& voxel = ws.grid.at((std::uint32_t)x, (std::uint32_t)y, (std::uint32_t)z);
              voxel.id = 1; voxel.color = color; ws.cell_area[cell_index] = tri.area;
            }
          }
        }
      }
      voxel_mesh_input_t mesh = greedy_mesh_grid_impl(ws.grid, nullptr, {}, 1.f, &ws.mask);
      for (vertex_t& vertex : mesh.vertices) { vertex.position = (fan::vec3(vertex.position) - grid_offset) / scale + bmin; }
      mesh.transform = model.transform;
      mesh.base_color = fan::vec3(1, 1, 1);
      return mesh;
    }
    object_handle_t add_voxelized_model(const scene_model_t& model, const voxelized_model_properties_t& properties = {}, std::source_location callers_path = std::source_location::current()) {
      voxel_mesh_input_t mesh = voxelize_model_to_mesh(model, properties, callers_path);
      return add_mesh(mesh);
    }
    object_handle_t add_voxelized_model(const std::string& path, const fan::mat4& transform = fan::mat4(1), std::uint32_t voxel_resolution = 64, std::source_location callers_path = std::source_location::current()) {
      scene_model_t model;
      model.path = path;
      model.transform = transform;
      voxelized_model_properties_t properties;
      properties.voxel_resolution = voxel_resolution;
      return add_voxelized_model(model, properties, callers_path);
    }
    struct voxelized_animation_clip_work_t {
      std::string name;
      f32_t duration_ms = 0;
      std::uint32_t frame_count = 0;
    };
    struct voxelized_animation_frame_work_t {
      std::uint32_t clip_index = 0;
      std::uint32_t frame_index = 0;
      f32_t time_ms = 0;
    };
    struct voxelized_animation_frame_result_t {
      std::uint32_t clip_index = 0;
      std::uint32_t frame_index = 0;
      f32_t time_ms = 0;
      voxel_mesh_input_t mesh;
      bool valid = false;
    };
    static std::uint32_t get_voxelized_animation_worker_count(std::uint32_t job_count) {
      std::uint32_t hw = std::max<std::uint32_t>(1, std::thread::hardware_concurrency());
      if (hw > 1) { --hw; }
      return std::max<std::uint32_t>(1, std::min(hw, job_count));
    }
    object_handle_t add_voxelized_animated_model(const std::string& path, const fan::mat4& transform = fan::mat4(1), std::uint32_t voxel_resolution = 32, std::source_location callers_path = std::source_location::current()) {
      fan::model::fms_t::properties_t fms_props;
      fms_props.path = path;
      fms_props.load_skeleton = true;
      fms_props.load_animations = true;
      fms_props.texture_loading = fan::model::fms_t::properties_t::texture_loading_e::wait;
      fan::model::fms_t fms(fms_props, callers_path);
      if (fms.animation_list.empty() || !fms.root_bone) { return add_voxelized_model(path, transform, voxel_resolution, callers_path); }

      std::vector<const fan::model::pm_texture_data_t*> active_textures;
      std::unordered_map<std::string, int> texture_layer_map;
      std::vector<int> mesh_albedo_textures(fms.meshes.size(), -1);
      active_textures.reserve(fms.meshes.size());
      texture_layer_map.reserve(fms.meshes.size() * 2);
      for (std::uint32_t mesh_index = 0; mesh_index < fms.meshes.size(); ++mesh_index) {
        mesh_albedo_textures[mesh_index] = fan::model::get_first_texture_index(
          fms.meshes[mesh_index],
          {fan::texture_type::base_color, fan::texture_type::diffuse, fan::texture_type::ambient, fan::texture_type::unknown},
          active_textures,
          texture_layer_map
        );
      }
      const std::vector<fan::model::cpu_texture_t> preloaded_textures = fan::model::copy_cpu_textures(active_textures);

      object_handle_t handle {(std::uint32_t)scene_models.size(), object_generation_counter++};
      scene_model_t scene_model {};
      scene_model.transform = transform;
      scene_model.animated = false;
      scene_models.push_back(scene_model);
      objects.push_back({.generation = handle.generation});
      object_t& object = objects.back();
      object.first_instance = (std::uint32_t)instances.size();
      object.animation_cache_index = (std::uint32_t)animation_caches.size();

      animation_cache_entry_t& cache = animation_caches.emplace_back();
      cache.default_clip = fms.active_anim;
      cache.bone_count = fms.bone_count;
      cache.sample_rate = animation_sample_rate;

      std::vector<voxelized_animation_clip_work_t> clip_jobs;
      std::vector<voxelized_animation_frame_work_t> frame_jobs;
      clip_jobs.reserve(fms.animation_list.size());

      for (auto& [clip_name, animation] : fms.animation_list) {
        std::uint32_t clip_index = (std::uint32_t)cache.clips.size();

        cache.clip_indices[clip_name] = clip_index;
        animation_clip_cache_t& clip = cache.clips.emplace_back();
        clip.name = clip_name;
        clip.duration_ms = animation.duration > 0 ? animation.duration : 1000.f;

        voxelized_animation_clip_work_t& clip_job = clip_jobs.emplace_back();
        clip_job.name = clip_name;
        clip_job.duration_ms = clip.duration_ms;
        clip_job.frame_count = std::max<std::uint32_t>(1, (std::uint32_t)std::ceil((clip.duration_ms / 1000.f) * animation_sample_rate));

        clip.frames.reserve(clip_job.frame_count);
        for (std::uint32_t frame = 0; frame < clip_job.frame_count; ++frame) {
          frame_jobs.push_back({
            .clip_index = clip_index,
            .frame_index = frame,
            .time_ms = get_cached_animation_sample_time(clip.duration_ms, frame, clip_job.frame_count)
          });
        }
      }

      std::vector<std::vector<voxelized_animation_frame_result_t>> frame_results(cache.clips.size());
      for (std::uint32_t i = 0; i < clip_jobs.size(); ++i) {
        frame_results[i].resize(clip_jobs[i].frame_count);
      }

      std::uint32_t worker_count = get_voxelized_animation_worker_count((std::uint32_t)frame_jobs.size());
      std::vector<std::future<std::vector<voxelized_animation_frame_result_t>>> futures;
      futures.reserve(worker_count);

      for (std::uint32_t worker_index = 0; worker_index < worker_count; ++worker_index) {
        futures.push_back(std::async(std::launch::async, [&, worker_index, worker_count]() {
          static std::mutex fms_load_mutex;

          std::unique_ptr<fan::model::fms_t> worker_fms;
          {
            std::scoped_lock lock(fms_load_mutex);
            worker_fms = std::make_unique<fan::model::fms_t>(fms_props, callers_path);
          }

          fan::model::fms_t& fms = *worker_fms;
          voxelizer_workspace_t workspace;

          scene_model_t deformed_model {};
          deformed_model.path = path;
          deformed_model.transform = transform;

          voxelized_model_properties_t vox_props {};
          vox_props.voxel_resolution = voxel_resolution;

          std::vector<voxelized_animation_frame_result_t> results;
          results.reserve((frame_jobs.size() + worker_count - 1) / worker_count);

          std::uint32_t active_clip_index = std::uint32_t(-1);

          for (std::uint32_t job_index = worker_index; job_index < frame_jobs.size(); job_index += worker_count) {
            const voxelized_animation_frame_work_t& job = frame_jobs[job_index];
            const voxelized_animation_clip_work_t& clip_job = clip_jobs[job.clip_index];

            if (active_clip_index != job.clip_index) {
              for (auto& other : fms.animation_list) { other.second.weight = 0; }
              auto found = fms.animation_list.find(clip_job.name);
              if (found == fms.animation_list.end()) { fan::throw_error("voxelized animation clip missing in worker"); }
              found->second.weight = 1.f;
              fms.active_anim = clip_job.name;
              active_clip_index = job.clip_index;
            }

            fms.dt = job.time_ms;
            fms.fk_calculate_poses();
            std::vector<fan::mat4> transforms = fms.fk_calculate_transformations();
            fms.calculate_modified_vertices(transforms);

            std::swap(fms.meshes, fms.calculated_meshes);
            voxel_mesh_input_t vox_mesh = voxelize_model_to_mesh(deformed_model, vox_props, callers_path, &fms, &preloaded_textures, &mesh_albedo_textures, &workspace);
            std::swap(fms.meshes, fms.calculated_meshes);

            results.push_back({
              .clip_index = job.clip_index,
              .frame_index = job.frame_index,
              .time_ms = job.time_ms,
              .mesh = std::move(vox_mesh),
              .valid = true
            });
          }

          return results;
        }));
      }

      for (auto& future : futures) {
        for (auto& result : future.get()) {
          frame_results[result.clip_index][result.frame_index] = std::move(result);
        }
      }

      for (std::uint32_t clip_index = 0; clip_index < cache.clips.size(); ++clip_index) {
        animation_clip_cache_t& clip = cache.clips[clip_index];
        for (std::uint32_t frame = 0; frame < frame_results[clip_index].size(); ++frame) {
          voxelized_animation_frame_result_t& result = frame_results[clip_index][frame];
          if (!result.valid) { fan::throw_error("voxelized animation frame failed"); }
          std::uint32_t model_index = append_mesh_model(result.mesh);
          clip.frames.push_back({model_index, 1, result.time_ms});
        }
      }

      std::uint32_t first_clip_index = 0;
      auto default_clip = cache.clip_indices.find(cache.default_clip);
      if (default_clip != cache.clip_indices.end()) { first_clip_index = default_clip->second; }

      if (!cache.clips.empty() && !cache.clips[first_clip_index].frames.empty()) {
        const animation_frame_t& first_frame = cache.clips[first_clip_index].frames.front();
        add_instance(first_frame.first_model, transform);
        object.instance_count = 1;
        apply_object_ray_mask(handle.index);
        object.animation_frame = 0;
        object.animation_name = cache.default_clip;
      }
      if (ctx && ready) {
        scene_geometry_dirty = true;
        tlas_dirty = true;
        reset_accumulation();
      }
      return handle;
    }
    object_handle_t add_model(const scene_model_t& model, std::source_location callers_path = std::source_location::current()) {
      object_handle_t handle {(std::uint32_t)scene_models.size(), object_generation_counter++};
      scene_models.push_back(model);
      scene_models.back().callers_path = callers_path;
      objects.push_back({.generation = handle.generation, .first_instance = 0, .instance_count = 0});
      if (!ctx || !ready) { return handle; }
      bool geometry_changed = load_scene_model(handle.index);
      scene_geometry_dirty = scene_geometry_dirty || geometry_changed;
      tlas_dirty = true;
      reset_accumulation();
      return handle;
    }
    object_handle_t add_model(const std::string& path, const fan::mat4& transform = fan::mat4(1), std::source_location callers_path = std::source_location::current()) {
      scene_model_t m;
      m.path = path;
      m.transform = transform;
      return add_model(m, callers_path);
    }
    object_handle_t add_animated_model(const std::string& path, const fan::mat4& transform = fan::mat4(1), std::source_location callers_path = std::source_location::current()) {
      scene_model_t m;
      m.path = path;
      m.transform = transform;
      m.animated = true;
      return add_model(m, callers_path);
    }
    object_handle_t add_animated_model(const std::string& path, const fan::mat4& transform, const std::string& animation_name, f32_t animation_time_offset, f32_t animation_speed = 1.f, std::source_location callers_path = std::source_location::current()) {
      scene_model_t m;
      m.path = path;
      m.transform = transform;
      m.animation_name = animation_name;
      m.animation_time_offset = animation_time_offset;
      m.animation_speed = animation_speed;
      m.animated = true;
      return add_model(m, callers_path);
    }
    void clear_scene_models() {
      scene_models.clear();
      procedural_meshes.clear();
      objects.clear();
      selected_object = {};
      if (!ctx || !ready) { return; }
      instances.clear();
      animated_models.clear();
      bone_matrices.clear();
      tlas_dirty = true;
      tlas_rebuild_dirty = false;
      reset_accumulation();
    }
    bool is_object_valid(object_handle_t handle) const {
      return handle.valid() && handle.index < objects.size() && objects[handle.index].generation == handle.generation;
    }
    bool remove_object(object_handle_t handle) {
      if (!is_object_valid(handle)) { return false; }
      if (handle.index >= scene_models.size()) { return false; }
      object_t& object = objects[handle.index];
      std::uint32_t erase_begin = object.first_instance;
      std::uint32_t erase_count = object.instance_count;
      if (erase_count != 0 && erase_begin < instances.size()) {
        erase_count = std::min<std::uint32_t>(erase_count, (std::uint32_t)instances.size() - erase_begin);
        instances.erase(instances.begin() + erase_begin, instances.begin() + erase_begin + erase_count);
        for (std::uint32_t i = 0; i < objects.size(); ++i) {
          if (i == handle.index) { continue; }
          object_t& other = objects[i];
          if (other.instance_count != 0 && other.first_instance > erase_begin) { other.first_instance -= std::min(erase_count, other.first_instance - erase_begin); }
        }
      }
      scene_model_t& model = scene_models[handle.index];
      std::uint32_t procedural_index = model.procedural_mesh_index;
      model.path.clear();
      model.animated = false;
      model.procedural_mesh_index = std::uint32_t(-1);
      if (procedural_index != std::uint32_t(-1) && procedural_index < procedural_meshes.size()) {
        bool referenced = false;
        for (const scene_model_t& other : scene_models) {
          if (other.procedural_mesh_index == procedural_index) {
            referenced = true;
            break;
          }
        }
        if (!referenced) {
          procedural_meshes.erase(procedural_meshes.begin() + procedural_index);
          for (scene_model_t& other : scene_models) {
            if (other.procedural_mesh_index != std::uint32_t(-1) && other.procedural_mesh_index > procedural_index) { --other.procedural_mesh_index; }
          }
        }
      }
      object.first_instance = 0;
      object.instance_count = 0;
      object.animated_model_index = std::uint32_t(-1);
      object.animation_cache_index = std::uint32_t(-1);
      object.animation_name.clear();
      object.animation_frame = std::uint32_t(-1);
      ++object.generation;
      if (object.generation == 0) { ++object.generation; }
      if (selected_object.index == handle.index) { selected_object = {}; }
      tlas_dirty = true;
      tlas_rebuild_dirty = true;
      reset_accumulation();
      return true;
    }
    bool destroy_object_geometry(object_handle_t handle) {
      if (!is_object_valid(handle)) { return false; }
      wait_idle();

      std::uint32_t model_index = std::uint32_t(-1);
      const object_t& object = objects[handle.index];
      if (object.instance_count != 0 && object.first_instance < instances.size()) {
        model_index = instances[object.first_instance].model_index;
      }

      bool removed = remove_object(handle);

      if (ctx && model_index != std::uint32_t(-1)) {
        if (model_index < blas_list.size()) {
          blas_list[model_index].destroy(*ctx);
        }
        if (model_index < models.size()) {
          model_t& model = models[model_index];
          release_scene_range(vertex_free_ranges, model.first_vertex, model.vertex_capacity);
          release_scene_range(index_free_ranges, model.first_index, model.index_capacity);
          release_scene_range(primitive_free_ranges, model.first_primitive, model.blas_primitive_capacity);
          model.index_count = 0;
          model.vertex_count = 0;
          model.vertex_capacity = 0;
          model.index_capacity = 0;
          model.blas_primitive_count = 0;
          model.blas_primitive_capacity = 0;
          model.material_count = 0;
          model.material_capacity = 0;
          model.has_bounds = false;
        }
      }

      return removed;
    }
    std::size_t pending_mesh_upload_count() const {
      return pending_mesh_uploads.size();
    }
    bool has_pending_mesh_uploads() const {
      return !pending_mesh_uploads.empty();
    }
    void wait_idle() {
      if (ctx) { vkDeviceWaitIdle(ctx->device); }
    }

    fan::mat4 get_transform(object_handle_t handle) const {
      if (!is_object_valid(handle)) { fan::throw_error("invalid ray tracing object handle"); }
      return scene_models[handle.index].transform;
    }
    bool set_transform(object_handle_t handle, const fan::mat4& transform, bool rebuild_now = false) {
      if (!is_object_valid(handle)) { return false; }
      scene_models[handle.index].transform = transform;
      object_t& object = objects[handle.index];
      for (std::uint32_t i = 0; i < object.instance_count; ++i) { instances[object.first_instance + i].transform = transform; }
      tlas_dirty = true;
      reset_accumulation();
      if (rebuild_now) { flush_transform_updates(); }
      return true;
    }
    bool set_transform_deferred(object_handle_t handle, const fan::mat4& transform) {
      return set_transform(handle, transform, false);
    }
    bool set_ray_mask(object_handle_t handle, std::uint32_t mask) {
      if (!is_object_valid(handle)) { return false; }
      object_t& object = objects[handle.index];
      object.ray_mask = mask;
      apply_object_ray_mask(handle.index);
      tlas_dirty = true;
      tlas_rebuild_dirty = true;
      reset_accumulation();
      return true;
    }
    bool set_object_animation_frame(std::uint32_t object_index, f32_t time_seconds) {
      object_t& object = objects[object_index];
      if (object.animation_cache_index == std::uint32_t(-1) || object.animation_cache_index >= animation_caches.size()) { return false; }
      animation_cache_entry_t& cache = animation_caches[object.animation_cache_index];
      std::string clip_name = object.animation_name.empty() ? cache.default_clip : object.animation_name;
      auto clip_found = cache.clip_indices.find(clip_name);
      if (clip_found == cache.clip_indices.end()) { return false; }
      animation_clip_cache_t& clip = cache.clips[clip_found->second];
      if (clip.frames.empty()) { return false; }
      f32_t duration = clip.duration_ms > 0.f ? clip.duration_ms : 1000.f;
      f32_t local_time = std::fmod((time_seconds + object.animation_time_offset) * object.animation_speed * 1000.f, duration);
      if (local_time < 0.f) { local_time += duration; }
      std::uint32_t frame = std::min<std::uint32_t>((std::uint32_t)((local_time / duration) * clip.frames.size()), (std::uint32_t)clip.frames.size() - 1);
      if (object.animation_frame == frame) { return true; }
      const animation_frame_t& frame_data = clip.frames[frame];
      for (std::uint32_t i = 0; i < object.instance_count; ++i) { instances[object.first_instance + i].model_index = frame_data.first_model + i; }
      object.animation_frame = frame;
      tlas_dirty = true;
      tlas_rebuild_dirty = true;
      reset_accumulation();
      return true;
    }
    bool set_animation(object_handle_t handle, f32_t time_seconds, const std::string& animation_name = {}, f32_t weight = 1.f) {
      if (!is_object_valid(handle)) { return false; }
      object_t& object = objects[handle.index];
      if (object.animation_cache_index != std::uint32_t(-1)) {
        if (!animation_name.empty()) { object.animation_name = animation_name; }
        return set_object_animation_frame(handle.index, time_seconds);
      }
      if (object.animated_model_index == std::uint32_t(-1) || object.animated_model_index >= animated_models.size()) { return false; }
      animated_model_t& animated_model = animated_models[object.animated_model_index];
      if (!animated_model.fms) { return false; }
      fan::model::fms_t& fms = *animated_model.fms;
      if (!fms.root_bone || fms.bone_count == 0 || fms.animation_list.empty()) { return false; }
      std::string name = animation_name.empty() ? fms.active_anim : animation_name;
      auto found = fms.animation_list.find(name);
      if (found == fms.animation_list.end()) { return false; }
      for (auto& animation : fms.animation_list) { animation.second.weight = 0; }
      fms.active_anim = name;
      found->second.weight = weight;
      fms.dt = time_seconds * 1000.f;
      if (fms.bone_transforms.size() != fms.bone_count) { fms.bone_transforms.resize(fms.bone_count, fan::mat4(1)); }
      fms.fk_calculate_poses();
      std::vector<fan::mat4> transforms = fms.fk_calculate_transformations();
      if (transforms.size() < animated_model.bone_count) { transforms.resize(animated_model.bone_count, fan::mat4(1)); }
      std::copy_n(transforms.begin(), animated_model.bone_count, bone_matrices.begin() + animated_model.first_bone);
      bone_buffer_dirty = true;
      animated_model.dirty = true;
      animation_vertices_dirty = true;
      reset_accumulation();
      return true;
    }
    bool set_animation(object_handle_t handle, const std::string& animation_name = {}, f32_t weight = 1.f) {
      return set_animation(handle, attached_engine->start_time.seconds(), animation_name, weight);
    }
    bool set_animation_offset(object_handle_t handle, f32_t animation_time_offset) {
      if (!is_object_valid(handle)) { return false; }
      objects[handle.index].animation_time_offset = animation_time_offset;
      return true;
    }
    bool set_animation_speed(object_handle_t handle, f32_t animation_speed) {
      if (!is_object_valid(handle)) { return false; }
      objects[handle.index].animation_speed = animation_speed;
      return true;
    }
    void update_shared_animations(f32_t time_seconds) {
      for (std::uint32_t i = 0; i < objects.size(); ++i) { set_object_animation_frame(i, time_seconds); }
    }
    void flush_transform_updates() {
      if (!tlas_dirty || !ctx || !ready) { return; }
      if (scene_geometry_dirty) {
        rebuild_scene_geometry();
        scene_geometry_dirty = false;
        tlas_dirty = false;
        return;
      }
      if (!update_tlas_transforms()) { rebuild_tlas(); }
      tlas_dirty = false;
      tlas_rebuild_dirty = false;
    }
    void flush_transform_updates(VkCommandBuffer cmd) {
      if (!tlas_dirty || !ctx || !ready) { return; }
      if (scene_geometry_dirty) { return; }
      if (can_update_tlas_transforms()) {
        if (tlas_rebuild_dirty) { record_tlas_rebuild(cmd); }
        else { record_tlas_transform_update(cmd); }
        tlas_dirty = false;
        tlas_rebuild_dirty = false;
      }
    }
    void initialize_animated_model_pose(animated_model_t& animated_model) {
      fan::model::fms_t& fms = *animated_model.fms;
      if (!fms.root_bone || fms.bone_count == 0) { return; }
      if (fms.bone_transforms.size() != fms.bone_count) { fms.bone_transforms.resize(fms.bone_count, fan::mat4(1)); }
      if (!fms.animation_list.empty() && !fms.active_anim.empty()) {
        auto found = fms.animation_list.find(fms.active_anim);
        if (found != fms.animation_list.end()) {
          for (auto& animation : fms.animation_list) { animation.second.weight = 0; }
          found->second.weight = 1.f;
          fms.dt = 0;
          fms.fk_calculate_poses();
          fms.bone_transforms = fms.fk_calculate_transformations();
        }
      }
      else {
        fms.update_bone_transforms();
      }
      if (fms.bone_transforms.size() < animated_model.bone_count) { fms.bone_transforms.resize(animated_model.bone_count, fan::mat4(1)); }
      std::copy_n(fms.bone_transforms.begin(), animated_model.bone_count, bone_matrices.begin() + animated_model.first_bone);
      bone_buffer_dirty = true;
      animated_model.dirty = true;
      animation_vertices_dirty = true;
    }
    std::string make_animation_cache_key(const scene_model_t& model) const {
      return make_model_cache_key(model) + std::string("#animated#") + std::to_string((int)animation_sample_rate);
    }
    f32_t get_cached_animation_sample_time(f32_t duration_ms, std::uint32_t frame, std::uint32_t frame_count) const {
      if (duration_ms <= 0 || frame_count == 0) { return 0; }
      f32_t t = duration_ms * (((f32_t)frame + 0.5f) / (f32_t)frame_count);
      return std::min(t, std::max(0.f, duration_ms - 0.001f));
    }
    void append_pose_frame(fan::model::fms_t& fms, animation_clip_cache_t& clip, f32_t time_ms) {
      std::uint32_t first_bone = (std::uint32_t)bone_matrices.size();
      std::uint32_t bone_count = fms.bone_count;
      if (bone_count != 0) { bone_matrices.resize(first_bone + bone_count, fan::mat4(1)); }
      if (fms.bone_transforms.size() != fms.bone_count) { fms.bone_transforms.resize(fms.bone_count, fan::mat4(1)); }
      fms.dt = time_ms;
      fms.fk_calculate_poses();
      std::vector<fan::mat4> transforms = fms.fk_calculate_transformations();
      if (transforms.size() < bone_count) { transforms.resize(bone_count, fan::mat4(1)); }
      std::copy_n(transforms.begin(), bone_count, bone_matrices.begin() + first_bone);
      model_cache_entry_t range = load_model_from_fms(fms, first_bone, bone_count, true);
      animated_models.emplace_back();
      animated_model_t& animated_model = animated_models.back();
      animated_model.first_model = range.first_model;
      animated_model.model_count = range.model_count;
      animated_model.first_bone = first_bone;
      animated_model.bone_count = bone_count;
      animated_model.dirty = false;
      clip.frames.push_back({range.first_model, range.model_count, time_ms});
    }
    std::uint32_t get_or_create_animation_cache(const scene_model_t& model, bool& created) {
      std::string key = make_animation_cache_key(model);
      auto found_cache = animation_cache.find(key);
      if (found_cache != animation_cache.end()) { created = false; return found_cache->second; }
      created = true;
      fan::model::fms_t::properties_t properties;
      properties.path = model.path;
      properties.texture_path = model.texture_path;
      properties.fix_uv_diagonals = model.fix_uv_diagonals;
      fan::model::fms_t fms(properties, model.callers_path);
      std::uint32_t cache_index = (std::uint32_t)animation_caches.size();
      animation_cache[key] = cache_index;
      animation_cache_entry_t& cache = animation_caches.emplace_back();
      cache.key = key;
      cache.default_clip = fms.active_anim;
      cache.bone_count = fms.bone_count;
      cache.sample_rate = animation_sample_rate;
      for (auto& [clip_name, animation] : fms.animation_list) {
        cache.clip_indices[clip_name] = (std::uint32_t)cache.clips.size();
        animation_clip_cache_t& clip = cache.clips.emplace_back();
        clip.name = clip_name;
        clip.duration_ms = animation.duration > 0 ? animation.duration : 1000.f;
        for (auto& other : fms.animation_list) { other.second.weight = 0; }
        animation.weight = 1.f;
        fms.active_anim = clip_name;
        std::uint32_t frame_count = std::max<std::uint32_t>(1, (std::uint32_t)std::ceil((clip.duration_ms / 1000.f) * animation_sample_rate));
        for (std::uint32_t frame = 0; frame < frame_count; ++frame) {
          f32_t time_ms = get_cached_animation_sample_time(clip.duration_ms, frame, frame_count);
          append_pose_frame(fms, clip, time_ms);
        }
      }
      if (cache.clips.empty()) {
        animation_clip_cache_t& clip = cache.clips.emplace_back();
        clip.name = "bind";
        clip.duration_ms = 1000.f;
        cache.default_clip = clip.name;
        cache.clip_indices[clip.name] = 0;
        fms.update_bone_transforms();
        append_pose_frame(fms, clip, 0.f);
      }
      return cache_index;
    }
    bool load_animated_scene_model(std::uint32_t object_index) {
      scene_model_t& model = scene_models[object_index];
      object_t& object = objects[object_index];
      object.first_instance = (std::uint32_t)instances.size();
      bool created = false;
      object.animation_cache_index = get_or_create_animation_cache(model, created);
      animation_cache_entry_t& cache = animation_caches[object.animation_cache_index];
      object.animation_name = model.animation_name.empty() ? cache.default_clip : model.animation_name;
      object.animation_time_offset = model.animation_time_offset;
      object.animation_speed = model.animation_speed;
      object.animation_frame = std::uint32_t(-1);
      auto clip_found = cache.clip_indices.find(object.animation_name);
      if (clip_found == cache.clip_indices.end()) {
        object.animation_name = cache.default_clip;
        clip_found = cache.clip_indices.find(object.animation_name);
      }
      if (clip_found == cache.clip_indices.end() || cache.clips[clip_found->second].frames.empty()) { return created; }
      const animation_frame_t& frame = cache.clips[clip_found->second].frames.front();
      for (std::uint32_t i = 0; i < frame.model_count; ++i) { add_instance(frame.first_model + i, model.transform); }
      object.instance_count = (std::uint32_t)instances.size() - object.first_instance;
      apply_object_ray_mask(object_index);
      object.animation_frame = 0;
      return created;
    }
    void add_internal_empty_scene_mesh() {
      voxel_mesh_input_t mesh;
      mesh.vertices.resize(3);
      mesh.indices = {0, 1, 2};
      mesh.vertices[0].position = fan::vec3(-0.001f, 0.f, -0.001f);
      mesh.vertices[1].position = fan::vec3( 0.001f, 0.f, -0.001f);
      mesh.vertices[2].position = fan::vec3( 0.000f, 0.f,  0.001f);
      for (auto& v : mesh.vertices) {
        v.normal = fan::vec3(0.f, 1.f, 0.f);
        v.texcoord = fan::vec2(0.f);
        v.color = pack_vertex_color(fan::vec3(0.f));
      }
      mesh.base_color = fan::vec3(0.f);
      mesh.transform = fan::translate(fan::vec3(0.f, -1000000.f, 0.f));
      scene_model_t scene_model {};
      scene_model.transform = mesh.transform;
      scene_model.procedural_mesh_index = (std::uint32_t)procedural_meshes.size();
      procedural_meshes.push_back(std::move(mesh));
      scene_models.push_back(scene_model);
      objects.push_back({.generation = object_generation_counter++});
    }
    bool load_scene_model(std::uint32_t object_index) {
      scene_model_t& model = scene_models[object_index];
      if (model.procedural_mesh_index != std::uint32_t(-1)) {
        if (model.procedural_mesh_index >= procedural_meshes.size()) { fan::throw_error("ray tracing procedural mesh index is invalid"); }
        return append_mesh_geometry(procedural_meshes[model.procedural_mesh_index], object_index);
      }
      if (model.path.empty()) { return false; }
      if (model.animated) { return load_animated_scene_model(object_index); }
      object_t& object = objects[object_index];
      object.first_instance = (std::uint32_t)instances.size();
      std::string key = make_model_cache_key(model);
      auto found = model_cache.find(key);
      if (found != model_cache.end()) {
        add_cached_model_instances(model, found->second);
        object.instance_count = (std::uint32_t)instances.size() - object.first_instance;
        apply_object_ray_mask(object_index);
        return false;
      }
      fan::model::fms_t::properties_t properties;
      properties.path = model.path;
      properties.texture_path = model.texture_path;
      properties.fix_uv_diagonals = model.fix_uv_diagonals;
      fan::model::fms_t fms(properties, model.callers_path);
      model_cache[key] = load_model_from_fms(fms);
      add_cached_model_instances(model, model_cache[key]);
      object.instance_count = (std::uint32_t)instances.size() - object.first_instance;
      apply_object_ray_mask(object_index);
      return true;
    }
    void destroy_buffer(fan::vulkan::context_t::buffer_t& buffer) {
      ctx->destroy_buffer(buffer);
    }
    void destroy_tlas_resources() {
      tlas.destroy(*ctx); tlas_capacity = 0;
      destroy_buffer(tlas_instance_buffer); tlas_instance_capacity = 0;
      if (tlas_instance_staging_buffer.mapped) {
        ctx->unmap_buffer(tlas_instance_staging_buffer);
      }
      destroy_buffer(tlas_instance_staging_buffer);
      destroy_buffer(tlas_scratch_buffer); tlas_scratch_capacity = 0; tlas_scratch_peak_size = 0;
      tlas_instance_count = 0;
      tlas_instance_staging_size = 0;
    }
    void destroy_scene_geometry_resources() {
      destroy_acceleration_structures_only();
      destroy_buffer(source_vertex_buffer); source_vertex_capacity = 0;
      destroy_buffer(vertex_buffer); vertex_capacity = 0;
      destroy_buffer(index_buffer); index_capacity = 0;
      destroy_buffer(material_buffer); material_capacity = 0;
      destroy_buffer(material_index_buffer); material_index_capacity = 0;
      if (bone_buffer.mapped) {
        ctx->unmap_buffer(bone_buffer);
      }
      destroy_buffer(bone_buffer);
    }
    void rebuild_tlas() {
      if (!ctx) { return; }
      vkDeviceWaitIdle(ctx->device);
      if (tlas.handle) {
        auto destroy_as = (PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(ctx->device, "vkDestroyAccelerationStructureKHR");
        destroy_as(ctx->device, tlas.handle, nullptr);
        tlas.handle = VK_NULL_HANDLE;
        tlas.device_address = 0;
      }
      create_tlas();
      update_tlas_descriptor();
      reset_accumulation();
      tlas_dirty = false;
      tlas_rebuild_dirty = false;
    }
    void update_tlas_descriptor() {
      if (!descriptor_set) { return; }
      VkWriteDescriptorSetAccelerationStructureKHR as_info {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
        .accelerationStructureCount = 1,
        .pAccelerationStructures = &tlas.handle
      };
      VkWriteDescriptorSet write {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = &as_info,
        .dstSet = descriptor_set,
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR
      };
      vkUpdateDescriptorSets(ctx->device, 1, &write, 0, nullptr);
    }
    void update_scene_buffers_descriptor() {
      if (!descriptor_set) { return; }
      VkDescriptorBufferInfo material_info {material_buffer, 0, VK_WHOLE_SIZE};
      VkDescriptorBufferInfo vertex_info {vertex_buffer, 0, VK_WHOLE_SIZE};
      VkDescriptorBufferInfo index_info {index_buffer, 0, VK_WHOLE_SIZE};
      VkDescriptorBufferInfo mat_idx_info {material_index_buffer, 0, VK_WHOLE_SIZE};
      VkWriteDescriptorSet writes[4] {};
      set_descriptor_buffer_write(writes[0], 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &material_info);
      set_descriptor_buffer_write(writes[1], 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &vertex_info);
      set_descriptor_buffer_write(writes[2], 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &index_info);
      set_descriptor_buffer_write(writes[3], 9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &mat_idx_info);
      vkUpdateDescriptorSets(ctx->device, 4, writes, 0, nullptr);
    }
    void rebuild_scene_geometry() {
      if (!ctx) { return; }
      vkDeviceWaitIdle(ctx->device);

      if (has_gpu_only_models()) {
        destroy_acceleration_structures_only();
        create_blas_for_models();
        create_tlas();
        update_tlas_descriptor();
        update_scene_buffers_descriptor();
        update_rt_textures_descriptor();
        update_skinning_descriptor();
        reset_accumulation();
        tlas_dirty = false;
        tlas_rebuild_dirty = false;
        return;
      }

      destroy_scene_geometry_resources();
      create_source_vertex_buffer();
      create_vertex_buffer();
      create_index_buffer();
      create_material_buffer();
      create_material_index_buffer();
      create_bone_buffer();
      create_blas_for_models();
      create_tlas();
      update_tlas_descriptor();
      update_scene_buffers_descriptor();
      update_rt_textures_descriptor();
      update_skinning_descriptor();
      reset_accumulation();
      tlas_dirty = false;
      tlas_rebuild_dirty = false;
    }
    void set_light(const fan::vec3& position, const fan::vec3& color, f32_t intensity) {
      light_position = position;
      light_color = color;
      light_intensity = intensity;
      reset_accumulation();
      if (!ctx || !light_buffer || !light_buffer.mapped) { return; }
      light_ubo_t ubo {};
      ubo.position = light_position;
      ubo.color = light_color;
      ubo.intensity = light_intensity;
      std::memcpy(light_buffer.mapped, &ubo, sizeof(ubo));
    }
    void set_light() { set_light(light_position, light_color, light_intensity); }
    void reload_pipeline() {
      if (!ctx) { return; }
      vkDeviceWaitIdle(ctx->device);
      if (pipeline) { vkDestroyPipeline(ctx->device, pipeline, nullptr); }
      pipeline = VK_NULL_HANDLE;
      destroy_buffer(shader_binding_table);
      create_pipeline();
      create_sbt();
      reset_accumulation();
    }
    bool update_camera_from_engine() {
      auto camera_handle = fan::graphics::get_perspective_render_view().camera;
      auto camera_data = ctx->camera_get(camera_handle);
      if (!camera_buffer.mapped) { return false; }
      rt_camera_t vp {};
      vp.projection = camera_data.projection;
      vp.view = camera_data.view;
      vp.inv_projection = camera_data.projection.inverse();
      vp.inv_view = camera_data.view.inverse();
      vp.ray = fan::vec4(camera_data.znear, camera_data.zfar, 0.f, 0.f);
      bool changed = !last_camera_ubo_valid || std::memcmp(&last_camera_ubo, &vp, sizeof(vp)) != 0;
      std::memcpy(camera_buffer.mapped, &vp, sizeof(vp));
      last_camera_ubo = vp;
      last_camera_ubo_valid = true;
      return changed;
    }
    void open(fan::vulkan::context_t& main_ctx, const fan::vec2ui& sz) {
      ctx = &main_ctx;
      size = sz;
      load_functions();
      ctx->create_buffer(sizeof(rt_camera_t) * 16, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, camera_buffer);
      fan::vulkan::validate(ctx->map_buffer(camera_buffer, &camera_buffer.mapped));
      ctx->create_buffer(sizeof(time_ubo_t), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, time_buffer);
      fan::vulkan::validate(ctx->map_buffer(time_buffer, &time_buffer.mapped));
      ctx->create_buffer(sizeof(light_ubo_t), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, light_buffer);
      fan::vulkan::validate(ctx->map_buffer(light_buffer, &light_buffer.mapped));
      set_light();
      create_exposure_ubo();
      create_luminance_buffer(size.x, size.y);
      create_pick_buffer();
      if (scene_models.empty()) { add_internal_empty_scene_mesh(); }
      for (std::uint32_t i = 0; i < scene_models.size(); ++i) { load_scene_model(i); }
      create_source_vertex_buffer();
      create_vertex_buffer();
      create_index_buffer();
      create_material_buffer();
      create_material_index_buffer();
      create_bone_buffer();
      create_blas_for_models();
      create_tlas();
      create_output_image();
      create_accum_image();
      create_pipeline();
      create_sbt();
      create_descriptor_set();
      create_accum_pipeline();
      create_accum_descriptor_set();
      create_luminance_pipeline();
      create_luminance_descriptor_set();
      create_skinning_pipeline();
      create_skinning_descriptor_set();
      update_camera_from_engine();
      update_rt_textures_descriptor();
      reset_accumulation();
      tlas_dirty = false;
      tlas_rebuild_dirty = false;
    }
    void open(fan::graphics::engine_t& engine, const engine_open_properties_t& properties = {}) {
      attached_engine = &engine;
      auto output_size = properties.size;
      if (output_size.x == 0 || output_size.y == 0) { output_size = engine.window.get_size(); }
      output_sprite_enabled = properties.create_output_sprite;
      open(engine.context.vk, output_size);
      ready = true;
      sync_output_sprite();
      attach_engine_callbacks(engine);
    }
    void detach_engine() {
      if (attached_engine && update_callback_registered) { attached_engine->remove_update_callback(update_callback_handle); }
      update_callback_registered = false;
      resize_handle.remove();
      if (pre_begin_callback_registered && registered_vk_context && pre_begin_cmd_cb_index < registered_vk_context->pre_begin_cmd_cb.size()) { registered_vk_context->pre_begin_cmd_cb[pre_begin_cmd_cb_index] = []() {}; }
      pre_begin_callback_registered = false;
      if (command_callback_registered && registered_vk_context && begin_cmd_cb_index < registered_vk_context->begin_cmd_cb.size()) { registered_vk_context->begin_cmd_cb[begin_cmd_cb_index] = [](VkCommandBuffer) {}; }
      command_callback_registered = false;
      registered_vk_context = nullptr;
      attached_engine = nullptr;
      pending_resize = false;
      ready = false;
      if (output_sprite) { output_sprite.erase(); }
    }
    void update() {
      if (!attached_engine) { return; }
      if (pending_resize) {
        pending_resize = false;
        vkDeviceWaitIdle(attached_engine->context.vk.device);
        close();
        open(attached_engine->context.vk, pending_size);
        ready = true;
        sync_output_sprite();
      }
      if (!ready) { return; }
      bool update_animation_state = update_animations && (update_camera || !pause_animations_with_camera);
      if (update_animation_state) { update_shared_animations(attached_engine->start_time.seconds()); }
      bool camera_changed = update_camera ? update_camera_from_engine() : false;
      on_camera_updated(camera_changed);
      sync_output_sprite();
    }
  #if defined(FAN_GUI)
    bool get_object_world_bounds(std::uint32_t object_index, fan::vec3& min, fan::vec3& max) const {
      if (object_index >= objects.size()) { return false; }
      const object_t& object = objects[object_index];
      bool found_bounds = false;
      for (std::uint32_t i = 0; i < object.instance_count; ++i) {
        std::uint32_t instance_index = object.first_instance + i;
        if (instance_index >= instances.size()) { continue; }
        const instance_t& instance = instances[instance_index];
        if (instance.model_index >= models.size()) { continue; }
        const model_t& model = models[instance.model_index];
        if (!model.has_bounds) { continue; }
        fan::math::d3::aabb_t world_bounds = fan::math::d3::transform_aabb({model.aabb_min, model.aabb_max}, instance.transform);
        if (!found_bounds) {
          min = world_bounds.min;
          max = world_bounds.max;
          found_bounds = true;
        }
        else {
          min = min.min(world_bounds.min);
          max = max.max(world_bounds.max);
        }
      }
      return found_bounds;
    }
    object_handle_t pick_object_gizmo() const {
      if (!attached_engine || !show_object_gizmo) { return {}; }
      auto& rv = attached_engine->perspective_render_view;
      auto& camera = attached_engine->camera_get(rv.camera);
      fan::ray3_t ray = attached_engine->convert_mouse_to_ray(attached_engine->camera_get_position(rv.camera), camera.projection, camera.view);
      f32_t closest_t = std::numeric_limits<f32_t>::max();
      object_handle_t closest {};
      for (std::uint32_t i = 0; i < objects.size(); ++i) {
        fan::vec3 min;
        fan::vec3 max;
        if (!get_object_world_bounds(i, min, max)) { continue; }
        f32_t hit_t = 0.f;
        if (!fan::math::d3::ray_intersects_aabb(ray, min, max, hit_t)) { continue; }
        if (hit_t < closest_t) {
          closest_t = hit_t;
          closest = {i, objects[i].generation};
        }
      }
      return closest;
    }
    void update_object_gizmo_hotkeys() {
      if (!attached_engine || light_gizmo_gui_blocks_pick || !attached_engine->window.is_cursor_enabled()) { return; }
      if (attached_engine->is_key_clicked(fan::key_1)) { object_gizmo_mode = fan::graphics::gui::gizmo::transform_mode::translate; }
      if (attached_engine->is_key_clicked(fan::key_2)) { object_gizmo_mode = fan::graphics::gui::gizmo::transform_mode::rotate; }
      if (attached_engine->is_key_clicked(fan::key_3)) { object_gizmo_mode = fan::graphics::gui::gizmo::transform_mode::scale; }
    }
    bool mouse_hits_light_gizmo(f32_t radius) {
      auto& rv = attached_engine->perspective_render_view;
      auto& camera = attached_engine->camera_get(rv.camera);
      fan::ray3_t ray = attached_engine->convert_mouse_to_ray(attached_engine->camera_get_position(rv.camera), camera.projection, camera.view);
      return fan::math::d3::ray_intersects_sphere(ray, light_position, radius);
    }
    void render_light_gizmo() {
      if (!attached_engine || (!show_light_gizmo && !show_object_gizmo)) { return; }
      auto& rv = attached_engine->perspective_render_view;
      auto& camera = attached_engine->camera_get(rv.camera);
      fan::vec2 viewport_position = attached_engine->viewport_get_position(rv.viewport);
      fan::vec2 viewport_size = attached_engine->viewport_get_size(rv.viewport);
      if (viewport_size.x <= 0 || viewport_size.y <= 0) { return; }
      fan::mat4 light_transform = fan::translate(light_position);
      fan::graphics::gui::gizmo::set_orthographic(false);
      fan::graphics::gui::gizmo::set_drawlist();
      fan::graphics::gui::gizmo::set_rect(viewport_position, viewport_size);
      if (show_light_gizmo && light_gizmo_selected) {
        if (fan::graphics::gui::gizmo::manipulate(camera.view, camera.projection, fan::graphics::gui::gizmo::operation::translate, fan::graphics::gui::gizmo::mode::world, light_transform)) {
          set_light(light_transform.get_translation(), light_color, light_intensity);
        }
      }
      if (show_object_gizmo && is_object_valid(selected_object)) {
        fan::mat4 object_transform = scene_models[selected_object.index].transform;
        if (fan::graphics::gui::gizmo::manipulate(camera.view, camera.projection, fan::graphics::gui::gizmo::operation_from_transform_mode(object_gizmo_mode), fan::graphics::gui::gizmo::mode::world, object_transform)) {
          set_transform(selected_object, object_transform, false);
        }
      }
      fan::vec2 mouse_position = attached_engine->get_mouse_position();
      bool can_select = attached_engine->window.is_cursor_enabled() && attached_engine->is_mouse_clicked() && attached_engine->inside(rv.viewport, mouse_position) && !light_gizmo_gui_blocks_pick && !fan::graphics::gui::gizmo::is_using_any();
      if (can_select) {
        bool click_is_on_visible_gizmo = ((show_light_gizmo && light_gizmo_selected) || (show_object_gizmo && is_object_valid(selected_object))) && fan::graphics::gui::gizmo::is_over();
        if (!click_is_on_visible_gizmo) {
          f32_t pick_radius = std::max(light_indicator_radius, 1.f);
          bool hit_light = show_light_gizmo && mouse_hits_light_gizmo(pick_radius);
          light_gizmo_selected = hit_light;
          selected_object = hit_light ? object_handle_t {} : pick_object_gizmo();
        }
      }
    }
    memory_debug_t memory_debug() const {
      memory_debug_t out;
      constexpr f32_t mb = 1024.f * 1024.f;
      auto to_mb = [](std::size_t bytes) -> f32_t { return (f32_t)bytes / (1024.f * 1024.f); };

      out.model_count = models.size();
      out.instance_count = instances.size();
      out.texture_count = texture_ids.size();

      std::size_t blas_bytes = 0;
      for (const auto& blas : blas_list) {
        if (blas.handle != VK_NULL_HANDLE) {
          ++out.blas_count;
          blas_bytes += (std::size_t)blas.buffer.size;
        }
      }

      std::size_t scene_buffer_bytes = 0;
      scene_buffer_bytes += (std::size_t)source_vertex_buffer.size;
      scene_buffer_bytes += (std::size_t)vertex_buffer.size;
      scene_buffer_bytes += (std::size_t)index_buffer.size;
      scene_buffer_bytes += (std::size_t)material_buffer.size;
      scene_buffer_bytes += (std::size_t)material_index_buffer.size;
      out.rt_gpu_source_buffer_mb = to_mb((std::size_t)source_vertex_buffer.size);
      out.rt_gpu_vertex_buffer_mb = to_mb((std::size_t)vertex_buffer.size);
      out.rt_gpu_index_buffer_mb = to_mb((std::size_t)index_buffer.size);

      std::uint64_t active_vertex_capacity = 0;
      std::uint64_t active_index_capacity = 0;
      for (const model_t& model : models) {
        active_vertex_capacity += model.vertex_capacity;
        active_index_capacity += model.index_capacity;
      }
      std::uint64_t free_vertex_capacity = scene_range_free_count(vertex_free_ranges);
      std::uint64_t free_index_capacity = scene_range_free_count(index_free_ranges);
      std::size_t vertex_used_bytes = (std::size_t)active_vertex_capacity * sizeof(vertex_t);
      std::size_t index_used_bytes = (std::size_t)active_index_capacity * sizeof(std::uint32_t);
      std::size_t vertex_free_bytes = (std::size_t)free_vertex_capacity * sizeof(vertex_t);
      std::size_t index_free_bytes = (std::size_t)free_index_capacity * sizeof(std::uint32_t);
      std::size_t global_buffer_bytes = (std::size_t)vertex_buffer.size + (std::size_t)index_buffer.size;
      std::size_t global_used_bytes = vertex_used_bytes + index_used_bytes;
      out.rt_global_vertex_used_mb = to_mb(vertex_used_bytes);
      out.rt_global_vertex_free_mb = to_mb(vertex_free_bytes);
      out.rt_global_index_used_mb = to_mb(index_used_bytes);
      out.rt_global_index_free_mb = to_mb(index_free_bytes);
      out.rt_global_vertex_free_ranges = vertex_free_ranges.size();
      out.rt_global_index_free_ranges = index_free_ranges.size();
      out.rt_global_wasted_mb = global_buffer_bytes > global_used_bytes ? to_mb(global_buffer_bytes - global_used_bytes) : 0.f;

      scene_buffer_bytes += (std::size_t)bone_buffer.size;
      scene_buffer_bytes += (std::size_t)tlas_instance_buffer.size;
      scene_buffer_bytes += (std::size_t)tlas_instance_staging_buffer.size;

      std::size_t cpu_cached_bytes = 0;
      std::size_t cpu_source_vertex_bytes = source_vertex_data.capacity() * sizeof(source_vertex_t);
      std::size_t cpu_vertex_bytes = vertex_data.capacity() * sizeof(vertex_t);
      std::size_t cpu_index_bytes = index_data.capacity() * sizeof(std::uint32_t);
      std::size_t cpu_material_bytes = materials.capacity() * sizeof(material_info_t);
      cpu_cached_bytes += cpu_source_vertex_bytes;
      cpu_cached_bytes += cpu_vertex_bytes;
      cpu_cached_bytes += cpu_index_bytes;
      cpu_cached_bytes += cpu_material_bytes;
      cpu_cached_bytes += material_indices_per_primitive.capacity() * sizeof(std::uint32_t);
      cpu_cached_bytes += bone_matrices.capacity() * sizeof(fan::mat4);
      cpu_cached_bytes += models.capacity() * sizeof(model_t);
      cpu_cached_bytes += instances.capacity() * sizeof(instance_t);
      cpu_cached_bytes += scene_models.capacity() * sizeof(scene_model_t);
      cpu_cached_bytes += objects.capacity() * sizeof(object_t);
      for (const auto& mesh : procedural_meshes) {
        cpu_cached_bytes += mesh.vertices.capacity() * sizeof(vertex_t);
        cpu_cached_bytes += mesh.indices.capacity() * sizeof(std::uint32_t);
        cpu_cached_bytes += mesh.materials.capacity() * sizeof(material_info_t);
        cpu_cached_bytes += mesh.primitive_material_indices.capacity() * sizeof(std::uint32_t);
      }
      for (const auto& job : pending_mesh_uploads) {
        for (const auto& copy : job.copies) {
          cpu_cached_bytes += (std::size_t)copy.staging.size;
        }
      }
      for (const auto& retired : retired_mesh_uploads) {
        for (const auto& staging : retired.staging_buffers) {
          cpu_cached_bytes += (std::size_t)staging.size;
        }
      }

      std::size_t image_bytes = 0;
      auto add_image_allocation = [&](const auto& nr) {
        if (!ctx) { return; }
        auto& img = ctx->image_get(nr);
        if (img.image_allocation) {
          VmaAllocationInfo info {};
          vmaGetAllocationInfo(ctx->allocator, img.image_allocation, &info);
          image_bytes += (std::size_t)info.size;
        }
        if (img.staging_allocation) {
          VmaAllocationInfo info {};
          vmaGetAllocationInfo(ctx->allocator, img.staging_allocation, &info);
          image_bytes += (std::size_t)info.size;
        }
      };
      for (auto tex_id : texture_ids) {
        add_image_allocation(tex_id);
      }
      if (output_image_valid) { add_image_allocation(output_image); }
      if (accum_image_valid) { add_image_allocation(accum_image); }

      out.rt_blas_mb = to_mb(blas_bytes);
      out.rt_tlas_mb = to_mb((std::size_t)tlas.buffer.size);
      out.rt_blas_scratch_mb = to_mb((std::size_t)blas_scratch_buffer.size);
      out.rt_blas_scratch_peak_mb = to_mb((std::size_t)blas_scratch_peak_size);
      out.rt_tlas_scratch_mb = to_mb((std::size_t)tlas_scratch_buffer.size);
      out.rt_tlas_scratch_peak_mb = to_mb((std::size_t)tlas_scratch_peak_size);
      out.rt_scene_buffers_mb = to_mb(scene_buffer_bytes);
      out.rt_cpu_cached_mb = to_mb(cpu_cached_bytes);
      out.rt_cpu_source_vertices_mb = to_mb(cpu_source_vertex_bytes);
      out.rt_cpu_vertices_mb = to_mb(cpu_vertex_bytes);
      out.rt_cpu_indices_mb = to_mb(cpu_index_bytes);
      out.rt_cpu_materials_mb = to_mb(cpu_material_bytes);
      out.textures_images_mb = to_mb(image_bytes);
      return out;
    }
    void render_gui(const char* window_name = "ray tracing") {
      fan::graphics::gui::checkbox("update camera", &update_camera);
      fan::graphics::gui::checkbox("update animations", &update_animations);
      fan::graphics::gui::checkbox("pause animations with camera", &pause_animations_with_camera);
      bool shading_changed = false;
      shading_changed |= fan::graphics::gui::checkbox("auto exposure", &enable_auto_exposure);
      shading_changed |= fan::graphics::gui::checkbox("gi bounce", &enable_gi);
      shading_changed |= fan::graphics::gui::checkbox("reflections", &enable_reflections);
      shading_changed |= fan::graphics::gui::checkbox("shadows", &enable_shadows);
      shading_changed |= fan::graphics::gui::drag("Ambient Strength", &ambient_strength, 0.01f, 0.f, 1.f);
      shading_changed |= fan::graphics::gui::drag("Shadow Strength", &shadow_strength, 0.01f, 0.f, 1.f);
      shading_changed |= fan::graphics::gui::drag("Wrap Strength", &wrap_strength, 0.01f, 0.f, 1.f);
      shading_changed |= fan::graphics::gui::checkbox("light indicator", &show_light_indicator);
      shading_changed |= fan::graphics::gui::drag("Light Indicator Radius", &light_indicator_radius, 0.1f, 0.1f, 100.f);
      shading_changed |= fan::graphics::gui::checkbox("light gizmo", &show_light_gizmo);
      fan::graphics::gui::checkbox("light selected", &light_gizmo_selected);
      fan::graphics::gui::checkbox("object gizmo", &show_object_gizmo);
      object_gizmo_mode = std::clamp(object_gizmo_mode, 0, fan::graphics::gui::gizmo::transform_mode::count - 1);
      fan::graphics::gui::combo("object gizmo mode", &object_gizmo_mode, fan::graphics::gui::gizmo::transform_mode_names, fan::graphics::gui::gizmo::transform_mode::count);
      fan::graphics::gui::text(std::format("selected object: {}", is_object_valid(selected_object) ? (int)selected_object.index : -1));
      if (fan::graphics::gui::button("clear object selection")) { selected_object = {}; }
      if (!show_object_gizmo) { selected_object = {}; }
      if (shading_changed) {
        write_exposure_ubo();
        reset_accumulation();
      }
      bool light_changed = false;
      light_changed |= fan::graphics::gui::drag("Light Position", &light_position);
      light_changed |= fan::graphics::gui::drag("Light Color", &light_color);
      light_changed |= fan::graphics::gui::drag("Light Intensity", &light_intensity);
      if (light_changed) { set_light(); }
      light_gizmo_gui_blocks_pick = fan::graphics::gui::is_window_hovered(fan::graphics::gui::hovered_flags_child_windows | fan::graphics::gui::hovered_flags_allow_when_blocked_by_popup | fan::graphics::gui::hovered_flags_allow_when_blocked_by_active_item) || fan::graphics::gui::is_any_item_active();
      update_object_gizmo_hotkeys();
      render_light_gizmo();
    }
  #endif
    void attach_engine_callbacks(fan::graphics::engine_t& engine) {
      if (!registered_vk_context) { registered_vk_context = &engine.context.vk; }
      if (!pre_begin_callback_registered) {
        pre_begin_cmd_cb_index = registered_vk_context->pre_begin_cmd_cb.size();
        registered_vk_context->pre_begin_cmd_cb.push_back([this]() {
          gc_retired_mesh_uploads();
          upload_bone_buffer_if_dirty();
          if (tlas_dirty && !can_update_tlas_transforms()) { flush_transform_updates(); }
          if (attached_engine) { update_exposure(attached_engine->get_delta_time()); }
        });
        pre_begin_callback_registered = true;
      }
      if (!command_callback_registered) {
        begin_cmd_cb_index = registered_vk_context->begin_cmd_cb.size();
        registered_vk_context->begin_cmd_cb.push_back([this](VkCommandBuffer cmd) {
          if (ready) {
            record_gpu_animation_updates(cmd);
            record_pending_mesh_uploads(cmd);
            flush_transform_updates(cmd);
            record_trace_rays(cmd);
          }
        });
        command_callback_registered = true;
      }
      if (!update_callback_registered) {
        update_callback_handle = engine.add_update_callback_front([this](void*) { update(); });
        update_callback_registered = true;
      }
      resize_handle = engine.window.add_resize_callback([this](const auto& d) {
        pending_size = d.size;
        pending_resize = true;
        ready = false;
      });
    }
    void sync_output_sprite() {
      if (!attached_engine || !output_sprite_enabled || !accum_image_valid) { return; }
      fan::vec2 window_size = fan::vec2(attached_engine->window.get_size());
      if (!output_sprite) {
        output_sprite = fan::graphics::sprite_t {{
          .position = fan::vec3(window_size / 2.f, 0),
          .size = window_size / 2.f,
          .image = accum_image,
          .tc_position = fan::vec2(0.f, 1.f),
          .tc_size = fan::vec2(1.f, -1.f)
        }};
        return;
      }
      output_sprite.set_image(accum_image);
      output_sprite.set_position(fan::vec3(window_size / 2.f, 0));
      output_sprite.set_size(window_size / 2.f);
    }
    void record_gpu_animation_updates(VkCommandBuffer cmd) {
      if (!animation_vertices_dirty || !skinning_pipeline || !skinning_descriptor_set) { return; }
      if (bone_buffer) {
        VkBufferMemoryBarrier bone_barrier {
          .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
          .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
          .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .buffer = bone_buffer,
          .offset = 0,
          .size = VK_WHOLE_SIZE
        };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &bone_barrier, 0, nullptr);
      }
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, skinning_pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, skinning_pipeline_layout, 0, 1, &skinning_descriptor_set, 0, nullptr);
      bool dispatched = false;
      for (animated_model_t& animated_model : animated_models) {
        if (!animated_model.dirty) { continue; }
        for (std::uint32_t i = 0; i < animated_model.model_count; ++i) {
          std::uint32_t model_index = animated_model.first_model + i;
          const model_t& model = models[model_index];
          if (model.vertex_count == 0 || model.bone_count == 0) { continue; }
          skinning_push_constants_t pc {
            .first_vertex = model.first_vertex,
            .vertex_count = model.vertex_count,
            .first_bone = model.first_bone,
            .bone_count = model.bone_count
          };
          vkCmdPushConstants(cmd, skinning_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
          vkCmdDispatch(cmd, (model.vertex_count + 63) / 64, 1, 1);
          dispatched = true;
        }
      }
      if (!dispatched) {
        animation_vertices_dirty = false;
        for (animated_model_t& animated_model : animated_models) { animated_model.dirty = false; }
        return;
      }
      VkBufferMemoryBarrier skin_barrier {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_SHADER_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = vertex_buffer,
        .offset = 0,
        .size = VK_WHOLE_SIZE
      };
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 0, nullptr, 1, &skin_barrier, 0, nullptr);
      bool updated = false;
      for (animated_model_t& animated_model : animated_models) {
        if (!animated_model.dirty) { continue; }
        for (std::uint32_t i = 0; i < animated_model.model_count; ++i) {
          record_blas_update(cmd, animated_model.first_model + i);
          updated = true;
          VkMemoryBarrier build_barrier {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
            .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR
          };
          vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &build_barrier, 0, nullptr, 0, nullptr);
        }
        animated_model.dirty = false;
      }
      if (updated) {
        VkMemoryBarrier trace_barrier {
          .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
          .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
          .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR
        };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 1, &trace_barrier, 0, nullptr, 0, nullptr);
      }
      animation_vertices_dirty = false;
    }
    void record_trace_rays(VkCommandBuffer cmd) {
      static auto start_time = std::chrono::steady_clock::now();
      time_ubo_t t {};
      t.time = std::chrono::duration<f32_t>(std::chrono::steady_clock::now() - start_time).count();
      t.frame_index = frame_index;
      std::memcpy(time_buffer.mapped, &t, sizeof(t));
      if (current_layout != VK_IMAGE_LAYOUT_GENERAL) {
        ctx->insert_image_barrier(cmd, ctx->image_get(output_image).image_index, current_layout, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR);
        current_layout = VK_IMAGE_LAYOUT_GENERAL;
      }
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
      VkDeviceAddress sbt_addr = get_buffer_address(shader_binding_table);
      VkStridedDeviceAddressRegionKHR rgen_region {.deviceAddress = sbt_addr + rgen_offset, .stride = group_stride, .size = group_stride};
      VkStridedDeviceAddressRegionKHR miss_region {.deviceAddress = sbt_addr + miss_offset, .stride = group_stride, .size = group_stride * 2};
      VkStridedDeviceAddressRegionKHR hit_region {.deviceAddress = sbt_addr + hit_offset, .stride = group_stride, .size = group_stride};
      VkStridedDeviceAddressRegionKHR callable_region {};
      vkCmdTraceRaysKHR(cmd, &rgen_region, &miss_region, &hit_region, &callable_region, size.x, size.y, 1);
      {
        ctx->insert_image_barrier(cmd, ctx->image_get(output_image).image_index, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        current_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      }
      if (enable_auto_exposure) { dispatch_luminance_compute(cmd, luminance_descriptor_set, size.x, size.y); }
      if (accum_layout != VK_IMAGE_LAYOUT_GENERAL) {
        ctx->insert_image_barrier(cmd, ctx->image_get(accum_image).image_index, accum_layout, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        accum_layout = VK_IMAGE_LAYOUT_GENERAL;
      }
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, accum_pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, accum_pipeline_layout, 0, 1, &accum_descriptor_set, 0, nullptr);
      vkCmdPushConstants(cmd, accum_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(std::uint32_t), &frame_index);
      vkCmdDispatch(cmd, (size.x + 7) / 8, (size.y + 7) / 8, 1);
      accumulation_reset_pending = false;
      {
        ctx->insert_image_barrier(cmd, ctx->image_get(accum_image).image_index, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        accum_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      }
    }
    void on_camera_updated(bool camera_moved) {
      if (camera_moved) { reset_accumulation(); return; }
      if (!accumulation_reset_pending) { frame_index++; }
    }
    void trace_rays_before_shapes() { record_trace_rays(ctx->command_buffers[ctx->current_frame]); }
    void create_accum_image() { accum_image = create_rt_image(accum_image_valid, accum_layout, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, true); }
    void create_compute_pipeline(const char* shader_path, const VkDescriptorSetLayoutBinding* bindings, std::uint32_t binding_count, std::uint32_t push_constant_size, VkDescriptorSetLayout& descriptor_layout_out, VkPipelineLayout& pipeline_layout_out, VkPipeline& pipeline_out) {
      std::vector<VkDescriptorBindingFlags> binding_flags(binding_count);
      for (auto& flags : binding_flags) {
        flags =
          VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
          VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT |
          VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
      }
      VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags_info {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .bindingCount = binding_count,
        .pBindingFlags = binding_flags.data()
      };
      VkDescriptorSetLayoutCreateInfo layout_info {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = &binding_flags_info,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        .bindingCount = binding_count,
        .pBindings = bindings
      };
      fan::vulkan::validate(vkCreateDescriptorSetLayout(ctx->device, &layout_info, nullptr, &descriptor_layout_out));
      VkPushConstantRange pcr {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = push_constant_size};
      VkPipelineLayoutCreateInfo pl_info {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &descriptor_layout_out, .pushConstantRangeCount = push_constant_size ? 1u : 0u, .pPushConstantRanges = push_constant_size ? &pcr : nullptr};
      fan::vulkan::validate(vkCreatePipelineLayout(ctx->device, &pl_info, nullptr, &pipeline_layout_out));
      VkShaderModule comp = load_shader(shader_path, shaderc_glsl_compute_shader);
      VkPipelineShaderStageCreateInfo stage {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = comp, .pName = "main"};
      VkComputePipelineCreateInfo pi {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .stage = stage, .layout = pipeline_layout_out};
      fan::vulkan::validate(vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &pi, nullptr, &pipeline_out));
      vkDestroyShaderModule(ctx->device, comp, nullptr);
    }
    void create_skinning_pipeline() {
      VkDescriptorSetLayoutBinding bindings[] = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
      };
      create_compute_pipeline("shaders/vulkan/ray_tracing/skin.comp", bindings, (std::uint32_t)std::size(bindings), sizeof(skinning_push_constants_t), skinning_descriptor_layout, skinning_pipeline_layout, skinning_pipeline);
    }
    void update_skinning_descriptor() {
      if (!skinning_descriptor_set || !source_vertex_buffer || !vertex_buffer || !bone_buffer) { return; }
      VkDescriptorBufferInfo source_info {source_vertex_buffer, 0, VK_WHOLE_SIZE};
      VkDescriptorBufferInfo vertex_info {vertex_buffer, 0, VK_WHOLE_SIZE};
      VkDescriptorBufferInfo bone_info {bone_buffer, 0, VK_WHOLE_SIZE};
      VkWriteDescriptorSet writes[] = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, skinning_descriptor_set, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &source_info, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, skinning_descriptor_set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &vertex_info, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, skinning_descriptor_set, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bone_info, nullptr }
      };
      vkUpdateDescriptorSets(ctx->device, 3, writes, 0, nullptr);
    }
    void create_skinning_descriptor_set() {
      VkDescriptorPoolSize pool_size {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 3};
      VkDescriptorPoolCreateInfo pool_info {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT, .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &pool_size};
      fan::vulkan::validate(vkCreateDescriptorPool(ctx->device, &pool_info, nullptr, &skinning_descriptor_pool));
      VkDescriptorSetAllocateInfo alloc_info {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = skinning_descriptor_pool, .descriptorSetCount = 1, .pSetLayouts = &skinning_descriptor_layout};
      fan::vulkan::validate(vkAllocateDescriptorSets(ctx->device, &alloc_info, &skinning_descriptor_set));
      update_skinning_descriptor();
    }
    void create_accum_pipeline() {
      VkDescriptorSetLayoutBinding bindings[] = {
        { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
      };
      create_compute_pipeline("shaders/vulkan/ray_tracing/accumulate.comp", bindings, (std::uint32_t)std::size(bindings), sizeof(std::uint32_t), accum_descriptor_layout, accum_pipeline_layout, accum_pipeline);
    }
    void create_accum_descriptor_set() {
      VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 }
      };
      VkDescriptorPoolCreateInfo pool_info {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT, .maxSets = 1, .poolSizeCount = 2, .pPoolSizes = pool_sizes};
      fan::vulkan::validate(vkCreateDescriptorPool(ctx->device, &pool_info, nullptr, &accum_descriptor_pool));
      VkDescriptorSetAllocateInfo alloc_info {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = accum_descriptor_pool, .descriptorSetCount = 1, .pSetLayouts = &accum_descriptor_layout};
      fan::vulkan::validate(vkAllocateDescriptorSets(ctx->device, &alloc_info, &accum_descriptor_set));
      auto& out_img = ctx->image_get(output_image);
      VkDescriptorImageInfo current_info {.sampler = out_img.sampler, .imageView = out_img.image_view, .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
      auto& acc_img = ctx->image_get(accum_image);
      VkDescriptorImageInfo accum_info {.sampler = VK_NULL_HANDLE, .imageView = acc_img.image_view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
      VkWriteDescriptorSet writes[] = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, accum_descriptor_set, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &current_info, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, accum_descriptor_set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &accum_info, nullptr, nullptr }
      };
      vkUpdateDescriptorSets(ctx->device, 2, writes, 0, nullptr);
    }
    void create_luminance_pipeline() {
      VkDescriptorSetLayoutBinding bindings[] = {
        { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
      };
      create_compute_pipeline("shaders/vulkan/ray_tracing/luminance_reduce.comp", bindings, (std::uint32_t)std::size(bindings), sizeof(int) * 4, luminance_descriptor_layout, luminance_pipeline_layout, luminance_pipeline);
    }
    void create_luminance_descriptor_set() {
      VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 }
      };
      VkDescriptorPoolCreateInfo pool_info {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT, .maxSets = 1, .poolSizeCount = 2, .pPoolSizes = pool_sizes};
      fan::vulkan::validate(vkCreateDescriptorPool(ctx->device, &pool_info, nullptr, &luminance_descriptor_pool));
      VkDescriptorSetAllocateInfo alloc_info {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = luminance_descriptor_pool, .descriptorSetCount = 1, .pSetLayouts = &luminance_descriptor_layout};
      fan::vulkan::validate(vkAllocateDescriptorSets(ctx->device, &alloc_info, &luminance_descriptor_set));
      auto& hdr_img = ctx->image_get(output_image);
      VkDescriptorImageInfo hdr_info {.sampler = hdr_img.sampler, .imageView = hdr_img.image_view, .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
      VkDescriptorBufferInfo lum_info {.buffer = luminance_buffer, .offset = 0, .range = VK_WHOLE_SIZE};
      VkWriteDescriptorSet writes[] = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, luminance_descriptor_set, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &hdr_info, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, luminance_descriptor_set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lum_info, nullptr }
      };
      vkUpdateDescriptorSets(ctx->device, 2, writes, 0, nullptr);
    }
    void create_luminance_buffer(std::uint32_t width, std::uint32_t height) {
      luminance_group_x = (width + 15) / 16;
      luminance_group_y = (height + 15) / 16;
      luminance_group_count = luminance_group_x * luminance_group_y;
      luminance_ready = false;
      VkDeviceSize size = sizeof(f32_t) * luminance_group_count;
      ctx->create_buffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, luminance_buffer);
      fan::vulkan::validate(ctx->map_buffer(luminance_buffer, &luminance_buffer.mapped));
    }
    void dispatch_luminance_compute(VkCommandBuffer cmd, VkDescriptorSet luminance_set, std::uint32_t width, std::uint32_t height) {
      if (luminance_group_x == 0 || luminance_group_y == 0) { return; }
      vkCmdFillBuffer(cmd, luminance_buffer, 0, sizeof(f32_t) * luminance_group_count, 0);
      VkBufferMemoryBarrier clearBarrier {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .buffer = luminance_buffer,
        .offset = 0,
        .size = VK_WHOLE_SIZE
      };
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &clearBarrier, 0, nullptr);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, luminance_pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, luminance_pipeline_layout, 0, 1, &luminance_set, 0, nullptr);
      struct PC { int w, h, gx, gy; } pc {(int)width, (int)height, (int)luminance_group_x, (int)luminance_group_y};
      vkCmdPushConstants(cmd, luminance_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
      vkCmdDispatch(cmd, luminance_group_x, luminance_group_y, 1);
      luminance_ready = true;
      VkBufferMemoryBarrier read_barrier {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = luminance_buffer,
        .offset = 0,
        .size = sizeof(f32_t) * luminance_group_count
      };
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &read_barrier, 0, nullptr);
    }
    void write_exposure_ubo() {
      if (!ctx || !exposure_ubo || !exposure_ubo.mapped) { return; }
      exposure_ubo_t ubo {
        .exposure = exposure,
        .enable_gi = enable_gi ? 1.f : 0.f,
        .enable_reflections = enable_reflections ? 1.f : 0.f,
        .enable_shadows = enable_shadows ? 1.f : 0.f,
        .ambient_strength = std::clamp(ambient_strength, 0.f, 1.f),
        .shadow_strength = std::clamp(shadow_strength, 0.f, 1.f),
        .wrap_strength = std::clamp(wrap_strength, 0.f, 1.f),
        .show_light_indicator = show_light_indicator ? 1.f : 0.f,
        .light_indicator_radius = std::max(light_indicator_radius, 0.1f)
      };
      std::memcpy(exposure_ubo.mapped, &ubo, sizeof(ubo));
    }
    void update_exposure(f32_t dt) {
      if (!ctx || !enable_auto_exposure || !luminance_buffer || luminance_group_count == 0 || !luminance_buffer.mapped) {
        luminance_ready = false;
        write_exposure_ubo();
        return;
      }
      if (!luminance_ready) {
        write_exposure_ubo();
        return;
      }
      std::vector<f32_t> partial(luminance_group_count);
      std::memcpy(partial.data(), luminance_buffer.mapped, sizeof(f32_t) * luminance_group_count);
      double total = 0.0;
      for (f32_t v : partial) { total += v; }
      double avgL = total / double(size.x * size.y);
      if (avgL < 1e-6) { avgL = 1e-6; }
      f32_t middle_gray = 0.06f;
      target_exposure = middle_gray / f32_t(avgL);
      f32_t lambda = 1.0f - std::exp(-adaptation_speed * dt);
      exposure = exposure + lambda * (target_exposure - exposure);
      exposure = std::clamp(exposure, 0.0001f, 20.0f);
      write_exposure_ubo();
    }
    void create_pick_buffer() {
      ctx->create_buffer(sizeof(pick_result_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, pick_buffer);
      fan::vulkan::validate(ctx->map_buffer(pick_buffer, &pick_buffer.mapped));
      std::memset(pick_buffer.mapped, 0, sizeof(pick_result_t));
    }
    bool get_pick_result(pick_result_t& out) {
      if (!ctx || !pick_buffer || !pick_buffer.mapped) { return false; }
      ctx->invalidate_buffer(pick_buffer, 0, sizeof(pick_result_t));
      std::memcpy(&out, pick_buffer.mapped, sizeof(out));
      return out.position.w > 0.5f;
    }
    void create_exposure_ubo() {
      ctx->create_buffer(sizeof(exposure_ubo_t), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, exposure_ubo);
      fan::vulkan::validate(ctx->map_buffer(exposure_ubo, &exposure_ubo.mapped));
      write_exposure_ubo();
    }
    void close() {
      if (!ctx) { return; }
      ready = false;
      vkDeviceWaitIdle(ctx->device);
      for (auto& geom : mesh_geometries) { geom.destroy(*ctx); }
      destroy_scene_geometry_resources();
      auto destroy_pipe = [&](auto& p, auto& l, auto& d, auto& pool) {
        if (p) { vkDestroyPipeline(ctx->device, p, nullptr); }
        if (l) { vkDestroyPipelineLayout(ctx->device, l, nullptr); }
        if (d) { vkDestroyDescriptorSetLayout(ctx->device, d, nullptr); }
        if (pool) { vkDestroyDescriptorPool(ctx->device, pool, nullptr); }
        p = VK_NULL_HANDLE; l = VK_NULL_HANDLE; d = VK_NULL_HANDLE; pool = VK_NULL_HANDLE;
      };
      destroy_pipe(pipeline, pipeline_layout, descriptor_layout, descriptor_pool);
      destroy_pipe(accum_pipeline, accum_pipeline_layout, accum_descriptor_layout, accum_descriptor_pool);
      destroy_pipe(luminance_pipeline, luminance_pipeline_layout, luminance_descriptor_layout, luminance_descriptor_pool);
      destroy_pipe(skinning_pipeline, skinning_pipeline_layout, skinning_descriptor_layout, skinning_descriptor_pool);
      skinning_descriptor_set = VK_NULL_HANDLE;

      if (camera_buffer.mapped) { ctx->unmap_buffer(camera_buffer); }
      if (time_buffer.mapped) { ctx->unmap_buffer(time_buffer); }
      if (luminance_buffer.mapped) { ctx->unmap_buffer(luminance_buffer); }
      if (pick_buffer.mapped) { ctx->unmap_buffer(pick_buffer); }
      if (exposure_ubo.mapped) { ctx->unmap_buffer(exposure_ubo); }
      if (light_buffer.mapped) { ctx->unmap_buffer(light_buffer); }

      destroy_buffer(shader_binding_table);
      destroy_buffer(sbt_staging);
      destroy_buffer(camera_buffer);
      destroy_buffer(time_buffer);
      destroy_buffer(luminance_buffer);
      destroy_buffer(pick_buffer);
      destroy_buffer(exposure_ubo);
      destroy_buffer(light_buffer);

      last_camera_ubo_valid = false;

      for (auto tex_id : texture_ids) { ctx->image_erase(tex_id); }
      if (output_image_valid) { ctx->image_erase(output_image); }
      if (accum_image_valid) { ctx->image_erase(accum_image); }

      blas_list.clear();
      model_cache.clear();
      mesh_geometries.clear();
      submeshes.clear();
      models.clear();
      instances.clear();
      animated_models.clear();
      for (auto& object : objects) {
        object.first_instance = 0; object.instance_count = 0;
        object.animated_model_index = std::uint32_t(-1); object.animation_cache_index = std::uint32_t(-1);
        object.animation_name.clear(); object.animation_frame = std::uint32_t(-1);
      }
      animation_cache.clear();
      animation_caches.clear();
      rt_texture_cache.clear();
      materials.clear();
      scene_vertex_count = 0;
      scene_index_count = 0;
      scene_material_count = 0;
      scene_primitive_count = 0;
      vertex_free_ranges.clear();
      index_free_ranges.clear();
      primitive_free_ranges.clear();
      source_vertex_data.clear();
      vertex_data.clear();
      index_data.clear();
      bone_matrices.clear();
      texture_ids.clear();
      rt_texture_infos.clear();
      material_indices_per_primitive.clear();
      current_material_index = 0;
      vertex_offset = 0;
      tlas_instance_count = 0;
      reset_accumulation();
      tlas_dirty = false;
      tlas_rebuild_dirty = false;
      animation_vertices_dirty = false;
      bone_buffer_dirty = false;
      scene_geometry_dirty = false;
      selected_object = {};
      luminance_ready = false;
      output_image_valid = false;
      accum_image_valid = false;
      tlas_instance_staging_size = 0;
      luminance_group_count = 0;
      luminance_group_x = 0;
      luminance_group_y = 0;
      ctx = nullptr;
    }

    std::vector<submesh_t> submeshes;
    std::vector<model_t> models;
    std::vector<instance_t> instances;
    std::vector<scene_model_t> scene_models;
    std::vector<voxel_mesh_input_t> procedural_meshes;
    std::vector<object_t> objects;
    std::unordered_map<std::string, model_cache_entry_t> model_cache;
    std::unordered_map<std::string, std::uint32_t> animation_cache;
    std::vector<animation_cache_entry_t> animation_caches;

    static constexpr std::uint32_t instance_count = 1;
    static constexpr std::uint32_t max_textures = 512;
    f32_t animation_sample_rate = 12.f;
    engine_open_properties_t pending_open_properties;
    fan::vulkan::context_t* ctx = nullptr;
    fan::vec2ui size {};
    std::vector<acceleration_structure_t> blas_list;
    std::vector<shapes::gpu_mesh_t> mesh_geometries;
    std::vector<animated_model_t> animated_models;
    acceleration_structure_t tlas;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    fan::vulkan::context_t::buffer_t shader_binding_table;
    fan::vulkan::context_t::buffer_t sbt_staging;
    std::uint32_t rgen_offset = 0;
    std::uint32_t miss_offset = 0;
    std::uint32_t hit_offset = 0;
    std::uint32_t shadow_miss_offset = 0;
    std::uint32_t shadow_hit_offset = 0;
    std::uint32_t handle_size_aligned = 0;
    std::uint32_t group_stride = 0;
    fan::graphics::image_t accum_image;
    bool accum_image_valid = false;
    VkImageLayout accum_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkPipeline accum_pipeline = VK_NULL_HANDLE;
    VkPipelineLayout accum_pipeline_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout accum_descriptor_layout = VK_NULL_HANDLE;
    VkDescriptorPool accum_descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet accum_descriptor_set = VK_NULL_HANDLE;
    std::uint32_t frame_index = 0;
    bool accumulation_reset_pending = true;
    fan::graphics::image_t output_image;
    bool output_image_valid = false;
    VkImageLayout current_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    fan::vulkan::context_t::buffer_t camera_buffer;
    rt_camera_t last_camera_ubo {};
    bool last_camera_ubo_valid = false;
    fan::vulkan::context_t::buffer_t time_buffer;
    fan::vulkan::context_t::buffer_t tlas_instance_buffer;
    fan::vulkan::context_t::buffer_t tlas_instance_staging_buffer;
    VkDeviceSize tlas_instance_staging_size = 0;
    std::uint32_t tlas_instance_count = 0;
    fan::vulkan::context_t::buffer_t tlas_scratch_buffer;
    fan::vulkan::context_t::buffer_t blas_scratch_buffer;
    VkDeviceSize blas_scratch_size = 0;
    VkDeviceSize blas_scratch_peak_size = 0;
    VkDeviceSize tlas_scratch_peak_size = 0;
    fan::vulkan::context_t::buffer_t source_vertex_buffer;
    VkDeviceSize source_vertex_capacity = 0;
    fan::vulkan::context_t::buffer_t vertex_buffer;
    VkDeviceSize vertex_capacity = 0;
    fan::vulkan::context_t::buffer_t index_buffer;
    VkDeviceSize index_capacity = 0;
    fan::vulkan::context_t::buffer_t material_buffer;
    VkDeviceSize material_capacity = 0;
    fan::vulkan::context_t::buffer_t bone_buffer;
    std::vector<material_info_t> materials;
    std::vector<scene_range_t> vertex_free_ranges;
    std::vector<scene_range_t> index_free_ranges;
    std::vector<scene_range_t> primitive_free_ranges;
    VkDeviceSize scene_vertex_count = 0;
    VkDeviceSize scene_index_count = 0;
    VkDeviceSize scene_material_count = 0;
    VkDeviceSize scene_primitive_count = 0;
    std::vector<source_vertex_t> source_vertex_data;
    std::vector<vertex_t> vertex_data;
    std::vector<std::uint32_t> index_data;
    std::vector<fan::mat4> bone_matrices;
    std::vector<fan::graphics::image_nr_t> texture_ids;
    std::vector<VkDescriptorImageInfo> rt_texture_infos;
    std::unordered_map<std::string, std::int32_t> rt_texture_cache;
    fan::vulkan::context_t::buffer_t material_index_buffer;
    VkDeviceSize material_index_capacity = 0;
    VkDeviceSize tlas_instance_capacity = 0;
    VkDeviceSize tlas_capacity = 0;
    VkDeviceSize tlas_scratch_capacity = 0;
    std::vector<std::uint32_t> material_indices_per_primitive;
    std::size_t current_material_index = 0;
    std::uint32_t vertex_offset = 0;
    std::uint32_t object_generation_counter = 1;
    bool tlas_dirty = false;
    bool tlas_rebuild_dirty = false;
    bool scene_geometry_dirty = false;
    bool animation_vertices_dirty = false;
    bool bone_buffer_dirty = false;
    f32_t exposure = 1.f;
    f32_t target_exposure = 1.f;
    f32_t adaptation_speed = 4.f;
    bool enable_auto_exposure = false;
    bool enable_gi = false;
    bool enable_reflections = false;
    bool enable_shadows = true;
    f32_t ambient_strength = 0.8f;
    f32_t shadow_strength = 0.8f;
    f32_t wrap_strength = 0.35f;
    bool show_light_indicator = true;
    f32_t light_indicator_radius = 6.f;
    bool show_light_gizmo = true;
    bool light_gizmo_selected = true;
    bool light_gizmo_gui_blocks_pick = false;
    bool show_object_gizmo = true;
    object_handle_t selected_object;
    int object_gizmo_mode = 0;
    bool update_camera = true;
    bool update_animations = true;
    bool pause_animations_with_camera = true;
    fan::vec3 light_position = fan::vec3(5.0f, 10.0f, 5.0f);
    fan::vec3 light_color = fan::vec3(1.0f, 1.0f, 1.0f);
    f32_t light_intensity = 3.0f;
    fan::graphics::sprite_t output_sprite;
    fan::graphics::engine_t* attached_engine = nullptr;
    bool output_sprite_enabled = false;
    bool ready = false;
    bool incremental_upload_batch = false;
    bool incremental_upload_had_changes = false;
    bool incremental_upload_needs_tlas_rebuild = false;
    bool pending_resize = false;
    fan::vec2ui pending_size {};
    fan::window_t::resize_handle_t resize_handle;
    fan::graphics::engine_t::update_callback_handle_t update_callback_handle;
    bool update_callback_registered = false;
    fan::vulkan::context_t* registered_vk_context = nullptr;
    std::size_t pre_begin_cmd_cb_index = 0;
    std::vector<pending_mesh_upload_t> pending_mesh_uploads;
    std::vector<retired_mesh_upload_t> retired_mesh_uploads;
    bool pre_begin_callback_registered = false;
    std::size_t begin_cmd_cb_index = 0;
    bool command_callback_registered = false;
    fan::vulkan::context_t::buffer_t luminance_buffer;
    fan::vulkan::context_t::buffer_t pick_buffer;
    std::uint32_t luminance_group_x = 0;
    std::uint32_t luminance_group_y = 0;
    std::uint32_t luminance_group_count = 0;
    bool luminance_ready = false;
    fan::vulkan::context_t::buffer_t exposure_ubo;
    VkDescriptorSetLayout luminance_descriptor_layout = VK_NULL_HANDLE;
    VkDescriptorPool luminance_descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet luminance_descriptor_set = VK_NULL_HANDLE;
    VkPipelineLayout luminance_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline luminance_pipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout skinning_descriptor_layout = VK_NULL_HANDLE;
    VkDescriptorPool skinning_descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet skinning_descriptor_set = VK_NULL_HANDLE;
    VkPipelineLayout skinning_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline skinning_pipeline = VK_NULL_HANDLE;
    fan::vulkan::context_t::buffer_t light_buffer;
    PFN_vkGetBufferDeviceAddressKHR vkGetBufferDeviceAddressKHR = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR = nullptr;
    PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR vkGetAccelerationStructureDeviceAddressKHR = nullptr;
    PFN_vkCreateRayTracingPipelinesKHR vkCreateRayTracingPipelinesKHR = nullptr;
    PFN_vkGetRayTracingShaderGroupHandlesKHR vkGetRayTracingShaderGroupHandlesKHR = nullptr;
    PFN_vkCmdTraceRaysKHR vkCmdTraceRaysKHR = nullptr;
  };
}

#endif