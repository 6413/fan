module;

#if defined(FAN_3D) && defined(FAN_VULKAN)
#include <vulkan/vulkan.h>
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
import fan.print.error;
export namespace fan::graphics::vulkan::ray_tracing {
  struct acceleration_structure_t {
    void destroy(fan::vulkan::context_t& ctx) {
      if (handle) {
        auto vkDestroyAS = (PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(ctx.device, "vkDestroyAccelerationStructureKHR");
        vkDestroyAS(ctx.device, handle, nullptr);
      }
      if (buffer) vkDestroyBuffer(ctx.device, buffer, nullptr);
      if (memory) vkFreeMemory(ctx.device, memory, nullptr);
      handle = VK_NULL_HANDLE;
      buffer = VK_NULL_HANDLE;
      memory = VK_NULL_HANDLE;
      device_address = 0;
    }
    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceAddress device_address = 0;
  };
  #pragma pack(push, 1)
  struct material_info_t {
    std::int32_t albedo_texture_id = -1;
    std::int32_t normal_texture_id = -1;
    std::int32_t metallic_texture_id = -1;
    std::int32_t roughness_texture_id = -1;
    fan::vec3 base_color = fan::vec3(1.0f, 1.0f, 1.0f);
    f32_t pad1;
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
      std::uint32_t first_bone = 0;
      std::uint32_t bone_count = 0;
      bool animated = false;
    };
    struct instance_t {
      std::uint32_t model_index;
      fan::mat4 transform;
    };
    struct object_handle_t {
      std::uint32_t index = std::uint32_t(-1);
      std::uint32_t generation = 0;
      bool valid() const {
        return index != std::uint32_t(-1);
      }
    };
    struct scene_model_t {
      std::string path;
      fan::mat4 transform = fan::mat4(1);
      std::string texture_path = "models/textures";
      std::source_location callers_path;
      bool fix_uv_diagonals = false;
      bool animated = false;
    };
    struct model_cache_entry_t {
      std::uint32_t first_model = 0;
      std::uint32_t model_count = 0;
    };
    struct object_t {
      std::uint32_t generation = 0;
      std::uint32_t first_instance = 0;
      std::uint32_t instance_count = 0;
      std::uint32_t animated_model_index = std::uint32_t(-1);
    };
    struct engine_open_properties_t {
      fan::vec2ui size{};
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
    std::vector<submesh_t> submeshes;
    std::vector<model_t> models;
    std::vector<instance_t> instances;
    std::vector<scene_model_t> scene_models;
    std::vector<object_t> objects;
    std::unordered_map<std::string, model_cache_entry_t> model_cache;
    context_t() = default;
    context_t(fan::graphics::engine_t& engine, const engine_open_properties_t& properties = {}) {
      attached_engine = &engine;
      pending_open_properties = properties;
      engine.single_queue.push_back([this, &engine]() {
        auto sz = pending_open_properties.size;
        if (sz.x == 0 || sz.y == 0) sz = engine.window.get_size();
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
    struct time_ubo_t {
      f32_t time = 0;
    };
    struct rt_camera_t {
      fan::mat4 projection;
      fan::mat4 view;
      fan::mat4 inv_projection;
      fan::mat4 inv_view;
    };
    struct exposure_ubo_t {
      f32_t exposure = 1.f;
      f32_t enable_gi = 0.f;
      f32_t enable_reflections = 0.f;
      f32_t pad0 = 0.f;
    };
    VkDeviceAddress get_buffer_address(VkBuffer buffer) const {
      VkBufferDeviceAddressInfoKHR info{};
      info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
      info.buffer = buffer;
      return vkGetBufferDeviceAddressKHR(ctx->device, &info);
    }
    void load_functions() {
      vkGetBufferDeviceAddressKHR = (PFN_vkGetBufferDeviceAddressKHR)vkGetDeviceProcAddr(ctx->device, "vkGetBufferDeviceAddressKHR");
      vkCmdBuildAccelerationStructuresKHR = (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(ctx->device, "vkCmdBuildAccelerationStructuresKHR");
      vkGetAccelerationStructureBuildSizesKHR = (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(ctx->device, "vkGetAccelerationStructureBuildSizesKHR");
      vkCreateAccelerationStructureKHR = (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(ctx->device, "vkCreateAccelerationStructureKHR");
      vkGetAccelerationStructureDeviceAddressKHR = (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(ctx->device, "vkGetAccelerationStructureDeviceAddressKHR");
      vkCreateRayTracingPipelinesKHR = (PFN_vkCreateRayTracingPipelinesKHR)vkGetDeviceProcAddr(ctx->device, "vkCreateRayTracingPipelinesKHR");
      vkGetRayTracingShaderGroupHandlesKHR = (PFN_vkGetRayTracingShaderGroupHandlesKHR)vkGetDeviceProcAddr(ctx->device, "vkGetRayTracingShaderGroupHandlesKHR");
      vkCmdTraceRaysKHR = (PFN_vkCmdTraceRaysKHR)vkGetDeviceProcAddr(ctx->device, "vkCmdTraceRaysKHR");
    }
    void fill_blas_build_info(
      std::uint32_t model_index,
      VkAccelerationStructureGeometryTrianglesDataKHR& triangles,
      VkAccelerationStructureGeometryKHR& geometry,
      VkAccelerationStructureBuildGeometryInfoKHR& build_info
    ) const {
      const model_t& model = models[model_index];
      triangles.sType       = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
      triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
      triangles.vertexData.deviceAddress = get_buffer_address(vertex_buffer);
      triangles.vertexStride             = sizeof(vertex_t);
      triangles.maxVertex                = model.vertex_count ? model.first_vertex + model.vertex_count - 1 : 0;
      triangles.indexType                = VK_INDEX_TYPE_UINT32;
      triangles.indexData.deviceAddress = get_buffer_address(index_buffer);
      geometry.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
      geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
      geometry.geometry.triangles = triangles;
      geometry.flags        = 0;
      build_info.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
      build_info.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
      build_info.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
      build_info.geometryCount = 1;
      build_info.pGeometries   = &geometry;
    }
    void create_blas_for_models() {
      if (models.empty()) return;
      std::vector<VkAccelerationStructureBuildSizesInfoKHR> sizes(models.size());
      std::vector<std::uint32_t> primitive_counts(models.size());
      VkDeviceSize scratch_size = 0;
      blas_list.resize(models.size());
      for (std::uint32_t i = 0; i < models.size(); ++i) {
        const model_t& model = models[i];
        primitive_counts[i] = model.index_count / 3;
        VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
        VkAccelerationStructureGeometryKHR geometry{};
        VkAccelerationStructureBuildGeometryInfoKHR build_info{};
        fill_blas_build_info(i, triangles, geometry, build_info);
        sizes[i].sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        vkGetAccelerationStructureBuildSizesKHR(ctx->device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build_info, &primitive_counts[i], &sizes[i]);
        scratch_size = std::max(scratch_size, std::max(sizes[i].buildScratchSize, sizes[i].updateScratchSize));
        acceleration_structure_t& blas = blas_list[i];
        ctx->create_buffer(sizes[i].accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, blas.buffer, blas.memory);
        VkAccelerationStructureCreateInfoKHR create_info{};
        create_info.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        create_info.buffer = blas.buffer;
        create_info.size   = sizes[i].accelerationStructureSize;
        create_info.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        fan::vulkan::validate(vkCreateAccelerationStructureKHR(ctx->device, &create_info, nullptr, &blas.handle));
      }
      destroy_buffer(blas_scratch_buffer, blas_scratch_memory);
      blas_scratch_size = scratch_size;
      ctx->create_buffer(blas_scratch_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, blas_scratch_buffer, blas_scratch_memory);
      VkDeviceAddress scratch_address = get_buffer_address(blas_scratch_buffer);
      VkCommandBuffer cmd = ctx->begin_single_time_commands();
      for (std::uint32_t i = 0; i < models.size(); ++i) {
        const model_t& model = models[i];
        VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
        VkAccelerationStructureGeometryKHR geometry{};
        VkAccelerationStructureBuildGeometryInfoKHR build_info{};
        fill_blas_build_info(i, triangles, geometry, build_info);
        build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        build_info.dstAccelerationStructure = blas_list[i].handle;
        build_info.scratchData.deviceAddress = scratch_address;
        VkAccelerationStructureBuildRangeInfoKHR range_info{};
        range_info.primitiveCount  = primitive_counts[i];
        range_info.primitiveOffset = model.first_index * sizeof(std::uint32_t);
        range_info.firstVertex     = 0;
        range_info.transformOffset = 0;
        const VkAccelerationStructureBuildRangeInfoKHR* range_infos = &range_info;
        vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build_info, &range_infos);
        if (i + 1 < models.size()) {
          VkMemoryBarrier barrier{};
          barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
          barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
          barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
          vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        }
      }
      ctx->end_single_time_commands(cmd);

      for (auto& blas : blas_list) {
        VkAccelerationStructureDeviceAddressInfoKHR addr_info{};
        addr_info.sType                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        addr_info.accelerationStructure = blas.handle;
        blas.device_address             = vkGetAccelerationStructureDeviceAddressKHR(ctx->device, &addr_info);
      }
    }
    void record_blas_update(VkCommandBuffer cmd, std::uint32_t model_index) {
      if (model_index >= models.size() || model_index >= blas_list.size() || !blas_scratch_buffer) return;
      const model_t& model = models[model_index];
      VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
      VkAccelerationStructureGeometryKHR geometry{};
      VkAccelerationStructureBuildGeometryInfoKHR build_info{};
      fill_blas_build_info(model_index, triangles, geometry, build_info);
      build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
      build_info.srcAccelerationStructure = blas_list[model_index].handle;
      build_info.dstAccelerationStructure = blas_list[model_index].handle;
      build_info.scratchData.deviceAddress = get_buffer_address(blas_scratch_buffer);
      VkAccelerationStructureBuildRangeInfoKHR range_info{};
      range_info.primitiveCount = model.index_count / 3;
      range_info.primitiveOffset = model.first_index * sizeof(std::uint32_t);
      const VkAccelerationStructureBuildRangeInfoKHR* range_infos = &range_info;
      vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build_info, &range_infos);
    }
    std::vector<VkAccelerationStructureInstanceKHR> make_tlas_instances() const {
      std::uint32_t instance_count = (std::uint32_t)instances.size();
      std::vector<VkAccelerationStructureInstanceKHR> vk_instances(instance_count);
      for (std::uint32_t i = 0; i < instance_count; ++i) {
        const instance_t& inst = instances[i];
        const model_t& model = models[inst.model_index];
        VkTransformMatrixKHR t{};
        fan::mat4 m = inst.transform;
        for (std::uint32_t row = 0; row < 3; ++row) {
          for (std::uint32_t column = 0; column < 4; ++column) {
            t.matrix[row][column] = m[column][row];
          }
        }
        vk_instances[i].transform = t;
        vk_instances[i].instanceCustomIndex = model.first_primitive;
        vk_instances[i].mask = 0xFF;
        vk_instances[i].instanceShaderBindingTableRecordOffset = 0;
        vk_instances[i].flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        vk_instances[i].accelerationStructureReference = blas_list[inst.model_index].device_address;
      }
      return vk_instances;
    }
    void ensure_tlas_instance_staging(VkDeviceSize size) {
      if (tlas_instance_staging_buffer && tlas_instance_staging_size >= size) return;
      destroy_buffer(tlas_instance_staging_buffer, tlas_instance_staging_memory);
      ctx->create_buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, tlas_instance_staging_buffer, tlas_instance_staging_memory);
      vkMapMemory(ctx->device, tlas_instance_staging_memory, 0, size, 0, &tlas_instance_staging_mapped);
      tlas_instance_staging_size = size;
    }
    void upload_tlas_instances_to_staging(const std::vector<VkAccelerationStructureInstanceKHR>& vk_instances, VkDeviceSize size) {
      ensure_tlas_instance_staging(size);
      std::memcpy(tlas_instance_staging_mapped, vk_instances.data(), (std::size_t)size);
    }
    bool can_update_tlas_transforms() const {
      return ctx && tlas.handle && tlas_instance_buffer && tlas_scratch_buffer && tlas_instance_count != 0 && tlas_instance_count == (std::uint32_t)instances.size();
    }
    void record_tlas_transform_update(VkCommandBuffer cmd) {
      std::vector<VkAccelerationStructureInstanceKHR> vk_instances = make_tlas_instances();
      VkDeviceSize instance_size = sizeof(VkAccelerationStructureInstanceKHR) * tlas_instance_count;
      upload_tlas_instances_to_staging(vk_instances, instance_size);

      VkAccelerationStructureGeometryInstancesDataKHR instances_data{};
      instances_data.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
      instances_data.arrayOfPointers = VK_FALSE;
      instances_data.data.deviceAddress = get_buffer_address(tlas_instance_buffer);
      VkAccelerationStructureGeometryKHR geometry{};
      geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
      geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
      geometry.geometry.instances = instances_data;
      VkAccelerationStructureBuildGeometryInfoKHR build_info{};
      build_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
      build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
      build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
      build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
      build_info.srcAccelerationStructure = tlas.handle;
      build_info.dstAccelerationStructure = tlas.handle;
      build_info.geometryCount = 1;
      build_info.pGeometries = &geometry;
      build_info.scratchData.deviceAddress = get_buffer_address(tlas_scratch_buffer);
      VkAccelerationStructureBuildRangeInfoKHR range_info{};
      range_info.primitiveCount = tlas_instance_count;
      const VkAccelerationStructureBuildRangeInfoKHR* range_infos = &range_info;

      VkBufferCopy copy_region{};
      copy_region.size = instance_size;
      vkCmdCopyBuffer(cmd, tlas_instance_staging_buffer, tlas_instance_buffer, 1, &copy_region);
      VkBufferMemoryBarrier instance_barrier{};
      instance_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
      instance_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      instance_barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
      instance_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      instance_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      instance_barrier.buffer = tlas_instance_buffer;
      instance_barrier.offset = 0;
      instance_barrier.size = instance_size;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 0, nullptr, 1, &instance_barrier, 0, nullptr);
      vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build_info, &range_infos);
      VkMemoryBarrier build_barrier{};
      build_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      build_barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
      build_barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 1, &build_barrier, 0, nullptr, 0, nullptr);
      frame_index = 0;
    }
    void create_tlas() {
      tlas_instance_count = (std::uint32_t)instances.size();
      if (tlas_instance_count == 0) return;
      std::vector<VkAccelerationStructureInstanceKHR> vk_instances = make_tlas_instances();
      VkDeviceSize instance_size = sizeof(VkAccelerationStructureInstanceKHR) * tlas_instance_count;
      upload_tlas_instances_to_staging(vk_instances, instance_size);
      ctx->create_buffer(instance_size, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, tlas_instance_buffer, tlas_instance_memory);
      ctx->copy_buffer(tlas_instance_staging_buffer, tlas_instance_buffer, instance_size);
      VkAccelerationStructureGeometryInstancesDataKHR instances_data{};
      instances_data.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
      instances_data.arrayOfPointers = VK_FALSE;
      instances_data.data.deviceAddress = get_buffer_address(tlas_instance_buffer);
      VkAccelerationStructureGeometryKHR geometry{};
      geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
      geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
      geometry.geometry.instances = instances_data;
      geometry.flags = 0;
      VkAccelerationStructureBuildGeometryInfoKHR build_info{};
      build_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
      build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
      build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
      build_info.geometryCount = 1;
      build_info.pGeometries = &geometry;
      VkAccelerationStructureBuildSizesInfoKHR size_info{};
      size_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
      vkGetAccelerationStructureBuildSizesKHR(ctx->device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build_info, &tlas_instance_count, &size_info);
      ctx->create_buffer(size_info.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, tlas.buffer, tlas.memory);
      VkAccelerationStructureCreateInfoKHR create_info{};
      create_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
      create_info.buffer = tlas.buffer;
      create_info.size = size_info.accelerationStructureSize;
      create_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
      fan::vulkan::validate(vkCreateAccelerationStructureKHR(ctx->device, &create_info, nullptr, &tlas.handle));
      VkDeviceSize scratch_size = std::max(size_info.buildScratchSize, size_info.updateScratchSize);
      ctx->create_buffer(scratch_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, tlas_scratch_buffer, tlas_scratch_memory);
      build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
      build_info.dstAccelerationStructure = tlas.handle;
      build_info.scratchData.deviceAddress = get_buffer_address(tlas_scratch_buffer);
      VkAccelerationStructureBuildRangeInfoKHR range_info{};
      range_info.primitiveCount = tlas_instance_count;
      const VkAccelerationStructureBuildRangeInfoKHR* range_infos = &range_info;
      VkCommandBuffer cmd = ctx->begin_single_time_commands();
      vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build_info, &range_infos);
      ctx->end_single_time_commands(cmd);
      VkAccelerationStructureDeviceAddressInfoKHR addr_info{};
      addr_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
      addr_info.accelerationStructure = tlas.handle;
      tlas.device_address = vkGetAccelerationStructureDeviceAddressKHR(ctx->device, &addr_info);
    }
    bool update_tlas_transforms() {
      if (!can_update_tlas_transforms()) return false;
      VkCommandBuffer cmd = ctx->begin_single_time_commands();
      record_tlas_transform_update(cmd);
      ctx->end_single_time_commands(cmd);
      return true;
    }
    fan::graphics::image_t create_rt_image(bool& valid, VkImageLayout& tracked_layout, VkPipelineStageFlags dst_stage, bool clear) {
      fan::graphics::image_t image = ctx->image_create();
      valid = true;
      auto& img = ctx->image_get(image);
      VkImageCreateInfo image_info{};
      image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
      image_info.imageType = VK_IMAGE_TYPE_2D;
      image_info.format = VK_FORMAT_R16G16B16A16_SFLOAT;
      image_info.extent = { size.x, size.y, 1 };
      image_info.mipLevels = 1;
      image_info.arrayLayers = 1;
      image_info.samples = VK_SAMPLE_COUNT_1_BIT;
      image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
      image_info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      fan::vulkan::image_create(*ctx, size, image_info.format, VK_IMAGE_TILING_OPTIMAL, image_info.usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, img.image_index, img.image_memory);
      VkImageViewCreateInfo view_info{};
      view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      view_info.image = img.image_index;
      view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
      view_info.format = image_info.format;
      view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      view_info.subresourceRange.baseMipLevel = 0;
      view_info.subresourceRange.levelCount = 1;
      view_info.subresourceRange.baseArrayLayer = 0;
      view_info.subresourceRange.layerCount = 1;
      fan::vulkan::validate(vkCreateImageView(ctx->device, &view_info, nullptr, &img.image_view));
      ctx->create_texture_sampler(img.sampler, {});
      VkCommandBuffer cmd = ctx->begin_single_time_commands();
      ctx->insert_image_barrier(cmd, img.image_index, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, dst_stage);
      if (clear) {
        VkClearColorValue clear_color{};
        clear_color.float32[3] = 1.0f;
        VkImageSubresourceRange range{};
        range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        range.baseMipLevel = 0;
        range.levelCount = 1;
        range.baseArrayLayer = 0;
        range.layerCount = 1;
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
      VkDescriptorSetLayoutBinding bindings[11]{};
      auto bnd = [&](int i, VkDescriptorType t, int c, VkShaderStageFlags s) { bindings[i] = { (std::uint32_t)i, t, (std::uint32_t)c, s, nullptr }; };
      bnd(0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
      bnd(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR);
      bnd(2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR);
      bnd(3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_RAYGEN_BIT_KHR);
      bnd(4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, max_textures, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
      bnd(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
      bnd(6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
      bnd(7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
      bnd(8, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
      bnd(9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
      bnd(10, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);

      VkDescriptorSetLayoutCreateInfo layout_info{};
      layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
      layout_info.bindingCount = std::size(bindings);
      layout_info.pBindings = bindings;
      fan::vulkan::validate(vkCreateDescriptorSetLayout(ctx->device, &layout_info, nullptr, &descriptor_layout));

      VkPipelineLayoutCreateInfo pipeline_layout_info{};
      pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
      pipeline_layout_info.setLayoutCount = 1;
      pipeline_layout_info.pSetLayouts = &descriptor_layout;
      fan::vulkan::validate(vkCreatePipelineLayout(ctx->device, &pipeline_layout_info, nullptr, &pipeline_layout));

      VkShaderModule rgen   = load_shader("shaders/vulkan/ray_tracing/raygen.rgen", shaderc_glsl_raygen_shader);
      VkShaderModule miss   = load_shader("shaders/vulkan/ray_tracing/miss.rmiss", shaderc_glsl_miss_shader);
      VkShaderModule chit   = load_shader("shaders/vulkan/ray_tracing/closesthit.rchit", shaderc_glsl_closesthit_shader);
      VkShaderModule shadow = load_shader("shaders/vulkan/ray_tracing/shadow.rmiss", shaderc_glsl_miss_shader);
      VkShaderModule shany  = load_shader("shaders/vulkan/ray_tracing/shadow_anyhit.rahit", shaderc_glsl_anyhit_shader);

      VkPipelineShaderStageCreateInfo stages[5]{};
      auto stg = [&](int i, VkShaderStageFlagBits s, VkShaderModule m) {
        stages[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[i].stage = s;
        stages[i].module = m;
        stages[i].pName = "main";
      };
      stg(0, VK_SHADER_STAGE_RAYGEN_BIT_KHR, rgen);
      stg(1, VK_SHADER_STAGE_MISS_BIT_KHR, miss);
      stg(2, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, chit);
      stg(3, VK_SHADER_STAGE_MISS_BIT_KHR, shadow);
      stg(4, VK_SHADER_STAGE_ANY_HIT_BIT_KHR, shany);

      VkRayTracingShaderGroupCreateInfoKHR groups[4]{};
      auto grp = [&](int i, VkRayTracingShaderGroupTypeKHR t, std::uint32_t g, std::uint32_t c, std::uint32_t a) {
        groups[i].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        groups[i].type = t;
        groups[i].generalShader = g;
        groups[i].closestHitShader = c;
        groups[i].anyHitShader = a;
        groups[i].intersectionShader = VK_SHADER_UNUSED_KHR;
      };
      grp(0, VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 0, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR);
      grp(1, VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 1, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR);
      grp(2, VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR, VK_SHADER_UNUSED_KHR, 2, 4);
      grp(3, VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 3, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR);

      VkRayTracingPipelineCreateInfoKHR pipeline_info{};
      pipeline_info.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
      pipeline_info.stageCount = 5;
      pipeline_info.pStages = stages;
      pipeline_info.groupCount = 4;
      pipeline_info.pGroups = groups;
      pipeline_info.maxPipelineRayRecursionDepth = 3;
      pipeline_info.layout = pipeline_layout;
      fan::vulkan::validate(vkCreateRayTracingPipelinesKHR(ctx->device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline));

      vkDestroyShaderModule(ctx->device, rgen, nullptr);
      vkDestroyShaderModule(ctx->device, miss, nullptr);
      vkDestroyShaderModule(ctx->device, chit, nullptr);
      vkDestroyShaderModule(ctx->device, shadow, nullptr);
      vkDestroyShaderModule(ctx->device, shany, nullptr);
    }
    void create_sbt(){
      VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt_props{};
      rt_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
      VkPhysicalDeviceProperties2 props{};
      props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
      props.pNext = &rt_props;
      vkGetPhysicalDeviceProperties2(ctx->physical_device, &props);
      const std::uint32_t handle_size  = rt_props.shaderGroupHandleSize;
      const std::uint32_t handle_align = rt_props.shaderGroupHandleAlignment;
      const std::uint32_t base_align   = rt_props.shaderGroupBaseAlignment;
      handle_size_aligned = (handle_size + handle_align - 1) & ~(handle_align - 1);
      const std::uint32_t group_count      = 4;
      const std::uint32_t aligned_group_sz = (handle_size_aligned + base_align - 1) & ~(base_align - 1);
      const std::uint32_t sbt_size         = group_count * aligned_group_sz;
      std::vector<std::uint8_t> handles(group_count * handle_size);
      fan::vulkan::validate(vkGetRayTracingShaderGroupHandlesKHR(ctx->device, pipeline, 0, group_count, group_count * handle_size, handles.data()));
      ctx->create_buffer(sbt_size, VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, shader_binding_table, sbt_memory);
      ctx->create_buffer(sbt_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, sbt_staging, sbt_staging_memory);
      void* data;
      vkMapMemory(ctx->device, sbt_staging_memory, 0, sbt_size, 0, &data);
      std::uint8_t* sbt_ptr = static_cast<std::uint8_t*>(data);
      std::memset(sbt_ptr, 0, sbt_size);
      group_stride = aligned_group_sz;
      rgen_offset        = 0 * aligned_group_sz;
      miss_offset        = 1 * aligned_group_sz;
      shadow_miss_offset = 2 * aligned_group_sz;
      hit_offset         = 3 * aligned_group_sz;
      std::memcpy(sbt_ptr + rgen_offset,        handles.data() + 0 * handle_size, handle_size);
      std::memcpy(sbt_ptr + miss_offset,        handles.data() + 1 * handle_size, handle_size);
      std::memcpy(sbt_ptr + shadow_miss_offset, handles.data() + 3 * handle_size, handle_size);
      std::memcpy(sbt_ptr + hit_offset,         handles.data() + 2 * handle_size, handle_size);
      vkUnmapMemory(ctx->device, sbt_staging_memory);
      ctx->copy_buffer(sbt_staging, shader_binding_table, sbt_size);
    }
    void set_descriptor_image_write(VkWriteDescriptorSet& write, std::uint32_t binding, VkDescriptorType type, const VkDescriptorImageInfo* info) const {
      write = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set, binding, 0, 1, type, info, nullptr, nullptr };
    }
    void set_descriptor_buffer_write(VkWriteDescriptorSet& write, std::uint32_t binding, VkDescriptorType type, const VkDescriptorBufferInfo* info) const {
      write = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set, binding, 0, 1, type, nullptr, info, nullptr };
    }
    void create_descriptor_set(){
      VkDescriptorPoolSize pool_sizes[5]{};
      pool_sizes[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
      pool_sizes[0].descriptorCount = 1;
      pool_sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      pool_sizes[1].descriptorCount = 1;
      pool_sizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      pool_sizes[2].descriptorCount = 4;
      pool_sizes[3].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      pool_sizes[3].descriptorCount = max_textures;
      pool_sizes[4].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      pool_sizes[4].descriptorCount = 5;

      VkDescriptorPoolCreateInfo pool_info{};
      pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
      pool_info.maxSets = 1;
      pool_info.poolSizeCount = 5;
      pool_info.pPoolSizes = pool_sizes;
      fan::vulkan::validate(vkCreateDescriptorPool(ctx->device, &pool_info, nullptr, &descriptor_pool));

      VkDescriptorSetAllocateInfo alloc_info{};
      alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
      alloc_info.descriptorPool = descriptor_pool;
      alloc_info.descriptorSetCount = 1;
      alloc_info.pSetLayouts = &descriptor_layout;
      fan::vulkan::validate(vkAllocateDescriptorSets(ctx->device, &alloc_info, &descriptor_set));

      VkDescriptorImageInfo image_info{};
      image_info.imageView = ctx->image_get(output_image).image_view;
      image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

      VkDescriptorBufferInfo cam_info{camera_buffer, 0, sizeof(rt_camera_t) * 16};
      VkDescriptorBufferInfo time_info{time_buffer, 0, sizeof(time_ubo_t)};
      VkDescriptorBufferInfo light_info{light_buffer, 0, sizeof(light_ubo_t)};
      VkDescriptorBufferInfo exposure_info{exposure_ubo, 0, sizeof(exposure_ubo_t)};

      VkWriteDescriptorSet writes[5]{};
      set_descriptor_image_write(writes[0], 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &image_info);
      set_descriptor_buffer_write(writes[1], 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &cam_info);
      set_descriptor_buffer_write(writes[2], 3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &time_info);
      set_descriptor_buffer_write(writes[3], 8, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &light_info);
      set_descriptor_buffer_write(writes[4], 10, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &exposure_info);
      vkUpdateDescriptorSets(ctx->device, 5, writes, 0, nullptr);

      update_tlas_descriptor();
      update_scene_buffers_descriptor();
      update_rt_textures_descriptor();
    }
    void update_rt_textures_descriptor() {
      std::vector<VkDescriptorImageInfo> infos(max_textures);
      auto& dummy = ctx->image_get(fan::graphics::ctx().default_texture);
      for (std::uint32_t i = 0; i < max_textures; i++) {
        if (i < rt_texture_infos.size()) infos[i] = rt_texture_infos[i];
        else {
          infos[i].sampler = dummy.sampler;
          infos[i].imageView = dummy.image_view;
          infos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
      }
      VkWriteDescriptorSet write{};
      write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      write.dstSet = descriptor_set;
      write.dstBinding = 4;
      write.descriptorCount = max_textures;
      write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      write.pImageInfo = infos.data();
      vkUpdateDescriptorSets(ctx->device, 1, &write, 0, nullptr);
    }
    void create_source_vertex_buffer() {
      ctx->upload_buffer(source_vertex_data, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, source_vertex_buffer, source_vertex_memory);
    }
    void create_vertex_buffer() {
      ctx->upload_buffer(vertex_data, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_TRANSFER_DST_BIT, vertex_buffer, vertex_memory);
    }
    void create_index_buffer() {
      ctx->upload_buffer(index_data, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_TRANSFER_DST_BIT, index_buffer, index_memory);
    }
    void create_material_buffer() {
      ctx->upload_buffer(materials, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, material_buffer, material_memory);
    }
    void create_material_index_buffer() {
      if (material_indices_per_primitive.empty()) return;
      if (material_index_buffer) {
        vkDestroyBuffer(ctx->device, material_index_buffer, nullptr);
        vkFreeMemory(ctx->device, material_index_memory, nullptr);
      }
      ctx->upload_buffer(material_indices_per_primitive, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, material_index_buffer, material_index_memory);
    }
    void create_bone_buffer() {
      destroy_buffer(bone_buffer, bone_memory);
      bone_mapped = nullptr;
      std::uint32_t count = std::max<std::uint32_t>((std::uint32_t)bone_matrices.size(), 1);
      VkDeviceSize size = sizeof(fan::mat4) * count;
      ctx->create_buffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, bone_buffer, bone_memory);
      vkMapMemory(ctx->device, bone_memory, 0, size, 0, &bone_mapped);
      if (bone_matrices.empty()) {
        fan::mat4 identity(1);
        std::memcpy(bone_mapped, &identity, sizeof(identity));
      }
      else {
        std::memcpy(bone_mapped, bone_matrices.data(), sizeof(fan::mat4) * bone_matrices.size());
      }
    }
    model_cache_entry_t load_model_from_fms(fan::model::fms_t& fms, std::uint32_t first_bone = 0, std::uint32_t bone_count = 0, bool animated = false) {
      std::uint32_t first_model = (std::uint32_t)models.size();
      std::uint32_t added_index_count = 0;
      for (const auto& mesh : fms.meshes) {
        added_index_count += (std::uint32_t)mesh.indices.size();
      }
      std::uint32_t needed_primitive_count = ((std::uint32_t)index_data.size() + added_index_count) / 3;
      if (material_indices_per_primitive.size() < needed_primitive_count) {
        material_indices_per_primitive.resize(needed_primitive_count);
      }
      for (std::uint32_t mesh_idx = 0; mesh_idx < fms.meshes.size(); mesh_idx++) {
        const auto& src_mesh = fms.meshes[mesh_idx];
        model_t model{};
        model.first_index = (std::uint32_t)index_data.size();
        model.index_count = (std::uint32_t)src_mesh.indices.size();
        model.first_vertex = (std::uint32_t)vertex_data.size();
        model.vertex_count = (std::uint32_t)src_mesh.vertices.size();
        model.first_bone = first_bone;
        model.bone_count = bone_count;
        model.animated = animated;
        std::uint32_t first_vertex = model.first_vertex;
        for (const auto& v : src_mesh.vertices) {
          vertex_t out{};
          out.position = v.position;
          out.normal   = v.normal;
          out.texcoord = v.uv;
          out.color = fan::vec3(v.color.x, v.color.y, v.color.z);
          source_vertex_t src{};
          src.position = out.position;
          src.normal = out.normal;
          src.texcoord = out.texcoord;
          src.color = out.color;
          src.bone_ids = v.bone_ids;
          src.bone_weights = v.bone_weights;
          for (int i = 0; i < 4; ++i) {
            if (src.bone_ids[i] >= 0) {
              src.bone_ids[i] += first_bone;
            }
          }
          source_vertex_data.push_back(src);
          vertex_data.push_back(out);
        }
        for (std::uint32_t idx : src_mesh.indices) {
          index_data.push_back(idx + first_vertex);
        }
        material_info_t mat;
        mat.base_color            = fan::vec3(1,1,1);
        if (mesh_idx < fms.material_data_vector.size()) {
          const auto& md = fms.material_data_vector[mesh_idx];
          const fan::color* c = &md.color[fan::texture_type::base_color];
          if ((*c)[0] == 1 && (*c)[1] == 1 && (*c)[2] == 1 && (*c)[3] == 1) {
            c = &md.color[fan::texture_type::diffuse];
          }
          mat.base_color = fan::vec3((*c)[0], (*c)[1], (*c)[2]);
        }
        auto load_rt_texture = [&](const std::string& name) -> std::int32_t {
          if (name.empty()) return -1;
          auto it = fan::model::cached_texture_data.find(name);
          if (it == fan::model::cached_texture_data.end()) return -1;
          const auto& td = it->second;
          if (!td.valid()) return -1;
          fan::image::info_t ii;
          ii.data     = (void*)td.data.get();
          ii.size     = td.size;
          ii.channels = td.channels;
          auto tex = ctx->image_load(ii);
          texture_ids.push_back(tex);
          auto& img = ctx->image_get(tex);
          VkDescriptorImageInfo di{};
          di.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
          di.imageView   = img.image_view;
          di.sampler     = img.sampler;
          rt_texture_infos.push_back(di);
          return (std::int32_t)rt_texture_infos.size() - 1;
        };
        auto load_first_rt_texture = [&](std::initializer_list<std::uint32_t> types) -> std::int32_t {
          for (auto type : types) {
            const std::string& tn = src_mesh.texture_names[type];
            if (tn.empty()) continue;
            std::int32_t slot = load_rt_texture(tn);
            if (slot >= 0) return slot;
          }
          return -1;
        };
        {
          std::int32_t slot = load_first_rt_texture({ fan::texture_type::base_color, fan::texture_type::diffuse });
          if (slot >= 0) mat.albedo_texture_id = slot;
        }
        {
          std::int32_t slot = load_first_rt_texture({ fan::texture_type::normals, fan::texture_type::normal_camera });
          if (slot >= 0) mat.normal_texture_id = slot;
        }
        {
          const std::string& tn = src_mesh.texture_names[fan::texture_type::metalness];
          if (!tn.empty()) {
            std::int32_t slot = load_rt_texture(tn);
            if (slot >= 0) mat.metallic_texture_id = slot;
          }
        }
        {
          const std::string& tn = src_mesh.texture_names[fan::texture_type::diffuse_roughness];
          if (!tn.empty()) {
            std::int32_t slot = load_rt_texture(tn);
            if (slot >= 0) mat.roughness_texture_id = slot;
          }
        }
        model.material_index = (std::uint32_t)materials.size();
        materials.push_back(mat);
        std::uint32_t mesh_first_index   = model.first_index;
        std::uint32_t mesh_index_count   = model.index_count;
        std::uint32_t mesh_primitive_cnt = mesh_index_count / 3;
        model.first_primitive       = mesh_first_index / 3;
        for (std::uint32_t p = 0; p < mesh_primitive_cnt; ++p) {
          material_indices_per_primitive[model.first_primitive + p] = model.material_index;
        }
        models.push_back(model);
      }
      return { first_model, (std::uint32_t)models.size() - first_model };
    }
    void add_instance(std::uint32_t model_index, const fan::mat4& transform) {
      instance_t inst{};
      inst.model_index = model_index;
      inst.transform = transform;
      instances.push_back(inst);
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
      for (std::uint32_t i = 0; i < entry.model_count; ++i) {
        add_instance(entry.first_model + i, model.transform);
      }
    }
    object_handle_t add_model(const scene_model_t& model, std::source_location callers_path = std::source_location::current()) {
      object_handle_t handle { (std::uint32_t)scene_models.size(), object_generation_counter++ };
      scene_models.push_back(model);
      scene_models.back().callers_path = callers_path;
      objects.push_back({ .generation = handle.generation, .first_instance = 0, .instance_count = 0 });
      if (!ctx || !ready) return handle;
      bool geometry_changed = load_scene_model(handle.index);
      scene_geometry_dirty = scene_geometry_dirty || geometry_changed;
      tlas_dirty = true;
      frame_index = 0;
      return handle;
    }
    object_handle_t add_model(
      const std::string& path, 
      const fan::mat4& transform = fan::mat4(1), 
      std::source_location callers_path = std::source_location::current()) 
    {
      scene_model_t m;
      m.path = path;
      m.transform = transform;
      return add_model(m, callers_path);
    }
    object_handle_t add_animated_model(
      const std::string& path,
      const fan::mat4& transform = fan::mat4(1),
      std::source_location callers_path = std::source_location::current())
    {
      scene_model_t m;
      m.path = path;
      m.transform = transform;
      m.animated = true;
      return add_model(m, callers_path);
    }
    void clear_scene_models() {
      scene_models.clear();
      objects.clear();
      if (!ctx || !ready) return;
      instances.clear();
      animated_models.clear();
      bone_matrices.clear();
      tlas_dirty = true;
      frame_index = 0;
    }
    bool is_object_valid(object_handle_t handle) const {
      return handle.valid() && handle.index < objects.size() && objects[handle.index].generation == handle.generation;
    }
    fan::mat4 get_transform(object_handle_t handle) const {
      if (!is_object_valid(handle)) fan::throw_error("invalid ray tracing object handle");
      return scene_models[handle.index].transform;
    }
    bool set_transform(object_handle_t handle, const fan::mat4& transform, bool rebuild_now = false) {
      if (!is_object_valid(handle)) return false;
      scene_models[handle.index].transform = transform;
      object_t& object = objects[handle.index];
      for (std::uint32_t i = 0; i < object.instance_count; ++i) {
        instances[object.first_instance + i].transform = transform;
      }
      tlas_dirty = true;
      frame_index = 0;
      if (rebuild_now) flush_transform_updates();
      return true;
    }
    bool set_transform_deferred(object_handle_t handle, const fan::mat4& transform) {
      return set_transform(handle, transform, false);
    }
    bool set_animation(object_handle_t handle, f32_t time_seconds, const std::string& animation_name = {}, f32_t weight = 1.f) {
      if (!is_object_valid(handle)) return false;
      object_t& object = objects[handle.index];
      if (object.animated_model_index == std::uint32_t(-1) || object.animated_model_index >= animated_models.size()) return false;
      animated_model_t& animated_model = animated_models[object.animated_model_index];
      fan::model::fms_t& fms = *animated_model.fms;
      if (!fms.root_bone || fms.bone_count == 0 || fms.animation_list.empty()) return false;
      std::string name = animation_name.empty() ? fms.active_anim : animation_name;
      auto found = fms.animation_list.find(name);
      if (found == fms.animation_list.end()) return false;
      for (auto& animation : fms.animation_list) {
        animation.second.weight = 0;
      }
      fms.active_anim = name;
      found->second.weight = weight;
      fms.dt = time_seconds * 1000.f;
      if (fms.bone_transforms.size() != fms.bone_count) {
        fms.bone_transforms.resize(fms.bone_count, fan::mat4(1));
      }
      fms.fk_calculate_poses();
      std::vector<fan::mat4> transforms = fms.fk_calculate_transformations();
      if (transforms.size() < animated_model.bone_count) {
        transforms.resize(animated_model.bone_count, fan::mat4(1));
      }
      std::copy_n(transforms.begin(), animated_model.bone_count, bone_matrices.begin() + animated_model.first_bone);
      if (bone_mapped) {
        std::memcpy(
          static_cast<std::byte*>(bone_mapped) + sizeof(fan::mat4) * animated_model.first_bone,
          transforms.data(),
          sizeof(fan::mat4) * animated_model.bone_count
        );
      }
      animated_model.dirty = true;
      animation_vertices_dirty = true;
      frame_index = 0;
      return true;
    }
    bool set_animation(object_handle_t handle, const std::string& animation_name = {}, f32_t weight = 1.f) {
      return set_animation(handle, attached_engine->start_time.seconds(), animation_name, weight);
    }
    void flush_transform_updates() {
      if (!tlas_dirty || !ctx || !ready) return;
      if (scene_geometry_dirty) {
        rebuild_scene_geometry();
        scene_geometry_dirty = false;
        tlas_dirty = false;
        return;
      }
      if (!update_tlas_transforms()) rebuild_tlas();
      tlas_dirty = false;
    }
    void flush_transform_updates(VkCommandBuffer cmd) {
      if (!tlas_dirty || !ctx || !ready) return;
      if (scene_geometry_dirty) return;
      if (can_update_tlas_transforms()) {
        record_tlas_transform_update(cmd);
        tlas_dirty = false;
      }
    }
    void initialize_animated_model_pose(animated_model_t& animated_model) {
      fan::model::fms_t& fms = *animated_model.fms;
      if (!fms.root_bone || fms.bone_count == 0) return;
      if (fms.bone_transforms.size() != fms.bone_count) {
        fms.bone_transforms.resize(fms.bone_count, fan::mat4(1));
      }
      if (!fms.animation_list.empty() && !fms.active_anim.empty()) {
        auto found = fms.animation_list.find(fms.active_anim);
        if (found != fms.animation_list.end()) {
          for (auto& animation : fms.animation_list) {
            animation.second.weight = 0;
          }
          found->second.weight = 1.f;
          fms.dt = 0;
          fms.fk_calculate_poses();
          fms.bone_transforms = fms.fk_calculate_transformations();
        }
      }
      else {
        fms.update_bone_transforms();
      }
      if (fms.bone_transforms.size() < animated_model.bone_count) {
        fms.bone_transforms.resize(animated_model.bone_count, fan::mat4(1));
      }
      std::copy_n(fms.bone_transforms.begin(), animated_model.bone_count, bone_matrices.begin() + animated_model.first_bone);
      animated_model.dirty = true;
      animation_vertices_dirty = true;
    }
    bool load_animated_scene_model(std::uint32_t object_index) {
      scene_model_t& model = scene_models[object_index];
      object_t& object = objects[object_index];
      object.first_instance = (std::uint32_t)instances.size();
      fan::model::fms_t::properties_t properties;
      properties.path = model.path;
      properties.texture_path = model.texture_path;
      properties.fix_uv_diagonals = model.fix_uv_diagonals;
      auto fms = std::make_unique<fan::model::fms_t>(properties, model.callers_path);
      std::uint32_t first_bone = (std::uint32_t)bone_matrices.size();
      std::uint32_t bone_count = fms->bone_count;
      if (bone_count != 0) {
        bone_matrices.resize(first_bone + bone_count, fan::mat4(1));
      }
      std::uint32_t first_model = (std::uint32_t)models.size();
      model_cache_entry_t range = load_model_from_fms(*fms, first_bone, bone_count, true);
      for (std::uint32_t i = 0; i < range.model_count; ++i) {
        add_instance(range.first_model + i, model.transform);
      }
      object.instance_count = (std::uint32_t)instances.size() - object.first_instance;
      object.animated_model_index = (std::uint32_t)animated_models.size();
      animated_models.emplace_back();
      animated_model_t& animated_model = animated_models.back();
      animated_model.fms = std::move(fms);
      animated_model.first_model = first_model;
      animated_model.model_count = (std::uint32_t)models.size() - first_model;
      animated_model.first_bone = first_bone;
      animated_model.bone_count = bone_count;
      initialize_animated_model_pose(animated_model);
      return true;
    }
    bool load_scene_model(std::uint32_t object_index) {
      scene_model_t& model = scene_models[object_index];
      if (model.animated) {
        return load_animated_scene_model(object_index);
      }
      object_t& object = objects[object_index];
      object.first_instance = (std::uint32_t)instances.size();
      std::string key = make_model_cache_key(model);
      auto found = model_cache.find(key);
      if (found != model_cache.end()) {
        add_cached_model_instances(model, found->second);
        object.instance_count = (std::uint32_t)instances.size() - object.first_instance;
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
      return true;
    }
    void destroy_buffer(VkBuffer& buffer, VkDeviceMemory& memory) {
      if (buffer) {
        vkDestroyBuffer(ctx->device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
      }
      if (memory) {
        vkFreeMemory(ctx->device, memory, nullptr);
        memory = VK_NULL_HANDLE;
      }
    }
    void destroy_tlas_resources() {
      tlas.destroy(*ctx);
      destroy_buffer(tlas_instance_buffer, tlas_instance_memory);
      destroy_buffer(tlas_instance_staging_buffer, tlas_instance_staging_memory);
      destroy_buffer(tlas_scratch_buffer, tlas_scratch_memory);
      tlas_instance_count = 0;
      tlas_instance_staging_size = 0;
    }
    void destroy_scene_geometry_resources() {
      for (auto& blas : blas_list) blas.destroy(*ctx);
      blas_list.clear();
      destroy_tlas_resources();
      destroy_buffer(blas_scratch_buffer, blas_scratch_memory);
      blas_scratch_size = 0;
      destroy_buffer(source_vertex_buffer, source_vertex_memory);
      destroy_buffer(vertex_buffer, vertex_memory);
      destroy_buffer(index_buffer, index_memory);
      destroy_buffer(material_buffer, material_memory);
      destroy_buffer(material_index_buffer, material_index_memory);
      destroy_buffer(bone_buffer, bone_memory);
      bone_mapped = nullptr;
    }
    void update_tlas_descriptor() {
      if (!descriptor_set) return;
      VkWriteDescriptorSetAccelerationStructureKHR as_info{};
      as_info.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
      as_info.accelerationStructureCount = 1;
      as_info.pAccelerationStructures = &tlas.handle;
      VkWriteDescriptorSet write{};
      write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      write.pNext = &as_info;
      write.dstSet = descriptor_set;
      write.dstBinding = 0;
      write.descriptorCount = 1;
      write.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
      vkUpdateDescriptorSets(ctx->device, 1, &write, 0, nullptr);
    }
    void update_scene_buffers_descriptor() {
      if (!descriptor_set) return;
      VkDescriptorBufferInfo material_info{material_buffer, 0, VK_WHOLE_SIZE};
      VkDescriptorBufferInfo vertex_info{vertex_buffer, 0, VK_WHOLE_SIZE};
      VkDescriptorBufferInfo index_info{index_buffer, 0, VK_WHOLE_SIZE};
      VkDescriptorBufferInfo mat_idx_info{material_index_buffer, 0, VK_WHOLE_SIZE};
      VkWriteDescriptorSet writes[4]{};
      set_descriptor_buffer_write(writes[0], 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &material_info);
      set_descriptor_buffer_write(writes[1], 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &vertex_info);
      set_descriptor_buffer_write(writes[2], 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &index_info);
      set_descriptor_buffer_write(writes[3], 9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &mat_idx_info);
      vkUpdateDescriptorSets(ctx->device, 4, writes, 0, nullptr);
    }
    void rebuild_tlas() {
      if (!ctx) return;
      vkDeviceWaitIdle(ctx->device);
      destroy_tlas_resources();
      create_tlas();
      update_tlas_descriptor();
      frame_index = 0;
      tlas_dirty = false;
    }
    void rebuild_scene_geometry() {
      if (!ctx) return;
      vkDeviceWaitIdle(ctx->device);
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
      frame_index = 0;
      tlas_dirty = false;
    }
    void set_light(const fan::vec3& position, const fan::vec3& color, f32_t intensity) {
      light_position = position;
      light_color = color;
      light_intensity = intensity;
      if (!ctx || !light_buffer || !light_mapped) return;
      light_ubo_t ubo{};
      ubo.position = light_position;
      ubo.color = light_color;
      ubo.intensity = light_intensity;
      std::memcpy(light_mapped, &ubo, sizeof(ubo));
    }
    void set_light() {
      set_light(light_position, light_color, light_intensity);
    }
    void reload_pipeline() {
      if (!ctx) return;
      vkDeviceWaitIdle(ctx->device);
      if (pipeline) vkDestroyPipeline(ctx->device, pipeline, nullptr);
      pipeline = VK_NULL_HANDLE;
      destroy_buffer(shader_binding_table, sbt_memory);
      create_pipeline();
      create_sbt();
      frame_index = 0;
    }
    void update_camera_from_engine(){
      auto camera_handle = fan::graphics::get_perspective_render_view().camera;
      auto camera_data = ctx->camera_get(camera_handle);
      if (!camera_mapped) return;
      rt_camera_t vp{};
      vp.projection = camera_data.projection;
      vp.view = camera_data.view;
      vp.inv_projection = camera_data.projection.inverse();
      vp.inv_view = camera_data.view.inverse();
      std::memcpy(camera_mapped, &vp, sizeof(vp));
    }
    void open(fan::vulkan::context_t& main_ctx, const fan::vec2ui& sz) {
      ctx = &main_ctx;
      size = sz;
      load_functions();

      ctx->create_buffer(sizeof(rt_camera_t) * 16, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, camera_buffer, camera_memory);
      vkMapMemory(ctx->device, camera_memory, 0, sizeof(rt_camera_t) * 16, 0, &camera_mapped);

      ctx->create_buffer(sizeof(time_ubo_t), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, time_buffer, time_memory);
      vkMapMemory(ctx->device, time_memory, 0, sizeof(time_ubo_t), 0, &time_mapped);

      ctx->create_buffer(sizeof(light_ubo_t), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, light_buffer, light_memory);
      vkMapMemory(ctx->device, light_memory, 0, sizeof(light_ubo_t), 0, &light_mapped);

      set_light();
      create_exposure_ubo(); 
      create_luminance_buffer(size.x, size.y);

      if (scene_models.empty()) fan::throw_error("ray tracing scene has no models; call add_model() before open()");
      for (std::uint32_t i = 0; i < scene_models.size(); ++i) load_scene_model(i);
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
      frame_index = 0;
      tlas_dirty = false;
    }
    void open(fan::graphics::engine_t& engine, const engine_open_properties_t& properties = {}) {
      attached_engine = &engine;
      auto output_size = properties.size;
      if (output_size.x == 0 || output_size.y == 0) output_size = engine.window.get_size();
      output_sprite_enabled = properties.create_output_sprite;
      open(engine.context.vk, output_size);
      ready = true;
      sync_output_sprite();
      attach_engine_callbacks(engine);
    }
    void detach_engine() {
      if (attached_engine && update_callback_registered) attached_engine->remove_update_callback(update_callback_handle);
      update_callback_registered = false;
      resize_handle.remove();
      if (pre_begin_callback_registered && registered_vk_context && pre_begin_cmd_cb_index < registered_vk_context->pre_begin_cmd_cb.size()) registered_vk_context->pre_begin_cmd_cb[pre_begin_cmd_cb_index] = []() {};
      pre_begin_callback_registered = false;
      if (command_callback_registered && registered_vk_context && begin_cmd_cb_index < registered_vk_context->begin_cmd_cb.size()) registered_vk_context->begin_cmd_cb[begin_cmd_cb_index] = [](VkCommandBuffer) {};
      command_callback_registered = false;
      registered_vk_context = nullptr;
      attached_engine = nullptr;
      pending_resize = false;
      ready = false;
      if (output_sprite) output_sprite.erase();
    }
    void update() {
      if (!attached_engine) return;
      if (pending_resize) {
        pending_resize = false;
        vkDeviceWaitIdle(attached_engine->context.vk.device);
        close();
        open(attached_engine->context.vk, pending_size);
        ready = true;
        sync_output_sprite();
      }
      if (!ready) return;
      if (tlas_dirty && !can_update_tlas_transforms()) flush_transform_updates();
      on_camera_updated(update_camera);
      update_exposure(attached_engine->get_delta_time());
      if (update_camera) update_camera_from_engine();
      sync_output_sprite();
    }
#if defined(FAN_GUI)
    void render_gui(const char* window_name = "ray tracing") {
      fan::graphics::gui::begin(window_name);
      fan::graphics::gui::checkbox("update camera", &update_camera);
      fan::graphics::gui::checkbox("auto exposure", &enable_auto_exposure);
      fan::graphics::gui::checkbox("gi bounce", &enable_gi);
      fan::graphics::gui::checkbox("reflections", &enable_reflections);
      bool light_changed = false;
      light_changed |= fan::graphics::gui::drag("Light Position", &light_position);
      light_changed |= fan::graphics::gui::drag("Light Color", &light_color);
      light_changed |= fan::graphics::gui::drag("Light Intensity", &light_intensity);
      if (light_changed) set_light();
      fan::graphics::gui::end();
    }
#endif
    void attach_engine_callbacks(fan::graphics::engine_t& engine) {
      if (!registered_vk_context) registered_vk_context = &engine.context.vk;
      if (!pre_begin_callback_registered) {
        pre_begin_cmd_cb_index = registered_vk_context->pre_begin_cmd_cb.size();
        registered_vk_context->pre_begin_cmd_cb.push_back([this]() {
          if (tlas_dirty && !can_update_tlas_transforms()) flush_transform_updates();
        });
        pre_begin_callback_registered = true;
      }
      if (!command_callback_registered) {
        begin_cmd_cb_index = registered_vk_context->begin_cmd_cb.size();
        registered_vk_context->begin_cmd_cb.push_back([this](VkCommandBuffer cmd) {
          if (ready) {
            record_gpu_animation_updates(cmd);
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
      if (!attached_engine || !output_sprite_enabled || !accum_image_valid) return;
      fan::vec2 window_size = fan::vec2(attached_engine->window.get_size());
      if (!output_sprite) {
        output_sprite = fan::graphics::sprite_t{{
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
      if (!animation_vertices_dirty || !skinning_pipeline || !skinning_descriptor_set) return;
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, skinning_pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, skinning_pipeline_layout, 0, 1, &skinning_descriptor_set, 0, nullptr);
      bool dispatched = false;
      for (animated_model_t& animated_model : animated_models) {
        if (!animated_model.dirty) continue;
        for (std::uint32_t i = 0; i < animated_model.model_count; ++i) {
          std::uint32_t model_index = animated_model.first_model + i;
          const model_t& model = models[model_index];
          if (model.vertex_count == 0 || model.bone_count == 0) continue;
          skinning_push_constants_t pc{};
          pc.first_vertex = model.first_vertex;
          pc.vertex_count = model.vertex_count;
          pc.first_bone = model.first_bone;
          pc.bone_count = model.bone_count;
          vkCmdPushConstants(cmd, skinning_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
          vkCmdDispatch(cmd, (model.vertex_count + 63) / 64, 1, 1);
          dispatched = true;
        }
      }
      if (!dispatched) {
        animation_vertices_dirty = false;
        for (animated_model_t& animated_model : animated_models) {
          animated_model.dirty = false;
        }
        return;
      }
      VkMemoryBarrier skin_barrier{};
      skin_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      skin_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      skin_barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &skin_barrier, 0, nullptr, 0, nullptr);
      bool updated = false;
      for (animated_model_t& animated_model : animated_models) {
        if (!animated_model.dirty) continue;
        for (std::uint32_t i = 0; i < animated_model.model_count; ++i) {
          record_blas_update(cmd, animated_model.first_model + i);
          updated = true;
          VkMemoryBarrier build_barrier{};
          build_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
          build_barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
          build_barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
          vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &build_barrier, 0, nullptr, 0, nullptr);
        }
        animated_model.dirty = false;
      }
      if (updated) {
        VkMemoryBarrier trace_barrier{};
        trace_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        trace_barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        trace_barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 1, &trace_barrier, 0, nullptr, 0, nullptr);
      }
      animation_vertices_dirty = false;
    }
    void record_trace_rays(VkCommandBuffer cmd) {
      static auto start_time = std::chrono::steady_clock::now();
      time_ubo_t t{ std::chrono::duration<f32_t>(std::chrono::steady_clock::now() - start_time).count() };
      std::memcpy(time_mapped, &t, sizeof(t));

      if (current_layout != VK_IMAGE_LAYOUT_GENERAL) {
        ctx->insert_image_barrier(cmd, ctx->image_get(output_image).image_index, current_layout, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR);
        current_layout = VK_IMAGE_LAYOUT_GENERAL;
      }
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
      VkDeviceAddress sbt_addr = get_buffer_address(shader_binding_table);
      VkStridedDeviceAddressRegionKHR rgen_region{};
      rgen_region.deviceAddress = sbt_addr + rgen_offset;
      rgen_region.stride = group_stride;
      rgen_region.size   = group_stride;
      VkStridedDeviceAddressRegionKHR miss_region{};
      miss_region.deviceAddress = sbt_addr + miss_offset;
      miss_region.stride        = group_stride;
      miss_region.size          = group_stride * 2;
      VkStridedDeviceAddressRegionKHR hit_region{};
      hit_region.deviceAddress = sbt_addr + hit_offset;
      hit_region.stride        = group_stride;
      hit_region.size          = group_stride;
      VkStridedDeviceAddressRegionKHR callable_region{};
      vkCmdTraceRaysKHR(cmd, &rgen_region, &miss_region, &hit_region, &callable_region, size.x, size.y, 1);
      {
        ctx->insert_image_barrier(cmd, ctx->image_get(output_image).image_index, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        current_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      }
      if (enable_auto_exposure) dispatch_luminance_compute(cmd, luminance_descriptor_set, size.x, size.y);
      if (accum_layout != VK_IMAGE_LAYOUT_GENERAL) {
        ctx->insert_image_barrier(cmd, ctx->image_get(accum_image).image_index, accum_layout, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        accum_layout = VK_IMAGE_LAYOUT_GENERAL;
      }
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, accum_pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, accum_pipeline_layout, 0, 1, &accum_descriptor_set, 0, nullptr);
      vkCmdPushConstants(cmd, accum_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(std::uint32_t), &frame_index);
      std::uint32_t gx = (size.x + 7) / 8;
      std::uint32_t gy = (size.y + 7) / 8;
      vkCmdDispatch(cmd, gx, gy, 1);
      {
        ctx->insert_image_barrier(cmd, ctx->image_get(accum_image).image_index, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        accum_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      }
    }
    void on_camera_updated(bool camera_moved) {
      if (camera_moved) frame_index = 0;
      else frame_index++;
    }
    void trace_rays_before_shapes(){
      record_trace_rays(ctx->command_buffers[ctx->current_frame]);
    }
    void create_accum_image() {
      accum_image = create_rt_image(accum_image_valid, accum_layout, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, true);
    }
    void create_compute_pipeline(
      const char* shader_path,
      const VkDescriptorSetLayoutBinding* bindings,
      std::uint32_t binding_count,
      std::uint32_t push_constant_size,
      VkDescriptorSetLayout& descriptor_layout_out,
      VkPipelineLayout& pipeline_layout_out,
      VkPipeline& pipeline_out
    ) {
      VkDescriptorSetLayoutCreateInfo layout_info{};
      layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
      layout_info.bindingCount = binding_count;
      layout_info.pBindings = bindings;
      fan::vulkan::validate(vkCreateDescriptorSetLayout(ctx->device, &layout_info, nullptr, &descriptor_layout_out));
      VkPushConstantRange pcr{};
      pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      pcr.offset = 0;
      pcr.size = push_constant_size;
      VkPipelineLayoutCreateInfo pl_info{};
      pl_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
      pl_info.setLayoutCount = 1;
      pl_info.pSetLayouts = &descriptor_layout_out;
      pl_info.pushConstantRangeCount = push_constant_size ? 1 : 0;
      pl_info.pPushConstantRanges = push_constant_size ? &pcr : nullptr;
      fan::vulkan::validate(vkCreatePipelineLayout(ctx->device, &pl_info, nullptr, &pipeline_layout_out));
      VkShaderModule comp = load_shader(shader_path, shaderc_glsl_compute_shader);
      VkPipelineShaderStageCreateInfo stage{};
      stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
      stage.module = comp;
      stage.pName = "main";
      VkComputePipelineCreateInfo pi{};
      pi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
      pi.stage = stage;
      pi.layout = pipeline_layout_out;
      fan::vulkan::validate(vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &pi, nullptr, &pipeline_out));
      vkDestroyShaderModule(ctx->device, comp, nullptr);
    }
    void create_skinning_pipeline() {
      VkDescriptorSetLayoutBinding bindings[3]{};
      auto bnd = [&](int i) { bindings[i] = { (std::uint32_t)i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; };
      bnd(0);
      bnd(1);
      bnd(2);
      create_compute_pipeline(
        "shaders/vulkan/ray_tracing/skin.comp",
        bindings,
        (std::uint32_t)std::size(bindings),
        sizeof(skinning_push_constants_t),
        skinning_descriptor_layout,
        skinning_pipeline_layout,
        skinning_pipeline
      );
    }
    void update_skinning_descriptor() {
      if (!skinning_descriptor_set || !source_vertex_buffer || !vertex_buffer || !bone_buffer) return;
      VkDescriptorBufferInfo source_info{source_vertex_buffer, 0, VK_WHOLE_SIZE};
      VkDescriptorBufferInfo vertex_info{vertex_buffer, 0, VK_WHOLE_SIZE};
      VkDescriptorBufferInfo bone_info{bone_buffer, 0, VK_WHOLE_SIZE};
      VkWriteDescriptorSet writes[3]{};
      writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[0].dstSet = skinning_descriptor_set;
      writes[0].dstBinding = 0;
      writes[0].descriptorCount = 1;
      writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[0].pBufferInfo = &source_info;
      writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[1].dstSet = skinning_descriptor_set;
      writes[1].dstBinding = 1;
      writes[1].descriptorCount = 1;
      writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[1].pBufferInfo = &vertex_info;
      writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[2].dstSet = skinning_descriptor_set;
      writes[2].dstBinding = 2;
      writes[2].descriptorCount = 1;
      writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[2].pBufferInfo = &bone_info;
      vkUpdateDescriptorSets(ctx->device, 3, writes, 0, nullptr);
    }
    void create_skinning_descriptor_set() {
      VkDescriptorPoolSize pool_size{};
      pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      pool_size.descriptorCount = 3;
      VkDescriptorPoolCreateInfo pool_info{};
      pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
      pool_info.maxSets = 1;
      pool_info.poolSizeCount = 1;
      pool_info.pPoolSizes = &pool_size;
      fan::vulkan::validate(vkCreateDescriptorPool(ctx->device, &pool_info, nullptr, &skinning_descriptor_pool));
      VkDescriptorSetAllocateInfo alloc_info{};
      alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
      alloc_info.descriptorPool = skinning_descriptor_pool;
      alloc_info.descriptorSetCount = 1;
      alloc_info.pSetLayouts = &skinning_descriptor_layout;
      fan::vulkan::validate(vkAllocateDescriptorSets(ctx->device, &alloc_info, &skinning_descriptor_set));
      update_skinning_descriptor();
    }
    void create_accum_pipeline() {
      VkDescriptorSetLayoutBinding bindings[2]{};
      auto bnd = [&](int i, VkDescriptorType t) { bindings[i] = { (std::uint32_t)i, t, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; };
      bnd(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
      bnd(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
      create_compute_pipeline(
        "shaders/vulkan/ray_tracing/accumulate.comp",
        bindings,
        (std::uint32_t)std::size(bindings),
        sizeof(std::uint32_t),
        accum_descriptor_layout,
        accum_pipeline_layout,
        accum_pipeline
      );
    }
    void create_accum_descriptor_set() {
      VkDescriptorPoolSize pool_sizes[2]{};
      pool_sizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      pool_sizes[0].descriptorCount = 1;
      pool_sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      pool_sizes[1].descriptorCount = 1;
      VkDescriptorPoolCreateInfo pool_info{};
      pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
      pool_info.maxSets = 1;
      pool_info.poolSizeCount = 2;
      pool_info.pPoolSizes = pool_sizes;
      fan::vulkan::validate(vkCreateDescriptorPool(ctx->device, &pool_info, nullptr, &accum_descriptor_pool));
      VkDescriptorSetAllocateInfo alloc_info{};
      alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
      alloc_info.descriptorPool = accum_descriptor_pool;
      alloc_info.descriptorSetCount = 1;
      alloc_info.pSetLayouts = &accum_descriptor_layout;
      fan::vulkan::validate(vkAllocateDescriptorSets(ctx->device, &alloc_info, &accum_descriptor_set));
      auto& out_img = ctx->image_get(output_image);
      VkDescriptorImageInfo current_info{};
      current_info.sampler = out_img.sampler;
      current_info.imageView = out_img.image_view;
      current_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      auto& acc_img = ctx->image_get(accum_image);
      VkDescriptorImageInfo accum_info{};
      accum_info.sampler = VK_NULL_HANDLE; 
      accum_info.imageView = acc_img.image_view;
      accum_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
      VkWriteDescriptorSet writes[2]{};
      writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[0].dstSet = accum_descriptor_set;
      writes[0].dstBinding = 0;
      writes[0].descriptorCount = 1;
      writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      writes[0].pImageInfo = &current_info;
      writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[1].dstSet = accum_descriptor_set;
      writes[1].dstBinding = 1;
      writes[1].descriptorCount = 1;
      writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      writes[1].pImageInfo = &accum_info;
      vkUpdateDescriptorSets(ctx->device, 2, writes, 0, nullptr);
    }
    void create_luminance_pipeline() {
      VkDescriptorSetLayoutBinding bindings[2]{};
      auto bnd = [&](int i, VkDescriptorType t) { bindings[i] = { (std::uint32_t)i, t, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; };
      bnd(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
      bnd(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
      create_compute_pipeline(
        "shaders/vulkan/ray_tracing/luminance_reduce.comp",
        bindings,
        (std::uint32_t)std::size(bindings),
        sizeof(int) * 4,
        luminance_descriptor_layout,
        luminance_pipeline_layout,
        luminance_pipeline
      );
    }
    void create_luminance_descriptor_set() {
      VkDescriptorPoolSize pool_sizes[2]{};
      pool_sizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      pool_sizes[0].descriptorCount = 1;
      pool_sizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      pool_sizes[1].descriptorCount = 1;
      VkDescriptorPoolCreateInfo pool_info{};
      pool_info.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
      pool_info.maxSets       = 1;
      pool_info.poolSizeCount = 2;
      pool_info.pPoolSizes    = pool_sizes;
      fan::vulkan::validate(vkCreateDescriptorPool(ctx->device, &pool_info, nullptr, &luminance_descriptor_pool));
      VkDescriptorSetAllocateInfo alloc_info{};
      alloc_info.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
      alloc_info.descriptorPool     = luminance_descriptor_pool;
      alloc_info.descriptorSetCount = 1;
      alloc_info.pSetLayouts        = &luminance_descriptor_layout;
      fan::vulkan::validate(vkAllocateDescriptorSets(ctx->device, &alloc_info, &luminance_descriptor_set));
      auto& hdr_img = ctx->image_get(output_image);
      VkDescriptorImageInfo hdr_info{};
      hdr_info.sampler     = hdr_img.sampler;
      hdr_info.imageView   = hdr_img.image_view;
      hdr_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      VkDescriptorBufferInfo lum_info{};
      lum_info.buffer = luminance_buffer;
      lum_info.range  = VK_WHOLE_SIZE;
      VkWriteDescriptorSet writes[2]{};
      writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[0].dstSet          = luminance_descriptor_set;
      writes[0].dstBinding      = 0;
      writes[0].descriptorCount = 1;
      writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      writes[0].pImageInfo      = &hdr_info;
      writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[1].dstSet          = luminance_descriptor_set;
      writes[1].dstBinding      = 1;
      writes[1].descriptorCount = 1;
      writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[1].pBufferInfo     = &lum_info;
      vkUpdateDescriptorSets(ctx->device, 2, writes, 0, nullptr);
    }
    void create_luminance_buffer(std::uint32_t width, std::uint32_t height) {
      luminance_group_x = (width  + 15) / 16;
      luminance_group_y = (height + 15) / 16;
      luminance_group_count = luminance_group_x * luminance_group_y;
      VkDeviceSize size = sizeof(f32_t) * luminance_group_count;
      ctx->create_buffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, luminance_buffer, luminance_memory);
      vkMapMemory(ctx->device, luminance_memory, 0, size, 0, &luminance_mapped);
    }
    void dispatch_luminance_compute(VkCommandBuffer cmd, VkDescriptorSet luminance_set, std::uint32_t width, std::uint32_t height) {
      if (luminance_group_x == 0 || luminance_group_y == 0) return;
      vkCmdFillBuffer(cmd, luminance_buffer, 0, sizeof(f32_t) * luminance_group_count, 0);
      VkBufferMemoryBarrier clearBarrier{};
      clearBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
      clearBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      clearBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      clearBarrier.buffer = luminance_buffer;
      clearBarrier.offset = 0;
      clearBarrier.size = VK_WHOLE_SIZE;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &clearBarrier, 0, nullptr);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, luminance_pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, luminance_pipeline_layout, 0, 1, &luminance_set, 0, nullptr);
      struct PC { int w, h, gx, gy; } pc;
      pc.w = width;
      pc.h = height;
      pc.gx = luminance_group_x;
      pc.gy = luminance_group_y;
      vkCmdPushConstants(cmd, luminance_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
      vkCmdDispatch(cmd, luminance_group_x, luminance_group_y, 1);
      VkBufferMemoryBarrier read_barrier{};
      read_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
      read_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      read_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
      read_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      read_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      read_barrier.buffer = luminance_buffer;
      read_barrier.offset = 0;
      read_barrier.size = sizeof(f32_t) * luminance_group_count;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &read_barrier, 0, nullptr);
    }
    void write_exposure_ubo() {
      if (!ctx || !exposure_ubo || !exposure_mapped) return;
      exposure_ubo_t ubo{};
      ubo.exposure = exposure;
      ubo.enable_gi = enable_gi ? 1.f : 0.f;
      ubo.enable_reflections = enable_reflections ? 1.f : 0.f;
      std::memcpy(exposure_mapped, &ubo, sizeof(ubo));
    }
    void update_exposure(f32_t dt) {
      if (!ctx || !enable_auto_exposure || !luminance_buffer || luminance_group_count == 0 || !luminance_mapped) {
        write_exposure_ubo();
        return;
      }
      std::vector<f32_t> partial(luminance_group_count);
      std::memcpy(partial.data(), luminance_mapped, sizeof(f32_t) * luminance_group_count);
      double total = 0.0;
      for (f32_t v : partial) total += v;
      double avgL = total / double(size.x * size.y);
      if (avgL < 1e-6) avgL = 1e-6;
      f32_t middle_gray = 0.06f;
      target_exposure = middle_gray / f32_t(avgL);
      f32_t lambda = 1.0f - std::exp(-adaptation_speed * dt);
      exposure = exposure + lambda * (target_exposure - exposure);
      exposure = std::clamp(exposure, 0.0001f, 20.0f);
      write_exposure_ubo();
    }
    void create_exposure_ubo() {
      ctx->create_buffer(sizeof(exposure_ubo_t), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, exposure_ubo, exposure_ubo_memory);
      vkMapMemory(ctx->device, exposure_ubo_memory, 0, sizeof(exposure_ubo_t), 0, &exposure_mapped);
      write_exposure_ubo();
    }
    void close(){
      if(!ctx) return;
      ready = false;
      vkDeviceWaitIdle(ctx->device);
      for(auto& geom : mesh_geometries) geom.destroy(*ctx);
      destroy_scene_geometry_resources();

      auto destroy_pipe = [&](auto& p, auto& l, auto& d, auto& pool) {
        if (p) vkDestroyPipeline(ctx->device, p, nullptr);
        if (l) vkDestroyPipelineLayout(ctx->device, l, nullptr);
        if (d) vkDestroyDescriptorSetLayout(ctx->device, d, nullptr);
        if (pool) vkDestroyDescriptorPool(ctx->device, pool, nullptr);
        p = VK_NULL_HANDLE;
        l = VK_NULL_HANDLE;
        d = VK_NULL_HANDLE;
        pool = VK_NULL_HANDLE;
      };
      destroy_pipe(pipeline, pipeline_layout, descriptor_layout, descriptor_pool);
      destroy_pipe(accum_pipeline, accum_pipeline_layout, accum_descriptor_layout, accum_descriptor_pool);
      destroy_pipe(luminance_pipeline, luminance_pipeline_layout, luminance_descriptor_layout, luminance_descriptor_pool);
      destroy_pipe(skinning_pipeline, skinning_pipeline_layout, skinning_descriptor_layout, skinning_descriptor_pool);
      skinning_descriptor_set = VK_NULL_HANDLE;

      destroy_buffer(shader_binding_table, sbt_memory);
      destroy_buffer(sbt_staging, sbt_staging_memory);
      destroy_buffer(camera_buffer, camera_memory);
      destroy_buffer(time_buffer, time_memory);
      destroy_buffer(luminance_buffer, luminance_memory);
      destroy_buffer(exposure_ubo, exposure_ubo_memory);
      destroy_buffer(light_buffer, light_memory);

      camera_mapped = nullptr;
      time_mapped = nullptr;
      tlas_instance_staging_mapped = nullptr;
      luminance_mapped = nullptr;
      exposure_mapped = nullptr;
      light_mapped = nullptr;
      bone_mapped = nullptr;

      for(auto tex_id : texture_ids) ctx->image_erase(tex_id);
      if(output_image_valid) ctx->image_erase(output_image);
      if(accum_image_valid) ctx->image_erase(accum_image);

      blas_list.clear();
      model_cache.clear();
      mesh_geometries.clear();
      submeshes.clear();
      models.clear();
      instances.clear();
      animated_models.clear();
      for (auto& object : objects) {
        object.first_instance = 0;
        object.instance_count = 0;
        object.animated_model_index = std::uint32_t(-1);
      }
      materials.clear();
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
      frame_index = 0;
      tlas_dirty = false;
      animation_vertices_dirty = false;
      output_image_valid = false;
      accum_image_valid = false;
      tlas_instance_staging_size = 0;
      luminance_group_count = 0;
      luminance_group_x = 0;
      luminance_group_y = 0;
      ctx = nullptr;
    }
  #pragma pack(push, 1)
    struct vertex_t {
      fan::vec3 position; f32_t pad0;
      fan::vec3 normal; f32_t pad1;
      fan::vec2 texcoord; fan::vec2 pad2;
      fan::vec3 color; f32_t pad3;
    };
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
    static constexpr std::uint32_t instance_count = 1;
    static constexpr std::uint32_t max_textures = 512;
    engine_open_properties_t pending_open_properties;
    fan::vulkan::context_t* ctx = nullptr;
    fan::vec2ui size{};
    std::vector<acceleration_structure_t> blas_list;
    std::vector<shapes::gpu_mesh_t> mesh_geometries;
    std::vector<animated_model_t> animated_models;
    acceleration_structure_t tlas;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkBuffer shader_binding_table = VK_NULL_HANDLE;
    VkDeviceMemory sbt_memory = VK_NULL_HANDLE;
    VkBuffer sbt_staging = VK_NULL_HANDLE;
    VkDeviceMemory sbt_staging_memory = VK_NULL_HANDLE;
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
    fan::graphics::image_t output_image;
    bool output_image_valid = false;
    VkImageLayout current_layout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkBuffer camera_buffer = VK_NULL_HANDLE;
    VkDeviceMemory camera_memory = VK_NULL_HANDLE;
    VkBuffer time_buffer = VK_NULL_HANDLE;
    VkDeviceMemory time_memory = VK_NULL_HANDLE;
    VkBuffer tlas_instance_buffer = VK_NULL_HANDLE;
    VkDeviceMemory tlas_instance_memory = VK_NULL_HANDLE;
    VkBuffer tlas_instance_staging_buffer = VK_NULL_HANDLE;
    VkDeviceMemory tlas_instance_staging_memory = VK_NULL_HANDLE;
    VkDeviceSize tlas_instance_staging_size = 0;
    std::uint32_t tlas_instance_count = 0;
    VkBuffer tlas_scratch_buffer = VK_NULL_HANDLE;
    VkDeviceMemory tlas_scratch_memory = VK_NULL_HANDLE;
    VkBuffer blas_scratch_buffer = VK_NULL_HANDLE;
    VkDeviceMemory blas_scratch_memory = VK_NULL_HANDLE;
    VkDeviceSize blas_scratch_size = 0;
    VkBuffer source_vertex_buffer = VK_NULL_HANDLE;
    VkDeviceMemory source_vertex_memory = VK_NULL_HANDLE;
    VkBuffer vertex_buffer = VK_NULL_HANDLE;
    VkDeviceMemory vertex_memory = VK_NULL_HANDLE;
    VkBuffer index_buffer = VK_NULL_HANDLE;
    VkDeviceMemory index_memory = VK_NULL_HANDLE;
    VkBuffer material_buffer = VK_NULL_HANDLE;
    VkDeviceMemory material_memory = VK_NULL_HANDLE;
    VkBuffer bone_buffer = VK_NULL_HANDLE;
    VkDeviceMemory bone_memory = VK_NULL_HANDLE;
    std::vector<material_info_t> materials;
    std::vector<source_vertex_t> source_vertex_data;
    std::vector<vertex_t> vertex_data;
    std::vector<std::uint32_t> index_data;
    std::vector<fan::mat4> bone_matrices;
    std::vector<fan::graphics::image_nr_t> texture_ids;
    std::vector<VkDescriptorImageInfo> rt_texture_infos;
    VkBuffer material_index_buffer = VK_NULL_HANDLE;
    VkDeviceMemory material_index_memory = VK_NULL_HANDLE;
    std::vector<std::uint32_t> material_indices_per_primitive;
    std::size_t current_material_index = 0;
    std::uint32_t vertex_offset = 0;
    std::uint32_t object_generation_counter = 1;
    bool tlas_dirty = false;
    bool scene_geometry_dirty = false;
    bool animation_vertices_dirty = false;
    f32_t exposure = 1.f;
    f32_t target_exposure = 1.f;
    f32_t adaptation_speed = 4.f;
    bool enable_auto_exposure = false;
    bool enable_gi = false;
    bool enable_reflections = false;
    bool update_camera = true;
    fan::vec3 light_position = fan::vec3(5.0f, 10.0f, 5.0f);
    fan::vec3 light_color = fan::vec3(1.0f, 1.0f, 1.0f);
    f32_t light_intensity = 3.0f;
    fan::graphics::sprite_t output_sprite;
    fan::graphics::engine_t* attached_engine = nullptr;
    bool output_sprite_enabled = false;
    bool ready = false;
    bool pending_resize = false;
    fan::vec2ui pending_size{};
    fan::window_t::resize_handle_t resize_handle;
    fan::graphics::engine_t::update_callback_handle_t update_callback_handle;
    bool update_callback_registered = false;
    fan::vulkan::context_t* registered_vk_context = nullptr;
    std::size_t pre_begin_cmd_cb_index = 0;
    bool pre_begin_callback_registered = false;
    std::size_t begin_cmd_cb_index = 0;
    bool command_callback_registered = false;
    VkBuffer luminance_buffer = VK_NULL_HANDLE;
    VkDeviceMemory luminance_memory = VK_NULL_HANDLE;
    std::uint32_t luminance_group_x = 0;
    std::uint32_t luminance_group_y = 0;
    std::uint32_t luminance_group_count = 0;
    VkBuffer exposure_ubo = VK_NULL_HANDLE;
    VkDeviceMemory exposure_ubo_memory = VK_NULL_HANDLE;
    VkDescriptorSetLayout luminance_descriptor_layout = VK_NULL_HANDLE;
    VkDescriptorPool      luminance_descriptor_pool   = VK_NULL_HANDLE;
    VkDescriptorSet       luminance_descriptor_set    = VK_NULL_HANDLE;
    VkPipelineLayout      luminance_pipeline_layout   = VK_NULL_HANDLE;
    VkPipeline            luminance_pipeline          = VK_NULL_HANDLE;
    VkDescriptorSetLayout skinning_descriptor_layout = VK_NULL_HANDLE;
    VkDescriptorPool skinning_descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet skinning_descriptor_set = VK_NULL_HANDLE;
    VkPipelineLayout skinning_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline skinning_pipeline = VK_NULL_HANDLE;
    VkBuffer light_buffer = VK_NULL_HANDLE;
    VkDeviceMemory light_memory = VK_NULL_HANDLE;

    void* camera_mapped = nullptr;
    void* time_mapped = nullptr;
    void* light_mapped = nullptr;
    void* exposure_mapped = nullptr;
    void* luminance_mapped = nullptr;
    void* bone_mapped = nullptr;
    void* tlas_instance_staging_mapped = nullptr;

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