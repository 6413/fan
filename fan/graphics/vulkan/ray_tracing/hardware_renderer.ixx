module;

#include <vulkan/vulkan.h>
#include <shaderc/shaderc.hpp>
#include <vector>
#include <cstring>
#include <cmath>

export module fan.graphics.vulkan.ray_tracing.hardware_renderer;

import fan.graphics.vulkan.core;
import fan.graphics;
import fan.graphics.vulkan.ray_tracing.shapes;

export namespace fan::graphics::vulkan::ray_tracing {

  struct acceleration_structure_t {
    void destroy(fan::vulkan::context_t& ctx) {
      if (handle) {
        auto vkDestroyAS = (PFN_vkDestroyAccelerationStructureKHR)
          vkGetDeviceProcAddr(ctx.device, "vkDestroyAccelerationStructureKHR");
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

  struct context_t {

    struct time_ubo_t {
      f32_t time = 0;
    };

    VkDeviceAddress get_buffer_address(VkBuffer buffer) const {
      VkBufferDeviceAddressInfoKHR info{};
      info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
      info.buffer = buffer;
      return vkGetBufferDeviceAddressKHR(ctx->device, &info);
    }

    void load_functions() {
      vkGetBufferDeviceAddressKHR = (PFN_vkGetBufferDeviceAddressKHR)
        vkGetDeviceProcAddr(ctx->device, "vkGetBufferDeviceAddressKHR");
      vkCmdBuildAccelerationStructuresKHR = (PFN_vkCmdBuildAccelerationStructuresKHR)
        vkGetDeviceProcAddr(ctx->device, "vkCmdBuildAccelerationStructuresKHR");
      vkGetAccelerationStructureBuildSizesKHR = (PFN_vkGetAccelerationStructureBuildSizesKHR)
        vkGetDeviceProcAddr(ctx->device, "vkGetAccelerationStructureBuildSizesKHR");
      vkCreateAccelerationStructureKHR = (PFN_vkCreateAccelerationStructureKHR)
        vkGetDeviceProcAddr(ctx->device, "vkCreateAccelerationStructureKHR");
      vkGetAccelerationStructureDeviceAddressKHR = (PFN_vkGetAccelerationStructureDeviceAddressKHR)
        vkGetDeviceProcAddr(ctx->device, "vkGetAccelerationStructureDeviceAddressKHR");
      vkCreateRayTracingPipelinesKHR = (PFN_vkCreateRayTracingPipelinesKHR)
        vkGetDeviceProcAddr(ctx->device, "vkCreateRayTracingPipelinesKHR");
      vkGetRayTracingShaderGroupHandlesKHR = (PFN_vkGetRayTracingShaderGroupHandlesKHR)
        vkGetDeviceProcAddr(ctx->device, "vkGetRayTracingShaderGroupHandlesKHR");
      vkCmdTraceRaysKHR = (PFN_vkCmdTraceRaysKHR)
        vkGetDeviceProcAddr(ctx->device, "vkCmdTraceRaysKHR");
    }

    void create_blas() {
      VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
      triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
      triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
      triangles.vertexData.deviceAddress = get_buffer_address(mesh_geom.vertex_buffer);
      triangles.vertexStride = sizeof(fan::vec3);
      triangles.maxVertex = mesh_geom.vertex_count - 1;
      triangles.indexType = VK_INDEX_TYPE_UINT32;
      triangles.indexData.deviceAddress = get_buffer_address(mesh_geom.index_buffer);

      VkAccelerationStructureGeometryKHR geometry{};
      geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
      geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
      geometry.geometry.triangles = triangles;
      geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

      VkAccelerationStructureBuildGeometryInfoKHR build_info{};
      build_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
      build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
      build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR |
        VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
      build_info.geometryCount = 1;
      build_info.pGeometries = &geometry;

      uint32_t primitive_count = mesh_geom.index_count / 3;

      VkAccelerationStructureBuildSizesInfoKHR size_info{};
      size_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
      vkGetAccelerationStructureBuildSizesKHR(ctx->device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &build_info, &primitive_count, &size_info);

      ctx->create_buffer(size_info.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, blas.buffer, blas.memory);

      VkAccelerationStructureCreateInfoKHR create_info{};
      create_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
      create_info.buffer = blas.buffer;
      create_info.size = size_info.accelerationStructureSize;
      create_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
      fan::vulkan::validate(vkCreateAccelerationStructureKHR(ctx->device, &create_info, nullptr, &blas.handle));

      VkBuffer scratch_buffer;
      VkDeviceMemory scratch_memory;
      ctx->create_buffer(size_info.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | 
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        scratch_buffer, scratch_memory);

      build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
      build_info.dstAccelerationStructure = blas.handle;
      build_info.scratchData.deviceAddress = get_buffer_address(scratch_buffer);

      VkAccelerationStructureBuildRangeInfoKHR range_info{};
      range_info.primitiveCount = primitive_count;
      const VkAccelerationStructureBuildRangeInfoKHR* range_infos = &range_info;

      VkCommandBuffer cmd = ctx->begin_single_time_commands();
      vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build_info, &range_infos);
      ctx->end_single_time_commands(cmd);

      vkDestroyBuffer(ctx->device, scratch_buffer, nullptr);
      vkFreeMemory(ctx->device, scratch_memory, nullptr);

      VkAccelerationStructureDeviceAddressInfoKHR addr_info{};
      addr_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
      addr_info.accelerationStructure = blas.handle;
      blas.device_address = vkGetAccelerationStructureDeviceAddressKHR(ctx->device, &addr_info);
    }

    void update_tlas(f32_t time) {
      if (!tlas.handle || !tlas_instance_buffer || tlas_instance_count == 0) return;

      static struct {
        std::vector<VkAccelerationStructureInstanceKHR> instances;
        VkBuffer staging = VK_NULL_HANDLE;
        VkDeviceMemory staging_mem = VK_NULL_HANDLE;
        void* mapped = nullptr;
        uint32_t cached_count = 0;
      } s;

      VkDeviceSize instance_size = sizeof(VkAccelerationStructureInstanceKHR) * tlas_instance_count;

      if (s.cached_count != tlas_instance_count) {
        if (s.staging) vkDestroyBuffer(ctx->device, s.staging, nullptr);
        if (s.staging_mem) vkFreeMemory(ctx->device, s.staging_mem, nullptr);
        s.instances.resize(tlas_instance_count);
        ctx->create_buffer(instance_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          s.staging, s.staging_mem);
        vkMapMemory(ctx->device, s.staging_mem, 0, instance_size, 0, &s.mapped);
        s.cached_count = tlas_instance_count;
      }

      const f32_t spacing = 1.0f;
      const int dim = (int)std::ceil(std::pow(f32_t(tlas_instance_count), 1.0f / 3.0f));
      const f32_t offset = spacing * f32_t(dim) * 0.5f;
      const f32_t base_angle = time * 0.5f;
      const f32_t radius = 1.0f;

      for (uint32_t i = 0; i < tlas_instance_count; ++i) {
        const int ix = int(i % dim);
        const int iy = int((i / dim) % dim);
        const int iz = int((i / (dim * dim)) % dim);
        f32_t angle = base_angle + f32_t(i) * 0.01f;

        VkTransformMatrixKHR t{};
        t.matrix[0][0] = 1.0f; t.matrix[0][3] = f32_t(ix) * spacing - offset + radius * std::cos(angle);
        t.matrix[1][1] = 1.0f; t.matrix[1][3] = f32_t(iy) * spacing - offset + radius * std::sin(angle);
        t.matrix[2][2] = 1.0f; t.matrix[2][3] = f32_t(iz) * spacing - offset;

        VkAccelerationStructureInstanceKHR& inst = s.instances[i];
        inst.transform = t;
        inst.instanceCustomIndex = i;
        inst.mask = 0xFF;
        inst.instanceShaderBindingTableRecordOffset = 0;
        inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        inst.accelerationStructureReference = blas.device_address;
      }

      std::memcpy(s.mapped, s.instances.data(), (size_t)instance_size);
      ctx->copy_buffer(s.staging, tlas_instance_buffer, instance_size);

      VkAccelerationStructureGeometryInstancesDataKHR instances_data{};
      instances_data.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
      instances_data.arrayOfPointers = VK_FALSE;
      instances_data.data.deviceAddress = get_buffer_address(tlas_instance_buffer);

      VkAccelerationStructureGeometryKHR geometry{};
      geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
      geometry.pNext = nullptr;
      geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
      geometry.geometry.instances = instances_data;
      geometry.flags = 0;

      VkAccelerationStructureBuildGeometryInfoKHR build_info{};
      build_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
      build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
      build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR |
        VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
      build_info.geometryCount = 1;
      build_info.pGeometries = &geometry;
      build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
      build_info.srcAccelerationStructure = tlas.handle;
      build_info.dstAccelerationStructure = tlas.handle;
      build_info.scratchData.deviceAddress = get_buffer_address(tlas_scratch_buffer);

      VkAccelerationStructureBuildRangeInfoKHR range_info{};
      range_info.primitiveCount = tlas_instance_count;
      VkAccelerationStructureBuildRangeInfoKHR* range_infos[] = { &range_info };

      VkCommandBuffer cmd = ctx->begin_single_time_commands();
      vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build_info,
        (const VkAccelerationStructureBuildRangeInfoKHR* const*)range_infos);
      ctx->end_single_time_commands(cmd);
    }

    void create_tlas() {
      tlas_instance_count = instance_count;
      std::vector<VkAccelerationStructureInstanceKHR> instances(instance_count);

      const f32_t spacing = 1.0f;
      const int dim = std::ceil(std::pow(instance_count, 1.0f/3.0f));
      const f32_t offset = spacing * dim / 2.0f;

      for (uint32_t i = 0; i < instance_count; ++i) {
        VkTransformMatrixKHR t{};
        t.matrix[0][0] = 1.0f; t.matrix[0][3] = (i % dim) * spacing - offset;
        t.matrix[1][1] = 1.0f; t.matrix[1][3] = ((i / dim) % dim) * spacing - offset;
        t.matrix[2][2] = 1.0f; t.matrix[2][3] = ((i / (dim*dim)) % dim) * spacing - offset;

        instances[i].transform = t;
        instances[i].instanceCustomIndex = i;
        instances[i].mask = 0xFF;
        instances[i].instanceShaderBindingTableRecordOffset = 0;
        instances[i].flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        instances[i].accelerationStructureReference = blas.device_address;
      }

      VkDeviceSize instance_size = sizeof(VkAccelerationStructureInstanceKHR) * instance_count;
      VkBuffer staging;
      VkDeviceMemory staging_mem;
      ctx->create_buffer(instance_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        staging, staging_mem);

      void* data;
      vkMapMemory(ctx->device, staging_mem, 0, instance_size, 0, &data);
      std::memcpy(data, instances.data(), (size_t)instance_size);
      vkUnmapMemory(ctx->device, staging_mem);

      ctx->create_buffer(instance_size,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, tlas_instance_buffer, tlas_instance_memory);

      ctx->copy_buffer(staging, tlas_instance_buffer, instance_size);
      vkDestroyBuffer(ctx->device, staging, nullptr);
      vkFreeMemory(ctx->device, staging_mem, nullptr);

      VkAccelerationStructureGeometryInstancesDataKHR instances_data{};
      instances_data.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
      instances_data.arrayOfPointers = VK_FALSE;
      instances_data.data.deviceAddress = get_buffer_address(tlas_instance_buffer);

      VkAccelerationStructureGeometryKHR geometry{};
      geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
      geometry.pNext = nullptr;
      geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
      geometry.geometry.instances = instances_data;
      geometry.flags = 0;

      VkAccelerationStructureBuildGeometryInfoKHR build_info{};
      build_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
      build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
      build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR |
        VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
      build_info.geometryCount = 1;
      build_info.pGeometries = &geometry;

      VkAccelerationStructureBuildSizesInfoKHR size_info{};
      size_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
      vkGetAccelerationStructureBuildSizesKHR(ctx->device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &build_info, &instance_count, &size_info);

      ctx->create_buffer(size_info.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, tlas.buffer, tlas.memory);

      VkAccelerationStructureCreateInfoKHR create_info{};
      create_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
      create_info.buffer = tlas.buffer;
      create_info.size = size_info.accelerationStructureSize;
      create_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
      fan::vulkan::validate(vkCreateAccelerationStructureKHR(ctx->device, &create_info, nullptr, &tlas.handle));

      ctx->create_buffer(size_info.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        tlas_scratch_buffer, tlas_scratch_memory);

      build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
      build_info.dstAccelerationStructure = tlas.handle;
      build_info.scratchData.deviceAddress = get_buffer_address(tlas_scratch_buffer);

      VkAccelerationStructureBuildRangeInfoKHR range_info{};
      range_info.primitiveCount = instance_count;
      const VkAccelerationStructureBuildRangeInfoKHR* range_infos = &range_info;

      VkCommandBuffer cmd = ctx->begin_single_time_commands();
      vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build_info, &range_infos);
      ctx->end_single_time_commands(cmd);

      VkAccelerationStructureDeviceAddressInfoKHR addr_info{};
      addr_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
      addr_info.accelerationStructure = tlas.handle;
      tlas.device_address = vkGetAccelerationStructureDeviceAddressKHR(ctx->device, &addr_info);
    }

    void create_output_image() {
      output_image = ctx->image_create();       
      auto& img = ctx->image_get(output_image); 

      VkImageCreateInfo image_info{};
      image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
      image_info.imageType = VK_IMAGE_TYPE_2D;
      image_info.format = VK_FORMAT_R8G8B8A8_UNORM;
      image_info.extent = { size.x, size.y, 1 };
      image_info.mipLevels = 1;
      image_info.arrayLayers = 1;
      image_info.samples = VK_SAMPLE_COUNT_1_BIT;
      image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
      image_info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

      fan::vulkan::image_create(*ctx, size, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_TILING_OPTIMAL,
        image_info.usage,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        img.image_index, img.image_memory);

      VkImageViewCreateInfo view_info{};
      view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      view_info.image = img.image_index;
      view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
      view_info.format = VK_FORMAT_R8G8B8A8_UNORM;
      view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      view_info.subresourceRange.baseMipLevel = 0;
      view_info.subresourceRange.levelCount = 1;
      view_info.subresourceRange.baseArrayLayer = 0;
      view_info.subresourceRange.layerCount = 1;
      fan::vulkan::validate(vkCreateImageView(ctx->device, &view_info, nullptr, &img.image_view));

      ctx->create_texture_sampler(img.sampler, {});

      VkCommandBuffer cmd = ctx->begin_single_time_commands();
      VkImageMemoryBarrier barrier{};
      barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
      barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.image = img.image_index;
      barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      barrier.subresourceRange.baseMipLevel = 0;
      barrier.subresourceRange.levelCount = 1;
      barrier.subresourceRange.baseArrayLayer = 0;
      barrier.subresourceRange.layerCount = 1;
      barrier.srcAccessMask = 0;
      barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;

      vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
      );
      ctx->end_single_time_commands(cmd);

      current_layout = VK_IMAGE_LAYOUT_GENERAL;
    }


    VkShaderModule load_shader(const char* path, shaderc_shader_kind kind) {
      std::string code = fan::graphics::read_shader(path);
      auto spirv = fan::vulkan::context_t::compile_file(path, kind, code);
      return ctx->create_shader_module(spirv);
    }

    void create_pipeline() {
      VkDescriptorSetLayoutBinding bindings[4]{};
      bindings[0].binding = 0;
      bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
      bindings[0].descriptorCount = 1;
      bindings[0].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

      bindings[1].binding = 1;
      bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      bindings[1].descriptorCount = 1;
      bindings[1].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

      bindings[2].binding = 2;
      bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      bindings[2].descriptorCount = 1;
      bindings[2].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

      bindings[3].binding = 3;
      bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      bindings[3].descriptorCount = 1;
      bindings[3].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_RAYGEN_BIT_KHR;

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

      VkShaderModule rgen = load_shader("shaders/vulkan/rt/raygen.rgen", shaderc_glsl_raygen_shader);
      VkShaderModule miss = load_shader("shaders/vulkan/rt/miss.rmiss", shaderc_glsl_miss_shader);
      VkShaderModule chit = load_shader("shaders/vulkan/rt/closesthit.rchit", shaderc_glsl_closesthit_shader);

      VkPipelineShaderStageCreateInfo stages[3]{};
      stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      stages[0].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
      stages[0].module = rgen;
      stages[0].pName = "main";

      stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      stages[1].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
      stages[1].module = miss;
      stages[1].pName = "main";

      stages[2].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      stages[2].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
      stages[2].module = chit;
      stages[2].pName = "main";

      VkRayTracingShaderGroupCreateInfoKHR groups[3]{};
      groups[0].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
      groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
      groups[0].generalShader = 0;
      groups[0].closestHitShader = VK_SHADER_UNUSED_KHR;
      groups[0].anyHitShader = VK_SHADER_UNUSED_KHR;
      groups[0].intersectionShader = VK_SHADER_UNUSED_KHR;

      groups[1].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
      groups[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
      groups[1].generalShader = 1;
      groups[1].closestHitShader = VK_SHADER_UNUSED_KHR;
      groups[1].anyHitShader = VK_SHADER_UNUSED_KHR;
      groups[1].intersectionShader = VK_SHADER_UNUSED_KHR;

      groups[2].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
      groups[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
      groups[2].generalShader = VK_SHADER_UNUSED_KHR;
      groups[2].closestHitShader = 2;
      groups[2].anyHitShader = VK_SHADER_UNUSED_KHR;
      groups[2].intersectionShader = VK_SHADER_UNUSED_KHR;

      VkRayTracingPipelineCreateInfoKHR pipeline_info{};
      pipeline_info.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
      pipeline_info.stageCount = 3;
      pipeline_info.pStages = stages;
      pipeline_info.groupCount = 3;
      pipeline_info.pGroups = groups;
      pipeline_info.maxPipelineRayRecursionDepth = 3;
      pipeline_info.layout = pipeline_layout;
      fan::vulkan::validate(vkCreateRayTracingPipelinesKHR(ctx->device, VK_NULL_HANDLE, VK_NULL_HANDLE,
        1, &pipeline_info, nullptr, &pipeline));

      vkDestroyShaderModule(ctx->device, rgen, nullptr);
      vkDestroyShaderModule(ctx->device, miss, nullptr);
      vkDestroyShaderModule(ctx->device, chit, nullptr);
    }

    void create_sbt() {
      VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt_props{};
      rt_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
      VkPhysicalDeviceProperties2 props{};
      props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
      props.pNext = &rt_props;
      vkGetPhysicalDeviceProperties2(ctx->physical_device, &props);

      const uint32_t handle_size = rt_props.shaderGroupHandleSize;
      const uint32_t handle_align = rt_props.shaderGroupHandleAlignment;
      const uint32_t base_align = rt_props.shaderGroupBaseAlignment;
      handle_size_aligned = (handle_size + handle_align - 1) & ~(handle_align - 1);

      const uint32_t group_count = 3;
      const uint32_t aligned_group_size = (handle_size_aligned + base_align - 1) & ~(base_align - 1);
      const uint32_t sbt_size = group_count * aligned_group_size;

      std::vector<uint8_t> handles(group_count * handle_size);
      fan::vulkan::validate(vkGetRayTracingShaderGroupHandlesKHR(ctx->device, pipeline, 0, 
        group_count, group_count * handle_size, handles.data()));

      ctx->create_buffer(sbt_size, VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, shader_binding_table, sbt_memory);

      ctx->create_buffer(sbt_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        sbt_staging, sbt_staging_memory);

      void* data;
      vkMapMemory(ctx->device, sbt_staging_memory, 0, sbt_size, 0, &data);
      uint8_t* sbt_ptr = static_cast<uint8_t*>(data);
      std::memset(sbt_ptr, 0, sbt_size);

      rgen_offset = 0;
      miss_offset = aligned_group_size;
      hit_offset = 2 * aligned_group_size;

      std::memcpy(sbt_ptr + rgen_offset, handles.data() + 0 * handle_size, handle_size);
      std::memcpy(sbt_ptr + miss_offset, handles.data() + 1 * handle_size, handle_size);
      std::memcpy(sbt_ptr + hit_offset, handles.data() + 2 * handle_size, handle_size);
      vkUnmapMemory(ctx->device, sbt_staging_memory);

      ctx->copy_buffer(sbt_staging, shader_binding_table, sbt_size);
      group_stride = aligned_group_size;
    }

    void create_descriptor_set() {
      VkDescriptorPoolSize pool_sizes[3]{};
      pool_sizes[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
      pool_sizes[0].descriptorCount = 1;
      pool_sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      pool_sizes[1].descriptorCount = 1;
      pool_sizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      pool_sizes[2].descriptorCount = 2;

      VkDescriptorPoolCreateInfo pool_info{};
      pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
      pool_info.maxSets = 1;
      pool_info.poolSizeCount = 3;
      pool_info.pPoolSizes = pool_sizes;
      fan::vulkan::validate(vkCreateDescriptorPool(ctx->device, &pool_info, nullptr, &descriptor_pool));

      VkDescriptorSetAllocateInfo alloc_info{};
      alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
      alloc_info.descriptorPool = descriptor_pool;
      alloc_info.descriptorSetCount = 1;
      alloc_info.pSetLayouts = &descriptor_layout;
      fan::vulkan::validate(vkAllocateDescriptorSets(ctx->device, &alloc_info, &descriptor_set));

      VkWriteDescriptorSetAccelerationStructureKHR as_info{};
      as_info.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
      as_info.accelerationStructureCount = 1;
      as_info.pAccelerationStructures = &tlas.handle;

      VkDescriptorImageInfo image_info{};
      image_info.imageView = ctx->image_get(output_image).image_view;
      image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

      VkDescriptorBufferInfo cam_info{};
      cam_info.buffer = camera_buffer;
      cam_info.range = sizeof(fan::vulkan::context_t::view_projection_t) * 16;

      VkDescriptorBufferInfo time_info{};
      time_info.buffer = time_buffer;
      time_info.range = sizeof(time_ubo_t);

      VkWriteDescriptorSet writes[4]{};
      writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[0].pNext = &as_info;
      writes[0].dstSet = descriptor_set;
      writes[0].dstBinding = 0;
      writes[0].descriptorCount = 1;
      writes[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

      writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[1].dstSet = descriptor_set;
      writes[1].dstBinding = 1;
      writes[1].descriptorCount = 1;
      writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      writes[1].pImageInfo = &image_info;

      writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[2].dstSet = descriptor_set;
      writes[2].dstBinding = 2;
      writes[2].descriptorCount = 1;
      writes[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      writes[2].pBufferInfo = &cam_info;

      writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[3].dstSet = descriptor_set;
      writes[3].dstBinding = 3;
      writes[3].descriptorCount = 1;
      writes[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      writes[3].pBufferInfo = &time_info;

      vkUpdateDescriptorSets(ctx->device, (uint32_t)std::size(writes), writes, 0, nullptr);
    }

    void update_camera_from_engine() {
      auto camera_handle = fan::graphics::get_perspective_render_view().camera;
      auto camera_data = ctx->camera_get(camera_handle);
      fan::vulkan::context_t::view_projection_t vp{};
      vp.view = camera_data.m_view;
      vp.projection = camera_data.m_projection;
      void* data;
      vkMapMemory(ctx->device, camera_memory, 0, sizeof(vp), 0, &data);
      std::memcpy(data, &vp, sizeof(vp));
      vkUnmapMemory(ctx->device, camera_memory);
    }

    void open(fan::vulkan::context_t& main_ctx, const fan::vec2ui& sz) {
      ctx = &main_ctx;
      size = sz;
      load_functions();

      ctx->create_buffer(sizeof(fan::vulkan::context_t::view_projection_t) * 16,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        camera_buffer, camera_memory);

      ctx->create_buffer(sizeof(time_ubo_t), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        time_buffer, time_memory);

      //static fan_3d::model::fms_t fms({.path = "models/player.gltf"});
      //shapes::triangle_mesh_t mesh(fms.meshes[0]);
      shapes::triangle_mesh_t mesh = shapes::make_sphere(fan::vec3(0), 0.1f);
      mesh_geom.upload(*ctx, mesh);


      create_blas();
      create_tlas();
      create_output_image();
      create_pipeline();
      create_sbt();
      create_descriptor_set();
      update_camera_from_engine();
    }

    void record_trace_rays(VkCommandBuffer cmd) {
      static fan::time::timer timer{true};
      time_ubo_t t{timer.seconds()};
      void* data;
      vkMapMemory(ctx->device, time_memory, 0, sizeof(t), 0, &data);
      std::memcpy(data, &t, sizeof(t));
      vkUnmapMemory(ctx->device, time_memory);

      if (current_layout != VK_IMAGE_LAYOUT_GENERAL) {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = current_layout;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = ctx->image_get(output_image).image_index;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
          VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        current_layout = VK_IMAGE_LAYOUT_GENERAL;
      }

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline_layout,
        0, 1, &descriptor_set, 0, nullptr);

      VkDeviceAddress sbt_addr = get_buffer_address(shader_binding_table);
      VkStridedDeviceAddressRegionKHR rgen_region{};
      rgen_region.deviceAddress = sbt_addr + rgen_offset;
      rgen_region.stride = group_stride;
      rgen_region.size = group_stride;

      VkStridedDeviceAddressRegionKHR miss_region{};
      miss_region.deviceAddress = sbt_addr + miss_offset;
      miss_region.stride = group_stride;
      miss_region.size = group_stride;

      VkStridedDeviceAddressRegionKHR hit_region{};
      hit_region.deviceAddress = sbt_addr + hit_offset;
      hit_region.stride = group_stride;
      hit_region.size = group_stride;

      VkStridedDeviceAddressRegionKHR callable_region{};

      vkCmdTraceRaysKHR(cmd, &rgen_region, &miss_region, &hit_region, &callable_region,
        size.x, size.y, 1);

      VkImageMemoryBarrier post_barrier{};
      post_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      post_barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
      post_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      post_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      post_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      post_barrier.image = ctx->image_get(output_image).image_index;
      post_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      post_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      post_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &post_barrier);
      current_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    void trace_rays_before_shapes() {
      //VkCommandBuffer cmd = ctx->begin_single_time_commands();
      record_trace_rays(ctx->command_buffers[ctx->current_frame]);
      //ctx->end_single_time_commands(cmd);
    }

    void close() {
      if (!ctx) return;
      vkDeviceWaitIdle(ctx->device);

      mesh_geom.destroy(*ctx);
      blas.destroy(*ctx);
      tlas.destroy(*ctx);

      if (pipeline) vkDestroyPipeline(ctx->device, pipeline, nullptr);
      if (pipeline_layout) vkDestroyPipelineLayout(ctx->device, pipeline_layout, nullptr);
      if (descriptor_layout) vkDestroyDescriptorSetLayout(ctx->device, descriptor_layout, nullptr);
      if (descriptor_pool) vkDestroyDescriptorPool(ctx->device, descriptor_pool, nullptr);
      if (shader_binding_table) vkDestroyBuffer(ctx->device, shader_binding_table, nullptr);
      if (sbt_memory) vkFreeMemory(ctx->device, sbt_memory, nullptr);
      if (sbt_staging) vkDestroyBuffer(ctx->device, sbt_staging, nullptr);
      if (sbt_staging_memory) vkFreeMemory(ctx->device, sbt_staging_memory, nullptr);
      if (camera_buffer) vkDestroyBuffer(ctx->device, camera_buffer, nullptr);
      if (camera_memory) vkFreeMemory(ctx->device, camera_memory, nullptr);
      if (time_buffer) vkDestroyBuffer(ctx->device, time_buffer, nullptr);
      if (time_memory) vkFreeMemory(ctx->device, time_memory, nullptr);
      if (tlas_instance_buffer) vkDestroyBuffer(ctx->device, tlas_instance_buffer, nullptr);
      if (tlas_instance_memory) vkFreeMemory(ctx->device, tlas_instance_memory, nullptr);
      if (tlas_scratch_buffer) vkDestroyBuffer(ctx->device, tlas_scratch_buffer, nullptr);
      if (tlas_scratch_memory) vkFreeMemory(ctx->device, tlas_scratch_memory, nullptr);

      pipeline = VK_NULL_HANDLE;
      pipeline_layout = VK_NULL_HANDLE;
      descriptor_layout = VK_NULL_HANDLE;
      descriptor_pool = VK_NULL_HANDLE;
      shader_binding_table = VK_NULL_HANDLE;
      sbt_memory = VK_NULL_HANDLE;
      sbt_staging = VK_NULL_HANDLE;
      sbt_staging_memory = VK_NULL_HANDLE;
      camera_buffer = VK_NULL_HANDLE;
      camera_memory = VK_NULL_HANDLE;
      time_buffer = VK_NULL_HANDLE;
      time_memory = VK_NULL_HANDLE;
      tlas_instance_buffer = VK_NULL_HANDLE;
      tlas_instance_memory = VK_NULL_HANDLE;
      tlas_scratch_buffer = VK_NULL_HANDLE;
      tlas_scratch_memory = VK_NULL_HANDLE;

      ctx = nullptr;
    }

    static constexpr uint32_t instance_count = 1000000;

    fan::vulkan::context_t* ctx = nullptr;
    fan::vec2ui size{};
    acceleration_structure_t blas;
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
    uint32_t rgen_offset = 0;
    uint32_t miss_offset = 0;
    uint32_t hit_offset = 0;
    uint32_t handle_size_aligned = 0;
    uint32_t group_stride = 0;
    fan::graphics::image_t output_image;
    VkImageLayout current_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkBuffer camera_buffer = VK_NULL_HANDLE;
    VkDeviceMemory camera_memory = VK_NULL_HANDLE;
    VkBuffer time_buffer = VK_NULL_HANDLE;
    VkDeviceMemory time_memory = VK_NULL_HANDLE;
    VkBuffer tlas_instance_buffer = VK_NULL_HANDLE;
    VkDeviceMemory tlas_instance_memory = VK_NULL_HANDLE;
    uint32_t tlas_instance_count = 0;
    VkBuffer tlas_scratch_buffer = VK_NULL_HANDLE;
    VkDeviceMemory tlas_scratch_memory = VK_NULL_HANDLE;
    shapes::gpu_mesh_t mesh_geom;

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