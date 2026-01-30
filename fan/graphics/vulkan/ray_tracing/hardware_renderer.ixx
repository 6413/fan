module;

#if defined(FAN_3D) && defined(FAN_VULKAN)

#include <vulkan/vulkan.h>
#include <shaderc/shaderc.hpp>
#include <vector>
#include <cstring>
#include <iostream>
#include <cmath>

#include <fstream>

#endif

export module fan.graphics.vulkan.ray_tracing.hardware_renderer;

#if defined(FAN_3D) && defined(FAN_VULKAN)

import fan.graphics.vulkan.core;
import fan.graphics;
import fan.graphics.vulkan.ray_tracing.shapes;
import fan.random;
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
#pragma pack(push, 1)
  struct material_info_t {
    int32_t albedo_texture_id = -1;
    int32_t normal_texture_id = -1;
    int32_t metallic_texture_id = -1;
    int32_t roughness_texture_id = -1;
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
      uint32_t first_vertex;
      uint32_t vertex_count;
      uint32_t first_index;
      uint32_t index_count;
    };
    struct model_t {
      uint32_t first_index;
      uint32_t index_count;
      uint32_t material_index;
      uint32_t first_primitive;
    };
    struct instance_t {
      uint32_t model_index;
      fan::mat4 transform;
    };
    std::vector<submesh_t> submeshes;
    std::vector<model_t> models;
    std::vector<instance_t> instances;
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
    void create_blas_for_model(uint32_t model_index) {
      const model_t& model = models[model_index];
      uint32_t first_index     = model.first_index;
      uint32_t index_count     = model.index_count;
      uint32_t primitive_count = index_count / 3;
      VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
      triangles.sType       = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
      triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
      triangles.vertexData.deviceAddress = get_buffer_address(vertex_buffer);
      triangles.vertexStride             = sizeof(vertex_t);
      triangles.maxVertex                = static_cast<uint32_t>(vertex_data.size()) - 1;
      triangles.indexType                = VK_INDEX_TYPE_UINT32;
      triangles.indexData.deviceAddress = get_buffer_address(index_buffer);
      VkAccelerationStructureGeometryKHR geometry{};
      geometry.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
      geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
      geometry.geometry.triangles = triangles;
      geometry.flags        = 0;
      VkAccelerationStructureBuildGeometryInfoKHR build_info{};
      build_info.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
      build_info.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
      build_info.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
      build_info.geometryCount = 1;
      build_info.pGeometries   = &geometry;
      VkAccelerationStructureBuildSizesInfoKHR size_info{};
      size_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
      vkGetAccelerationStructureBuildSizesKHR(
        ctx->device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &build_info,
        &primitive_count,
        &size_info
      );
      acceleration_structure_t blas;
      ctx->create_buffer(
        size_info.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        blas.buffer,
        blas.memory
      );
      VkAccelerationStructureCreateInfoKHR create_info{};
      create_info.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
      create_info.buffer = blas.buffer;
      create_info.size   = size_info.accelerationStructureSize;
      create_info.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
      fan::vulkan::validate(
        vkCreateAccelerationStructureKHR(ctx->device, &create_info, nullptr, &blas.handle)
      );
      VkBuffer scratch_buffer;
      VkDeviceMemory scratch_memory;
      ctx->create_buffer(
        size_info.buildScratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        scratch_buffer,
        scratch_memory
      );
      build_info.mode                    = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
      build_info.dstAccelerationStructure = blas.handle;
      build_info.scratchData.deviceAddress = get_buffer_address(scratch_buffer);
      VkAccelerationStructureBuildRangeInfoKHR range_info{};
      range_info.primitiveCount  = primitive_count;
      range_info.primitiveOffset = first_index * sizeof(uint32_t);
      range_info.firstVertex     = 0;
      range_info.transformOffset = 0;
      const VkAccelerationStructureBuildRangeInfoKHR* range_infos = &range_info;
      VkCommandBuffer cmd = ctx->begin_single_time_commands();
      vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build_info, &range_infos);
      ctx->end_single_time_commands(cmd);
      vkDestroyBuffer(ctx->device, scratch_buffer, nullptr);
      vkFreeMemory(ctx->device, scratch_memory, nullptr);
      VkAccelerationStructureDeviceAddressInfoKHR addr_info{};
      addr_info.sType                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
      addr_info.accelerationStructure = blas.handle;
      blas.device_address             = vkGetAccelerationStructureDeviceAddressKHR(ctx->device, &addr_info);
      blas_list.push_back(blas);
    }
    void create_tlas() {
      tlas_instance_count = (uint32_t)instances.size();
      if (tlas_instance_count == 0) {
        return;
      }
      std::vector<VkAccelerationStructureInstanceKHR> vk_instances(tlas_instance_count);
      for (uint32_t i = 0; i < tlas_instance_count; ++i) {
        const instance_t& inst = instances[i];
        const model_t& model = models[inst.model_index];
        VkTransformMatrixKHR t{};
        fan::mat4 m = inst.transform;
        t.matrix[0][0] = m[0][0]; t.matrix[0][1] = m[1][0]; t.matrix[0][2] = m[2][0]; t.matrix[0][3] = m[3][0];
        t.matrix[1][0] = m[0][1]; t.matrix[1][1] = m[1][1]; t.matrix[1][2] = m[2][1]; t.matrix[1][3] = m[3][1];
        t.matrix[2][0] = m[0][2]; t.matrix[2][1] = m[1][2]; t.matrix[2][2] = m[2][2]; t.matrix[2][3] = m[3][2];
        vk_instances[i].transform = t;
        vk_instances[i].instanceCustomIndex = model.first_primitive;
        vk_instances[i].mask = 0xFF;
        vk_instances[i].instanceShaderBindingTableRecordOffset = 0;
        vk_instances[i].flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        vk_instances[i].accelerationStructureReference = blas_list[inst.model_index].device_address;
      }
      VkDeviceSize instance_size = sizeof(VkAccelerationStructureInstanceKHR) * tlas_instance_count;
      VkBuffer staging;
      VkDeviceMemory staging_mem;
      ctx->create_buffer(
        instance_size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        staging,
        staging_mem
      );
      void* data;
      vkMapMemory(ctx->device, staging_mem, 0, instance_size, 0, &data);
      std::memcpy(data, vk_instances.data(), (size_t)instance_size);
      vkUnmapMemory(ctx->device, staging_mem);
      ctx->create_buffer(
        instance_size,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        tlas_instance_buffer,
        tlas_instance_memory
      );
      ctx->copy_buffer(staging, tlas_instance_buffer, instance_size);
      vkDestroyBuffer(ctx->device, staging, nullptr);
      vkFreeMemory(ctx->device, staging_mem, nullptr);
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
      build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
      build_info.geometryCount = 1;
      build_info.pGeometries = &geometry;
      VkAccelerationStructureBuildSizesInfoKHR size_info{};
      size_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
      vkGetAccelerationStructureBuildSizesKHR(
        ctx->device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &build_info,
        &tlas_instance_count,
        &size_info
      );
      ctx->create_buffer(
        size_info.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        tlas.buffer,
        tlas.memory
      );
      VkAccelerationStructureCreateInfoKHR create_info{};
      create_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
      create_info.buffer = tlas.buffer;
      create_info.size = size_info.accelerationStructureSize;
      create_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
      fan::vulkan::validate(vkCreateAccelerationStructureKHR(ctx->device, &create_info, nullptr, &tlas.handle));
      ctx->create_buffer(
        size_info.buildScratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        tlas_scratch_buffer,
        tlas_scratch_memory
      );
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
    void create_output_image() {
      output_image = ctx->image_create();       
      auto& img = ctx->image_get(output_image); 
      VkImageCreateInfo image_info{};
      image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
      image_info.imageType = VK_IMAGE_TYPE_2D;
      image_info.format = VK_FORMAT_R16G16B16A16_SFLOAT;
      image_info.extent = { size.x, size.y, 1 };
      image_info.mipLevels = 1;
      image_info.arrayLayers = 1;
      image_info.samples = VK_SAMPLE_COUNT_1_BIT;
      image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
      image_info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      fan::vulkan::image_create(*ctx, size, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_TILING_OPTIMAL,
        image_info.usage,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        img.image_index, img.image_memory);
      VkImageViewCreateInfo view_info{};
      view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      view_info.image = img.image_index;
      view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
      view_info.format = VK_FORMAT_R16G16B16A16_SFLOAT;
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
      VkDescriptorSetLayoutBinding bindings[11]{};
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
      bindings[4].binding = 4;
      bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      bindings[4].descriptorCount = max_textures;
      bindings[4].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
      bindings[5].binding = 5;
      bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[5].descriptorCount = 1;
      bindings[5].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
      bindings[6].binding = 6;
      bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[6].descriptorCount = 1;
      bindings[6].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
      bindings[7].binding = 7;
      bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[7].descriptorCount = 1;
      bindings[7].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
      bindings[8].binding = 8;
      bindings[8].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      bindings[8].descriptorCount = 1;
      bindings[8].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
      bindings[9].binding = 9;
      bindings[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[9].descriptorCount = 1;
      bindings[9].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
      bindings[10].binding = 10;
      bindings[10].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      bindings[10].descriptorCount = 1;
      bindings[10].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
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
      VkShaderModule rgen   = load_shader("shaders/vulkan/ray_tracing/raygen.rgen",           shaderc_glsl_raygen_shader);
      VkShaderModule miss   = load_shader("shaders/vulkan/ray_tracing/miss.rmiss",           shaderc_glsl_miss_shader);
      VkShaderModule chit   = load_shader("shaders/vulkan/ray_tracing/closesthit.rchit",     shaderc_glsl_closesthit_shader);
      VkShaderModule shadow = load_shader("shaders/vulkan/ray_tracing/shadow.rmiss",         shaderc_glsl_miss_shader);
      VkShaderModule shany  = load_shader("shaders/vulkan/ray_tracing/shadow_anyhit.rahit",  shaderc_glsl_anyhit_shader);
      VkPipelineShaderStageCreateInfo stages[5]{};
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
      stages[3].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      stages[3].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
      stages[3].module = shadow;
      stages[3].pName = "main";
      stages[4].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      stages[4].stage = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
      stages[4].module = shany;
      stages[4].pName = "main";
      VkRayTracingShaderGroupCreateInfoKHR groups[4]{};
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
      groups[2].anyHitShader = 4;
      groups[2].intersectionShader = VK_SHADER_UNUSED_KHR;
      groups[3].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
      groups[3].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
      groups[3].generalShader = 3;
      groups[3].closestHitShader = VK_SHADER_UNUSED_KHR;
      groups[3].anyHitShader = VK_SHADER_UNUSED_KHR;
      groups[3].intersectionShader = VK_SHADER_UNUSED_KHR;
      VkRayTracingPipelineCreateInfoKHR pipeline_info{};
      pipeline_info.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
      pipeline_info.stageCount = 5;
      pipeline_info.pStages = stages;
      pipeline_info.groupCount = 4;
      pipeline_info.pGroups = groups;
      pipeline_info.maxPipelineRayRecursionDepth = 3;
      pipeline_info.layout = pipeline_layout;
      fan::vulkan::validate(
        vkCreateRayTracingPipelinesKHR(
          ctx->device,
          VK_NULL_HANDLE,
          VK_NULL_HANDLE,
          1,
          &pipeline_info,
          nullptr,
          &pipeline
        )
      );
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
      const uint32_t handle_size  = rt_props.shaderGroupHandleSize;
      const uint32_t handle_align = rt_props.shaderGroupHandleAlignment;
      const uint32_t base_align   = rt_props.shaderGroupBaseAlignment;
      handle_size_aligned = (handle_size + handle_align - 1) & ~(handle_align - 1);
      const uint32_t group_count      = 4;
      const uint32_t aligned_group_sz = (handle_size_aligned + base_align - 1) & ~(base_align - 1);
      const uint32_t sbt_size         = group_count * aligned_group_sz;
      std::vector<uint8_t> handles(group_count * handle_size);
      fan::vulkan::validate(
        vkGetRayTracingShaderGroupHandlesKHR(
          ctx->device,
          pipeline,
          0,
          group_count,
          group_count * handle_size,
          handles.data()
        )
      );
      ctx->create_buffer(
        sbt_size,
        VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT    |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        shader_binding_table,
        sbt_memory
      );
      ctx->create_buffer(
        sbt_size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        sbt_staging,
        sbt_staging_memory
      );
      void* data;
      vkMapMemory(ctx->device, sbt_staging_memory, 0, sbt_size, 0, &data);
      uint8_t* sbt_ptr = static_cast<uint8_t*>(data);
      std::memset(sbt_ptr, 0, sbt_size);
      group_stride = aligned_group_sz;
      rgen_offset        = 0;
      miss_offset        = 1 * aligned_group_sz;
      shadow_miss_offset = 2 * aligned_group_sz;
      hit_offset         = 3 * aligned_group_sz;
      shadow_hit_offset  = 4 * aligned_group_sz;
      std::memcpy(sbt_ptr + rgen_offset,        handles.data() + 0 * handle_size, handle_size);
      std::memcpy(sbt_ptr + miss_offset,        handles.data() + 1 * handle_size, handle_size);
      std::memcpy(sbt_ptr + shadow_miss_offset, handles.data() + 3 * handle_size, handle_size);
      std::memcpy(sbt_ptr + hit_offset,         handles.data() + 2 * handle_size, handle_size);
      vkUnmapMemory(ctx->device, sbt_staging_memory);
      ctx->copy_buffer(sbt_staging, shader_binding_table, sbt_size);
      VkDeviceAddress sbt_addr = get_buffer_address(shader_binding_table);
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
      pool_sizes[4].descriptorCount = 3;
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
      VkDescriptorBufferInfo material_info{};
      material_info.buffer = material_buffer;
      material_info.range = VK_WHOLE_SIZE;
      VkDescriptorBufferInfo vertex_info{};
      vertex_info.buffer = vertex_buffer;
      vertex_info.range = VK_WHOLE_SIZE;
      VkDescriptorBufferInfo index_info{};
      index_info.buffer = index_buffer;
      index_info.range = VK_WHOLE_SIZE;
      std::vector<VkDescriptorImageInfo> texture_infos(max_textures);
      auto& dummy_img = ctx->image_get(fan::graphics::ctx().default_texture);
      for (uint32_t i = 0; i < max_textures; i++) {
        if (i < rt_texture_infos.size()) {
          texture_infos[i] = rt_texture_infos[i];
        }
        else {
          texture_infos[i].sampler = dummy_img.sampler;
          texture_infos[i].imageView = dummy_img.image_view;
          texture_infos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
      }
      VkWriteDescriptorSet writes[10]{};
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
      writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[4].dstSet = descriptor_set;
      writes[4].dstBinding = 4;
      writes[4].descriptorCount = texture_infos.size();
      writes[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      writes[4].pImageInfo = texture_infos.data();
      writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[5].dstSet = descriptor_set;
      writes[5].dstBinding = 5;
      writes[5].descriptorCount = 1;
      writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[5].pBufferInfo = &material_info;
      writes[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[6].dstSet = descriptor_set;
      writes[6].dstBinding = 6;
      writes[6].descriptorCount = 1;
      writes[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[6].pBufferInfo = &vertex_info;
      writes[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[7].dstSet = descriptor_set;
      writes[7].dstBinding = 7;
      writes[7].descriptorCount = 1;
      writes[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[7].pBufferInfo = &index_info;
      VkDescriptorBufferInfo light_info{};
      light_info.buffer = light_buffer;
      light_info.range = sizeof(light_ubo_t);
      writes[8].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[8].dstSet = descriptor_set;
      writes[8].dstBinding = 8;
      writes[8].descriptorCount = 1;
      writes[8].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      writes[8].pBufferInfo = &light_info;
      VkDescriptorBufferInfo mat_idx_info{};
      mat_idx_info.buffer = material_index_buffer;
      mat_idx_info.range = VK_WHOLE_SIZE;
      writes[9].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[9].dstSet = descriptor_set;
      writes[9].dstBinding = 9;
      writes[9].descriptorCount = 1;
      writes[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[9].pBufferInfo = &mat_idx_info;
      vkUpdateDescriptorSets(ctx->device, (uint32_t)std::size(writes), writes, 0, nullptr);
      {
        VkDescriptorBufferInfo exposure_info{};
        exposure_info.buffer = exposure_ubo;
        exposure_info.range = sizeof(float) * 4;
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = descriptor_set;
        write.dstBinding = 10;
        write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        write.descriptorCount = 1;
        write.pBufferInfo = &exposure_info;
        vkUpdateDescriptorSets(ctx->device, 1, &write, 0, nullptr);
      }
    }
    void update_rt_textures_descriptor() {
      std::vector<VkDescriptorImageInfo> infos(max_textures);
      auto& dummy = ctx->image_get(fan::graphics::ctx().default_texture);
      for (uint32_t i = 0; i < max_textures; i++) {
        if (i < rt_texture_infos.size()) {
          infos[i] = rt_texture_infos[i];
        }
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
    void create_vertex_buffer() {
      VkDeviceSize size = sizeof(vertex_t) * vertex_data.size();
      VkBuffer staging;
      VkDeviceMemory staging_mem;
      ctx->create_buffer(
        size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        staging,
        staging_mem
      );
      void* data;
      vkMapMemory(ctx->device, staging_mem, 0, size, 0, &data);
      memcpy(data, vertex_data.data(), (size_t)size);
      vkUnmapMemory(ctx->device, staging_mem);
      ctx->create_buffer(
        size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        vertex_buffer,
        vertex_memory
      );
      ctx->copy_buffer(staging, vertex_buffer, size);
      vkDestroyBuffer(ctx->device, staging, nullptr);
      vkFreeMemory(ctx->device, staging_mem, nullptr);
    }
    void create_index_buffer() {
      VkDeviceSize size = sizeof(uint32_t) * index_data.size();
      VkBuffer staging;
      VkDeviceMemory staging_mem;
      ctx->create_buffer(
        size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        staging,
        staging_mem
      );
      void* data;
      vkMapMemory(ctx->device, staging_mem, 0, size, 0, &data);
      memcpy(data, index_data.data(), (size_t)size);
      vkUnmapMemory(ctx->device, staging_mem);
      ctx->create_buffer(
        size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        index_buffer,
        index_memory
      );
      ctx->copy_buffer(staging, index_buffer, size);
      vkDestroyBuffer(ctx->device, staging, nullptr);
      vkFreeMemory(ctx->device, staging_mem, nullptr);
    }
    void create_material_buffer() {
      VkDeviceSize size = sizeof(material_info_t) * materials.size();
      VkBuffer staging;
      VkDeviceMemory staging_mem;
      ctx->create_buffer(
        size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        staging,
        staging_mem
      );
      void* data;
      vkMapMemory(ctx->device, staging_mem, 0, size, 0, &data);
      memcpy(data, materials.data(), (size_t)size);
      vkUnmapMemory(ctx->device, staging_mem);
      ctx->create_buffer(
        size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        material_buffer,
        material_memory
      );
      ctx->copy_buffer(staging, material_buffer, size);
      vkDestroyBuffer(ctx->device, staging, nullptr);
      vkFreeMemory(ctx->device, staging_mem, nullptr);
    }
    void create_material_index_buffer() {
      if (material_indices_per_primitive.empty()) {
        return;
      }
      VkDeviceSize size = sizeof(uint32_t) * material_indices_per_primitive.size();
      VkBuffer staging;
      VkDeviceMemory staging_mem;
      ctx->create_buffer(
        size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        staging,
        staging_mem
      );
      void* data;
      vkMapMemory(ctx->device, staging_mem, 0, size, 0, &data);
      std::memcpy(data, material_indices_per_primitive.data(), (size_t)size);
      vkUnmapMemory(ctx->device, staging_mem);
      if (material_index_buffer) {
        vkDestroyBuffer(ctx->device, material_index_buffer, nullptr);
        vkFreeMemory(ctx->device, material_index_memory, nullptr);
      }
      ctx->create_buffer(
        size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        material_index_buffer,
        material_index_memory
      );
      ctx->copy_buffer(staging, material_index_buffer, size);
      vkDestroyBuffer(ctx->device, staging, nullptr);
      vkFreeMemory(ctx->device, staging_mem, nullptr);
    }
    void load_model_from_fms(fan::model::fms_t& fms) {
      for (uint32_t mesh_idx = 0; mesh_idx < fms.meshes.size(); mesh_idx++) {
        const auto& src_mesh = fms.meshes[mesh_idx];
        model_t model{};
        model.first_index = (uint32_t)index_data.size();
        model.index_count = (uint32_t)src_mesh.indices.size();
        uint32_t first_vertex = (uint32_t)vertex_data.size();
        for (const auto& v : src_mesh.vertices) {
          vertex_t out{};
          out.position = v.position;
          out.normal   = v.normal;
          out.texcoord = v.uv;
          vertex_data.push_back(out);
        }
        for (uint32_t idx : src_mesh.indices) {
          index_data.push_back(idx + first_vertex);
        }
        material_info_t mat;
        mat.base_color            = fan::vec3(1,1,1);
        if (mesh_idx < fms.material_data_vector.size()) {
          const auto& md = fms.material_data_vector[mesh_idx];
          fan::vec4 c = md.color[fan::texture_type::base_color];
          if (c.x == 1 && c.y == 1 && c.z == 1 && c.w == 1) {
            c = md.color[fan::texture_type::diffuse];
          }
          mat.base_color = fan::vec3(c.x, c.y, c.z);
        }
        auto load_rt_texture = [&](const std::string& name) -> int32_t {
          if (name.empty()) return -1;
          auto it = fan::model::cached_texture_data.find(name);
          if (it == fan::model::cached_texture_data.end()) return -1;
          const auto& td = it->second;
          if (td.data.empty()) return -1;
          fan::image::info_t ii;
          ii.data     = (void*)td.data.data();
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
          return (int32_t)rt_texture_infos.size() - 1;
        };
        {
          const std::string& tn = src_mesh.texture_names[fan::texture_type::diffuse];
          if (!tn.empty()) {
            int32_t slot = load_rt_texture(tn);
            if (slot >= 0) {
              mat.albedo_texture_id = slot;
            }
          }
        }
        {
          const std::string& tn = src_mesh.texture_names[fan::texture_type::normals];
          if (!tn.empty()) {
            int32_t slot = load_rt_texture(tn);
            if (slot >= 0) {
              mat.normal_texture_id = slot;
            }
          }
        }
        {
          const std::string& tn = src_mesh.texture_names[fan::texture_type::metalness];
          if (!tn.empty()) {
            int32_t slot = load_rt_texture(tn);
            if (slot >= 0) {
              mat.metallic_texture_id = slot;
            }
          }
        }
        {
          const std::string& tn = src_mesh.texture_names[fan::texture_type::diffuse_roughness];
          if (!tn.empty()) {
            int32_t slot = load_rt_texture(tn);
            if (slot >= 0) {
              mat.roughness_texture_id = slot;
            }
          }
        }
        model.material_index = (uint32_t)materials.size();
        materials.push_back(mat);
        uint32_t mesh_first_index   = model.first_index;
        uint32_t mesh_index_count   = model.index_count;
        uint32_t mesh_primitive_cnt = mesh_index_count / 3;
        model.first_primitive       = mesh_first_index / 3;
        uint32_t needed_prim = model.first_primitive + mesh_primitive_cnt;
        if (material_indices_per_primitive.size() < needed_prim) {
          material_indices_per_primitive.resize(needed_prim);
        }
        for (uint32_t p = 0; p < mesh_primitive_cnt; ++p) {
          material_indices_per_primitive[model.first_primitive + p] = model.material_index;
        }
        models.push_back(model);
      }
    }
    void add_instance(uint32_t model_index, const fan::mat4& transform) {
      instance_t inst{};
      inst.model_index = model_index;
      inst.transform = transform;
      instances.push_back(inst);
    }
    void add_model(const std::string& path, const fan::mat4& transform) {
      fan::model::fms_t fms({ .path = path });
      uint32_t first_model = (uint32_t)models.size();
      load_model_from_fms(fms);
      uint32_t last_model  = (uint32_t)models.size();
      for (uint32_t i = first_model; i < last_model; ++i) {
        instance_t inst{};
        inst.model_index = i;
        inst.transform   = transform;
        instances.push_back(inst);
      }
    }
    void update_camera_from_engine(){
      auto camera_handle = fan::graphics::get_perspective_render_view().camera;
      auto camera_data = ctx->camera_get(camera_handle);
      void* data;
      fan::vulkan::context_t::view_projection_t vp{};
      vp.projection = camera_data.m_projection;
      vp.view = camera_data.m_view;
      vkMapMemory(ctx->device, camera_memory, 0, sizeof(vp), 0, &data);
      memcpy(data, &vp, sizeof(vp));
      vkUnmapMemory(ctx->device, camera_memory);
    }
    void open(fan::vulkan::context_t& main_ctx, const fan::vec2ui& sz) {
      ctx = &main_ctx;
      size = sz;
      load_functions();
      ctx->create_buffer(
        sizeof(fan::vulkan::context_t::view_projection_t) * 16,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        camera_buffer,
        camera_memory
      );
      ctx->create_buffer(
        sizeof(time_ubo_t),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        time_buffer,
        time_memory
      );
      ctx->create_buffer(
        sizeof(light_ubo_t),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        light_buffer,
        light_memory
      );
      create_exposure_ubo(); 
      create_luminance_buffer(size.x, size.y);
      
      
      add_model("models/NewSponza_Main_glTF_003.gltf", fan::mat4(1).scale(0.1f).translate(fan::vec3(0.1, -0.1, 0.1)));
    /*  add_model("models/Fox.glb", fan::mat4(1).scale(0.001f).translate(fan::vec3(0.1, -0.1, 0.1)));
      for (int i = 0; i < 100000; ++i) {
        add_instance(0, fan::mat4(1).scale(0.001f).translate(fan::vec3(fan::random::value(0.0, 10.0) - 5.0, fan::random::value(0.0, 10.0), fan::random::value(0.0, 10.0) - 5.0)).rotate(fan::random::vec3(-fan::math::two_pi, fan::math::two_pi)));
      }
      add_model("models/floor.fbx", fan::mat4(1).scale(0.001f).translate(fan::vec3(0.1, -0.1, 0.1)));
      add_model("models/wall.fbx", fan::mat4(1).scale(0.001f).translate(fan::vec3(0.1, -0.1, 0.1)));
      fan::print("models:", models.size(), "instances:", instances.size(),
        "materials:", materials.size(),
        "material_indices_per_primitive:", material_indices_per_primitive.size());*/

      //auto mesh = ray_tracing::shapes::make_sphere(0, 1);
      //mesh_geometries.resize(mesh_geometries.size() + 1);
      //mesh_geometries.back().upload(*ctx, mesh);
     /* add_model("models/sphere_optimized.fbx", fan::mat4(1).scale(0.0001f));
      for (int i = 0; i < 500; ++i) {
        for (int j = 0; j < 500; ++j) {
          for (int k = 0; k < 500; ++k) {
            add_instance(0, fan::mat4(1).scale(0.0007f).translate(fan::vec3(i, j, k)));
          }
        }
      }*/
      //add_instance(0, fan::mat4(1).translate(fan::vec3(0, 0, 0)));
      create_vertex_buffer();
      create_index_buffer();
      create_material_buffer();
      create_material_index_buffer();
      for (uint32_t i = 0; i < models.size(); i++) {
        create_blas_for_model(i);
      }
      for (size_t i = 0; i < materials.size(); i++) {
        const auto& m = materials[i];
        fan::print("material", i,
          "albedo_texture_id", m.albedo_texture_id,
          "base_color", m.base_color.x, m.base_color.y, m.base_color.z);
      }
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
      update_camera_from_engine();
      update_rt_textures_descriptor();
      frame_index = 0;
    }
    void record_trace_rays(VkCommandBuffer cmd) {
      static fan::time::timer timer{ true };
      time_ubo_t t{ timer.seconds() };
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
        barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;   
        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;  
        vkCmdPipelineBarrier(
          cmd,
          VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
          VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
          0,
          0, nullptr,
          0, nullptr,
          1, &barrier
        );
        current_layout = VK_IMAGE_LAYOUT_GENERAL;
      }
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);
      vkCmdBindDescriptorSets(
        cmd,
        VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
        pipeline_layout,
        0, 1, &descriptor_set,
        0, nullptr
      );
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
      vkCmdTraceRaysKHR(
        cmd,
        &rgen_region,
        &miss_region,
        &hit_region,
        &callable_region,
        size.x, size.y, 1
      );
      {
        VkImageMemoryBarrier lum_barrier{};
        lum_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        lum_barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        lum_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        lum_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        lum_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        lum_barrier.image = ctx->image_get(output_image).image_index;
        lum_barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        lum_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; 
        lum_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;  
        vkCmdPipelineBarrier(
          cmd,
          VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          0,
          0, nullptr,
          0, nullptr,
          1, &lum_barrier
        );
        current_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      }
      dispatch_luminance_compute(cmd, luminance_descriptor_set, size.x, size.y);
      {
        VkImageMemoryBarrier post_barrier{};
        post_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        post_barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        post_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        post_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        post_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        post_barrier.image = ctx->image_get(output_image).image_index;
        post_barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        post_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; 
        post_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;  
        vkCmdPipelineBarrier(
          cmd,
          VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          0,
          0, nullptr,
          0, nullptr,
          1, &post_barrier
        );
        current_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      }
      if (accum_layout != VK_IMAGE_LAYOUT_GENERAL) {
        VkImageMemoryBarrier acc_back{};
        acc_back.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        acc_back.oldLayout = accum_layout;
        acc_back.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        acc_back.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        acc_back.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        acc_back.image = ctx->image_get(accum_image).image_index;
        acc_back.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        acc_back.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;   
        acc_back.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;  
        vkCmdPipelineBarrier(
          cmd,
          VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          0,
          0, nullptr,
          0, nullptr,
          1, &acc_back
        );
        accum_layout = VK_IMAGE_LAYOUT_GENERAL;
      }
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, accum_pipeline);
      vkCmdBindDescriptorSets(
        cmd,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        accum_pipeline_layout,
        0, 1, &accum_descriptor_set,
        0, nullptr
      );
      vkCmdPushConstants(
        cmd,
        accum_pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(uint32_t),
        &frame_index
      );
      uint32_t gx = (size.x + 7) / 8;
      uint32_t gy = (size.y + 7) / 8;
      vkCmdDispatch(cmd, gx, gy, 1);
      {
        VkImageMemoryBarrier acc_to_sample{};
        acc_to_sample.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        acc_to_sample.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        acc_to_sample.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        acc_to_sample.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        acc_to_sample.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        acc_to_sample.image = ctx->image_get(accum_image).image_index;
        acc_to_sample.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        acc_to_sample.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; 
        acc_to_sample.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;  
        vkCmdPipelineBarrier(
          cmd,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
          0,
          0, nullptr,
          0, nullptr,
          1, &acc_to_sample
        );
        accum_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      }
    }
    void on_camera_updated(bool camera_moved) {
      if (camera_moved) {
        frame_index = 0;
      } else {
        frame_index++;
      }
    }
    void trace_rays_before_shapes(){
      record_trace_rays(ctx->command_buffers[ctx->current_frame]);
    }
    void create_accum_image() {
      accum_image = ctx->image_create();
      auto& img = ctx->image_get(accum_image);
      VkImageCreateInfo image_info{};
      image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
      image_info.imageType = VK_IMAGE_TYPE_2D;
      image_info.format = VK_FORMAT_R16G16B16A16_SFLOAT; 
      image_info.extent = { size.x, size.y, 1 };
      image_info.mipLevels = 1;
      image_info.arrayLayers = 1;
      image_info.samples = VK_SAMPLE_COUNT_1_BIT;
      image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
      image_info.usage =
        VK_IMAGE_USAGE_STORAGE_BIT |
        VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      fan::vulkan::image_create(
        *ctx,
        size,
        image_info.format,
        VK_IMAGE_TILING_OPTIMAL,
        image_info.usage,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        img.image_index,
        img.image_memory
      );
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
      fan::vulkan::validate(
        vkCreateImageView(ctx->device, &view_info, nullptr, &img.image_view)
      );
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
      barrier.dstAccessMask =
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR |
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
      );
      VkClearColorValue clear_color{};
      clear_color.float32[0] = 0.0f;
      clear_color.float32[1] = 0.0f;
      clear_color.float32[2] = 0.0f;
      clear_color.float32[3] = 1.0f;
      VkImageSubresourceRange range{};
      range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      range.baseMipLevel = 0;
      range.levelCount = 1;
      range.baseArrayLayer = 0;
      range.layerCount = 1;
      vkCmdClearColorImage(
        cmd,
        img.image_index,
        VK_IMAGE_LAYOUT_GENERAL,
        &clear_color,
        1,
        &range
      );
      ctx->end_single_time_commands(cmd);
      accum_layout = VK_IMAGE_LAYOUT_GENERAL;
    }
    void create_accum_pipeline() {
      VkDescriptorSetLayoutBinding bindings[2]{};
      bindings[0].binding = 0;
      bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      bindings[0].descriptorCount = 1;
      bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      bindings[1].binding = 1;
      bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      bindings[1].descriptorCount = 1;
      bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      VkDescriptorSetLayoutCreateInfo layout_info{};
      layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
      layout_info.bindingCount = 2;
      layout_info.pBindings = bindings;
      fan::vulkan::validate(
        vkCreateDescriptorSetLayout(ctx->device, &layout_info, nullptr, &accum_descriptor_layout)
      );
      VkPushConstantRange pcr{};
      pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      pcr.offset = 0;
      pcr.size = sizeof(uint32_t);
      VkPipelineLayoutCreateInfo pl_info{};
      pl_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
      pl_info.setLayoutCount = 1;
      pl_info.pSetLayouts = &accum_descriptor_layout;
      pl_info.pushConstantRangeCount = 1;
      pl_info.pPushConstantRanges = &pcr;
      fan::vulkan::validate(
        vkCreatePipelineLayout(ctx->device, &pl_info, nullptr, &accum_pipeline_layout)
      );
      VkShaderModule comp = load_shader(
        "shaders/vulkan/ray_tracing/accumulate.comp",
        shaderc_glsl_compute_shader
      );
      VkPipelineShaderStageCreateInfo stage{};
      stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
      stage.module = comp;
      stage.pName = "main";
      VkComputePipelineCreateInfo pi{};
      pi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
      pi.stage = stage;
      pi.layout = accum_pipeline_layout;
      fan::vulkan::validate(
        vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &pi, nullptr, &accum_pipeline)
      );
      vkDestroyShaderModule(ctx->device, comp, nullptr);
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
      fan::vulkan::validate(
        vkCreateDescriptorPool(ctx->device, &pool_info, nullptr, &accum_descriptor_pool)
      );
      VkDescriptorSetAllocateInfo alloc_info{};
      alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
      alloc_info.descriptorPool = accum_descriptor_pool;
      alloc_info.descriptorSetCount = 1;
      alloc_info.pSetLayouts = &accum_descriptor_layout;
      fan::vulkan::validate(
        vkAllocateDescriptorSets(ctx->device, &alloc_info, &accum_descriptor_set)
      );
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
      bindings[0].binding = 0;
      bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      bindings[0].descriptorCount = 1;
      bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      bindings[1].binding = 1;
      bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[1].descriptorCount = 1;
      bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      VkDescriptorSetLayoutCreateInfo layout_info{};
      layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
      layout_info.bindingCount = 2;
      layout_info.pBindings = bindings;
      fan::vulkan::validate(
        vkCreateDescriptorSetLayout(ctx->device, &layout_info, nullptr, &luminance_descriptor_layout)
      );
      VkPushConstantRange pcr{};
      pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      pcr.offset = 0;
      pcr.size = sizeof(int) * 4; 
      VkPipelineLayoutCreateInfo pl_info{};
      pl_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
      pl_info.setLayoutCount = 1;
      pl_info.pSetLayouts = &luminance_descriptor_layout;
      pl_info.pushConstantRangeCount = 1;
      pl_info.pPushConstantRanges = &pcr;
      fan::vulkan::validate(
        vkCreatePipelineLayout(ctx->device, &pl_info, nullptr, &luminance_pipeline_layout)
      );
      VkShaderModule comp = load_shader(
        "shaders/vulkan/ray_tracing/luminance_reduce.comp",
        shaderc_glsl_compute_shader
      );
      VkPipelineShaderStageCreateInfo stage{};
      stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
      stage.module = comp;
      stage.pName = "main";
      VkComputePipelineCreateInfo pi{};
      pi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
      pi.stage = stage;
      pi.layout = luminance_pipeline_layout;
      fan::vulkan::validate(
        vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &pi, nullptr, &luminance_pipeline)
      );
      vkDestroyShaderModule(ctx->device, comp, nullptr);
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
      fan::vulkan::validate(
        vkCreateDescriptorPool(ctx->device, &pool_info, nullptr, &luminance_descriptor_pool)
      );
      VkDescriptorSetAllocateInfo alloc_info{};
      alloc_info.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
      alloc_info.descriptorPool     = luminance_descriptor_pool;
      alloc_info.descriptorSetCount = 1;
      alloc_info.pSetLayouts        = &luminance_descriptor_layout;
      fan::vulkan::validate(
        vkAllocateDescriptorSets(ctx->device, &alloc_info, &luminance_descriptor_set)
      );
      auto& hdr_img = ctx->image_get(output_image);
      VkDescriptorImageInfo hdr_info{};
      hdr_info.sampler     = hdr_img.sampler;
      hdr_info.imageView   = hdr_img.image_view;
      hdr_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL; 
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
    void create_luminance_buffer(uint32_t width, uint32_t height) {
      uint32_t groupsX = (width  + 15) / 16;
      uint32_t groupsY = (height + 15) / 16;
      luminance_group_count = groupsX * groupsY;
      VkDeviceSize size = sizeof(float) * luminance_group_count;
      ctx->create_buffer(
        size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        luminance_buffer,
        luminance_memory
      );
    }
    void dispatch_luminance_compute(VkCommandBuffer cmd, VkDescriptorSet luminance_set,
      uint32_t width, uint32_t height) 
    {
      uint32_t groupsX = (width  + 15) / 16;
      uint32_t groupsY = (height + 15) / 16;
      vkCmdFillBuffer(cmd, luminance_buffer, 0, sizeof(float) * luminance_group_count, 0);
      VkBufferMemoryBarrier clearBarrier{};
      clearBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
      clearBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      clearBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      clearBarrier.buffer = luminance_buffer;
      clearBarrier.offset = 0;
      clearBarrier.size = VK_WHOLE_SIZE;
      vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0, nullptr,
        1, &clearBarrier,
        0, nullptr
      );
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, luminance_pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        luminance_pipeline_layout, 0, 1, &luminance_set, 0, nullptr);
      struct PC { int w, h, gx, gy; } pc;
      pc.w = width;
      pc.h = height;
      pc.gx = groupsX;
      pc.gy = groupsY;
      vkCmdPushConstants(cmd, luminance_pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
      vkCmdDispatch(cmd, groupsX, groupsY, 1);
    }
    void update_exposure(float dt) {
      std::vector<float> partial(luminance_group_count);
      void* data;
      vkMapMemory(ctx->device, luminance_memory, 0,
        sizeof(float) * luminance_group_count, 0, &data);
      std::memcpy(partial.data(), data, sizeof(float) * luminance_group_count);
      vkUnmapMemory(ctx->device, luminance_memory);
      double total = 0.0;
      for (float v : partial) total += v;
      double avgL = total / double(size.x * size.y);
      if (avgL < 1e-6) avgL = 1e-6;
      float middle_gray = 0.06f;
      target_exposure = middle_gray / float(avgL);
      float lambda = 1.0f - std::exp(-adaptation_speed * dt);
      exposure = exposure + lambda * (target_exposure - exposure);
      exposure = std::clamp(exposure, 0.0001f, 20.0f);
      struct ExposureUBO { float exposure, pad0, pad1, pad2; } ubo;
      ubo.exposure = exposure;
      void* mapped;
      vkMapMemory(ctx->device, exposure_ubo_memory, 0, sizeof(ubo), 0, &mapped);
      std::memcpy(mapped, &ubo, sizeof(ubo));
      vkUnmapMemory(ctx->device, exposure_ubo_memory);
    }
    void create_exposure_ubo() {
      ctx->create_buffer(
        sizeof(float) * 4,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        exposure_ubo,
        exposure_ubo_memory
      );
    }
    void close(){
      if(!ctx) return;
      vkDeviceWaitIdle(ctx->device);
      for(auto& geom : mesh_geometries){
        geom.destroy(*ctx);
      }
      for(auto& blas : blas_list){
        blas.destroy(*ctx);
      }
      tlas.destroy(*ctx);
      if(pipeline) vkDestroyPipeline(ctx->device, pipeline, nullptr);
      if(pipeline_layout) vkDestroyPipelineLayout(ctx->device, pipeline_layout, nullptr);
      if(descriptor_layout) vkDestroyDescriptorSetLayout(ctx->device, descriptor_layout, nullptr);
      if(descriptor_pool) vkDestroyDescriptorPool(ctx->device, descriptor_pool, nullptr);
      if(shader_binding_table) vkDestroyBuffer(ctx->device, shader_binding_table, nullptr);
      if(sbt_memory) vkFreeMemory(ctx->device, sbt_memory, nullptr);
      if(sbt_staging) vkDestroyBuffer(ctx->device, sbt_staging, nullptr);
      if(sbt_staging_memory) vkFreeMemory(ctx->device, sbt_staging_memory, nullptr);
      if(camera_buffer) vkDestroyBuffer(ctx->device, camera_buffer, nullptr);
      if(camera_memory) vkFreeMemory(ctx->device, camera_memory, nullptr);
      if(time_buffer) vkDestroyBuffer(ctx->device, time_buffer, nullptr);
      if(time_memory) vkFreeMemory(ctx->device, time_memory, nullptr);
      if(tlas_instance_buffer) vkDestroyBuffer(ctx->device, tlas_instance_buffer, nullptr);
      if(tlas_instance_memory) vkFreeMemory(ctx->device, tlas_instance_memory, nullptr);
      if(tlas_scratch_buffer) vkDestroyBuffer(ctx->device, tlas_scratch_buffer, nullptr);
      if(tlas_scratch_memory) vkFreeMemory(ctx->device, tlas_scratch_memory, nullptr);
      if(vertex_buffer) vkDestroyBuffer(ctx->device, vertex_buffer, nullptr);
      if(vertex_memory) vkFreeMemory(ctx->device, vertex_memory, nullptr);
      if(index_buffer) vkDestroyBuffer(ctx->device, index_buffer, nullptr);
      if(index_memory) vkFreeMemory(ctx->device, index_memory, nullptr);
      if(material_buffer) vkDestroyBuffer(ctx->device, material_buffer, nullptr);
      if(material_memory) vkFreeMemory(ctx->device, material_memory, nullptr);
      for(auto tex_id : texture_ids){
        ctx->image_erase(tex_id);
      }
      ctx = nullptr;
    }
  #pragma pack(push, 1)
    struct vertex_t {
      fan::vec3 position; f32_t pad0;
      fan::vec3 normal; f32_t pad1;
      fan::vec2 texcoord; fan::vec2 pad2;
      fan::vec3 color; f32_t pad3;
    };
  #pragma pack(pop)
    static constexpr uint32_t instance_count = 1;
    static constexpr uint32_t max_textures = 512;
    fan::vulkan::context_t* ctx = nullptr;
    fan::vec2ui size{};
    std::vector<acceleration_structure_t> blas_list;
    std::vector<shapes::gpu_mesh_t> mesh_geometries;
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
    uint32_t shadow_miss_offset = 0;
    uint32_t shadow_hit_offset = 0;
    uint32_t handle_size_aligned = 0;
    uint32_t group_stride = 0;
    fan::graphics::image_t accum_image;
    VkImageLayout accum_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkPipeline accum_pipeline = VK_NULL_HANDLE;
    VkPipelineLayout accum_pipeline_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout accum_descriptor_layout = VK_NULL_HANDLE;
    VkDescriptorPool accum_descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet accum_descriptor_set = VK_NULL_HANDLE;
    uint32_t frame_index = 0;
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
    VkBuffer vertex_buffer = VK_NULL_HANDLE;
    VkDeviceMemory vertex_memory = VK_NULL_HANDLE;
    VkBuffer index_buffer = VK_NULL_HANDLE;
    VkDeviceMemory index_memory = VK_NULL_HANDLE;
    VkBuffer material_buffer = VK_NULL_HANDLE;
    VkDeviceMemory material_memory = VK_NULL_HANDLE;
    std::vector<material_info_t> materials;
    std::vector<vertex_t> vertex_data;
    std::vector<uint32_t> index_data;
    std::vector<fan::graphics::image_nr_t> texture_ids;
    std::vector<VkDescriptorImageInfo> rt_texture_infos;
    VkBuffer material_index_buffer = VK_NULL_HANDLE;
    VkDeviceMemory material_index_memory = VK_NULL_HANDLE;
    std::vector<uint32_t> material_indices_per_primitive;
    size_t current_material_index = 0;
    uint32_t vertex_offset = 0;
    f32_t exposure = 1.f;
    f32_t target_exposure = 1.f;
    f32_t adaptation_speed = 4.f;
    VkBuffer luminance_buffer;
    VkDeviceMemory luminance_memory;
    uint32_t luminance_group_count;
    VkBuffer exposure_ubo;
    VkDeviceMemory exposure_ubo_memory;
    VkDescriptorSetLayout luminance_descriptor_layout = VK_NULL_HANDLE;
    VkDescriptorPool      luminance_descriptor_pool   = VK_NULL_HANDLE;
    VkDescriptorSet       luminance_descriptor_set    = VK_NULL_HANDLE;
    VkPipelineLayout      luminance_pipeline_layout   = VK_NULL_HANDLE;
    VkPipeline            luminance_pipeline          = VK_NULL_HANDLE;
    VkBuffer light_buffer; 
    VkDeviceMemory light_memory;
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