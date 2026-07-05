module;

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

export module fan.graphics.vulkan.ray_tracing.shapes;

import std;


import fan.types.vector;
import fan.types.matrix;
import fan.graphics.vulkan.core;


export namespace fan::graphics::vulkan::ray_tracing::shapes {

  struct triangle_mesh_t { 
    std::vector<fan::vec3> vertices; 
    std::vector<std::uint32_t> indices;
    triangle_mesh_t() = default;
  };

  struct gpu_mesh_t {
    fan::vulkan::context_t::buffer_t vertex_buffer;
    fan::vulkan::context_t::buffer_t index_buffer;
    std::uint32_t vertex_count = 0;
    std::uint32_t index_count = 0;
    void upload(fan::vulkan::context_t& ctx, const triangle_mesh_t& mesh) {
      vertex_count = (std::uint32_t)mesh.vertices.size();
      index_count = (std::uint32_t)mesh.indices.size();
      if (vertex_count == 0 || index_count == 0) {
        return;
      }

      ctx.upload_buffer(mesh.vertices, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, vertex_buffer);
      ctx.upload_buffer(mesh.indices, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, index_buffer);
    }

    void destroy(fan::vulkan::context_t& ctx) {
      ctx.destroy_buffer(vertex_buffer);
      ctx.destroy_buffer(index_buffer);
      vertex_count = 0;
      index_count = 0;
    }
  };
  triangle_mesh_t make_sphere(
    const fan::vec3& center,
    f32_t radius,
    std::uint32_t stacks = 20,
    std::uint32_t slices = 20
  ) {
    triangle_mesh_t mesh;
    mesh.vertices.reserve((stacks + 1) * (slices + 1));
    mesh.indices.reserve(stacks * slices * 6);
    for (std::uint32_t i = 0; i <= stacks; ++i) {
      f32_t v = (f32_t)i / (f32_t)stacks;
      f32_t phi = v * 3.1415926535f;

      for (std::uint32_t j = 0; j <= slices; ++j) {
        f32_t u = (f32_t)j / (f32_t)slices;
        f32_t theta = u * 2.0f * 3.1415926535f;

        fan::vec3 pos(
          radius * std::sin(phi) * std::cos(theta),
          radius * std::cos(phi),
          radius * std::sin(phi) * std::sin(theta)
        );
        mesh.vertices.push_back(center + pos);
      }
    }

    for (std::uint32_t i = 0; i < stacks; ++i) {
      for (std::uint32_t j = 0; j < slices; ++j) {
        std::uint32_t first = i * (slices + 1) + j;
        std::uint32_t second = first + slices + 1;

        mesh.indices.push_back(first);
        mesh.indices.push_back(second);
        mesh.indices.push_back(first + 1);

        mesh.indices.push_back(second);
        mesh.indices.push_back(second + 1);
        mesh.indices.push_back(first + 1);
      }
    }

    return mesh;
  }
  
}
