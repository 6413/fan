#if defined(loco_vulkan)

inline void fan::vulkan::viewport_t::open() {
  auto& context = gloco->get_context();
  viewport_reference = context.viewport_list.NewNode();
  context.viewport_list[viewport_reference].viewport_id = this;
}

inline void fan::vulkan::viewport_t::close() {
  auto& context = gloco->get_context();
  context.viewport_list.Recycle(viewport_reference);
}

inline void fan::vulkan::viewport_t::set(const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
  viewport_position = viewport_position_;
  viewport_size = viewport_size_;

  VkViewport viewport{};
  viewport.x = viewport_position.x;
  viewport.y = viewport_position.y;
  viewport.width = viewport_size.x;
  viewport.height = viewport_size.y;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

  auto& context = gloco->get_context();

  if (!context.command_buffer_in_use) {
    VkResult result = vkGetFenceStatus(context.device, context.in_flight_fences[context.current_frame]);
    if (result == VK_NOT_READY) {
      vkDeviceWaitIdle(context.device);
    }

    if (vkBeginCommandBuffer(context.command_buffers[context.current_frame], &beginInfo) != VK_SUCCESS) {
      fan::throw_error("failed to begin recording command buffer!");
    }
  }
  vkCmdSetViewport(context.command_buffers[context.current_frame], 0, 1, &viewport);

  if (!context.command_buffer_in_use) {
    if (vkEndCommandBuffer(context.command_buffers[context.current_frame]) != VK_SUCCESS) {
      fan::throw_error("failed to record command buffer!");
    }
    context.command_buffer_in_use = false;
  }
}

inline void fan::vulkan::viewport_t::set_viewport(const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
  auto& context = gloco->get_context();
  VkViewport viewport{};
  viewport.x = viewport_position_.x;
  viewport.y = viewport_position_.y;
  viewport.width = viewport_size_.x;
  viewport.height = viewport_size_.y;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

  if (!context.command_buffer_in_use) {
    VkResult result = vkGetFenceStatus(context.device, context.in_flight_fences[context.current_frame]);
    if (result == VK_NOT_READY) {
      vkDeviceWaitIdle(context.device);
    }

    if (vkBeginCommandBuffer(context.command_buffers[context.current_frame], &beginInfo) != VK_SUCCESS) {
      fan::throw_error("failed to begin recording command buffer!");
    }
  }
  vkCmdSetViewport(context.command_buffers[context.current_frame], 0, 1, &viewport);

  if (!context.command_buffer_in_use) {
    if (vkEndCommandBuffer(context.command_buffers[context.current_frame]) != VK_SUCCESS) {
      fan::throw_error("failed to record command buffer!");
    }
    context.command_buffer_in_use = false;
  }
}

#endif