  struct image_t {

    struct format {
      static constexpr auto b8g8r8a8_unorm = VK_FORMAT_B8G8R8A8_UNORM;
      static constexpr auto r8b8g8a8_unorm = VK_FORMAT_R8G8B8A8_UNORM;
      static constexpr auto r8_unorm = VK_FORMAT_R8_UNORM;
      static constexpr auto r8_uint = VK_FORMAT_R8_UINT;
      static constexpr auto r8g8b8a8_srgb = VK_FORMAT_R8G8B8A8_SRGB;
    };

    struct sampler_address_mode {
      static constexpr auto repeat = VK_SAMPLER_ADDRESS_MODE_REPEAT;
      static constexpr auto mirrored_repeat = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
      static constexpr auto clamp_to_edge = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      static constexpr auto clamp_to_border = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
      static constexpr auto mirrored_clamp_to_edge = VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
    };

    struct filter {
      static constexpr auto nearest = VK_FILTER_NEAREST;
      static constexpr auto linear = VK_FILTER_LINEAR;
    };

    struct load_properties_defaults {
      static constexpr VkSamplerAddressMode visual_output = sampler_address_mode::clamp_to_border;
      //static constexpr uint32_t internal_format = fan::opengl::GL_RGBA;
      static constexpr VkFormat format = format::r8b8g8a8_unorm;
      //static constexpr uint32_t type = fan::opengl::GL_UNSIGNED_BYTE;
      static constexpr VkFilter min_filter = filter::nearest;
      static constexpr VkFilter mag_filter = filter::nearest;
    };

    struct load_properties_t {
      constexpr load_properties_t() noexcept {}
      //constexpr load_properties_t(auto a, auto b, auto c, auto d, auto e)
        //: visual_output(a), internal_format(b), format(c), type(d), filter(e) {}
      VkSamplerAddressMode visual_output = load_properties_defaults::visual_output;
      //uintptr_t           internal_format = load_properties_defaults::internal_format;
      //uintptr_t           format = load_properties_defaults::format;
      //uintptr_t           type = load_properties_defaults::type;
      VkFormat format = load_properties_defaults::format;
      VkFilter           min_filter = load_properties_defaults::min_filter;
      VkFilter           mag_filter = load_properties_defaults::mag_filter;
      // unused opengl filler
      uint8_t internal_format = 0;
    };

    image_t() = default;

    image_t(const fan::webp::image_info_t image_info, load_properties_t p = load_properties_t()) {
      load(image_info, p);
    }

    auto* get_texture_data() {
      return &gloco->image_list[texture_reference];
    }

    static void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
      auto& context = gloco->get_context();
      VkCommandBuffer commandBuffer = context.beginSingleTimeCommands(&context);

      VkImageMemoryBarrier barrier{};
      barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      barrier.oldLayout = oldLayout;
      barrier.newLayout = newLayout;
      barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.image = image;
      barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      barrier.subresourceRange.baseMipLevel = 0;
      barrier.subresourceRange.levelCount = 1;
      barrier.subresourceRange.baseArrayLayer = 0;
      barrier.subresourceRange.layerCount = 1;

      VkPipelineStageFlags sourceStage;
      VkPipelineStageFlags destinationStage;

      if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
      }
      else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
      }
      else {
        throw std::invalid_argument("unsupported layout transition!");
      }

      vkCmdPipelineBarrier(
        commandBuffer,
        sourceStage, destinationStage,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
      );

      fan::vulkan::context_t::endSingleTimeCommands(&context, commandBuffer);
    }

    static void copyBufferToImage(VkBuffer buffer, VkImage image, VkFormat format, const fan::vec2ui& size, const fan::vec2ui& stride = 1) {
      auto& context = gloco->get_context();
      VkCommandBuffer commandBuffer = fan::vulkan::context_t::beginSingleTimeCommands(&context);

      uint32_t block_width = get_image_multiplier(format);
      uint32_t block_x = (block_width - 1) / block_width;
      uint32_t block_y = (block_width - 1) / block_width;
      uint32_t block_h = std::max(1u, (size.y + block_width - 1) / block_width);
      // Flush CPU and GPU caches if not coherent mapping.
      VkDeviceSize buffer_flush_offset = block_y * stride.x;
      VkDeviceSize buffer_flush_size = block_h * stride.x;

      /*
            VkBufferImageCopy region{};
      region.bufferOffset = 0;
      region.bufferRowLength = 0;
      region.bufferImageHeight = 0;
      region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      region.imageSubresource.mipLevel = 0;
      region.imageSubresource.baseArrayLayer = 0;
      region.imageSubresource.layerCount = 1;
      region.imageOffset = { 0, 0, 0 };
      region.imageExtent = {
          size.x,
          size.y,
          1
      };
      */

      VkBufferImageCopy region = {
          block_y * stride.x + block_x,// VkDeviceSize             bufferOffset
          size.x,                                        // uint32_t                 bufferRowLength
          0,                                              // uint32_t                 bufferImageHeight
          {0, 0, 0, 1},                  // VkImageSubresourceLayers imageSubresource
          {0, 0, 0},  // VkOffset3D               imageOffset
          {size.x, size.y, 1}                              // VkExtent3D               imageExtent
      };

      region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      region.imageSubresource.layerCount = 1;

      vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

      fan::vulkan::context_t::endSingleTimeCommands(&context, commandBuffer);
    }


    static void createTextureSampler(VkSampler& sampler, const load_properties_t& lp = load_properties_t()) {
      auto& context = gloco->get_context();
      VkPhysicalDeviceProperties properties{};
      vkGetPhysicalDeviceProperties(context.physicalDevice, &properties);

      VkSamplerCreateInfo samplerInfo{};
      samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
      samplerInfo.magFilter = lp.mag_filter;
      samplerInfo.minFilter = lp.min_filter;
      samplerInfo.addressModeU = lp.visual_output;
      samplerInfo.addressModeV = lp.visual_output;
      samplerInfo.addressModeW = lp.visual_output;
      samplerInfo.anisotropyEnable = VK_TRUE;
      samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
      samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
      samplerInfo.unnormalizedCoordinates = VK_FALSE;
      samplerInfo.compareEnable = VK_FALSE;
      samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
      samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

      if (vkCreateSampler(context.device, &samplerInfo, nullptr, &sampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture sampler!");
      }
    }


    auto& get() {
      auto& node = gloco->image_list[texture_reference];
      return node;
    }

    auto& create_texture(const fan::vec2ui& image_size, const load_properties_t& lp) {
      auto& context = gloco->get_context();

      texture_reference = gloco->image_list.NewNode();
      auto& node = gloco->image_list[texture_reference];

      createImage(
        image_size, 
        lp.format,
        VK_IMAGE_TILING_OPTIMAL, 
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 
        node.image,
        node.image_memory
      );
      node.image_view = createImageView(node.image, lp.format);
      createTextureSampler(node.sampler, lp);

      return node;
    }
    void erase_texture() {
      auto& context = gloco->get_context();
      auto texture_data = get_texture_data();
      vkDestroyImage(context.device, texture_data->image, 0);
      vkDestroyImageView(context.device, texture_data->image_view, 0);
      vkFreeMemory(context.device, texture_data->image_memory, nullptr);

      gloco->image_list.Recycle(texture_reference);
    }

    constexpr static uint32_t get_image_multiplier(VkFormat format) {
      switch (format) {
        case format::b8g8r8a8_unorm: {
          return 4;
        }
        case format::r8_unorm: {
          return 4;
        }
        case format::r8g8b8a8_srgb: {
          return 4;
        }
        case format::r8b8g8a8_unorm: {
          return 4;
        }
        default: {
          fan::throw_error("failed to find format for image multiplier");
        }
      }
    }

    bool load(const fan::webp::image_info_t image_info, load_properties_t p = load_properties_t()) {
      size = image_info.size;

      auto& context = gloco->get_context();

      auto image_multiplier = get_image_multiplier(p.format);

      VkDeviceSize imageSize = image_info.size.multiply() * image_multiplier;

      fan::vulkan::core::create_buffer(
        context, 
        imageSize, 
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
         //VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT ,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
        stagingBuffer, 
        stagingBufferMemory
      );

      vkMapMemory(context.device, stagingBufferMemory, 0, imageSize, 0, &data);
      memcpy(data, image_info.data, imageSize); // TODO  / 4 in yuv420p

      auto node = create_texture(image_info.size, p);

      transitionImageLayout(node.image, p.format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
      copyBufferToImage(stagingBuffer, node.image, p.format, image_info.size);
      transitionImageLayout(node.image, p.format, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

      return 0;
    }

    bool load(const fan::string& path, const load_properties_t& p = load_properties_t()) {
      auto& context = gloco->get_context();
      #if fan_assert_if_same_path_loaded_multiple_times

      static std::unordered_map<fan::string, bool> existing_images;

      if (existing_images.find(path) != existing_images.end()) {
        fan::throw_error("image already existing " + path);
      }

      existing_images[path] = 0;

      #endif

      fan::webp::image_info_t image_info;
      if (fan::webp::load(path, &image_info)) {
        return true;
      }

      //image_info.size *= 4;
      bool ret = load(image_info, p);
      fan::webp::free_image(image_info.data);

      return ret;
    }

    void reload_pixels(const fan::webp::image_info_t& image_info, load_properties_t p = load_properties_t()) {
      auto& context = gloco->get_context();
      auto image_multiplier = get_image_multiplier(p.format);

      VkDeviceSize imageSize = image_info.size.multiply() * image_multiplier;

      memcpy(data, image_info.data, imageSize / 4);

      auto& node = gloco->image_list[texture_reference];

      transitionImageLayout(node.image, p.format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
      copyBufferToImage(stagingBuffer, node.image, p.format, image_info.size);
      transitionImageLayout(node.image, p.format, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    //void reload(const fan::string& path, const load_properties_t& p = load_properties_t()) {
    //  fan::webp::image_info_t image_info;
    //  if (fan::webp::load(path, &image_info)) {
    //    return true;
    //  }

    //  size = image_info.size;

    //  auto& context = gloco->get_context();
    //  VkBuffer stagingBuffer;
    //  VkDeviceMemory stagingBufferMemory;

    //  VkDeviceSize imageSize = image_info.size.multiply() * 4;

    //  fan::vulkan::core::createBuffer(
    //    context, 
    //    imageSize, 
    //    VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
    //    // VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT 
    //    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
    //    stagingBuffer, 
    //    stagingBufferMemory
    //  );

    //  void* data;
    //  vkMapMemory(context->device, stagingBufferMemory, 0, imageSize, 0, &data);
    //  memcpy(data, image_info.data, imageSize);
    //  vkUnmapMemory(context->device, stagingBufferMemory);

    //  auto node = create_texture(loco, image_info.size, p);

    //  transitionImageLayout(loco, node.image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    //  copyBufferToImage(loco, stagingBuffer, node.image, image_info.size);
    //  transitionImageLayout(loco, node.image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    //  vkDestroyBuffer(context->device, stagingBuffer, nullptr);
    //  vkFreeMemory(context->device, stagingBufferMemory, nullptr);

    //  fan::webp::free_image(image_info.data);
    //}

    void unload() {
      erase_texture();
      auto& context = gloco->get_context();
      vkDestroyBuffer(context.device, stagingBuffer, nullptr);
      vkFreeMemory(context.device, stagingBufferMemory, nullptr);
    }

    static void createImage(const fan::vec2ui& image_size, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
      auto& context = gloco->get_context();
      VkImageCreateInfo imageInfo{};
      imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
      imageInfo.imageType = VK_IMAGE_TYPE_2D;
      imageInfo.extent.width = image_size.x;
      imageInfo.extent.height = image_size.y;
      imageInfo.extent.depth = 1;
      imageInfo.mipLevels = 1;
      imageInfo.arrayLayers = 1;
      imageInfo.format = format;
      imageInfo.tiling = tiling;
      imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      imageInfo.usage = usage;
      imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
      imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

      if (vkCreateImage(context.device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image!");
      }

      VkMemoryRequirements memRequirements;
      vkGetImageMemoryRequirements(context.device, image, &memRequirements);

      VkMemoryAllocateInfo allocInfo{};
      allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      allocInfo.allocationSize = memRequirements.size;
      allocInfo.memoryTypeIndex = fan::vulkan::core::findMemoryType(context, memRequirements.memoryTypeBits, properties);

      if (vkAllocateMemory(context.device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate image memory!");
      }

      vkBindImageMemory(context.device, image, imageMemory, 0);
    }

    static VkImageView createImageView(VkImage image, VkFormat format) {
      auto& context = gloco->get_context();
      VkImageViewCreateInfo viewInfo{};
      viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      viewInfo.image = image;
      viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
      viewInfo.format = format;
      viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      viewInfo.subresourceRange.baseMipLevel = 0;
      viewInfo.subresourceRange.levelCount = 1;
      viewInfo.subresourceRange.baseArrayLayer = 0;
      viewInfo.subresourceRange.layerCount = 1;

      VkImageView imageView;
      if (vkCreateImageView(context.device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture image view!");
      }

      return imageView;
    }

    fan::vec2ui size;
    image_list_NodeReference_t texture_reference;
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    void* data;
  };