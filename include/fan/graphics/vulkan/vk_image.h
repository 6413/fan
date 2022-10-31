  struct image_t {
    struct load_properties_t {

    };

    image_t() = default;

    image_t(loco_t* loco, const fan::webp::image_info_t image_info, load_properties_t p = load_properties_t()) {
      load(loco, image_info, p);
    }

    auto* get_texture_data(loco_t* loco) {
      return &loco->image_list[texture_reference];
    }

    static void transitionImageLayout(loco_t* loco, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
      auto context = loco->get_context();
      VkCommandBuffer commandBuffer = context->beginSingleTimeCommands(context);

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

      fan::vulkan::context_t::endSingleTimeCommands(context, commandBuffer);
    }

    static void copyBufferToImage(loco_t* loco, VkBuffer buffer, VkImage image, const fan::vec2ui& size) {
      auto context = loco->get_context();
      VkCommandBuffer commandBuffer = fan::vulkan::context_t::beginSingleTimeCommands(context);

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

      vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

      fan::vulkan::context_t::endSingleTimeCommands(context, commandBuffer);
    }


    static void createTextureSampler(loco_t* loco, VkSampler& sampler, const load_properties_t& lp) {
      auto context = loco->get_context();
      VkPhysicalDeviceProperties properties{};
      vkGetPhysicalDeviceProperties(context->physicalDevice, &properties);

      VkSamplerCreateInfo samplerInfo{};
      samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
      samplerInfo.magFilter = VK_FILTER_LINEAR;
      samplerInfo.minFilter = VK_FILTER_LINEAR;
      samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
      samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
      samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
      samplerInfo.anisotropyEnable = VK_TRUE;
      samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
      samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
      samplerInfo.unnormalizedCoordinates = VK_FALSE;
      samplerInfo.compareEnable = VK_FALSE;
      samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
      samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

      if (vkCreateSampler(context->device, &samplerInfo, nullptr, &sampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture sampler!");
      }
    }


    auto& get(loco_t* loco) {
      auto& node = loco->image_list[texture_reference];
      return node;
    }

    auto& create_texture(loco_t* loco, const fan::vec2ui& image_size, const load_properties_t& lp) {
      auto context = loco->get_context();

      texture_reference = loco->image_list.NewNode();
      auto& node = loco->image_list[texture_reference];

      createImage(
        loco, 
        image_size, 
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_TILING_OPTIMAL, 
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 
        node.image,
        node.image_memory
      );
      node.image_view = createImageView(loco, node.image, VK_FORMAT_R8G8B8A8_UNORM);
      createTextureSampler(loco, node.sampler, lp);

      return node;
    }
    void erase_texture(loco_t* loco) {
      auto* context = loco->get_context();
      auto texture_data = get_texture_data(loco);
      vkDestroyImage(context->device, texture_data->image, 0);
      vkDestroyImageView(context->device, texture_data->image_view, 0);
      vkFreeMemory(context->device, texture_data->image_memory, nullptr);

      loco->image_list.Recycle(texture_reference);
    }

    bool load(loco_t* loco, const fan::webp::image_info_t image_info, load_properties_t p = load_properties_t()) {
      size = image_info.size;

      auto context = loco->get_context();
      VkBuffer stagingBuffer;
      VkDeviceMemory stagingBufferMemory;

      VkDeviceSize imageSize = image_info.size.multiply() * 4;

      fan::vulkan::core::createBuffer(
        context, 
        imageSize, 
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
        stagingBuffer, 
        stagingBufferMemory
      );

      void* data;
      vkMapMemory(context->device, stagingBufferMemory, 0, imageSize, 0, &data);
      memcpy(data, image_info.data, imageSize);
      vkUnmapMemory(context->device, stagingBufferMemory);

      auto node = create_texture(loco, image_info.size, p);

      transitionImageLayout(loco, node.image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
      copyBufferToImage(loco, stagingBuffer, node.image, image_info.size);
      transitionImageLayout(loco, node.image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

      vkDestroyBuffer(context->device, stagingBuffer, nullptr);
      vkFreeMemory(context->device, stagingBufferMemory, nullptr);

      return 0;
    }

    bool load(loco_t* loco, const fan::string& path, const load_properties_t& p = load_properties_t()) {
      auto context = loco->get_context();
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

      bool ret = load(loco, image_info, p);
      fan::webp::free_image(image_info.data);

      return ret;
    }

    void unload(loco_t* loco) {
      erase_texture(loco);
    }

    static void createImage(loco_t* loco, const fan::vec2ui& image_size, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
      auto context = loco->get_context();
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

      if (vkCreateImage(context->device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image!");
      }

      VkMemoryRequirements memRequirements;
      vkGetImageMemoryRequirements(context->device, image, &memRequirements);

      VkMemoryAllocateInfo allocInfo{};
      allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      allocInfo.allocationSize = memRequirements.size;
      allocInfo.memoryTypeIndex = fan::vulkan::core::findMemoryType(context, memRequirements.memoryTypeBits, properties);

      if (vkAllocateMemory(context->device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate image memory!");
      }

      vkBindImageMemory(context->device, image, imageMemory, 0);
    }

    static VkImageView createImageView(loco_t* loco, VkImage image, VkFormat format) {
      auto context = loco->get_context();
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
      if (vkCreateImageView(context->device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture image view!");
      }

      return imageView;
    }

    fan::vec2ui size;
    image_list_NodeReference_t texture_reference;
  };