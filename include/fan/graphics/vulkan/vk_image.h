namespace fan {
  namespace vulkan {

    static constexpr fan::_vec4<fan::vec2> default_texture_coordinates = {
      vec2(0, 0),
      vec2(1, 0),
      vec2(1, 1),
      vec2(0, 1)
    };

    struct image_t {

      struct load_properties_t {

      };


      void create_texture(fan::vulkan::context_t* context, const fan::vec2ui& image_size) {
        texture_reference = context->image_list.NewNode();
        auto& node = context->image_list[texture_reference];

        createImage(
          context, 
          image_size, 
          VK_FORMAT_R8G8B8A8_SRGB, 
          VK_IMAGE_TILING_OPTIMAL, 
          VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 
          node.image,
          node.image_memory
        );
        node.image_view = createImageView(context, node.image, VK_FORMAT_R8G8B8A8_SRGB);
      }
      void erase_texture(fan::vulkan::context_t* context) {
        assert(0);
        //context->opengl.glDeleteTextures(1, get_texture(context));
        //vkDestroyImage(context->device, )
        //context->image_list.Recycle(texture_reference);
      }

      bool load(fan::vulkan::context_t* context, const fan::string& path, const load_properties_t& p = load_properties_t()) {

        #if fan_assert_if_same_path_loaded_multiple_times

        static std::unordered_map<fan::string, bool> existing_images;

        if (existing_images.find(path) != existing_images.end()) {
          fan::throw_error("image already existing " + path);
        }

        existing_images[path] = 0;

        #endif

        fan::webp::image_info_t image_info;
        if (fan::webp::load(path, &image_info)) {
          return 0;
        }
        create_texture(context, image_info.size);

        //bool ret = load(context, image_info, p);
        fan::webp::free_image(image_info.data);
        //fan::webp::free_image(image_info.data); leaks and double free sometimes
        return 0;
      }

      static void createImage(fan::vulkan::context_t* context, const fan::vec2ui& image_size, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
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

      static VkImageView createImageView(fan::vulkan::context_t* context, VkImage image, VkFormat format) {
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

      fan::vulkan::image_list_NodeReference_t texture_reference;
    };
  }
}

fan::vulkan::image_list_NodeReference_t::image_list_NodeReference_t(fan::vulkan::image_t* image) {
  NRI = image->texture_reference.NRI;
}