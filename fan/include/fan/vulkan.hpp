#pragma once

#if fan_renderer == fan_renderer_vulkan

#include <fan/io/file.hpp>

#include <set>
#include <optional>
#include <cstring>

#include <fan/system.hpp>

#ifdef fan_compiler_visual_studio
#pragma comment(lib, "lib/vulkan/vulkan-1.lib")
#endif

#include <fan/graphics/vulkan/vk_shader.hpp>
#include <fan/graphics/vulkan/vk_pipeline.hpp>
#include <fan/graphics/vulkan/vk_descriptor.hpp>

#include <fan/graphics/vulkan/vk_core.hpp>

#include <fan/types/matrix.hpp>

inline fan::mat4 projection(1);

inline fan::mat4 view(1); 

constexpr auto mb = 1000000;

constexpr auto gpu_stack(10 * mb); // mb

#ifndef fan_platform_unix
	#define fan_debug
#endif


namespace fan {

	class vulkan {
	public:

		struct QueueFamilyIndices {
			std::optional<uint32_t> graphicsFamily;
			std::optional<uint32_t> presentFamily;

			bool is_complete() const {
				return graphicsFamily.has_value() && presentFamily.has_value();
			}
		};

		struct SwapChainSupportDetails {
			VkSurfaceCapabilitiesKHR capabilities;
			std::vector<VkSurfaceFormatKHR> formats;
			std::vector<VkPresentModeKHR> presentModes;
		};


		vulkan(const fan::vec2i* window_size, void* handle)
			: 
			m_window_size(window_size) {

			reload_swapchain = false;

			create_instance();
			setupDebugMessenger();
			createSurface(handle);
			pickPhysicalDevice();
			createLogicalDevice();
			createSwapChain();

			staging_buffer = new staging_buffer_t(&device, &physicalDevice);
			staging_buffer->allocate(gpu_stack);

			texture_handler = new fan::gpu_memory::texture_handler(&device, &physicalDevice, &commandPool, &graphicsQueue);

			// create after initializing texture_handler
			texture_handler->descriptor_handler->recreate_descriptor_pool(swapChainImages.size());

			fan::vk::graphics::pipeline::flags_t line_flag;
			line_flag.topology = fan_2d::graphics::shape::line;
			line_flag.msaa_samples = &msaa_samples;

			fan::vk::graphics::pipeline::flags_t line_strip_flag;
			line_strip_flag.topology = fan_2d::graphics::shape::line_strip;
			line_strip_flag.msaa_samples = &msaa_samples;

			fan::vk::graphics::pipeline::flags_t triangle_flag;
			triangle_flag.topology = fan_2d::graphics::shape::triangle;
			triangle_flag.msaa_samples = &msaa_samples;

			fan::vk::graphics::pipeline::flags_t triangle_strip_flag;
			triangle_strip_flag.topology = fan_2d::graphics::shape::triangle_strip;
			triangle_strip_flag.msaa_samples = &msaa_samples;

			fan::vk::graphics::pipeline::flags_t triangle_fan_flag;
			triangle_fan_flag.topology = fan_2d::graphics::shape::triangle_fan;
			triangle_fan_flag.msaa_samples = &msaa_samples;

			pipelines.emplace_back(new std::remove_pointer_t<decltype(pipelines)::value_type>(&device, &renderPass, &texture_handler->descriptor_handler->descriptor_set_layout, line_flag));
			pipelines.emplace_back(new std::remove_pointer_t<decltype(pipelines)::value_type>(&device, &renderPass, &texture_handler->descriptor_handler->descriptor_set_layout, line_strip_flag));
			pipelines.emplace_back(new std::remove_pointer_t<decltype(pipelines)::value_type>(&device, &renderPass, &texture_handler->descriptor_handler->descriptor_set_layout, triangle_flag));
			pipelines.emplace_back(new std::remove_pointer_t<decltype(pipelines)::value_type>(&device, &renderPass, &texture_handler->descriptor_handler->descriptor_set_layout, triangle_strip_flag));
			pipelines.emplace_back(new std::remove_pointer_t<decltype(pipelines)::value_type>(&device, &renderPass, &texture_handler->descriptor_handler->descriptor_set_layout, triangle_fan_flag));

			createImageViews();

			createColorResources();
			createDepthResources();

			createRenderPass();
			createFramebuffers();
			createCommandPool();

			create_command_buffers();
			create_sync_objects();

			VkCommandBufferAllocateInfo alloc_info{};
			alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			alloc_info.commandPool = commandPool;
			alloc_info.commandBufferCount = 1;

			vkAllocateCommandBuffers(device, &alloc_info, &fan::gpu_memory::memory_command_buffer);
		}

		~vulkan() {

			for (int i = pipelines.size(); i--; ) {
				if (pipelines[i]) {
					delete pipelines[i];
					pipelines.erase(pipelines.end() - 1);
				}
			}

			if (staging_buffer) {
				delete staging_buffer;
				staging_buffer = nullptr;
			}


			cleanupSwapChain();

			//vkDestroyBuffer(m_device, indexBuffer, nullptr); 
			//vkFreeMemory(m_device, indexBufferMemory, nullptr);

			for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
				vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
				vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
				vkDestroyFence(device, inFlightFences[i], nullptr);
			}

			vkDestroyCommandPool(device, commandPool, nullptr);
			commandPool = nullptr;

			for (auto& imageView : swapChainImageViews) {
				vkDestroyImageView(device, imageView, nullptr);
				imageView = nullptr;
			}

			if (texture_handler) {
				delete texture_handler;
				texture_handler = nullptr;
			}

			vkDestroyImageView(device, depthImageView, nullptr);
			vkDestroyImage(device, depthImage, nullptr);
			vkFreeMemory(device, depthImageMemory, nullptr);

			vkDestroyImageView(device, colorImageView, nullptr);
			vkDestroyImage(device, colorImage, nullptr);
			vkFreeMemory(device, colorImageMemory, nullptr);

			vkDestroyDevice(device, nullptr);
			device = nullptr;

			if (enable_validation_layers) {
				DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
			}

			vkDestroySurfaceKHR(instance, surface,  nullptr);
			surface = nullptr;
			vkDestroyInstance(instance, nullptr);
			instance = nullptr;
		}

		void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
			VkBufferCreateInfo bufferInfo{};
			bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			bufferInfo.size = size;
			bufferInfo.usage = usage;
			bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

			if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
				throw std::runtime_error("failed to create buffer!");
			}

			VkMemoryRequirements memRequirements;
			vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

			VkMemoryAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			allocInfo.allocationSize = memRequirements.size;
			allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

			if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
				throw std::runtime_error("failed to allocate buffer memory!");
			}

			vkBindBufferMemory(device, buffer, bufferMemory, 0);
		}

		void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
			VkCommandBufferAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			allocInfo.commandPool = commandPool;
			allocInfo.commandBufferCount = 1;

			VkCommandBuffer commandBuffer;
			vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

			VkCommandBufferBeginInfo beginInfo{};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

			vkBeginCommandBuffer(commandBuffer, &beginInfo);

			VkBufferCopy copyRegion{};
			copyRegion.size = size;
			vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

			vkEndCommandBuffer(commandBuffer);

			VkSubmitInfo submitInfo{};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &commandBuffer;

			vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
			vkQueueWaitIdle(graphicsQueue);

			vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);

		}


		/*void createIndexBuffer() {
		VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(m_device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, indices.data(), (size_t)bufferSize);
		vkUnmapMemory(m_device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

		copyBuffer(stagingBuffer, indexBuffer, bufferSize);

		vkDestroyBuffer(m_device, stagingBuffer, nullptr);
		vkFreeMemory(m_device, stagingBufferMemory, nullptr);
		}*/

		uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {

			VkPhysicalDeviceMemoryProperties memProperties;
			vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

			for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
				if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
					return i;
				}
			}

			throw std::runtime_error("failed to find suitable memory type!");

		}

		// m_instance
		void create_instance() {
			VkApplicationInfo app_info{};
			app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
			app_info.pApplicationName = "application";
			app_info.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0); // VK_MAKE_VERSION
			app_info.pEngineName = "fan";
			app_info.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
			app_info.apiVersion = VK_API_VERSION_1_0;

			VkInstanceCreateInfo create_info{};
			create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
			create_info.pApplicationInfo = &app_info;

			std::vector<char*> extension_str = get_required_instance_extensions();

			create_info.enabledExtensionCount = extension_str.size();

			create_info.ppEnabledExtensionNames = extension_str.data();

			create_info.enabledLayerCount = 0;

			if (enable_validation_layers && !check_validation_layer_support()) {
				throw std::runtime_error("validation layers not available.");
			}

			VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};

			if (enable_validation_layers) {
				create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
				create_info.ppEnabledLayerNames = validation_layers.data();

				populateDebugMessengerCreateInfo(debugCreateInfo);
				create_info.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
			}
			else {
				create_info.enabledLayerCount = 0;

				create_info.pNext = nullptr;
			}

			if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS) {
				throw std::runtime_error("failed to create instance.");
			}
		}

		// debug
		void setupDebugMessenger() {

			if (!enable_validation_layers) return;

			VkDebugUtilsMessengerCreateInfoEXT createInfo;
			populateDebugMessengerCreateInfo(createInfo);

			if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
				throw std::runtime_error("failed to set up debug messenger!");
			}
		}

		// physical m_device
		void pickPhysicalDevice() {
			uint32_t device_count = 0;
			vkEnumeratePhysicalDevices(instance, &device_count, nullptr);

			if (device_count == 0) {
				throw std::runtime_error("failed to find GPUs with Vulkan support!");
			}

			std::vector<VkPhysicalDevice> devices(device_count);
			vkEnumeratePhysicalDevices(instance, &device_count, devices.data());

			int highest_score = -1;

			for (int i = 0; i < devices.size(); i++) {
				int score = get_device_score(devices[i]);

				if (highest_score < score) {
					highest_score = score;
					physicalDevice = devices[i];
				}
			}
			
			if (!(int)msaa_samples || msaa_samples == VK_SAMPLE_COUNT_1_BIT) {
				msaa_samples = VK_SAMPLE_COUNT_1_BIT;
			}
			else {
				msaa_samples = get_sample_count();
			}
		}

		// logical m_device
		void createLogicalDevice() {
			QueueFamilyIndices indices = findQueueFamilies(surface, physicalDevice);

			std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
			std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

			float queuePriority = 1;

			for (uint32_t queueFamily : uniqueQueueFamilies) {
				VkDeviceQueueCreateInfo queueCreateInfo{};
				queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
				queueCreateInfo.queueFamilyIndex = queueFamily;
				queueCreateInfo.queueCount = 1;
				queueCreateInfo.pQueuePriorities = &queuePriority;
				queueCreateInfos.push_back(queueCreateInfo);
			}

			VkPhysicalDeviceFeatures deviceFeatures{};

			deviceFeatures.samplerAnisotropy = VK_TRUE;

			VkDeviceCreateInfo createInfo{};
			createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

			createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
			createInfo.pQueueCreateInfos = queueCreateInfos.data();

			createInfo.pEnabledFeatures = &deviceFeatures;

			createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
			createInfo.ppEnabledExtensionNames = deviceExtensions.data();

			if (enable_validation_layers) {
				createInfo.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
				createInfo.ppEnabledLayerNames = validation_layers.data();
			}
			else {
				createInfo.enabledLayerCount = 0;
			}

			if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
				throw std::runtime_error("failed to create logical device");
			}

			vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
			vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
		}

		// surface
		void createSurface(void* handle) {

#ifdef fan_platform_windows

			VkWin32SurfaceCreateInfoKHR create_info{};
			create_info.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
			create_info.hwnd = (HWND)handle;

			create_info.hinstance = GetModuleHandle(nullptr);

			if (vkCreateWin32SurfaceKHR(instance, &create_info, nullptr, &surface) != VK_SUCCESS) {
				throw std::runtime_error("failed to create window surface");
			}

#elif defined(fan_platform_unix)

			VkXlibSurfaceCreateInfoKHR create_info{};
			create_info.sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
			create_info.window = (Window)handle;
			create_info.dpy = fan::sys::get_display();

			int x = 0;

			if ((x = vkCreateXlibSurfaceKHR(instance, &create_info, nullptr, &surface)) != VK_SUCCESS) {
				fan::print("error:", x);
				throw std::runtime_error("failed to create window surface");
			}

#endif
		}

		// swap chain
		void createSwapChain() {
			SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

			VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
			VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
			VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

			uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
			if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
				imageCount = swapChainSupport.capabilities.maxImageCount;
			}

			VkSwapchainCreateInfoKHR createInfo{};
			createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
			createInfo.surface = surface;

			createInfo.minImageCount = imageCount;
			createInfo.imageFormat = surfaceFormat.format;
			createInfo.imageColorSpace = surfaceFormat.colorSpace;
			createInfo.imageExtent = extent;
			createInfo.imageArrayLayers = 1;
			createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

			QueueFamilyIndices indices = findQueueFamilies(surface, physicalDevice);
			uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

			if (indices.graphicsFamily != indices.presentFamily) {
				createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
				createInfo.queueFamilyIndexCount = 2;
				createInfo.pQueueFamilyIndices = queueFamilyIndices;
			}
			else {
				createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
			}

			createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
			createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
			createInfo.presentMode = presentMode;
			createInfo.clipped = VK_TRUE;

			if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
				throw std::runtime_error("failed to create swap chain!");
			}

			vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
			swapChainImages.resize(imageCount);
			vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

			swapChainImageFormat = surfaceFormat.format;
			swapChainExtent = extent;
		}

		void erase_command_buffers() {
			for (int i = 0; i < commandBuffers.size(); i++) {
				vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers[i].size()), commandBuffers[i].data());
				commandBuffers[i].clear();
			}
			commandBuffers.clear();
		}

		void cleanupSwapChain() {
			for (auto& framebuffer : swapChainFramebuffers) {
				vkDestroyFramebuffer(device, framebuffer, nullptr);
				framebuffer = nullptr;
			}

			vkDestroyRenderPass(device, renderPass, nullptr);
			renderPass = nullptr;

			for (auto& imageView : swapChainImageViews) {
				vkDestroyImageView(device, imageView, nullptr);
				imageView = nullptr;
			}

			vkDestroySwapchainKHR(device, swapChain, nullptr);
			swapChain = nullptr;
		}

		void recreateSwapChain() {

			while (*m_window_size == 0) {
				// ?
			}

			vkDeviceWaitIdle(device);

			cleanupSwapChain();

			createSwapChain();
			createImageViews();
			createRenderPass();

			for (uint32_t j = 0; j < pipelines.size(); j++) {
				for (uint32_t i = 0; i < pipelines[j]->old_data.size(); i++) {
					pipelines[j]->recreate_pipeline(i, *m_window_size, swapChainExtent);
				}
			}

			createColorResources();
			createDepthResources();

			createFramebuffers();

			for (int i = 0; i < uniform_buffers.size(); i++) {
				uniform_buffers[i]->recreate(swapChainImages.size());
			}

			texture_handler->descriptor_handler->recreate_descriptor_pool(swapChainImages.size());

			for (int j = 0; j < uniform_buffers.size(); j++) {
				for (int i = 0; i < texture_handler->descriptor_handler->descriptor_sets.size() / swapChainImages.size(); i++) {
					texture_handler->descriptor_handler->update_descriptor_sets<fan::gpu_memory::uniform_handler>(
						i,
						device,
						uniform_buffers[j],
						texture_handler->descriptor_handler->descriptor_set_layout,
						texture_handler->descriptor_handler->descriptor_pool,
						texture_handler->image_views[i],
						texture_handler->texture_sampler,
						swapChainImages.size()
						);
				}
			}

			create_command_buffers();

			imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);
		}

		// image views
		void createImageViews() {

			swapChainImageViews.resize(swapChainImages.size(), nullptr);

			for (size_t i = 0; i < swapChainImages.size(); i++) {
				swapChainImageViews[i] = texture_handler->create_image_view(swapChainImages[i], swapChainImageFormat, 1);
			}
		}

		// render pass
		void createRenderPass() {
			VkAttachmentDescription colorAttachment{};
			colorAttachment.format = swapChainImageFormat;
			colorAttachment.samples = msaa_samples;
			colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

			VkAttachmentDescription depthAttachment{};
			depthAttachment.format = findDepthFormat();
			depthAttachment.samples = msaa_samples;
			depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

			VkAttachmentDescription colorAttachmentResolve{};
			colorAttachmentResolve.format = swapChainImageFormat;
			colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
			colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

			VkAttachmentReference colorAttachmentRef{};
			colorAttachmentRef.attachment = 0;
			colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

			VkAttachmentReference depthAttachmentRef{};
			depthAttachmentRef.attachment = 1;
			depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

			VkAttachmentReference colorAttachmentResolveRef{};
			colorAttachmentResolveRef.attachment = 2;
			colorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

			VkSubpassDescription subpass{};
			subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
			subpass.colorAttachmentCount = 1;
			subpass.pColorAttachments = &colorAttachmentRef;
			subpass.pDepthStencilAttachment = &depthAttachmentRef;
			subpass.pResolveAttachments = &colorAttachmentResolveRef;

			VkSubpassDependency dependency{};
			dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
			dependency.dstSubpass = 0;
			dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
			dependency.srcAccessMask = 0;
			dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
			dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

			std::array<VkAttachmentDescription, 3> attachments = {colorAttachment, depthAttachment, colorAttachmentResolve };
			VkRenderPassCreateInfo renderPassInfo{};
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
			renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
			renderPassInfo.pAttachments = attachments.data();

			//if (msaa_samples > VK_SAMPLE_COUNT_1_BIT) {
				renderPassInfo.subpassCount = 1;
				renderPassInfo.pSubpasses = &subpass;
				renderPassInfo.dependencyCount = 1;
				renderPassInfo.pDependencies = &dependency;
			//}

			if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
				throw std::runtime_error("failed to create render pass!");
			}
		}

		// framebuffer
		void createFramebuffers() {

			swapChainFramebuffers.resize(swapChainImageViews.size());

			for (size_t i = 0; i < swapChainImageViews.size(); i++) {
				std::array<VkImageView, 3> attachments = {
					colorImageView,
					depthImageView,
					swapChainImageViews[i]
				};

				VkFramebufferCreateInfo framebufferInfo{};
				framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
				framebufferInfo.renderPass = renderPass;
				framebufferInfo.attachmentCount = attachments.size();
				framebufferInfo.pAttachments = attachments.data();
				framebufferInfo.width = swapChainExtent.width;
				framebufferInfo.height = swapChainExtent.height;
				framebufferInfo.layers = 1;

				if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
					throw std::runtime_error("failed to create framebuffer!");
				}
			}
		}

		// command m_pool
		void createCommandPool() {
			QueueFamilyIndices queueFamilyIndices = findQueueFamilies(surface, physicalDevice);

			VkCommandPoolCreateInfo poolInfo{};
			poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
			poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
			poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

			if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
				throw std::runtime_error("failed to create graphics command pool!");
			}
		}

		// command buffers
		void create_command_buffers() {

			commandBuffers.resize(1);
			commandBuffers[0].resize(swapChainFramebuffers.size());

			VkCommandBufferAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			allocInfo.commandPool = commandPool;
			allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			allocInfo.commandBufferCount = (uint32_t)commandBuffers[0].size();

			if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers[0].data()) != VK_SUCCESS) {
				throw std::runtime_error("failed to allocate command buffers.");
			}

			VkCommandBufferBeginInfo beginInfo{};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

			for (int i = 0; i < commandBuffers[0].size(); i++) {

				if (vkBeginCommandBuffer(commandBuffers[0][i], &beginInfo) != VK_SUCCESS) {
					throw std::runtime_error("failed to begin recording command buffer!");
				}

				VkRenderPassBeginInfo renderPassInfo{};
				renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
				renderPassInfo.renderPass = renderPass;
				renderPassInfo.framebuffer = swapChainFramebuffers[i];
				renderPassInfo.renderArea.offset = { 0, 0 };
				renderPassInfo.renderArea.extent = swapChainExtent;

				std::array<VkClearValue, 2> clearValues{};
				clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
				clearValues[1].depthStencil = {1.0f, 0};

				renderPassInfo.clearValueCount = clearValues.size();
				renderPassInfo.pClearValues = clearValues.data();

				vkCmdBeginRenderPass(commandBuffers[0][i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

				for (int k = 0; k < std::size(draw_calls); k++) { 

					std::vector<std::pair<void*, std::pair<uint32_t, std::function<void(uint32_t, uint32_t, void*, fan_2d::graphics::shape)>>>> sorted(draw_calls[k].begin(), draw_calls[k].end());

					std::sort(sorted.begin(), sorted.end(), [&](const std::pair<void*, std::pair<uint32_t, std::function<void(uint32_t, uint32_t, void*, fan_2d::graphics::shape)>>>& a, const std::pair<void*, std::pair<uint32_t, std::function<void(uint32_t, uint32_t, void*, fan_2d::graphics::shape)>>>& b) {
						return a.second.first < b.second.first;
					}); // might not work with different shapes

					for (int l = sorted.size(); l--; ) {
						if (sorted[l].second.first == ~0) {
							continue;
						}
						if (sorted[l].second.second) {
							sorted[l].second.second(i, l, sorted[l].first, (fan_2d::graphics::shape)k);
						}
					}
				}

				vkCmdEndRenderPass(commandBuffers[0][i]);

				if (vkEndCommandBuffer(commandBuffers[0][i]) != VK_SUCCESS) {
					throw std::runtime_error("failed to record command buffer!");
				}
			}

		}

		// semaphores
		void create_sync_objects() {

			imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
			renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
			inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
			imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);
			data_edit_semaphore.resize(MAX_FRAMES_IN_FLIGHT);

			VkSemaphoreCreateInfo semaphoreInfo{};
			semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

			VkFenceCreateInfo fenceInfo{};
			fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
			fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

			for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
				if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
					vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
					vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
					throw std::runtime_error("failed to create synchronization objects for a frame!");
				}
			}

		}

		void draw_frame() {

			if (reload_swapchain) {
				this->recreateSwapChain();
				reload_swapchain = false;
			}

			vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

			uint32_t image_index = 0;
			VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &image_index);

			if (result == VK_ERROR_OUT_OF_DATE_KHR) {
				recreateSwapChain();
				return;
			}
			else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
				throw std::runtime_error("failed to acquire swap chain image!");
			}

			for (int i = 0; i < uniform_buffers.size(); i++) {
				uniform_buffers[i]->upload(image_index);
			}

			if (imagesInFlight[image_index] != VK_NULL_HANDLE) {
				vkWaitForFences(device, 1, &imagesInFlight[image_index], VK_TRUE, UINT64_MAX);
			}
			imagesInFlight[image_index] = inFlightFences[currentFrame];

			VkSubmitInfo submit_info{};
			submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

			VkSemaphore wait_semaphores[] = { imageAvailableSemaphores[currentFrame] };
			VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
			submit_info.waitSemaphoreCount = 1;
			submit_info.pWaitSemaphores = wait_semaphores;
			submit_info.pWaitDstStageMask = wait_stages;

			VkCommandBuffer command_buffers[] = { commandBuffers[0][image_index], fan::gpu_memory::memory_command_buffer };

			submit_info.commandBufferCount = 2;
			submit_info.pCommandBuffers = command_buffers;

			VkSemaphore signal_semaphores[] = { renderFinishedSemaphores[currentFrame] };
			submit_info.signalSemaphoreCount = 1;
			submit_info.pSignalSemaphores = signal_semaphores;

			vkResetFences(device, 1, &inFlightFences[currentFrame]);

			if (vkQueueSubmit(graphicsQueue, 1, &submit_info, inFlightFences[currentFrame]) != VK_SUCCESS) {
				throw std::runtime_error("failed to submit draw command buffer.");
			}

			VkPresentInfoKHR present_info{};
			present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

			present_info.waitSemaphoreCount = 1;
			present_info.pWaitSemaphores = signal_semaphores;

			VkSwapchainKHR swap_chains[] = { swapChain };
			present_info.swapchainCount = 1; 
			present_info.pSwapchains = swap_chains;

			present_info.pImageIndices = &image_index;

			result = vkQueuePresentKHR(presentQueue, &present_info);
			 
			if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || window_resized) {
				recreateSwapChain();
				window_resized = false;
			} 
			else if (result != VK_SUCCESS) {
				throw std::runtime_error("failed to present swap chain image!");
			}

			currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
			draw_order_id = 0;
		}

		//+m_instance helper functions ------------------------------------------------------------

		std::vector<char*> get_required_instance_extensions() {

			uint32_t extensions_count = 0;
			VkResult result = VK_SUCCESS;

			result = vkEnumerateInstanceExtensionProperties(nullptr, &extensions_count, nullptr);
			if ((result != VK_SUCCESS) ||
				(extensions_count == 0)) {
				throw std::runtime_error("Could not get the number of Instance extensions.");
			}

			std::vector<VkExtensionProperties> available_extensions;

			available_extensions.resize(extensions_count);

			result = vkEnumerateInstanceExtensionProperties(nullptr, &extensions_count, &available_extensions[0]);

			if ((result != VK_SUCCESS) ||
				(extensions_count == 0)) {
				throw std::runtime_error("Could not enumerate Instance extensions.");
			}

			std::vector<char*> extension_str(available_extensions.size());

			for (int i = 0; i < available_extensions.size(); i++) {
				extension_str[i] = new char[strlen(available_extensions[i].extensionName) + 1];
				std::memcpy(extension_str[i], available_extensions[i].extensionName, strlen(available_extensions[i].extensionName) + 1);
			}

			if (enable_validation_layers) {
				extension_str.push_back((char*)VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
			}

			return extension_str;
		}

		bool check_validation_layer_support() {
			uint32_t layer_count;
			vkEnumerateInstanceLayerProperties(&layer_count, nullptr);

			std::vector<VkLayerProperties> available_layers(layer_count);
			vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

			for (const char* layer_name : validation_layers) {
				bool layer_found = false;

				for (const auto& layerProperties : available_layers) {
					if (strcmp(layer_name, layerProperties.layerName) == 0) {
						layer_found = true;
						break;
					}
				}

				if (!layer_found) {
					return false;
				}
			}

			return true;
		}

		static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
			VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
			VkDebugUtilsMessageTypeFlagsEXT messageType,
			const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
			void* pUserData) {

			if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
				fan::print("validation layer:", pCallbackData->pMessage);
			}

			return VK_FALSE;
		}

		//-m_instance helper functions ------------------------------------------------------------

		//+debug helper functions ------------------------------------------------------------

		static VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
			auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
			if (func != nullptr) {
				return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
			}
			else {
				return VK_ERROR_EXTENSION_NOT_PRESENT;
			}
		}

		static void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
			auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
			if (func != nullptr) {
				func(instance, debugMessenger, pAllocator);
			}
		}

		void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
			createInfo = {};
			createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
			createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
			createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
			createInfo.pfnUserCallback = debugCallback;
		}

		//-debug helper functions ------------------------------------------------------------


		//+physical m_device helper functions ------------------------------------------------------------

		SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) const {
			SwapChainSupportDetails details;

			vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

			uint32_t formatCount;
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

			if (formatCount != 0) {
				details.formats.resize(formatCount);
				vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
			}

			uint32_t presentModeCount;
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

			if (presentModeCount != 0) {
				details.presentModes.resize(presentModeCount);
				vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
			}

			return details;
		}

		bool is_device_suitable(VkPhysicalDevice device) const {

			QueueFamilyIndices indices = findQueueFamilies(surface, device);

			bool extensions_supported = checkDeviceExtensionSupport(device);

			bool swap_chain_adequate = false;
			if (extensions_supported) {
				SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
				swap_chain_adequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
			}

			VkPhysicalDeviceFeatures supported_features;
			vkGetPhysicalDeviceFeatures(device, &supported_features);

			return indices.is_complete() && extensions_supported && swap_chain_adequate && supported_features.samplerAnisotropy;
		}

		bool checkDeviceExtensionSupport(VkPhysicalDevice device) const {

			uint32_t extension_count = 0;
			vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, nullptr);

			std::vector<VkExtensionProperties> availableExtensions(extension_count);
			vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, availableExtensions.data());

			std::set<std::string> required_extensions(deviceExtensions.begin(), deviceExtensions.end());

			for (const auto& extension : availableExtensions) {
				required_extensions.erase(extension.extensionName);
			}

			return required_extensions.empty();
		}

		int get_device_score(VkPhysicalDevice device) {

			VkPhysicalDeviceProperties device_properties;
			VkPhysicalDeviceFeatures device_features;

			vkGetPhysicalDeviceProperties(device, &device_properties);
			vkGetPhysicalDeviceFeatures(device, &device_features);

			int score = 0;

			// discrete gpus have better performance
			if (device_properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
				score += 1000;
			}

			// maximum image dimension gives higher resolution images
			score += device_properties.limits.maxImageDimension2D;

			if (!device_features.geometryShader) {
				return 0;
			}

			return score;
		}

		//-physical m_device helper functions ------------------------------------------------------------

		//+queue famlies helper functions ------------------------------------------------------------

		static QueueFamilyIndices findQueueFamilies(VkSurfaceKHR surface, VkPhysicalDevice device) {
			QueueFamilyIndices indices;

			uint32_t queueFamilyCount = 0;
			vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

			std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
			vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

			int i = 0;
			for (const auto& queueFamily : queueFamilies) {
				if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
					indices.graphicsFamily = i;
				}

				VkBool32 presentSupport = false;
				vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

				if (presentSupport) {
					indices.presentFamily = i;
				}

				if (indices.is_complete()) {
					break;
				}

				i++;
			}

			return indices;
		}

		//-queue famlies helper functions ------------------------------------------------------------

		//+swapchain helper functions ------------------------------------------------------------

		VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {

			/*for (const auto& availableFormat : availableFormats) {
				if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB ) {
					return availableFormat;
				}
			}*/

			return availableFormats[0];
		}

		VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
			for (const auto& available_present_mode : availablePresentModes) {
				if (available_present_mode == VK_PRESENT_MODE_IMMEDIATE_KHR && !vsync) {
					return VK_PRESENT_MODE_IMMEDIATE_KHR;
				}
				else if (available_present_mode == VK_PRESENT_MODE_FIFO_KHR && vsync) {
					return VK_PRESENT_MODE_FIFO_KHR;
				}
			}

			return availablePresentModes[0];
		}

		VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
			if (capabilities.currentExtent.width != UINT32_MAX) {
				return capabilities.currentExtent;
			}
			else {

				VkExtent2D actualExtent = {
					static_cast<uint32_t>((*m_window_size)[0]),
					static_cast<uint32_t>((*m_window_size)[1])
				};

				actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
				actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

				return actualExtent;
			}
		}

		//-swapchain helper functions ------------------------------------------------------------

		void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
			VkImageCreateInfo imageInfo{};
			imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
			imageInfo.imageType = VK_IMAGE_TYPE_2D;
			imageInfo.extent.width = width;
			imageInfo.extent.height = height;
			imageInfo.extent.depth = 1;
			imageInfo.mipLevels = mipLevels;
			imageInfo.arrayLayers = 1;
			imageInfo.format = format;
			imageInfo.tiling = tiling;
			imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			imageInfo.usage = usage;
			imageInfo.samples = numSamples;
			imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

			if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
				throw std::runtime_error("failed to create image!");
			}

			VkMemoryRequirements memRequirements;
			vkGetImageMemoryRequirements(device, image, &memRequirements);

			VkMemoryAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			allocInfo.allocationSize = memRequirements.size;
			allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

			if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
				throw std::runtime_error("failed to allocate image memory!");
			}

			vkBindImageMemory(device, image, imageMemory, 0);
		}

		VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels) {
			VkImageViewCreateInfo viewInfo{};
			viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			viewInfo.image = image;
			viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			viewInfo.format = format;
			viewInfo.subresourceRange.aspectMask = aspectFlags;
			viewInfo.subresourceRange.baseMipLevel = 0;
			viewInfo.subresourceRange.levelCount = mipLevels;
			viewInfo.subresourceRange.baseArrayLayer = 0;
			viewInfo.subresourceRange.layerCount = 1;

			VkImageView imageView;
			if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
				throw std::runtime_error("failed to create texture image view");
			}

			return imageView;
		}

		VkSampleCountFlagBits get_sample_count() {
			VkPhysicalDeviceProperties physicalDeviceProperties;
			vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);

			VkSampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
			if (counts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
			if (counts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
			if (counts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
			if (counts & VK_SAMPLE_COUNT_8_BIT) { return VK_SAMPLE_COUNT_8_BIT; }
			if (counts & VK_SAMPLE_COUNT_4_BIT) { return VK_SAMPLE_COUNT_4_BIT; }
			if (counts & VK_SAMPLE_COUNT_2_BIT) { return VK_SAMPLE_COUNT_2_BIT; }

			return VK_SAMPLE_COUNT_1_BIT;
		}

		VkFormat findDepthFormat() {
			return findSupportedFormat(
				{VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
				VK_IMAGE_TILING_OPTIMAL,
				VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
			);
		}

		void createColorResources() {
			VkFormat colorFormat = swapChainImageFormat;

			createImage(swapChainExtent.width, swapChainExtent.height, 1, msaa_samples, colorFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, colorImage, colorImageMemory);
			colorImageView = createImageView(colorImage, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
		}

		void createDepthResources() {
			VkFormat depthFormat = findDepthFormat();

			createImage(swapChainExtent.width, swapChainExtent.height, 1, msaa_samples, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
			depthImageView = createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, 1);
		}

		VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
			for (VkFormat format : candidates) {
				VkFormatProperties props;
				vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

				if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
					return format;
				} else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
					return format;
				}
			}

			throw std::runtime_error("failed to find supported format!");
		}

		void push_back_draw_call(const fan_2d::graphics::shape* shape, uint32_t shape_count, void* base, const std::function<void(uint32_t i, uint32_t j, void* base, fan_2d::graphics::shape)>& function) {

			for (int i = 0; i < shape_count; i++, shape++) {
				draw_calls[(int)*shape][base] = std::make_pair(~0, function);
			}
		}

		bool set_draw_call_order(uint32_t draw_order_id, const fan_2d::graphics::shape* shape, uint32_t shape_count, void* base) {

			bool edited = false;

			for (int i = 0; i < shape_count; i++, shape++) {
				auto& x = draw_calls[(int)*shape][base].first;

				if (x != draw_order_id) {
					x = draw_order_id;
					edited = true;
				}
			}

			return edited;
		}

		const std::vector<const char*> validation_layers = {
			"VK_LAYER_KHRONOS_validation"
		};

		const std::vector<const char*> deviceExtensions = {
			VK_KHR_SWAPCHAIN_EXTENSION_NAME
		};

		static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

#if fan_debug >= fan_debug_soft
		// decreases fps when enabled
		static constexpr bool enable_validation_layers = true;
#else
		static constexpr bool enable_validation_layers = false;
#endif

		const fan::vec2i* m_window_size = nullptr;

		VkInstance instance = nullptr;

		VkDebugUtilsMessengerEXT debugMessenger = nullptr;

		VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;

		VkDevice device = nullptr;

		VkQueue graphicsQueue = nullptr;
		VkQueue presentQueue = nullptr;

		VkSurfaceKHR surface = nullptr;

		VkSwapchainKHR swapChain = nullptr;

		std::vector<VkImage> swapChainImages;

		VkFormat swapChainImageFormat;
		VkExtent2D swapChainExtent;

		std::vector<VkImageView> swapChainImageViews;

		VkRenderPass renderPass = nullptr;

		std::vector<VkFramebuffer> swapChainFramebuffers;

		VkCommandPool commandPool = nullptr;

		std::vector<std::vector<VkCommandBuffer>> commandBuffers;

		std::vector<VkSemaphore> imageAvailableSemaphores;
		std::vector<VkSemaphore> renderFinishedSemaphores;

		std::vector<VkSemaphore> data_edit_semaphore;

		std::vector<VkFence> inFlightFences;
		std::vector<VkFence> imagesInFlight;

		std::vector<fan::vk::graphics::pipeline*> pipelines;

		size_t currentFrame = 0;

		std::unordered_map<void*, std::pair<uint32_t, std::function<void(uint32_t i, uint32_t j, void* base, fan_2d::graphics::shape shape)>>> draw_calls[5];

		bool window_resized = false;

		using staging_buffer_t = fan::gpu_memory::glsl_location_handler<fan::gpu_memory::buffer_type::staging>;

		staging_buffer_t* staging_buffer = nullptr;

		// allows multiple buffer edits when using offset
		uint64_t staging_buffer_offset = 0;

		fan::gpu_memory::texture_handler* texture_handler = nullptr;

		VkImageCreateInfo image_info{};

		std::vector<fan::gpu_memory::uniform_handler*> uniform_buffers;

		static inline fan_2d::graphics::shape draw_topology = fan_2d::graphics::shape::triangle;

		static inline VkSampleCountFlagBits msaa_samples;

		VkImage colorImage;
		VkDeviceMemory colorImageMemory;
		VkImageView colorImageView;

		VkImage depthImage;
		VkDeviceMemory depthImageMemory;
		VkImageView depthImageView;

		static inline bool reload_swapchain;

		bool vsync = true;

		uint32_t draw_order_id = 0;
	};

}

#endif