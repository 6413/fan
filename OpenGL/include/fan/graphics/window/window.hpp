#pragma once

#include <fan/types/types.hpp>

#ifdef fan_compiler_visual_studio
	#define _CRT_SECURE_NO_WARNINGS
#endif

#include <fan/graphics/renderer.hpp>

#include <fan/math/random.hpp>

#include <fan/types/vector.hpp>
#include <fan/types/matrix.hpp>

#include <fan/types/color.hpp>
#include <fan/time/time.hpp>
#include <fan/graphics/window/window_input.hpp>

#include <fan/graphics/vulkan/vk_pipeline.hpp>

#include <deque>
#include <codecvt>
#include <locale>
#include <climits>
#include <type_traits>
#include <any>
#include <optional>
#include <mutex>


//#if fan_renderer == fan_renderer_opengl

#define GLEW_STATIC
#include <GL/glew.h>

//#endif

#ifdef fan_platform_windows

	#include <Windows.h>

//#if fan_renderer == fan_renderer_opengl

	#include <GL/wglew.h>

	#pragma comment(lib, "opengl32.lib")
	#pragma comment(lib, "lib/glew/glew32s.lib")

	#pragma comment(lib, "Gdi32.lib")
	#pragma comment(lib, "User32.lib")

//#endif

#elif defined(fan_platform_unix)

	#include <iostream>
	#include <cstring>

	#include <X11/Xlib.h>
	#include <X11/Xutil.h>
	#include <X11/Xos.h>
	#include <X11/Xatom.h>
	#include <X11/keysym.h>
	#include <X11/XKBlib.h>
//#if fan_renderer == fan_renderer_opengl
	#include <GL/glxew.h>
//#endif

	#include <sys/time.h>
	#include <unistd.h>

#undef index

#endif

namespace fan {

#if fan_renderer == fan_renderer_vulkan

	class vulkan;

#endif

	#ifdef fan_platform_windows

	static void set_console_visibility(bool visible) {
		ShowWindow(GetConsoleWindow(), visible ? SW_SHOW : SW_HIDE);
	}

	using window_t = HWND;

	#define FAN_API static


	#elif defined(fan_platform_unix)

	#define FAN_API

	using window_t = Window;

	#endif

	template <typename T>
	constexpr auto get_flag_value(T value) {
		return (1 << value);
	}

	struct pair_hash
	{
		template <class T1, class T2>
		constexpr std::size_t operator() (const std::pair<T1, T2> &pair) const
		{
			return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
		}
	};

	class window;


	#ifdef fan_platform_windows
#if fan_renderer == fan_renderer_opengl

	constexpr int WINDOW_MINOR_VERSION = WGL_CONTEXT_MINOR_VERSION_ARB;
	constexpr int WINDOW_MAJOR_VERSION = WGL_CONTEXT_MAJOR_VERSION_ARB;

	constexpr int WINDOW_SAMPLE_BUFFER = WGL_SAMPLE_BUFFERS_ARB;
	constexpr int OPENGL_SAMPLES = WGL_SAMPLES_ARB;

#endif
	#elif defined(fan_platform_unix)

	constexpr int WINDOW_MINOR_VERSION = GLX_CONTEXT_MINOR_VERSION_ARB;
	constexpr int WINDOW_MAJOR_VERSION = GLX_CONTEXT_MAJOR_VERSION_ARB;

	constexpr int WINDOW_SAMPLE_BUFFER = GLX_SAMPLE_BUFFERS;
	constexpr int OPENGL_SAMPLES = GLX_SAMPLES;

	#endif

	fan::vec2i get_resolution();

	template <typename T>
	constexpr auto initialized(T value) {
		return value != (T)uninitialized;
	}

	void set_screen_resolution(const fan::vec2i& size);
	void reset_screen_resolution();

	uint_t get_screen_refresh_rate();

	inline std::unordered_map<std::pair<fan::window_t, std::string>, std::any, pair_hash> m_window_storage;

	inline std::unordered_map<fan::window_t, fan::window*> window_id_storage;

	fan::window* get_window_by_id(fan::window_t wid);
	void set_window_by_id(fan::window_t wid, fan::window* window);

	enum class key_state {
		press,
		release
	};

	class window {
	public:

		enum class mode {
			not_set,
			windowed,
			borderless,
			full_screen
		};

		struct resolutions {
			constexpr static fan::vec2i r_800x600 = fan::vec2(800, 600);
			constexpr static fan::vec2i r_1024x768 = fan::vec2i(1024, 768);
			constexpr static fan::vec2i r_1280x720 = fan::vec2i(1280, 720);
			constexpr static fan::vec2i r_1280x800 = fan::vec2i(1280, 800);
			constexpr static fan::vec2i r_1280x900 = fan::vec2i(1280, 900);
			constexpr static fan::vec2i r_1280x1024 = fan::vec2i(1280, 1024);
			constexpr static fan::vec2i r_1360x768 = fan::vec2(1360, 768);
			constexpr static fan::vec2i r_1440x900 = fan::vec2i(1440, 900);
			constexpr static fan::vec2i r_1600x900 = fan::vec2i(1600, 900);
			constexpr static fan::vec2i r_1600x1024 = fan::vec2i(1600, 1024);
			constexpr static fan::vec2i r_1680x1050 = fan::vec2i(1680, 1050);
			constexpr static fan::vec2i r_1920x1080 = fan::vec2i(1920, 1080);

			constexpr static auto size = 12;

			static constexpr fan::vec2i x[12] = { fan::vec2(800, 600),
				fan::vec2i(1024, 768),
				fan::vec2i(1280, 720),
				fan::vec2i(1280, 800),
				fan::vec2i(1280, 900),
				fan::vec2i(1280, 1024),
				fan::vec2(1360, 768),
				fan::vec2i(1440, 900),
				fan::vec2i(1600, 900),
				fan::vec2i(1600, 1024),
				fan::vec2i(1680, 1050),
				fan::vec2i(1920, 1080) 
			};

		};

		// required type alias for function return types
		using keys_callback_t = std::function<void(uint16_t, key_state)>;
		using key_callback_t = struct{

			uint16_t key;
			key_state state;

			std::function<void()> function;

		};

		using text_callback_t = std::function<void(fan::fstring::value_type key)>;

		using mouse_move_position_callback_t = std::function<void(const fan::vec2i& position)>;
		using scroll_callback_t = std::function<void(uint16_t key)>;

		struct flags {
			static constexpr int no_mouse = get_flag_value(0);
			static constexpr int no_resize = get_flag_value(1);
			static constexpr int anti_aliasing = get_flag_value(2);
			static constexpr int mode = get_flag_value(3);
			static constexpr int borderless = get_flag_value(4);
			static constexpr int full_screen = get_flag_value(5);
		};

		static constexpr const char* default_window_name = "window";
		static constexpr vec2i default_window_size = fan::vec2i(800, 600);
		static constexpr vec2i default_opengl_version = fan::vec2i(2, 1); // major minor
		static constexpr mode default_size_mode = mode::windowed;

		// for static value storing
		static constexpr int reserved_storage = -1;

		window(const fan::vec2i& window_size = fan::window::default_window_size, const std::string& name = default_window_name, uint64_t flags = 0);
		window(const window& window);
		window(window&& window);

		window& operator=(const window& window);
		window& operator=(window&& window);

		~window();

		void destroy() {
			#ifdef fan_platform_windows
		#if fan_renderer == fan_renderer_opengl
			wglDeleteContext(m_context);
		#endif

			#elif defined(fan_platform_unix)

		#if fan_renderer == fan_renderer_opengl
			glXDestroyContext(m_display, m_context);
		#endif
			XCloseDisplay(m_display);
			m_display = 0;

			#endif

		#if fan_renderer == fan_renderer_opengl
			m_context = 0;
		#endif

		}

		void execute(const std::function<void()>& function);

		void loop(const std::function<void()>& function);

		void swap_buffers() const;

		std::string get_name() const;
		void set_name(const std::string& name);

		void calculate_delta_time();
		f_t get_delta_time() const;

		fan::vec2i get_mouse_position() const;
		fan::vec2i get_previous_mouse_position() const;

		fan::vec2i get_size() const;
		fan::vec2i get_previous_size() const;
		void set_size(const fan::vec2i& size);

		fan::vec2i get_position() const;
		void set_position(const fan::vec2i& position);

		uint_t get_max_fps() const;
		void set_max_fps(uint_t fps);

		bool vsync_enabled() const;
		void set_vsync(bool value);

		// use fan::window::resolutions for window sizes
		void set_full_screen(const fan::vec2i& size = uninitialized);
		void set_windowed_full_screen(const fan::vec2i& size = uninitialized);
		void set_windowed(const fan::vec2i& size = uninitialized);

		void set_resolution(const fan::vec2i& size, const mode& mode) const;

		mode get_size_mode() const;
		void set_size_mode(const mode& mode);

		template <typename type_t>
		static type_t get_window_storage(const fan::window_t& window, const std::string& location);
		static void set_window_storage(const fan::window_t& window, const std::string& location, std::any data);

		template <uint_t flag, typename T = 
			typename std::conditional<flag & fan::window::flags::no_mouse, bool,
			typename std::conditional<flag & fan::window::flags::no_resize, bool,
			typename std::conditional<flag & fan::window::flags::anti_aliasing, int,
			typename std::conditional<flag & fan::window::flags::mode, fan::window::mode, int
			>>>>::type>
			static constexpr void set_flag_value(T value) {
			if constexpr(static_cast<bool>(flag & fan::window::flags::no_mouse)) {
				flag_values::m_no_mouse = value;
			}
			else if constexpr(static_cast<bool>(flag & fan::window::flags::no_resize)) {
				flag_values::m_no_resize = value;
			}
			else if constexpr(static_cast<bool>(flag & fan::window::flags::anti_aliasing)) {
				flag_values::m_samples = value;
			}
			else if constexpr(static_cast<bool>(flag & fan::window::flags::mode)) {
				if (value > fan::window::mode::full_screen) {
					fan::print("fan window error: failed to set window mode flag to: ", fan::eti(value));
					exit(1);
				}
				flag_values::m_size_mode = value;
			}
			else if constexpr (static_cast<bool>(flag & fan::window::flags::borderless)) {
				flag_values::m_size_mode = value ? fan::window::mode::borderless : flag_values::m_size_mode;
			}
			else if constexpr (static_cast<bool>(flag & fan::window::flags::full_screen)) {
				flag_values::m_size_mode = value ? fan::window::mode::full_screen : flag_values::m_size_mode;
			}
		}

		template <uint64_t flags>
		static constexpr void set_flags() {
			// clang requires manual casting (c++11-narrowing)
			if constexpr(static_cast<bool>(flags & fan::window::flags::no_mouse)) {
				fan::window::flag_values::m_no_mouse = true;
			}
			if constexpr (static_cast<bool>(flags & fan::window::flags::no_resize)) {
				fan::window::flag_values::m_no_resize = true;
			}
			if constexpr (static_cast<bool>(flags & fan::window::flags::anti_aliasing)) {
				fan::window::flag_values::m_samples = 8;
			}
			if constexpr (static_cast<bool>(flags & fan::window::flags::borderless)) {
				fan::window::flag_values::m_size_mode = fan::window::mode::borderless;
			}
			if constexpr (static_cast<bool>(flags & fan::window::flags::full_screen)) {
				fan::window::flag_values::m_size_mode = fan::window::mode::full_screen;
			}
		}

		void set_keys_callback(const keys_callback_t& function);
		void remove_keys_callback();

		std::deque<key_callback_t>::iterator add_key_callback(uint16_t key, key_state state, const std::function<void()>& function);
		void edit_key_callback(std::deque<key_callback_t>::iterator it, uint16_t key, key_state state);
		void remove_key_callback(std::deque<key_callback_t>::const_iterator it);

		void set_text_callback(const text_callback_t& function);
		void remove_text_callback();

		std::deque<std::function<void()>>::iterator add_close_callback(const std::function<void()>& function);
		void remove_close_callback(std::deque<std::function<void()>>::const_iterator it);

		std::deque<mouse_move_position_callback_t>::iterator add_mouse_move_callback(const mouse_move_position_callback_t& function);
		void remove_mouse_move_callback(std::deque<mouse_move_position_callback_t>::const_iterator it);

		std::deque<std::function<void()>>::iterator add_resize_callback(const std::function<void()>& function);
		void remove_resize_callback(std::deque<std::function<void()>>::const_iterator it);

		std::deque<std::function<void()>>::iterator add_move_callback(const std::function<void()>& function);
		void remove_move_callback(std::deque<std::function<void()>>::const_iterator it);

		void set_error_callback();

		void set_background_color(const fan::color& color);

		fan::window_t get_handle() const;

		// when finished getting fps returns fps otherwise 0
		uint_t get_fps(bool window_title = true, bool print = true);

		bool key_press(uint16_t key) const;

		bool open() const;
		void close();

		bool focused() const;

		void destroy_window();

		uint16_t get_current_key() const;

		fan::vec2i get_raw_mouse_offset() const;

		static void handle_events();

		void auto_close(bool state);

#if fan_renderer == fan_renderer_vulkan
		fan::vulkan* m_vulkan = nullptr;

		
#endif

	private:

		static constexpr fan::input banned_keys[]{
			fan::key_enter,
			fan::key_tab,
			fan::key_escape,
			fan::key_backspace,
			fan::key_delete
		};

		using keymap_t = std::unordered_map<uint16_t, bool>;
		using timer_interval_t = fan::milliseconds;

		static void window_input_action(fan::window_t window, uint16_t key);
		FAN_API void window_input_mouse_action(fan::window_t window, uint16_t key);
		FAN_API void window_input_up(fan::window_t window, uint16_t key);
		FAN_API void window_input_action_reset(fan::window_t window, uint16_t key);

		#ifdef fan_platform_windows

		static LRESULT CALLBACK window_proc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam);

		HDC m_hdc;

	#if fan_renderer == fan_renderer_opengl
		static inline HGLRC m_context;
	#endif


		#elif defined(fan_platform_unix)

	public:

		static Display* get_display();

	protected:

		inline static Display* m_display;
		inline static int m_screen;
		inline static Atom m_atom_delete_window;
		XSetWindowAttributes m_window_attribs;
		XVisualInfo* m_visual;

	//#if fan_renderer == fan_renderer_opengl
		inline static GLXContext m_context;
	//#endif

		XIM m_xim;
		XIC m_xic;

		#endif

		void reset_keys();

		void initialize_window(const std::string& name, const fan::vec2i& window_size, uint64_t flags);

		// crossplatform variables

		window_t m_window;

		keys_callback_t m_keys_callback;
		std::deque<key_callback_t> m_key_callback;
		text_callback_t m_text_callback;
		std::deque<mouse_move_position_callback_t> m_mouse_move_position_callback;
		std::deque<std::function<void()>> m_move_callback;
		std::deque<std::function<void()>> m_resize_callback;
		std::deque<std::function<void()>> m_close_callback;

		keymap_t m_keys_down;

		// for releasing key after pressing it in key callback
		keymap_t m_keys_action;
		keymap_t m_keys_reset;

		fan::vec2i m_size;
		fan::vec2i m_previous_size;

		fan::vec2i m_position;

		fan::vec2i m_mouse_position;

		uint_t m_max_fps;

		f_t m_fps_next_tick;
		bool m_received_fps;
		uint_t m_fps;
		fan::timer<> m_fps_timer;

		f_t m_last_frame;
		f_t m_current_frame;
		f_t m_delta_time;

		bool m_vsync;

		bool m_close;

		std::string m_name;

		uint_t m_flags;

		uint16_t m_current_key;
		uint64_t m_reserved_flags;

		fan::vec2i m_raw_mouse_offset;

		bool m_focused;

		bool m_auto_close;

		fan::color m_background_color;

		fan::vec2i m_previous_mouse_position;

		struct flag_values {

			static inline int m_minor_version = fan::uninitialized;
			static inline int m_major_version = fan::uninitialized;

			static inline bool m_no_mouse = false;
			static inline bool m_no_resize = false;

			static inline uint8_t m_samples = fan::uninitialized;

			static inline mode m_size_mode;

		};

	};

	namespace io {

		static fan::fstring get_clipboard_text(fan::window_t window) {

			fan::fstring copied_text;

			#ifdef fan_platform_windows

			if (!OpenClipboard(nullptr)) {
				throw std::runtime_error("failed to open clipboard");
			}

			HANDLE data = GetClipboardData(CF_UNICODETEXT);

			if (data == nullptr) {
				throw std::runtime_error("clipboard data was nullptr");
			}

			wchar_t* text = static_cast<wchar_t*>(GlobalLock(data));
			if (text == nullptr) {
				throw std::runtime_error("copyboard text was nullptr");
			}

			copied_text = text;

			GlobalUnlock(data);

			CloseClipboard();

			#elif defined(fan_platform_unix)

			typedef std::codecvt_utf8<wchar_t> convert_type;
			std::wstring_convert<convert_type, wchar_t> converter;

			Display *display = XOpenDisplay(NULL);

			if (!display) {
				throw std::runtime_error("failed to open display");
			}

			XEvent ev;
			XSelectionEvent *sev;

			Atom da, incr, type, sel, p;
			int di = 0;
			unsigned long size = 0, dul = 0;
			unsigned char *prop_ret = NULL;

			auto target_window = XCreateSimpleWindow(display, RootWindow(display, DefaultScreen(display)), -10, -10, 1, 1, 0, 0, 0);

			sel = XInternAtom(display, "CLIPBOARD", False);
			p = XInternAtom(display, "PENGUIN", False);

			XConvertSelection(display, sel, XInternAtom(display, "UTF8_STRING", False), p, target_window,
				CurrentTime);

			for (;;)
			{
				XNextEvent(display, &ev);
				switch (ev.type)
				{
					case SelectionNotify:
					{
						sev = (XSelectionEvent*)&ev.xselection;
						if (sev->property == None)
						{
							fan::print("Conversion could not be performed.");
						}
						goto g_done;
					}

				}
			}

			g_done:

			if (XGetWindowProperty(display, target_window, p, 0, 0, False, AnyPropertyType,
				&type, &di, &dul, &size, &prop_ret) != Success) {
				fan::print("failed");
			}

			incr = XInternAtom(display, "INCR", False);

			if (type == incr)
			{
				printf("INCR not implemented\n");
				return L"";
			}

			if (XGetWindowProperty(display, target_window, p, 0, size, False, AnyPropertyType,
				&da, &di, &dul, &dul, &prop_ret) != Success) {
				fan::print("failed data");
			}

			if (prop_ret) {
				copied_text = converter.from_bytes((char*)prop_ret);
				XFree(prop_ret);
			}
			else {
				fan::print("no prop");
			}

			XDeleteProperty(display, target_window, p);
			XDestroyWindow(display, target_window);
			XCloseDisplay(display);

			#endif

			return copied_text;
		}
	}
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if fan_renderer == fan_renderer_vulkan

#include <fan/io/file.hpp>

#include <set>
#include <optional>

#ifdef fan_compiler_visual_studio
	#pragma comment(lib, "lib/vulkan/vulkan-1.lib")
#endif

#include <fan/graphics/vulkan/vk_shader.hpp>

inline fan::mat4 projection(1);

#include <fan/graphics/vulkan/vk_core.hpp>

inline fan::mat4 view(1); 

constexpr auto mb = 1000000;

constexpr auto gpu_stack(10 * mb); // mb

struct UniformBufferObject {
	alignas(16) fan::mat4 view;
	alignas(16) fan::mat4 proj;
};

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


		vulkan(fan::window* window)
			: m_window(window), pipelines(&device, &renderPass, &descriptorSetLayout) {

			create_instance();
			setupDebugMessenger();
			createSurface();
			pickPhysicalDevice();
			createLogicalDevice();
			createSwapChain();
			createImageViews();
			createRenderPass();
			createDescriptorSetLayout();
			createFramebuffers();
			createCommandPool();
			
			//createIndexBuffer();
			createUniformBuffers();
			createDescriptorPool();
			createDescriptorSets();
			create_command_buffers();
			create_sync_objects();

			staging_buffer = new staging_buffer_t(&device,  &physicalDevice, gpu_stack);

			m_window->add_resize_callback([&] {
				window_resized = true;
			});
		}

		~vulkan() {

			pipelines.~pipelines();
			staging_buffer->~glsl_location_handler();

			cleanupSwapChain();

			vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
			descriptorSetLayout = nullptr;

			//vkDestroyBuffer(device, indexBuffer, nullptr); 
			//vkFreeMemory(device, indexBufferMemory, nullptr);

			for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
				vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
				vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
				vkDestroyFence(device, inFlightFences[i], nullptr);
			}

			vkDestroyCommandPool(device, commandPool, nullptr);
			commandPool = nullptr;

			vkDestroyDevice(device, nullptr);
			device = nullptr;

			if (enable_validation_layers) {
				DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
			}

			vkDestroySurfaceKHR(instance, surface, nullptr);
			surface = nullptr;
			vkDestroyInstance(instance, nullptr);
			instance = nullptr;

			//delete staging_buffer;
		}

		//VkBuffer indexBuffer;
	//	VkDeviceMemory indexBufferMemory;

		VkDescriptorSetLayout descriptorSetLayout = nullptr;

		std::vector<VkBuffer> uniformBuffers;
		std::vector<VkDeviceMemory> uniformBuffersMemory;

		VkDescriptorPool descriptorPool = nullptr;
		std::vector<VkDescriptorSet> descriptorSets;

		void createDescriptorSetLayout() {
			VkDescriptorSetLayoutBinding uboLayoutBinding{};
			uboLayoutBinding.binding = 0;
			uboLayoutBinding.descriptorCount = 1;
			uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			uboLayoutBinding.pImmutableSamplers = nullptr;
			uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

			VkDescriptorSetLayoutCreateInfo layoutInfo{};
			layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
			layoutInfo.bindingCount = 1;
			layoutInfo.pBindings = &uboLayoutBinding;

			if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
				throw std::runtime_error("failed to create descriptor set layout!");
			}

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
			vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
			memcpy(data, indices.data(), (size_t)bufferSize);
			vkUnmapMemory(device, stagingBufferMemory);

			createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

			copyBuffer(stagingBuffer, indexBuffer, bufferSize);

			vkDestroyBuffer(device, stagingBuffer, nullptr);
			vkFreeMemory(device, stagingBufferMemory, nullptr);
		}*/

		void createUniformBuffers() {
			VkDeviceSize bufferSize = sizeof(UniformBufferObject);

			uniformBuffers.resize(swapChainImages.size());
			uniformBuffersMemory.resize(swapChainImages.size());

			for (size_t i = 0; i < swapChainImages.size(); i++) {
				createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);
			}
		}

		void updateUniformBuffer(uint32_t currentImage) {

			UniformBufferObject ubo{};

			fan::vec2 window_size = m_window->get_size();

		/*	ubo.model = fan::mat4(1);

			ubo.model = fan::math::rotate(fan::mat4(1), m_window->get_delta_time() * fan::math::radians(90.0f), fan::vec3(0.0f, 0.0f, 1.0f));

			

			ubo.view = fan::math::look_at_right<fan::mat4, fan::vec3>(fan::vec3(2.0f, 2.0f, 2.0f), fan::vec3(0.0f, 0.0f, 0.0f), fan::vec3(0.0f, 0.0f, 1.0f));
			ubo.proj = fan::math::perspective<fan::mat4>(fan::math::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f);
			ubo.proj[1][1] *= -1;*/

			//static f32_t angle = 0;

			//angle += m_window->get_delta_time() * 10000;

			//glm::mat4 a = glm::rotate(glm::mat4(1.0f), (f32_t)(glm::radians(angle)), glm::vec3(0.0f, 0.0f, 1.0f));
			//m_camera->get_view_matrix(fan_2d::graphics::get_view_translation(m_camera->m_window->get_size(), glm::mat4(1)))
			//glm::mat4 a1(1);
			//a1 = glm::translate(a1, glm::vec3(0.0f, 0.0f, -3.0f));
			//glm::mat4 a2 = glm::mat4(1);
			//a2 = glm::ortho(0.0f, 800.0f, 0.0f, 600.0f, 0.1f, 0.5f);
			////a2[1][1] *= -1;

			static fan::vec3 abc(0, 0, 0.1);

			constexpr fan::vec3 front(0, 0, 1);

			constexpr fan::vec3 camera_position(0, 0, 0.1); // z needs to be + 0.1

			constexpr fan::vec3 world_up(0, 1, 0);

			ubo.view = fan::math::look_at_left<fan::mat4>(camera_position, camera_position + front, world_up);

			static f32_t offsetx = window_size.x / 2;
			static f32_t offsety = window_size.y / 2;

			ubo.proj = fan::math::ortho<fan::mat4>((f32_t)0, (f32_t)window_size.x, (f32_t)0, (f32_t)window_size.y, 0.1, 100);

			void* data;
			vkMapMemory(device, uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
			memcpy(data, &ubo, sizeof(ubo));
			vkUnmapMemory(device, uniformBuffersMemory[currentImage]);
		}

		void createDescriptorPool() {

			VkDescriptorPoolSize poolSize{};
			poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			poolSize.descriptorCount = static_cast<uint32_t>(swapChainImages.size());

			VkDescriptorPoolCreateInfo poolInfo{};
			poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
			poolInfo.poolSizeCount = 1;
			poolInfo.pPoolSizes = &poolSize;
			poolInfo.maxSets = static_cast<uint32_t>(swapChainImages.size());

			if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
				throw std::runtime_error("failed to create descriptor pool!");
			}

		}

		void createDescriptorSets() {

			std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size(), descriptorSetLayout);
			VkDescriptorSetAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			allocInfo.descriptorPool = descriptorPool;
			allocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size());
			allocInfo.pSetLayouts = layouts.data();

			descriptorSets.resize(swapChainImages.size());
			if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
				throw std::runtime_error("failed to allocate descriptor sets!");
			}

			for (size_t i = 0; i < swapChainImages.size(); i++) {
				VkDescriptorBufferInfo bufferInfo{};
				bufferInfo.buffer = uniformBuffers[i];
				bufferInfo.offset = 0;
				bufferInfo.range = sizeof(UniformBufferObject);

				VkWriteDescriptorSet descriptorWrite{};
				descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptorWrite.dstSet = descriptorSets[i];
				descriptorWrite.dstBinding = 0;
				descriptorWrite.dstArrayElement = 0;
				descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				descriptorWrite.descriptorCount = 1;
				descriptorWrite.pBufferInfo = &bufferInfo;

				vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
			}
		}

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
			app_info.pEngineName = "No Engine";
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

		// physical device
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

		}

		// logical device
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
				throw std::runtime_error("failed to create logical device!");
			}

			vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
			vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
		}

		// surface
		void createSurface() {

#ifdef fan_platform_windows

			VkWin32SurfaceCreateInfoKHR create_info{};
			create_info.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
			create_info.hwnd = m_window->get_handle();

			create_info.hinstance = GetModuleHandle(nullptr);

			if (vkCreateWin32SurfaceKHR(instance, &create_info, nullptr, &surface) != VK_SUCCESS) {
				throw std::runtime_error("failed to create window surface!");
			}

#elif defined(fan_platform_unix)

			VkXlibSurfaceCreateInfoKHR create_info{};
			create_info.sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
			create_info.window = m_window->get_handle();
			create_info.dpy = fan::window::get_display();

			if (vkCreateXlibSurfaceKHR(instance, &create_info, nullptr, &surface) != VK_SUCCESS) {
				throw std::runtime_error("failed to create window surface!");
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
			for (auto framebuffer : swapChainFramebuffers) {
				vkDestroyFramebuffer(device, framebuffer, nullptr);
			}

			vkDestroyRenderPass(device, renderPass, nullptr);

			for (auto imageView : swapChainImageViews) {
				vkDestroyImageView(device, imageView, nullptr);
			}

			vkDestroySwapchainKHR(device, swapChain, nullptr);

			for (size_t i = 0; i < swapChainImages.size(); i++) {
				vkDestroyBuffer(device, uniformBuffers[i], nullptr);
				vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
			}

			vkDestroyDescriptorPool(device, descriptorPool, nullptr);
		}

		void recreateSwapChain() {

			fan::vec2 window_size = m_window->get_size();

			while (window_size == 0) {
				window_size = m_window->get_size();
				m_window->handle_events();
			}

			vkDeviceWaitIdle(device);

			cleanupSwapChain();

			createSwapChain();
			createImageViews();
			createRenderPass();

			for (uint32_t i = 0; i < pipelines.old_data.size(); i++) {
				pipelines.recreate_pipeline(i, m_window->get_size());
			}

			createFramebuffers();
			createUniformBuffers();
			createDescriptorPool();
			createDescriptorSets();
			create_command_buffers();

			imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);
		}
		 
		// image views
		void createImageViews() {

			swapChainImageViews.resize(swapChainImages.size());

			for (size_t i = 0; i < swapChainImages.size(); i++) {
				VkImageViewCreateInfo createInfo{};
				createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
				createInfo.image = swapChainImages[i];
				createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
				createInfo.format = swapChainImageFormat;
				createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
				createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
				createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
				createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
				createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
				createInfo.subresourceRange.baseMipLevel = 0;
				createInfo.subresourceRange.levelCount = 1;
				createInfo.subresourceRange.baseArrayLayer = 0;
				createInfo.subresourceRange.layerCount = 1;

				if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
					throw std::runtime_error("failed to create image views!");
				}
			}
		}

		// render pass
		void createRenderPass() {
			VkAttachmentDescription colorAttachment{};
			colorAttachment.format = swapChainImageFormat;
			colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
			colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

			VkAttachmentReference colorAttachmentRef{};
			colorAttachmentRef.attachment = 0;
			colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

			VkSubpassDescription subpass{};
			subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
			subpass.colorAttachmentCount = 1;
			subpass.pColorAttachments = &colorAttachmentRef;

			VkSubpassDependency dependency{};
			dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
			dependency.dstSubpass = 0;
			dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			dependency.srcAccessMask = 0;
			dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

			VkRenderPassCreateInfo renderPassInfo{};
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
			renderPassInfo.attachmentCount = 1;
			renderPassInfo.pAttachments = &colorAttachment;
			renderPassInfo.subpassCount = 1;
			renderPassInfo.pSubpasses = &subpass;
			renderPassInfo.dependencyCount = 1;
			renderPassInfo.pDependencies = &dependency;

			if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
				throw std::runtime_error("failed to create render pass!");
			}
		}

		// framebuffer
		void createFramebuffers() {

			swapChainFramebuffers.resize(swapChainImageViews.size());

			for (size_t i = 0; i < swapChainImageViews.size(); i++) {
				VkImageView attachments[] = {
					swapChainImageViews[i]
				};

				VkFramebufferCreateInfo framebufferInfo{};
				framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
				framebufferInfo.renderPass = renderPass;
				framebufferInfo.attachmentCount = 1;
				framebufferInfo.pAttachments = attachments;
				framebufferInfo.width = swapChainExtent.width;
				framebufferInfo.height = swapChainExtent.height;
				framebufferInfo.layers = 1;

				if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
					throw std::runtime_error("failed to create framebuffer!");
				}
			}
		}

		// command pool
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

			commandBuffers.resize(2);
			commandBuffers[0].resize(swapChainFramebuffers.size());
			commandBuffers[1].resize(1, nullptr);

			VkCommandBufferAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			allocInfo.commandPool = commandPool;
			allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			allocInfo.commandBufferCount = (uint32_t)commandBuffers[0].size();

			if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers[0].data()) != VK_SUCCESS) {
				throw std::runtime_error("failed to allocate command buffers!");
			}

			for (int j = 0; j < pipelines.pipeline_info.size(); j++) {
				for (size_t i = 0; i < commandBuffers[0].size(); i++) {
					
					for (const auto& call : draw_calls) {

						if (call) {
							call(i, j);
						}

					}

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

			updateUniformBuffer(image_index);

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

			submit_info.commandBufferCount = 1;
			submit_info.pCommandBuffers = &commandBuffers[0][image_index];

			VkSemaphore signal_semaphores[] = { renderFinishedSemaphores[currentFrame] };
			submit_info.signalSemaphoreCount = 1;
			submit_info.pSignalSemaphores = signal_semaphores;

			vkResetFences(device, 1, &inFlightFences[currentFrame]);

			if (vkQueueSubmit(graphicsQueue, 1, &submit_info, inFlightFences[currentFrame]) != VK_SUCCESS) {
				throw std::runtime_error("failed to submit draw command buffer!");
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

			if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
				recreateSwapChain();
			}
			else if (result != VK_SUCCESS) {
				throw std::runtime_error("failed to present swap chain image!");
			}

			currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
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


		//+physical device helper functions ------------------------------------------------------------

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

			bool extensionsSupported = checkDeviceExtensionSupport(device);

			bool swapChainAdequate = false;
			if (extensionsSupported) {
				SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
				swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
			}

			return indices.is_complete() && extensionsSupported && swapChainAdequate;
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

		//-physical device helper functions ------------------------------------------------------------

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
			// VK_FORMAT_B8G8R8A8_SRGB
			/*for (const auto& availableFormat : availableFormats) {
				if (availableFormat.format == VK_FORMAT_R8G8B8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
					return availableFormat;
				}
			}*/

			return availableFormats[0];
		}

		VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {

			return VK_PRESENT_MODE_IMMEDIATE_KHR;

			for (const auto& availablePresentMode : availablePresentModes) {
				if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
					return availablePresentMode;
				}
			}

			//return VK_PRESENT_MODE_FIFO_KHR; sync to blank
		}

		VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
			if (capabilities.currentExtent.width != UINT32_MAX) {
				return capabilities.currentExtent;
			}
			else {

				auto window_size = m_window->get_size();

				VkExtent2D actualExtent = {
					static_cast<uint32_t>(window_size[0]),
					static_cast<uint32_t>(window_size[1])
				};

				actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
				actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

				return actualExtent;
			}
		}

		//-swapchain helper functions ------------------------------------------------------------

		void push_back_draw_call(const std::function<void(uint32_t i, uint32_t j)>& function) {
			draw_calls.emplace_back(function);
		}

		fan::window* m_window;

		const std::vector<const char*> validation_layers = {
			"VK_LAYER_KHRONOS_validation"
		};

		const std::vector<const char*> deviceExtensions = {
			VK_KHR_SWAPCHAIN_EXTENSION_NAME
		};

		static constexpr int MAX_FRAMES_IN_FLIGHT = 10;

#ifdef fan_debug
		// decreases fps when enabled
		static constexpr bool enable_validation_layers = true;
#else
		static constexpr bool enable_validation_layers = false;
#endif
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

		fan::vk::graphics::pipelines pipelines;

		size_t currentFrame = 0;

		std::vector<std::function<void(uint32_t i, uint32_t j)>> draw_calls;

		bool window_resized = false;

		using staging_buffer_t = fan::gpu_memory::glsl_location_handler<fan::gpu_memory::buffer_type::staging>;

		staging_buffer_t* staging_buffer = nullptr;

	};

}

#endif