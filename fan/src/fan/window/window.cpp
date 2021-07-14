#include <fan/window/window.hpp>
#include <fan/math/random.hpp>

#define fan_renderer_opengl 0
#define fan_renderer_vulkan 1


#ifndef fan_set_graphics_renderer
	#define fan_set_graphics_renderer fan_renderer_vulkan
#endif

#define fan_renderer fan_set_graphics_renderer

#ifdef fan_platform_windows

#include <windowsx.h>

#include <mbctype.h>

#include <hidusage.h>

#undef min
#undef max

#elif defined(fan_platform_unix)

#include <locale>
#include <codecvt>

#include <X11/extensions/Xrandr.h>

#include <errno.h>

#endif

#include <string>

#define stringify(name) #name

fan::vec2i fan::get_resolution() {
	#ifdef fan_platform_windows

	return fan::vec2i(GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN));

	#elif defined(fan_platform_unix) // close

	Display* display = XOpenDisplay(0);

	if (!display) {
		fan::print("failed to open display");
	}

	int screen = DefaultScreen(display);
	fan::vec2i resolution(DisplayWidth(display, screen), DisplayHeight(display, screen));

	XCloseDisplay(display);

	return resolution;

	#endif
}

void fan::set_screen_resolution(const fan::vec2i& size)
{
	#ifdef fan_platform_windows
	DEVMODE screen_settings;
	memset (&screen_settings, 0, sizeof (screen_settings));
	screen_settings.dmSize = sizeof (screen_settings);
	screen_settings.dmPelsWidth = size.x;
	screen_settings.dmPelsHeight = size.y;
	screen_settings.dmBitsPerPel = 32;
	screen_settings.dmFields = DM_BITSPERPEL | DM_PELSWIDTH | DM_PELSHEIGHT;
	ChangeDisplaySettings(&screen_settings, CDS_FULLSCREEN);

	#elif defined(fan_platform_unix)

	#endif

}

void fan::reset_screen_resolution() {

	#ifdef fan_platform_windows

	ChangeDisplaySettings(nullptr, CDS_RESET);

	#elif defined(fan_platform_unix)



	#endif

}

uint_t fan::get_screen_refresh_rate()
{

	#ifdef fan_platform_windows

	DEVMODE dmode = { 0 };

	EnumDisplaySettings(nullptr, ENUM_CURRENT_SETTINGS, &dmode);

	return dmode.dmDisplayFrequency;

	#elif defined(fan_platform_unix)

	Display *display = XOpenDisplay(NULL);

	Window root = RootWindow(display, 0);

	XRRScreenConfiguration *conf = XRRGetScreenInfo(display, root);

	short refresh_rate = XRRConfigCurrentRate(conf);

	XCloseDisplay(display);

	return refresh_rate;

	#endif

}

fan::window* fan::get_window_by_id(fan::window_t wid) {

	auto found = fan::window_id_storage.find(wid);

	if (found == fan::window_id_storage.end()) {
		return 0;
	}

	return found->second;
}

void fan::set_window_by_id(fan::window_t wid, fan::window* window) {

	auto found = fan::window_id_storage.find(wid);

	if (found == fan::window_id_storage.end()) {
		fan::window_id_storage.insert(std::make_pair(wid, window));
	}
	else {
		throw std::runtime_error("window already exists in storage");
	}

}

// doesn't copy manually set flags yet
fan::window::window(const fan::vec2i& window_size, const std::string& name, uint64_t flags)
	: m_size(window_size), m_position(fan::uninitialized), m_mouse_position(0), m_max_fps(0), m_received_fps(0),
	m_fps(0), m_last_frame(0), m_current_frame(0), m_delta_time(0), m_vsync(false), m_close(false),
	m_name(name), m_flags(flags), m_current_key(0), m_reserved_flags(0), m_focused(true), m_auto_close(true)
{
	if (flag_values::m_size_mode == fan::window::mode::not_set) {
		flag_values::m_size_mode = fan::window::default_size_mode;
	}
	if (!initialized(flag_values::m_major_version)) {
		flag_values::m_major_version = default_opengl_version.x;
	}
	if (!initialized(flag_values::m_minor_version)) {
		flag_values::m_minor_version = default_opengl_version.y;
	}
	if (!initialized(flag_values::m_samples)) {
		flag_values::m_samples = 0;
	}

	if (static_cast<bool>(flags & fan::window::flags::no_mouse)) {
		fan::window::flag_values::m_no_mouse = true;
	}
	if (static_cast<bool>(flags & fan::window::flags::no_resize)) {
		fan::window::flag_values::m_no_resize = true;
	}
	if (static_cast<bool>(flags & fan::window::flags::anti_aliasing)) {
		fan::window::flag_values::m_samples = 8;
	}
	if (static_cast<bool>(flags & fan::window::flags::borderless)) {
		fan::window::flag_values::m_size_mode = fan::window::mode::borderless;
	}
	if (static_cast<bool>(flags & fan::window::flags::full_screen)) {
		fan::window::flag_values::m_size_mode = fan::window::mode::full_screen;
	}

	m_fps_next_tick = fan::timer<timer_interval_t>::get_time();

	initialize_window(name, window_size, flags);

	this->calculate_delta_time();

#if fan_renderer == fan_renderer_vulkan
	
		m_vulkan = new fan::vulkan(this);

#elif fan_renderer == fan_renderer_opengl

	this->set_vsync(true);

#endif

}

fan::window::window(const window& window) : fan::window(window.m_size, window.m_name, window.m_flags) {}

fan::window::window(window&& window)
{
	this->operator=(std::move(window));
}

fan::window& fan::window::operator=(const window& window)
{

	this->destroy_window();

	this->initialize_window(window.m_name, window.m_size, window.m_flags);

	this->m_close = window.m_close;
	this->m_current_frame = window.m_current_frame;
	this->m_delta_time = window.m_delta_time;
	this->m_fps = window.m_fps;
	this->m_fps_timer = window.m_fps_timer;

	this->m_keys_action = window.m_keys_action;
	this->m_keys_callback = window.m_keys_callback;
	this->m_text_callback = window.m_text_callback;
	this->m_keys_down = window.m_keys_down;
	this->m_keys_reset = window.m_keys_reset;
	this->m_key_callback = window.m_key_callback;
	this->m_last_frame = window.m_last_frame;
	this->m_max_fps = window.m_max_fps;
	this->m_mouse_move_position_callback = window.m_mouse_move_position_callback;
	this->m_mouse_position = window.m_mouse_position;
	this->m_move_callback = window.m_move_callback;
	this->m_position = window.m_position;
	this->m_previous_size = window.m_previous_size;
	this->m_fps_next_tick = window.m_fps_next_tick;
	this->m_received_fps = window.m_received_fps;
	this->m_resize_callback = window.m_resize_callback;
	this->m_close_callback = window.m_close_callback;
	this->m_size = window.m_size;
	this->m_vsync = window.m_vsync;

	this->m_name = window.m_name;
	this->m_flags = window.m_flags;
	this->m_current_key = window.m_current_key;
	this->m_reserved_flags = window.m_reserved_flags;

	this->m_raw_mouse_offset = window.m_raw_mouse_offset;
	this->m_focused = window.m_focused;
	this->m_auto_close = window.m_auto_close;
	this->m_background_color = window.m_background_color;

	return *this;
}

fan::window& fan::window::operator=(window&& window)
{
	this->destroy_window();

	this->m_close = std::move(window.m_close);
	this->m_current_frame = std::move(window.m_current_frame);
	this->m_delta_time = std::move(window.m_delta_time);
	this->m_fps = std::move(window.m_fps);
	this->m_fps_timer = std::move(window.m_fps_timer);

	#if defined(fan_platform_windows)

	this->m_hdc = std::move(window.m_hdc);

#if fan_renderer == fan_renderer_opengl

	this->m_context = std::move(window.m_context);

#endif

	#elif defined(fan_platform_unix) 

	m_display = window.m_display;
	m_screen = window.m_screen;
	m_atom_delete_window = window.m_atom_delete_window;
	m_window_attribs = window.m_window_attribs;
	m_visual = window.m_visual;

	#endif

	this->m_keys_action = std::move(window.m_keys_action);
	this->m_keys_callback = window.m_keys_callback;
	this->m_text_callback = std::move(window.m_text_callback);
	this->m_keys_down = std::move(window.m_keys_down);
	this->m_keys_reset = std::move(window.m_keys_reset);
	this->m_key_callback = window.m_key_callback;
	this->m_last_frame = std::move(window.m_last_frame);
	this->m_max_fps = std::move(window.m_max_fps);
	this->m_mouse_move_position_callback = window.m_mouse_move_position_callback;
	this->m_mouse_position = std::move(window.m_mouse_position);
	this->m_move_callback = window.m_move_callback;
	this->m_position = std::move(window.m_position);
	this->m_previous_size = std::move(window.m_previous_size);
	this->m_fps_next_tick = std::move(window.m_fps_next_tick);
	this->m_received_fps = std::move(window.m_received_fps);
	this->m_resize_callback = window.m_resize_callback;
	this->m_close_callback = window.m_close_callback;
	this->m_size = std::move(window.m_size);
	this->m_vsync = std::move(window.m_vsync);
	this->m_window = std::move(window.m_window);

	this->m_name = std::move(window.m_name);
	this->m_flags = window.m_flags;
	this->m_current_key = window.m_current_key;
	this->m_reserved_flags = window.m_reserved_flags;
	this->m_raw_mouse_offset = std::move(window.m_raw_mouse_offset);
	this->m_focused = std::move(window.m_focused);
	this->m_auto_close = std::move(window.m_auto_close);
	this->m_background_color = std::move(this->m_background_color);

#if fan_renderer == fan_renderer_opengl

	#if defined(fan_platform_windows)

	wglMakeCurrent(m_hdc, m_context);

	#elif defined(fan_platform_unix)

	glXMakeCurrent(m_display, m_window, m_context);

	#endif

#endif

	return *this;
}

fan::window::~window()
{
#if fan_renderer == fan_renderer_vulkan

	if (m_vulkan) {
		delete m_vulkan;
	}

#endif

	this->destroy_window();
}

void fan::window::execute(const std::function<void()>& function)
{
	if (!this->open()) {
		return;
	}

#if fan_renderer == fan_renderer_opengl

	#if defined(fan_platform_windows)

	if (wglGetCurrentContext() != m_context) {
		wglMakeCurrent(m_hdc, m_context);
	}

	glViewport(0, 0, m_size.x, m_size.y);

	#elif defined(fan_platform_unix)

	if (glXGetCurrentContext() != m_context) {
	glXMakeCurrent(m_display, m_window, m_context);
	}

	glViewport(0, 0, m_size.x, m_size.y);

	#endif

	glClearColor(
		m_background_color.r,
		m_background_color.g,
		m_background_color.b,
		m_background_color.a
	);
	glClear(GL_COLOR_BUFFER_BIT);

#elif fan_renderer == fan_renderer_vulkan
	m_vulkan->draw_frame();
#endif

	this->calculate_delta_time();

	if (function) {
		function();
	}

	if (f_t fps = static_cast<f_t>(timer_interval_t::period::den) / get_max_fps()) {
		m_fps_next_tick += fps;
		auto time = timer_interval_t(static_cast<uint_t>(std::ceil(m_fps_next_tick - fan::timer<timer_interval_t>::get_time())));
		fan::delay(timer_interval_t(std::max(static_cast<decltype(time.count())>(0), time.count())));
	}

#if fan_renderer == fan_renderer_opengl
	this->swap_buffers();
#endif

	this->reset_keys();
}

void fan::window::loop(const std::function<void()>& function) {

	while (fan::window::open()) {

		this->execute(function);

		fan::window::handle_events();

	}

#if fan_renderer == fan_renderer_vulkan
	vkDeviceWaitIdle(m_vulkan->device);
#endif
}
#if fan_renderer == fan_renderer_opengl
void fan::window::swap_buffers() const
{

	#ifdef fan_platform_windows
	SwapBuffers(m_hdc);
	#elif defined(fan_platform_unix)
	glXSwapBuffers(m_display, m_window);
	#endif

}
#endif

std::string fan::window::get_name() const
{
	return m_name;
}

void fan::window::set_name(const std::string& name)
{

	m_name = name;

	#ifdef fan_platform_windows

	SetWindowTextA(m_window, m_name.c_str());

	#elif defined(fan_platform_unix)

	XStoreName(m_display, m_window, name.c_str());
	XSetIconName(m_display, m_window, name.c_str());

	#endif

}

void fan::window::calculate_delta_time()
{
	m_current_frame = fan::timer<microseconds>::get_time();
	m_delta_time = f_t(m_current_frame - m_last_frame) / 1000000;
	m_last_frame = m_current_frame;
}

f_t fan::window::get_delta_time() const
{
	return m_delta_time;
}

fan::vec2i fan::window::get_mouse_position() const
{
	return m_mouse_position;
}

fan::vec2i fan::window::get_previous_mouse_position() const
{
	return m_previous_mouse_position;
}

fan::vec2i fan::window::get_size() const
{
	return m_size;
}

fan::vec2i fan::window::get_previous_size() const 
{
	return m_previous_size;
}

void fan::window::set_size(const fan::vec2i& size)
{
	const fan::vec2i move_offset = (size - get_previous_size()) / 2;

	#ifdef fan_platform_windows

	const fan::vec2i position = this->get_position();

#if fan_renderer == fan_renderer_opengl
	glViewport(0, 0, size.x, size.y);
#endif

	if (!SetWindowPos(m_window, 0, position.x, position.y, size.x, size.y, SWP_NOZORDER | SWP_SHOWWINDOW)) {
		fan::print("fan window error: failed to set window position", GetLastError());
		exit(1);
	}

	#elif defined(fan_platform_unix)

	int result = XResizeWindow(m_display, m_window, size.x, size.y);

	if (result == BadValue || result == BadWindow) {
		fan::print("fan window error: failed to set window position");
		exit(1);
	}

	this->set_position(this->get_position() - move_offset);

	#endif

}

fan::vec2i fan::window::get_position() const
{
	return m_position;
}

void fan::window::set_position(const fan::vec2i& position)
{
	#ifdef fan_platform_windows

	if (!SetWindowPos(m_window, 0, position.x, position.y, 0, 0, SWP_NOSIZE | SWP_NOZORDER | SWP_SHOWWINDOW)) {
		fan::print("fan window error: failed to set window position", GetLastError());
		exit(1);
	}

	#elif defined(fan_platform_unix)

	int result = XMoveWindow(m_display, m_window, position.x, position.y);

	if (result == BadValue || result == BadWindow) {
		fan::print("fan window error: failed to set window position");
		exit(1);
	}

	#endif
}

uint_t fan::window::get_max_fps() const {
	return m_max_fps;
}

void fan::window::set_max_fps(uint_t fps) {
	m_max_fps = fps;
	m_fps_next_tick = fan::timer<timer_interval_t>::get_time();
}

bool fan::window::vsync_enabled() const {
	return m_vsync;
}

void fan::window::set_vsync(bool value) {

#if fan_renderer == fan_renderer_opengl

	#if defined(fan_platform_windows)

		wglMakeCurrent(m_hdc, m_context);

	#elif defined(fan_platform_unix)

		glXMakeCurrent(m_display, m_window, m_context);

	#endif

	#ifdef fan_platform_windows

		PFNWGLSWAPINTERVALEXTPROC swap_interval = (PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT");

		if (swap_interval) {
			swap_interval(value);
		}
		else {
			fan::print("vsync not supported");
		}

	#elif defined(fan_platform_unix)

		GLXDrawable drawable = glXGetCurrentDrawable();

		if (drawable) {
			if (glXSwapIntervalEXT) {
				glXSwapIntervalEXT(m_display, drawable, value);
			}
			else {
				fan::print("vsync not supported");
			}
		}
	#endif

#endif
	
	m_vsync = value;
}

void fan::window::set_full_screen(const fan::vec2i& size)
{
	fan::window::set_size_mode(fan::window::mode::full_screen);

	fan::vec2i new_size;

	if (size == uninitialized) {
		new_size = fan::get_resolution();
	}
	else {
		new_size = size;
	}

	#ifdef fan_platform_windows

	this->set_resolution(new_size, fan::window::get_size_mode());

	this->set_windowed_full_screen();

	#elif defined(fan_platform_unix)

	this->set_windowed_full_screen(); // yeah

	#endif

}

void fan::window::set_windowed_full_screen(const fan::vec2i& size)
{

	fan::window::set_size_mode(fan::window::mode::borderless);

	fan::vec2i new_size;

	if (size == uninitialized) {
		new_size = fan::get_resolution();
	}
	else {
		new_size = size;
	}

	#ifdef fan_platform_windows

	DWORD dwStyle = GetWindowLong(m_window, GWL_STYLE);

	MONITORINFO mi = { sizeof(mi) };

	if (GetMonitorInfo(MonitorFromWindow(m_window, MONITOR_DEFAULTTOPRIMARY), &mi)) {
		SetWindowLong(m_window, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
		SetWindowPos(
			m_window, HWND_TOP,
			mi.rcMonitor.left, mi.rcMonitor.top,
			new_size.x,
			new_size.y,
			SWP_NOOWNERZORDER | SWP_FRAMECHANGED
		);
	}

	#elif defined(fan_platform_unix)

	struct MwmHints {
		unsigned long flags;
		unsigned long functions;
		unsigned long decorations;
		long input_mode;
		unsigned long status;
	};

	enum {
		MWM_HINTS_FUNCTIONS = (1L << 0),
		MWM_HINTS_DECORATIONS =  (1L << 1),

		MWM_FUNC_ALL = (1L << 0),
		MWM_FUNC_RESIZE = (1L << 1),
		MWM_FUNC_MOVE = (1L << 2),
		MWM_FUNC_MINIMIZE = (1L << 3),
		MWM_FUNC_MAXIMIZE = (1L << 4),
		MWM_FUNC_CLOSE = (1L << 5)
	};

	Atom mwmHintsProperty = XInternAtom(m_display, "_MOTIF_WM_HINTS", 0);
	struct MwmHints hints;
	hints.flags = MWM_HINTS_DECORATIONS;
	hints.functions = 0;
	hints.decorations = 0;
	XChangeProperty(m_display, m_window, mwmHintsProperty, mwmHintsProperty, 32,
		PropModeReplace, (unsigned char *)&hints, 5);

	XMoveResizeWindow(m_display, m_window, 0, 0, size.x, size.y);

	#endif

}

void fan::window::set_windowed(const fan::vec2i& size)
{
	fan::window::set_size_mode(fan::window::mode::windowed);

	fan::vec2i new_size;
	if (size == uninitialized) {
		new_size = this->get_previous_size();
	}
	else {
		new_size = size;
	}

	#ifdef fan_platform_windows

	this->set_resolution(0, fan::window::get_size_mode());

	const fan::vec2i position = fan::get_resolution() / 2 - new_size / 2;

	ShowWindow(m_window, SW_SHOW);

	SetWindowLongPtr(m_window, GWL_STYLE, WS_OVERLAPPEDWINDOW | WS_VISIBLE);

	SetWindowPos(
		m_window,
		0,
		position.x,
		position.y,
		new_size.x,
		new_size.y,
		SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED
	);

	#elif defined(fan_platform_unix)



	#endif
}

void fan::window::set_resolution(const fan::vec2i& size, const mode& mode) const
{
	if (mode == mode::full_screen) {
		fan::set_screen_resolution(size);
	}
	else {
		fan::reset_screen_resolution();
	}
}

fan::window::mode fan::window::get_size_mode() const
{
	return flag_values::m_size_mode;
}

void fan::window::set_size_mode(const mode& mode)
{
	flag_values::m_size_mode = mode;
}

template <typename type>
type fan::window::get_window_storage(const fan::window_t& window, const std::string& location) {
	auto found = m_window_storage.find(std::make_pair(window, location));
	if (found == m_window_storage.end()) {
		return {};
	}
	return std::any_cast<type>(found->second);
}

void fan::window::set_window_storage(const fan::window_t& window, const std::string& location, std::any data)
{
	m_window_storage.insert_or_assign(std::make_pair(window, location), data);
}

void fan::window::set_keys_callback(const keys_callback_t& function)
{
	this->m_keys_callback = function;
}

void fan::window::remove_keys_callback()
{
	this->m_keys_callback = 0;
}

std::deque<fan::window::key_callback_t>::iterator fan::window::add_key_callback(uint16_t key, key_state state, const std::function<void()>& function)
{
	this->m_key_callback.emplace_back(key_callback_t{ key, state, function });

	return this->m_key_callback.end() - 1;
}

void fan::window::edit_key_callback(std::deque<key_callback_t>::iterator it, uint16_t key, key_state state)
{
	it->key = key;
	it->state = state;
}

void fan::window::remove_key_callback(std::deque<key_callback_t>::const_iterator it)
{
	this->m_key_callback.erase(it);
}

void fan::window::set_text_callback(const text_callback_t& function)
{
	m_text_callback = function;
}

void fan::window::remove_text_callback()
{
	m_text_callback = 0;
}

std::deque<std::function<void()>>::iterator fan::window::add_close_callback(const std::function<void()>& function)
{
	m_close_callback.emplace_back(function);

	return this->m_close_callback.end() - 1;
}

void fan::window::remove_close_callback(std::deque<std::function<void()>>::const_iterator it)
{
	this->m_close_callback.erase(it);
}

std::deque<fan::window::mouse_move_position_callback_t>::iterator fan::window::add_mouse_move_callback(const mouse_move_position_callback_t& function)
{
	this->m_mouse_move_position_callback.emplace_back(function);

	return this->m_mouse_move_position_callback.end() - 1;
}

void fan::window::remove_mouse_move_callback(std::deque<mouse_move_position_callback_t>::const_iterator it)
{
	this->m_mouse_move_position_callback.erase(it);
}

std::deque<std::function<void(const fan::vec2i&)>>::iterator fan::window::add_resize_callback(const std::function<void(const fan::vec2i& window_size)>& function)
{
	this->m_resize_callback.emplace_back(function);

	return this->m_resize_callback.end() - 1;
}

void fan::window::remove_resize_callback(std::deque<std::function<void(const fan::vec2i& window_size)>>::const_iterator it)
{
	this->m_resize_callback.erase(it);
}

std::deque<std::function<void()>>::iterator fan::window::add_move_callback(const std::function<void()>& function)
{
	this->m_move_callback.push_back(function);

	return m_move_callback.end() - 1;
}

void fan::window::remove_move_callback(std::deque<std::function<void()>>::const_iterator it)
{
	this->m_move_callback.erase(it);
}

#if fan_renderer == fan_renderer_opengl

void message_callback(GLenum source,
	GLenum type,
	GLuint id,
	GLenum severity,
	GLsizei length,
	const GLchar* message,
	const void* userParam )
{
	if (type == 33361) { // gl_static_draw
		return;
	}
	fan::print_no_space(type == GL_DEBUG_TYPE_ERROR ? "opengl error:" : "", type, ", severity:", severity, ", message:", message);
}

#endif

void fan::window::set_error_callback()
{
#if fan_renderer == fan_renderer_opengl

	glEnable(GL_DEBUG_OUTPUT);
	glDebugMessageCallback((GLDEBUGPROC)message_callback, 0);

#elif fan_renderer == fan_renderer_vulkan

#endif
}

void fan::window::set_background_color(const fan::color& color)
{
	m_background_color = color;
}

fan::window_t fan::window::get_handle() const
{
	return m_window;
}

uint_t fan::window::get_fps(bool window_name, bool print)
{
	if (!m_fps_timer.get_reset_time()) {
		m_fps_timer = fan::timer<>(fan::timer<>::start(), 1000);
	}

	if (m_received_fps) {
		m_fps = 0;
		m_received_fps = false;
	}

	if (m_fps_timer.finished()) {
		std::string fps_info;
		if (window_name || print) {
			fps_info.append(
				std::string("FPS: ") +
				std::to_string(m_fps) +
				std::string(" frame time: ") +
				std::to_string(1.0 / m_fps * 1000) +
				std::string(" ms")
			).c_str();
		}
		if (window_name) {
			this->set_name(fps_info.c_str());
		}
		if (print) {
			fan::print(fps_info);
		}

		m_fps_timer.restart();
		m_received_fps = true;
		return m_fps;
	}

	m_fps++;
	return 0;
}

bool fan::window::key_press(uint16_t key) const
{
	auto found = this->m_keys_down.find(key);
	if (found == this->m_keys_down.end()) {
		fan::print("fan window error: incorrect key used in key_press:", key);
		exit(1);
	}
	return found->second;
}

bool fan::window::open() const {
	return !m_close;
}

void fan::window::close() {
	m_close = true;
}

bool fan::window::focused() const
{
	#ifdef fan_platform_windows
	return m_focused;
	#elif defined(fan_platform_unix)
	return 1;
	#endif

}

void fan::window::destroy_window()
{
	window::close();

	#if defined(fan_platform_windows)

	if (!m_window || !m_hdc 
#if fan_renderer == fan_renderer_opengl
		|| !m_context
#endif
		) {
		return;
	}

	fan::window_id_storage.erase(this->m_window);

	PostQuitMessage(0);

#if fan_renderer == fan_renderer_opengl
	wglMakeCurrent(m_hdc, 0);
#endif

	ReleaseDC(m_window, m_hdc);
	DestroyWindow(m_window);

	m_hdc = 0;

	#elif defined(fan_platform_unix)

	if (!m_display || !m_visual || !m_window_attribs.colormap) {
		return;
	}

	fan::window_id_storage.erase(this->m_window);

	XFree(m_visual);
	XFreeColormap(m_display, m_window_attribs.colormap);
	XDestroyWindow(m_display, m_window);

	glXDestroyContext(m_display, m_context);

	XCloseDisplay(m_display);

	m_visual = 0;
	m_window_attribs.colormap = 0;

	#endif

	m_window = 0;
}

uint16_t fan::window::get_current_key() const
{
	return m_current_key;
}

fan::vec2i fan::window::get_raw_mouse_offset() const
{
	return m_raw_mouse_offset;
}

void fan::window::window_input_action(fan::window_t window, uint16_t key) {

	fan::window* fwindow;

	#ifdef fan_platform_windows

	fwindow = get_window_by_id(window);

	#elif defined(fan_platform_unix)

	fwindow = get_window_by_id(window);

	#endif

	auto found = fwindow->m_keys_reset.find(key);

	if (found == fwindow->m_keys_reset.end()) {
		fwindow->m_keys_action.insert_or_assign(key, true);
	}

	for (const auto& i : fwindow->m_key_callback) {
		if (key != i.key || i.state == key_state::release || !fwindow->m_keys_action[key]) {
			continue;
		}

		fwindow->m_keys_reset.insert_or_assign(key, true);

		if (i.function) {
			i.function();
		}
	}

}

void fan::window::window_input_mouse_action(fan::window_t window, uint16_t key)
{
	fan::window* fwindow;

	#ifdef fan_platform_windows

	fwindow = fan::get_window_by_id(window);

	#elif defined(fan_platform_unix)

	fwindow = this;

	#endif

	fan::window_input::get_keys(fwindow->m_keys_down, key, true);

	for (const auto& i : fwindow->m_key_callback) {
		if (key != i.key || i.state == key_state::release) {
			continue;
		}
		if (i.function) {
			i.function();
		}
	}

}

void fan::window::window_input_up(fan::window_t window, uint16_t key)
{
	fan::window* fwindow;
	#ifdef fan_platform_windows

	fwindow = fan::get_window_by_id(window);

	#elif defined(fan_platform_unix)

	fwindow = this;

	#endif

	fan::window_input::get_keys(fwindow->m_keys_down, key, false);

	if (key <= fan::input::key_menu) {
		fan::window::window_input_action_reset(window, key);
	}

	for (const auto& i : fwindow->m_key_callback) {
		if (key != i.key || i.state == key_state::press) {
			continue;
		}
		if (i.function) {
			i.function();
		}
	}
}

void fan::window::window_input_action_reset(fan::window_t window, uint16_t key)
{

	fan::window* fwindow;

	#ifdef fan_platform_windows

	fwindow = fan::get_window_by_id(window);


	#elif defined(fan_platform_unix)

	fwindow = this;

	#endif

	for (const auto& i : fwindow->m_keys_action) {
		fwindow->m_keys_action[i.first] = false;
	}

	fwindow->m_keys_reset.clear();

}

#ifdef fan_platform_windows

static void handle_special(WPARAM wparam, LPARAM lparam, uint16_t& key, bool down) {
	if (wparam == 0x10 || wparam == 0x11) {
		if (down) {
			switch (lparam) {
				case fan::special_lparam::lshift_lparam_down:
				{
					key = fan::input::key_left_shift;
					break;
				}
				case fan::special_lparam::rshift_lparam_down:
				{
					key = fan::input::key_right_shift;
					break;
				}
				case fan::special_lparam::lctrl_lparam_down:
				{
					key = fan::input::key_left_control;
					break;
				}
				case fan::special_lparam::rctrl_lparam_down:
				{
					key = fan::input::key_right_control;
					break;
				}
			}
		}
		else {
			switch (lparam) {
				case fan::special_lparam::lshift_lparam_up: // ?
				{
					key = fan::input::key_left_shift;
					break;
				}
				case fan::special_lparam::rshift_lparam_up:
				{
					key = fan::input::key_right_shift;
					break;
				}
				case fan::special_lparam::lctrl_lparam_up:
				{
					key = fan::input::key_left_control;
					break;
				}
				case fan::special_lparam::rctrl_lparam_up: // ? 
				{
					key = fan::input::key_right_control;
					break;
				}
			}
		}
	}
	else {
		key = fan::window_input::convert_keys_to_fan(wparam);
	}

}

LRESULT fan::window::window_proc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam)
{

	switch (msg) {
		case WM_MOUSEMOVE:
		{
			auto window = fan::get_window_by_id(hwnd);

			if (!window) {
				break;
			}

			const auto get_cursor_position = [&] {
				POINT p;
				GetCursorPos(&p);
				ScreenToClient(window->m_window, &p);

				return fan::vec2i(p.x, p.y);
			};

			const fan::vec2i position(get_cursor_position());

			const auto cursor_in_range = [](const fan::vec2i& position, const fan::vec2& window_size) {
				return position.x >= 0 && position.x < window_size.x&&
					position.y >= 0 && position.y < window_size.y;
			};

			window->m_previous_mouse_position = window->m_mouse_position;
			window->m_mouse_position = position;

			for (const auto& i : window->m_mouse_move_position_callback) {
				if (i) {
					i(window->m_mouse_position);
				}
			}

			break;
		}
		case WM_MOVE:
		{

			auto window = fan::get_window_by_id(hwnd);

			if (!window) {
				break;
			}

			window->m_position = fan::vec2i(
				static_cast<int>(static_cast<short>(LOWORD(lparam))),
				static_cast<int>(static_cast<short>(HIWORD(lparam)))
			);

			for (const auto& i : window->m_move_callback) {
				if (i) {
					i();
				}
			}

			break;
		}
		case WM_SIZE:
		{
			fan::window* fwindow = fan::get_window_by_id(hwnd);

			if (!fwindow) {
				break;
			}

			RECT rect;
			GetClientRect(hwnd, &rect);


			fwindow->m_previous_size = fwindow->m_size;
			fwindow->m_size = fan::vec2i(rect.right - rect.left, rect.bottom - rect.top);

#if fan_renderer == fan_renderer_opengl
			wglMakeCurrent(fwindow->m_hdc, m_context);
			glViewport(0, 0, fwindow->m_size.x, fwindow->m_size.y);
#endif


			for (const auto& i : fwindow->m_resize_callback) {
				if (i) {
					i(fwindow->m_size);
				}
			}

			break;
		}
		case WM_SETFOCUS:
		{
			fan::window* fwindow = fan::get_window_by_id(hwnd);

			if (!fwindow) {
				break;
			}

			fwindow->m_focused = true;
			break;
		}
		case WM_KILLFOCUS:
		{
			fan::window* fwindow = fan::get_window_by_id(hwnd);
			if (!fwindow) {
				break;
			}

			for (auto& i : fwindow->m_keys_down) {
				window_input_up(fwindow->m_window, i.first);
				i.second = false;
			}

			fwindow->m_focused = false;
			break;
		}
		case WM_SYSCOMMAND:
		{
			//auto fwindow = get_window_storage<fan::window*>(m_window, stringify(this_window));
			// disable alt action for window
			if (wparam == SC_KEYMENU && (lparam >> 16) <= 0) {
				return 0;
			}

			break;
		}
		case WM_DESTROY:
		{

			PostQuitMessage(0);

			break;
		}
		case WM_CLOSE:
		{
			fan::window* fwindow = fan::get_window_by_id(hwnd);

			//if (fwindow->key_press(fan::key_alt)) {
			//	return 0;
			//}

			for (int i = 0; i < fwindow->m_close_callback.size(); i++) {
				if (fwindow->m_close_callback[i]) {
					fwindow->m_close_callback[i]();
				}
			}

			if (fwindow->m_auto_close) {
				fwindow->close();
			}
			else {
				return 0;
			}


			break;
		}
	}

	return DefWindowProc(hwnd, msg, wparam, lparam);
}
#endif

#ifdef fan_platform_unix

Display* fan::window::get_display()
{
	return m_display;
}

#endif

void fan::window::reset_keys()
{
	m_keys_down[fan::input::mouse_scroll_up] = false;
	m_keys_down[fan::input::mouse_scroll_down] = false;

	for (auto& i : m_keys_action) {
		i.second = false;
	}
}

	#ifdef fan_platform_windows
#if fan_renderer == fan_renderer_opengl

	void init_windows_opengl_extensions()
	{

		auto str = fan::random::string(10);
		WNDCLASSA window_class = {
			.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC,
			.lpfnWndProc = DefWindowProcA,
			.hInstance = GetModuleHandle(0),
			.lpszClassName = str.c_str(),
		};

		if (!RegisterClassA(&window_class)) {
			fan::print("failed to register window");
			exit(1);
		}

		HWND temp_window = CreateWindowExA(
			0,
			window_class.lpszClassName,
			"temp_window",
			0,
			CW_USEDEFAULT,
			CW_USEDEFAULT,
			CW_USEDEFAULT,
			CW_USEDEFAULT,
			0,
			0,
			window_class.hInstance,
			0);

		if (!temp_window) {
			fan::print("failed to create window");
			exit(1);
		}

		HDC temp_dc = GetDC(temp_window);

		PIXELFORMATDESCRIPTOR pfd = {
			sizeof(pfd),
			1,
			PFD_TYPE_RGBA,
			PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,
			32,
			8,
			PFD_MAIN_PLANE,
			24,
			8,
		};

		int pixel_format = ChoosePixelFormat(temp_dc, &pfd);
		if (!pixel_format) {
			fan::print("failed to choose pixel format");
			exit(1);
		}
		if (!SetPixelFormat(temp_dc, pixel_format, &pfd)) {
			fan::print("failed to set pixel format");
			exit(1);
		}

		HGLRC temp_context = wglCreateContext(temp_dc);
		if (!temp_context) {
			fan::print("failed to create context");
			exit(1);
		}

		if (!wglMakeCurrent(temp_dc, temp_context)) {
			fan::print("failed to make current");
			exit(1);
		}

		wglCreateContextAttribsARB = (decltype(wglCreateContextAttribsARB))wglGetProcAddress(
			"wglCreateContextAttribsARB");
		wglChoosePixelFormatARB = (decltype(wglChoosePixelFormatARB))wglGetProcAddress(
			"wglChoosePixelFormatARB");

		wglMakeCurrent(temp_dc, 0);
		wglDeleteContext(temp_context);
		ReleaseDC(temp_window, temp_dc);
		DestroyWindow(temp_window);
	}
#endif
	#elif defined(fan_platform_unix)

	static bool isExtensionSupported(const char *extList, const char *extension) {
		const char *start;
		const char *where, *terminator;

		where = strchr(extension, ' ');
		if (where || *extension == '\0') {
			return false;
		}

		for (start=extList;;) {
			where = strstr(start, extension);

			if (!where) {
				break;
			}

			terminator = where + strlen(extension);

			if ( where == start || *(where - 1) == ' ' ) {
				if ( *terminator == ' ' || *terminator == '\0' ) {
					return true;
				}
			}	

			start = terminator;
		}

		return false;
	}
	#endif

void fan::window::initialize_window(const std::string& name, const fan::vec2i& window_size, uint64_t flags)
{

	#ifdef fan_platform_windows

	auto instance = GetModuleHandle(NULL);

	WNDCLASS wc = { 0 };

	auto str = fan::random::string(10);

	wc.lpszClassName = str.c_str();

	wc.lpfnWndProc = fan::window::window_proc;

	wc.hCursor = LoadCursor(NULL, IDC_ARROW);

	wc.hInstance = instance;

	RegisterClass(&wc);

	const bool full_screen = flag_values::m_size_mode == fan::window::mode::full_screen;
	const bool borderless = flag_values::m_size_mode == fan::window::mode::borderless;

	RECT rect = { 0, 0, window_size.x, window_size.y };
	AdjustWindowRect(&rect, full_screen || borderless ? WS_POPUP : WS_OVERLAPPEDWINDOW, FALSE);

	const fan::vec2i position = fan::get_resolution() / 2 - window_size / 2;

	if (full_screen) {
		this->set_resolution(window_size, fan::window::mode::full_screen);
	}

	m_window = CreateWindow(str.c_str(), name.c_str(),
		(flag_values::m_no_resize ? ((full_screen || borderless ? WS_POPUP : (WS_OVERLAPPED | WS_MINIMIZEBOX | WS_SYSMENU)) | WS_SYSMENU ) :
		(full_screen || borderless ? WS_POPUP : WS_OVERLAPPEDWINDOW)) | WS_VISIBLE,
		position.x, position.y,
		rect.right - rect.left, rect.bottom - rect.top,
		0, 0, 0, 0);

	if (!m_window) {
		fan::print("fan window error: failed to initialize window", GetLastError());
		exit(1);
	}

	RAWINPUTDEVICE r_id;
	r_id.usUsagePage = HID_USAGE_PAGE_GENERIC;
	r_id.usUsage = HID_USAGE_GENERIC_MOUSE;
	r_id.dwFlags = RIDEV_INPUTSINK;
	r_id.hwndTarget = m_window;

	BOOL result = RegisterRawInputDevices(&r_id, 1, sizeof(RAWINPUTDEVICE));

	if (!result) {
		fan::print("failed to register raw input:", result);
		exit(1);
	}

#if fan_renderer == fan_renderer_opengl

	m_hdc = GetDC(m_window);

	init_windows_opengl_extensions();

	int pixel_format_attribs[19] = {
		WGL_DRAW_TO_WINDOW_ARB, GL_TRUE,
		WGL_SUPPORT_OPENGL_ARB, GL_TRUE,
		WGL_DOUBLE_BUFFER_ARB, GL_TRUE,
		WGL_PIXEL_TYPE_ARB, WGL_TYPE_RGBA_ARB,
		WGL_COLOR_BITS_ARB, 32,
		WGL_DEPTH_BITS_ARB, 24,
		WGL_STENCIL_BITS_ARB, 8,
		fan::WINDOW_SAMPLE_BUFFER, true, // Number of buffers (must be 1 at time of writing)
		fan::OPENGL_SAMPLES, fan::window::flag_values::m_samples,        // Number of samples
		0
	};
	if (!fan::window::flag_values::m_samples) {
		// set back to zero to disable antialising
		for (int i = 0; i < 4; i++) {
			pixel_format_attribs[14 + i] = 0;
		}
	}
	int pixel_format;
	UINT num_formats;

	wglChoosePixelFormatARB(m_hdc, pixel_format_attribs, 0, 1, &pixel_format, &num_formats);
	if (!num_formats) {
		fan::print("failed to choose pixel format", GetLastError());
		exit(1);
	}

	PIXELFORMATDESCRIPTOR pfd;
	memset(&pfd, 0, sizeof(pfd));
	DescribePixelFormat(m_hdc, pixel_format, sizeof(pfd), &pfd);
	if (!SetPixelFormat(m_hdc, pixel_format, &pfd)) {
		fan::print("failed to set pixel format");
		exit(1);
	}

	const int gl_attributes[] = {
		fan::WINDOW_MINOR_VERSION, flag_values::m_minor_version,
		fan::WINDOW_MAJOR_VERSION, flag_values::m_major_version,
		WGL_CONTEXT_PROFILE_MASK_ARB,  WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
		0,
	};

	if (!m_context) {
	
		if (flag_values::m_major_version < 3) {
			m_context = wglCreateContext(m_hdc);
		}
		else {
			m_context = wglCreateContextAttribsARB(m_hdc, 0, gl_attributes);
		}

		if (!m_context) {
			fan::print("failed to create context");
			exit(1);
		}
	}

	if (!wglMakeCurrent(m_hdc, m_context)) {
		fan::print("failed to make current");
		exit(1);
	}

	static bool initialize_glew = true;

	if (initialize_glew) {
		if (glewInit() != GLEW_OK) { // maybe
			fan::print("failed to initialize glew");
			exit(1);
		}
		initialize_glew = false;
	}

#endif

	ShowCursor(!flag_values::m_no_mouse);
	if (flag_values::m_no_mouse) {
		auto middle = this->get_position() + this->get_size() / 2;
		SetCursorPos(middle.x, middle.y);
	}


	#elif defined(fan_platform_unix)

	// if vulkan
	XInitThreads();

	if (!m_display) {
		m_display = XOpenDisplay(NULL);
		if (!m_display) {
			throw std::runtime_error("failed to initialize window");
		}
	}

	static bool abc = true;

	if (abc) {
		m_screen = DefaultScreen(m_display);
		abc = false;
	}

	int minor_glx = 0, major_glx = 0;
	glXQueryVersion(m_display, &major_glx, &minor_glx);

	if (minor_glx < flag_values::m_minor_version && major_glx <= flag_values::m_major_version) {
		fan::print("fan window error: too low glx version");
		XCloseDisplay(m_display);
		exit(1);
	}

	int pixel_format_attribs[] = {
		GLX_X_RENDERABLE			 , True,
		GLX_DRAWABLE_TYPE			 , GLX_WINDOW_BIT,
		GLX_RENDER_TYPE			 , GLX_RGBA_BIT,
		GLX_X_VISUAL_TYPE			 , GLX_TRUE_COLOR,
		GLX_RED_SIZE				 , 8,
		GLX_GREEN_SIZE			 , 8,
		GLX_BLUE_SIZE				 , 8,
		GLX_ALPHA_SIZE			 , 8,
		GLX_DEPTH_SIZE			 , 24,
		GLX_STENCIL_SIZE			 , 8,
		GLX_DOUBLEBUFFER			 , True,
		fan::WINDOW_SAMPLE_BUFFER  , 1,
		fan::OPENGL_SAMPLES        , fan::window::flag_values::m_samples,
		None
	};

	if (!fan::window::flag_values::m_samples) {
		// set back to zero to disable antialising
		for (int i = 0; i < 4; i++) {
			pixel_format_attribs[22 + i] = 0;
		}
	}

	glXChooseFBConfig =
		(decltype(glXChooseFBConfig))
		glXGetProcAddress((GLubyte*)"glXChooseFBConfig");

	glXGetVisualFromFBConfig =
		(PFNGLXGETVISUALFROMFBCONFIGPROC)
		glXGetProcAddress((GLubyte*)"glXGetVisualFromFBConfig");
	glXCreateContextAttribsARB =
		(PFNGLXCREATECONTEXTATTRIBSARBPROC)
		glXGetProcAddress((GLubyte*)"glXCreateContextAttribsARB");

	glXGetFBConfigAttrib = (
		decltype(glXGetFBConfigAttrib))
		glXGetProcAddress((const GLubyte*)"glXGetFBConfigAttrib");

	int fbcount;

	auto fbc = glXChooseFBConfig(m_display, m_screen, pixel_format_attribs, &fbcount);

	if (!fbc) {
		fan::print("fan window error: failed to retreive framebuffer");
		XCloseDisplay(m_display);
		exit(1);
	}

	int best_fbc = -1, worst_fbc = -1, best_num_samp = -1, worst_num_samp = 999;
	for (int i = 0; i < fbcount; ++i) {
		XVisualInfo *vi = glXGetVisualFromFBConfig( m_display, fbc[i] );
		if ( vi != 0) {
			int samp_buf, samples;
			if (!glXGetFBConfigAttrib) {
				exit(1);
			}
			glXGetFBConfigAttrib( m_display, fbc[i], GLX_SAMPLE_BUFFERS, &samp_buf );
			glXGetFBConfigAttrib( m_display, fbc[i], GLX_SAMPLES       , &samples  );

			if ( best_fbc < 0 || (samp_buf && samples > best_num_samp) ) {
				best_fbc = i;
				best_num_samp = samples;
			}
			if ( worst_fbc < 0 || !samp_buf || samples < worst_num_samp )
				worst_fbc = i;
			worst_num_samp = samples;
		}
		XFree( vi );
	}

	GLXFBConfig bestFbc = fbc[ best_fbc ];

	XFree(fbc);

	m_visual = glXGetVisualFromFBConfig(m_display, bestFbc);

	if (!m_visual) {
		fan::print("fan window error: failed to create visual");
		XCloseDisplay(m_display);
		exit(1);
	}

	if (m_screen != m_visual->screen) {
		fan::print("fan window error: screen doesn't match with visual screen");
		XCloseDisplay(m_display);
		exit(1);
	}

	std::memset(&m_window_attribs, 0, sizeof(m_window_attribs));

	m_window_attribs.border_pixel = BlackPixel(m_display, m_screen);
	m_window_attribs.background_pixel = WhitePixel(m_display, m_screen);
	m_window_attribs.override_redirect = True;
	m_window_attribs.colormap = XCreateColormap(m_display, RootWindow(m_display, m_screen), m_visual->visual, AllocNone);
	m_window_attribs.event_mask = ExposureMask | KeyPressMask | ButtonPress |
		StructureNotifyMask | ButtonReleaseMask |
		KeyReleaseMask | EnterWindowMask | LeaveWindowMask |
		PointerMotionMask | Button1MotionMask | VisibilityChangeMask |
		ColormapChangeMask;


	const fan::vec2i position = fan::get_resolution() / 2 - window_size / 2;

	m_window = XCreateWindow(
		m_display, 
		RootWindow(m_display, m_screen), 
		position.x, 
		position.y,
		window_size.x, 
		window_size.y, 
		0,
		m_visual->depth, 
		InputOutput, 
		m_visual->visual, 
		CWBackPixel | CWColormap | CWBorderPixel | CWEventMask | CWCursor, 
		&m_window_attribs
	);

	if (flags & fan::window::flags::no_resize) {
		auto sh = XAllocSizeHints();
		sh->flags = PMinSize | PMaxSize;
		sh->min_width = sh->max_width = window_size.x;
		sh->min_height = sh->max_height = window_size.y;
		XSetWMSizeHints(m_display, m_window, sh, XA_WM_NORMAL_HINTS);
		XFree(sh);
	}

	this->set_name(name);

	if (!m_atom_delete_window) {
		m_atom_delete_window = XInternAtom(m_display, "WM_DELETE_WINDOW", False);
	}

	XSetWMProtocols(m_display, m_window, &m_atom_delete_window, 1);

	int gl_attribs[] = {
		fan::WINDOW_MINOR_VERSION, flag_values::m_minor_version,
		fan::WINDOW_MAJOR_VERSION, flag_values::m_major_version,
		GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
		0
	};

	bool initialize_context = !m_context;

	const char *glxExts = glXQueryExtensionsString(m_display, m_screen);
	if (!isExtensionSupported( glxExts, "GLX_ARB_create_context") && initialize_context) {
		std::cout << "GLX_ARB_create_context not supported\n";
		m_context = glXCreateNewContext(m_display, bestFbc, GLX_RGBA_TYPE, 0, True );
	}
	else if (initialize_context){
		m_context = glXCreateContextAttribsARB(m_display, bestFbc, 0, true, gl_attribs);
	}

	initialize_context = false;

	XSync(m_display, True);

#if fan_renderer == fan_renderer_opengl
	glXMakeCurrent(m_display, m_window, m_context);
#endif

	XClearWindow(m_display, m_window);
	XMapRaised(m_display, m_window);
	XAutoRepeatOn(m_display);

#if fan_renderer == fan_renderer_opengl

	static bool initialized = false;

	if (!initialized) {
		if (glewInit() != GLEW_OK) { // maybe
			fan::print("failed to initialize glew");
			exit(1);
		}
		initialized = true;
	}

#endif

	m_xim = XOpenIM(m_display, 0, 0, 0);

	if(!m_xim){
		// fallback to internal input method
		XSetLocaleModifiers("@im=none");
		m_xim = XOpenIM(m_display, 0, 0, 0);
	}

	m_xic = XCreateIC(m_xim,
		XNInputStyle,   XIMPreeditNothing | XIMStatusNothing,
		XNClientWindow, m_window,
		XNFocusWindow,  m_window,
		NULL);

	XSetICFocus(m_xic);

	#endif

	m_position = position;

	m_previous_size = m_size;

	for (int i = 0; i != fan::input::last; ++i) {
		m_keys_down[i] = false;
	}

#if fan_renderer == fan_renderer_opengl

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glViewport(0, 0, window_size.x, window_size.y);

#endif

	this->set_vsync(false);

	set_window_by_id(m_window, this);
}

void fan::window::handle_events() {

	#ifdef fan_platform_windows

	MSG msg{};

	while (PeekMessageW(&msg, 0, 0, 0, PM_REMOVE))
	{

		switch (msg.message) {
			case WM_KEYDOWN:
			{
				auto window = fan::get_window_by_id(msg.hwnd);

				if (!window) {
					break;
				}

				uint16_t key;

				handle_special(msg.wParam, msg.lParam, key, true);

				fan::window_input::get_keys(window->m_keys_down, key, true);

				fan::window::window_input_action(window->m_window, key);

				window->m_current_key = key;

				if (window->m_keys_callback) {
					window->m_keys_callback(key, key_state::press);
				}

				break;
			}
			case WM_CHAR:
			{

				auto window = fan::get_window_by_id(msg.hwnd);

				if (!window) {
					break;
				}

				if (msg.wParam < 8) {
					window->m_reserved_flags |= msg.wParam;
				}
				else {

					bool found = false;

					if (!found) {

						uint16_t fan_key = fan::window_input::convert_keys_to_fan(msg.wParam);

						bool found = false;

						for (auto i : banned_keys) {
							if (fan_key == i) {

								found = true;
								break;
							}
						}

						if (window->m_text_callback && !found) {
							window->m_text_callback(msg.wParam + (window->m_reserved_flags << 8));
						}

						window->m_reserved_flags = 0;
					}

				}

				break;
			}
			case WM_LBUTTONDOWN:
			{
				auto window = fan::get_window_by_id(msg.hwnd);

				if (!window) {
					break;
				}

				const uint16_t key = fan::input::mouse_left;

				fan::window::window_input_mouse_action(window->m_window, key);

				if (window->m_keys_callback) {
					window->m_keys_callback(key, key_state::press);
				}

				break;
			}
			case WM_RBUTTONDOWN:
			{
				auto window = fan::get_window_by_id(msg.hwnd);

				if (!window) {
					break;
				}

				const uint16_t key = fan::input::mouse_right;

				fan::window::window_input_mouse_action(window->m_window, key);

				if (window->m_keys_callback) {
					window->m_keys_callback(key, key_state::press);
				}

				break;
			}
			case WM_MBUTTONDOWN:
			{
				auto window = fan::get_window_by_id(msg.hwnd);

				if (!window) {
					break;
				}

				const uint16_t key = fan::input::mouse_middle;

				fan::window::window_input_mouse_action(window->m_window, key);

				if (window->m_keys_callback) {
					window->m_keys_callback(key, key_state::press);
				}

				break;
			}
			case WM_KEYUP:
			{
				auto window = fan::get_window_by_id(msg.hwnd);

				if (!window) {
					break;
				}

				uint16_t key = 0;

				handle_special(msg.wParam, msg.lParam, key, false);

				window_input_up(window->m_window, key);

				if (window->m_keys_callback) {
					window->m_keys_callback(key, key_state::release);
				}

				break;
			}
			case WM_MOUSEWHEEL:
			{
				auto fwKeys = GET_KEYSTATE_WPARAM(msg.wParam);
				auto zDelta = GET_WHEEL_DELTA_WPARAM(msg.wParam);

				auto window = fan::get_window_by_id(msg.hwnd);

				fan::window_input::get_keys(window->m_keys_down, zDelta < 0 ? fan::input::mouse_scroll_down : fan::input::mouse_scroll_up, true);

				fan::window::window_input_mouse_action(window->m_window, zDelta < 0 ? fan::input::mouse_scroll_down : fan::input::mouse_scroll_up);

				if (window->m_keys_callback) {
					window->m_keys_callback(zDelta < 0 ? fan::input::mouse_scroll_down : fan::input::mouse_scroll_up, key_state::press);
				}

				break;
			}

			case WM_INPUT:
			{
				auto window = fan::get_window_by_id(msg.hwnd);

				if (!window) {
					break;
				}

				if (!window->focused()) {
					break;
				}

				UINT size = sizeof(RAWINPUT);
				BYTE data[sizeof(RAWINPUT)];

				GetRawInputData((HRAWINPUT)msg.lParam, RID_INPUT, data, &size, sizeof(RAWINPUTHEADER));

				RAWINPUT* raw = (RAWINPUT*)data;

				static bool allow_outside = false;

				const auto cursor_in_range = [](const fan::vec2i& position, const fan::vec2& window_size) {
					return position.x >= 0 && position.x < window_size.x&&
						position.y >= 0 && position.y < window_size.y;
				};

				if (raw->header.dwType == RIM_TYPEMOUSE)
				{

					const auto get_cursor_position = [&] {
						POINT p;
						GetCursorPos(&p);
						ScreenToClient(window->m_window, &p);

						return fan::vec2i(p.x, p.y);
					};

					if (fan::is_flag(raw->data.mouse.usButtonFlags, RI_MOUSE_LEFT_BUTTON_DOWN) ||
						fan::is_flag(raw->data.mouse.usButtonFlags, RI_MOUSE_MIDDLE_BUTTON_DOWN) ||
						fan::is_flag(raw->data.mouse.usButtonFlags, RI_MOUSE_RIGHT_BUTTON_DOWN)
						) {

						const fan::vec2i position(get_cursor_position());

						if (cursor_in_range(position, window->get_size())) {
							allow_outside = true;
						}

						if (fan::window::flag_values::m_no_mouse) {
							RECT rect;
							GetClientRect(window->m_window, &rect);

							POINT ul;
							ul.x = rect.left;
							ul.y = rect.top;

							POINT lr;
							lr.x = rect.right;
							lr.y = rect.bottom;

							MapWindowPoints(window->m_window, nullptr, &ul, 1);
							MapWindowPoints(window->m_window, nullptr, &lr, 1);

							rect.left = ul.x;
							rect.top = ul.y;

							rect.right = lr.x;
							rect.bottom = lr.y;

							ClipCursor(&rect);
						}
						else {
							window->m_previous_mouse_position = window->m_mouse_position;
							window->m_mouse_position = position;
						}

					}

					else if (fan::is_flag(raw->data.mouse.usButtonFlags, RI_MOUSE_LEFT_BUTTON_UP)) {

						if (window->m_keys_callback) {
							window->m_keys_callback(fan::input::mouse_left, key_state::release);
						}

						window_input_up(window->m_window, fan::input::mouse_left); allow_outside = false;
					}

					else if (fan::is_flag(raw->data.mouse.usButtonFlags, RI_MOUSE_MIDDLE_BUTTON_UP)) {

						if (window->m_keys_callback) {
							window->m_keys_callback(fan::input::mouse_middle, key_state::release);
						}

						window_input_up(window->m_window, fan::input::mouse_middle); allow_outside = false;
					}

					else if (fan::is_flag(raw->data.mouse.usButtonFlags, RI_MOUSE_RIGHT_BUTTON_UP)) {

						if (window->m_keys_callback) {
							window->m_keys_callback(fan::input::mouse_right, key_state::release);
						}

						window_input_up(window->m_window, fan::input::mouse_right); allow_outside = false;
					}

					else if ((raw->data.mouse.usFlags & MOUSE_MOVE_RELATIVE) == MOUSE_MOVE_RELATIVE) {

						const fan::vec2i position(get_cursor_position());

						window->m_raw_mouse_offset = fan::vec2i(raw->data.mouse.lLastX, raw->data.mouse.lLastY);

						if ((!cursor_in_range(position, window->get_size()) && !allow_outside)) {
							break;
						}

						if (fan::window::flag_values::m_no_mouse) {
							RECT rect;
							GetClientRect(window->m_window, &rect);

							POINT ul;
							ul.x = rect.left;
							ul.y = rect.top;

							POINT lr;
							lr.x = rect.right;
							lr.y = rect.bottom;

							MapWindowPoints(window->m_window, nullptr, &ul, 1);
							MapWindowPoints(window->m_window, nullptr, &lr, 1);

							rect.left = ul.x;
							rect.top = ul.y;

							rect.right = lr.x;
							rect.bottom = lr.y;

							ClipCursor(&rect);
						}
					}

				}
				break;
			}
		}

		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}



	#elif defined(fan_platform_unix)

	XEvent event;

	int nevents = XEventsQueued(m_display, QueuedAfterReading);

	while (nevents--) {
		XNextEvent(m_display, &event);
		// if (XFilterEvent(&m_event, m_window))
		// 	continue;

		switch (event.type) {

			case Expose:
			{
				auto window = fan::get_window_by_id(event.xexpose.window);

				if (!window) {
					break;
				}

				XWindowAttributes attribs;
				XGetWindowAttributes(m_display, window->m_window, &attribs);

#if fan_renderer == fan_renderer_opengl

				glXMakeCurrent(m_display, window->m_window, m_context);

				glViewport(0, 0, attribs.width, attribs.height);

#endif

				window->m_previous_size = window->m_size;
				window->m_size = fan::vec2i(attribs.width, attribs.height);

				for (const auto& i : window->m_resize_callback) {
					if (i) {
						i(window->m_size);
					}
				}

				break;
			}
			// case ConfigureNotify:
			// {

			// 	for (const auto& i : window.m_move_callback) {
			// 		if (i) {
			// 			i();
			// 		}
			// 	}

			// 	break;
			// }
			case ClientMessage:
			{
				auto window = fan::get_window_by_id(event.xclient.window);

				if (!window) {
					break;
				}

				if (event.xclient.data.l[0] == (long)m_atom_delete_window) {
					for (uint_t i = 0; i < window->m_close_callback.size(); i++) {
						if (window->m_close_callback[i]) {
							window->m_close_callback[i]();
						}
					}

					if (window->m_auto_close) {
						window->destroy_window();
					}

				}

				break;
			}
			case KeyPress:
			{

				auto window = fan::get_window_by_id(event.xkey.window);

				if (!window) {
					break;
				}

				uint16_t key = fan::window_input::convert_keys_to_fan(event.xkey.keycode);

				fan::window_input::get_keys(window->m_keys_down, key, true);

				fan::window::window_input_action(window->m_window, key);

				window->m_current_key = key;

				if (window->m_keys_callback) {
					window->m_keys_callback(key, key_state::press);
				}

				KeySym keysym;

				char text[32] = {};

				Status status;
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
				std::wstring str;

				Xutf8LookupString(window->m_xic, &event.xkey, text, sizeof(text) - 1, &keysym, &status);

				str = converter.from_bytes(text);

				bool found = false;

				for (auto i : banned_keys) {
					if (key == i) {
						found = true;
						break;
					}
				}

				if (str.size() && !found) {
					if (window->m_text_callback) {
						if (!str.size()) {
							window->m_text_callback(key);
						}
						else {
							window->m_text_callback(str[0]);
						}
					}
				}

				break;
			}
			case KeyRelease:
			{

				auto window = fan::get_window_by_id(event.xkey.window);

				if (!window) {
					break;
				}

				if (XEventsQueued(window->m_display, QueuedAfterReading)) {
					XEvent nev;
					XPeekEvent(window->m_display, &nev);

					if (nev.type == KeyPress && nev.xkey.time == event.xkey.time &&
						nev.xkey.keycode == event.xkey.keycode) {
						break;
					}
				}

				const uint16_t key = fan::window_input::convert_keys_to_fan(event.xkey.keycode);

				window->window_input_up(window->m_window, key);

				if (window->m_keys_callback) {
					window->m_keys_callback(key, key_state::release);
				}

				break;
			}
			case MotionNotify:
			{

				auto window = fan::get_window_by_id(event.xmotion.window);

				if (!window) {
					break;
				}

				const fan::vec2i position(event.xmotion.x, event.xmotion.y);

				auto mouse_move_position_callback = window->m_mouse_move_position_callback;

				for (const auto& i : mouse_move_position_callback) {
					if (i) {
						i(position);
					}
				}

				window->m_previous_mouse_position = window->m_mouse_position;

				window->m_mouse_position = position;

				break;
			}
			case ButtonPress:
			{

				auto window = fan::get_window_by_id(event.xbutton.window);

				if (!window) {
					break;
				}

				uint16_t key = fan::window_input::convert_keys_to_fan(event.xbutton.button);

				window->window_input_mouse_action(window->m_window, key);

				if (window->m_keys_callback) {
					window->m_keys_callback(key, key_state::press);
				}

				break;
			}
			case ButtonRelease:
			{

				auto window = fan::get_window_by_id(event.xbutton.window);

				if (!window) {
					break;
				}

				if (XEventsQueued(window->m_display, QueuedAfterReading)) {
					XEvent nev;
					XPeekEvent(window->m_display, &nev);

					if (nev.type == ButtonPress && nev.xbutton.time == event.xbutton.time &&
						nev.xbutton.button == event.xbutton.button) {
						break;
					}
				}

				auto key = fan::window_input::convert_keys_to_fan(event.xbutton.button);

				window->window_input_up(window->m_window, key);

				if (window->m_keys_callback) {
					window->m_keys_callback(key, key_state::release);
				}

				break;
			}
			case FocusOut:
			{
				fan::window* window = fan::get_window_by_id(event.xfocus.window);

				if (!window) {
					break;
				}

				for (auto& i : window->m_keys_down) {
					window->window_input_up(window->m_window, i.first);
					i.second = false;
				}

				window->m_focused = false;
				break;
			}
		}
	}

	#endif

}

void fan::window::auto_close(bool state)
{
	m_auto_close = state;
}