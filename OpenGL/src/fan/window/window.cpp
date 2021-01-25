#include <fan/window/window.hpp>

#ifdef FAN_PLATFORM_WINDOWS
// mouse move event position getter
#include <windowsx.h>

#elif defined(FAN_PLATFORM_LINUX)

#endif

#include <string>

#define stringify(name) #name

fan::vec2i fan::get_resolution() {
#ifdef FAN_PLATFORM_WINDOWS

	return fan::vec2i(GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN));

#elif defined(FAN_PLATFORM_LINUX) // close

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

int fan::window::flag_values::m_minor_version = fan::uninitialized;
int fan::window::flag_values::m_major_version = fan::uninitialized;

bool fan::window::flag_values::m_no_mouse = false;
bool fan::window::flag_values::m_no_resize = false;

uint8_t fan::window::flag_values::m_samples = fan::uninitialized;
fan::window::mode fan::window::flag_values::m_size_mode = fan::window::mode::not_set;

// doesn't copy manually set flags yet
fan::window::window(const std::string& name, const fan::vec2i& window_size, uint64_t flags)
	: m_size(window_size), m_position(fan::uninitialized), m_mouse_position(0), m_max_fps(0), m_received_fps(0), 
	  m_fps(0), m_last_frame(0), m_current_frame(0), m_delta_time(0), m_vsync(false), m_close(false), m_name(name), m_flags(flags)
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

	initialize_window(name, window_size, flags);
	this->calculate_delta_time();
}

fan::window::window(const window& window) : fan::window(window.m_name, window.m_size, window.m_flags) {}

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
	this->m_keys_down = window.m_keys_down;
	this->m_keys_reset = window.m_keys_reset;
	this->m_key_callback = window.m_key_callback;
	this->m_last_frame = window.m_last_frame;
	this->m_max_fps = window.m_max_fps;
	this->m_mouse_move_callback = window.m_mouse_move_callback;
	this->m_mouse_move_position_callback = window.m_mouse_move_position_callback;
	this->m_mouse_position = window.m_mouse_position;
	this->m_move_callback = window.m_move_callback;
	this->m_position = window.m_position;
	this->m_previous_size = window.m_previous_size;
	this->m_received_fps = window.m_received_fps;
	this->m_resize_callback = window.m_resize_callback;
	this->m_scroll_callback = window.m_scroll_callback;
	this->m_close_callback = window.m_close_callback;
	this->m_size = window.m_size;
	this->m_vsync = window.m_vsync;
	this->m_window_storage = window.m_window_storage;
	this->m_name = window.m_name;
	this->m_flags = window.m_flags;

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

#if defined(FAN_PLATFORM_WINDOWS)

	this->m_hdc = std::move(window.m_hdc);
	this->m_context = std::move(window.m_context);

#elif defined(FAN_PLATFORM_LINUX) 

	m_display = window.m_display;
	m_event = window.m_event;
	m_screen = window.m_screen;
	m_atom_delete_window = window.m_atom_delete_window;
	m_window_attribs = window.m_window_attribs;
	m_context = window.m_context;
	m_visual = window.m_visual;

#endif

	this->m_keys_action = std::move(window.m_keys_action);
	this->m_keys_callback = window.m_keys_callback;
	this->m_keys_down = std::move(window.m_keys_down);
	this->m_keys_reset = std::move(window.m_keys_reset);
	this->m_key_callback = window.m_key_callback;
	this->m_last_frame = std::move(window.m_last_frame);
	this->m_max_fps = std::move(window.m_max_fps);
	this->m_mouse_move_callback = window.m_mouse_move_callback;
	this->m_mouse_move_position_callback = window.m_mouse_move_position_callback;
	this->m_mouse_position = std::move(window.m_mouse_position);
	this->m_move_callback = window.m_move_callback;
	this->m_position = std::move(window.m_position);
	this->m_previous_size = std::move(window.m_previous_size);
	this->m_received_fps = std::move(window.m_received_fps);
	this->m_resize_callback = window.m_resize_callback;
	this->m_scroll_callback = window.m_scroll_callback;
	this->m_close_callback = window.m_close_callback;
	this->m_size = std::move(window.m_size);
	this->m_vsync = std::move(window.m_vsync);
	this->m_window = std::move(window.m_window);
	this->m_window_storage = std::move(window.m_window_storage);
	this->m_name = std::move(window.m_name);
	this->m_flags = window.m_flags;

#if defined(FAN_PLATFORM_WINDOWS)
	wglMakeCurrent(m_hdc, m_context);
#elif defined(FAN_PLATFORM_LINUX)
	glXMakeCurrent(m_display, m_window, m_context);
#endif

	return *this;
}

fan::window::~window()
{
	this->destroy_window();
}

void fan::window::execute(const fan::color& background_color, const std::function<void()>& function)
{

	using timer_interval_t = fan::milliseconds;

	static f_t next_tick = fan::timer<timer_interval_t>::get_time();

	this->calculate_delta_time();

	this->handle_events();

	this->set_background_color(background_color);

	if (function) {
		function();
	}

	if (f_t fps = static_cast<f_t>(timer_interval_t::period::den) / get_max_fps()) {
		next_tick += fps;
		auto time = timer_interval_t(static_cast<uint_t>(std::ceil(next_tick - fan::timer<timer_interval_t>::get_time())));
		fan::delay(timer_interval_t(std::max(static_cast<decltype(time.count())>(0), time.count())));
	}

	this->swap_buffers();

	this->reset_keys();

	if (flag_values::m_no_mouse && this->focused()) {
		auto middle = this->get_position() + this->get_size() / 2;

		if (get_mouse_position() != middle) {
		#ifdef FAN_PLATFORM_WINDOWS
			SetCursorPos(middle.x, middle.y);
		#endif
		}
	}
}

void fan::window::loop(const fan::color& background_color, const std::function<void()>& function) {

	while (fan::window::open()) {
		this->execute(background_color, [&] {
			function();
		});
	}
}

void fan::window::swap_buffers() const
{

#ifdef FAN_PLATFORM_WINDOWS
	SwapBuffers(m_hdc);
#elif defined(FAN_PLATFORM_LINUX)
	glXSwapBuffers(m_display, m_window);
#endif

}

std::string fan::window::get_name() const
{
	return m_name;
}

void fan::window::set_name(const std::string& name)
{

	m_name = name;

#ifdef FAN_PLATFORM_WINDOWS

	SetWindowTextA(m_window, m_name.c_str());

#elif defined(FAN_PLATFORM_LINUX)

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

#ifdef FAN_PLATFORM_WINDOWS

	const fan::vec2i position = this->get_position();

	if (!SetWindowPos(m_window, 0, position.x - move_offset.x, position.y - move_offset.y, size.x, size.y, SWP_NOZORDER | SWP_SHOWWINDOW)) {
		fan::print("fan window error: failed to set window position", GetLastError());
		exit(1);
	}

#elif defined(FAN_PLATFORM_LINUX)

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
	#ifdef FAN_PLATFORM_WINDOWS

	if (!SetWindowPos(m_window, 0, position.x, position.y, 0, 0, SWP_NOSIZE | SWP_NOZORDER | SWP_SHOWWINDOW)) {
		fan::print("fan window error: failed to set window position", GetLastError());
		exit(1);
	}

#elif defined(FAN_PLATFORM_LINUX)

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
}

bool fan::window::vsync_enabled() const {
	return m_vsync;
}

void fan::window::vsync(bool value) {

#ifdef FAN_PLATFORM_WINDOWS

	PFNWGLSWAPINTERVALEXTPROC swap_interval = (PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT");

	if (swap_interval) {
		swap_interval(value);
	}
	else {
		fan::print("fan window error: invalid swap interval");
		exit(1);
	}

#elif defined(FAN_PLATFORM_LINUX)

	GLXDrawable drawable = glXGetCurrentDrawable();

	if (drawable) {
		if (glXSwapIntervalEXT) {
			glXSwapIntervalEXT(m_display, drawable, value);
		}
		else {
			fan::print("fan window error: invalid swap interval");
			exit(1);
		}
	}
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

#ifdef FAN_PLATFORM_WINDOWS

	this->set_resolution(new_size, fan::window::get_size_mode());

	this->set_windowed_full_screen();

#elif defined(FAN_PLATFORM_LINUX)

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

#ifdef FAN_PLATFORM_WINDOWS

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

#elif defined(FAN_PLATFORM_LINUX)

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

#ifdef FAN_PLATFORM_WINDOWS

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

#elif defined(FAN_PLATFORM_LINUX)

	

#endif
}

void fan::window::set_resolution(const fan::vec2i& size, const mode& mode) const
{
#ifdef FAN_PLATFORM_WINDOWS

	if (mode == mode::full_screen) {
		DEVMODE screen_settings;
		memset (&screen_settings, 0, sizeof (screen_settings));
		screen_settings.dmSize = sizeof (screen_settings);
		screen_settings.dmPelsWidth = size.x;
		screen_settings.dmPelsHeight = size.y;
		screen_settings.dmBitsPerPel = 32;
		screen_settings.dmFields = DM_BITSPERPEL | DM_PELSWIDTH | DM_PELSHEIGHT;
		ChangeDisplaySettings(&screen_settings, CDS_FULLSCREEN);
	}
	else {
		ChangeDisplaySettings(nullptr, CDS_RESET);
	}

#elif defined(FAN_PLATFORM_LINUX)

	//

#endif
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
#ifdef __GNUC__ // to prevent gcc bug
	auto found = m_window_storage.find(std::make_pair(window, location));
	if (found != m_window_storage.end()) {
		found->second = data;
	}
	else {
		m_window_storage.insert(std::make_pair(std::make_pair(window, location), data));
	}
	
#else

	m_window_storage.insert_or_assign(std::make_pair(window, location), data);

#endif
}

fan::window::keys_callback_t fan::window::get_keys_callback() const
{
	return this->m_keys_callback;
}

void fan::window::set_keys_callback(const keys_callback_t& function)
{
	this->m_keys_callback = function;
	set_window_storage(m_window, stringify(m_keys_callback), this->m_keys_callback);
}

fan::window::key_callback_t fan::window::get_key_callback(uint_t i) const
{
	return this->m_key_callback[i];
}

void fan::window::add_key_callback(uint16_t key, const std::function<void()>& function, bool on_release)
{
	this->m_key_callback.emplace_back(key_callback_t{ key, function, on_release });
	set_window_storage(m_window, stringify(m_key_callback), this->m_key_callback);
}

std::function<void()> fan::window::get_close_callback(uint_t i) const
{
	return m_close_callback[i];
}

void fan::window::add_close_callback(const std::function<void()>& function)
{
	m_close_callback.emplace_back(function);
	set_window_storage(m_window, stringify(m_close_callback), this->m_close_callback);
}

fan::window::mouse_move_position_callback_t fan::window::get_mouse_move_callback(uint_t i) const
{
	return this->m_mouse_move_position_callback[i];
}

void fan::window::add_mouse_move_callback(const mouse_move_position_callback_t& function)
{
	this->m_mouse_move_position_callback.emplace_back(function);
	set_window_storage(m_window, stringify(m_mouse_move_position_callback), this->m_mouse_move_position_callback);
}

void fan::window::add_mouse_move_callback(const mouse_move_callback_t& function)
{
	this->m_mouse_move_callback.emplace_back(function);
	set_window_storage(m_window, stringify(m_mouse_move_callback), this->m_mouse_move_callback);
}

fan::window::scroll_callback_t fan::window::get_scroll_callback(uint_t i) const
{
	return this->m_scroll_callback[i];
}

void fan::window::add_scroll_callback(const scroll_callback_t& function)
{
	this->m_scroll_callback.emplace_back(function);
	set_window_storage(m_window, stringify(m_scroll_callback), this->m_scroll_callback);
}

std::function<void()> fan::window::get_resize_callback(uint_t i) const
{
	return this->m_resize_callback[i];
}

void fan::window::add_resize_callback(const std::function<void()>& function)
{
	this->m_resize_callback.emplace_back(function);
	set_window_storage(m_window, stringify(m_resize_callback), this->m_resize_callback);
}

std::function<void()> fan::window::get_move_callback(uint_t i) const
{
	return this->m_move_callback[i];
}

void fan::window::add_move_callback(const std::function<void()>& function)
{
	this->m_move_callback.push_back(function);
	set_window_storage(m_window, stringify(m_move_callback), this->m_move_callback);
}

void fan::window::set_background_color(const fan::color& color)
{
	glClearColor(
		color.r,
		color.g,
		color.b,
		color.a
	);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

fan::window_t fan::window::get_handle() const
{
	return m_window;
}

uint_t fan::window::get_fps(bool window_name, bool print)
{
	if (!m_fps_timer.get_reset_time()) {
		m_fps_timer = fan::timer<>(fan::timer<microseconds>::start(), 1000);
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
#ifdef FAN_PLATFORM_WINDOWS
	return m_window == GetFocus();
#elif defined(FAN_PLATFORM_LINUX)
	return 1;
#endif
	
}

void fan::window::destroy_window()
{
	window::close();

#if defined(FAN_PLATFORM_WINDOWS)

	if (!m_window || !m_hdc || !m_context) {
		return;
	}

	PostQuitMessage(0);
	wglMakeCurrent(m_hdc, 0);
    wglDeleteContext(m_context);
    ReleaseDC(m_window, m_hdc);
    DestroyWindow(m_window);
	//CloseWindow(m_window);

	m_hdc = 0;
	m_context = 0;

#elif defined(FAN_PLATFORM_LINUX)

	if (!m_display || !m_context || !m_visual || !m_window_attribs.colormap) {
		return;
	}

	glXDestroyContext(m_display, m_context);

	XFree(m_visual);
	XFreeColormap(m_display, m_window_attribs.colormap);
	XDestroyWindow(m_display, m_window);
	XCloseDisplay(m_display);

	m_atom_delete_window = 0;
	m_display = 0;
	m_context = 0;
	m_visual = 0;
	m_window_attribs.colormap = 0;

#endif

	m_window = 0;
}

void fan::window::window_input_action(fan::window_t window, uint16_t key) {
#ifdef FAN_PLATFORM_WINDOWS
	keymap_t* keys_action = get_window_storage<keymap_t*>(window, stringify(m_keys_action));
	keymap_t* keys_reset = get_window_storage<keymap_t*>(window, stringify(m_keys_reset));

	auto found = keys_reset->find(key);
			
	if (found == keys_reset->end()) {
		keys_action->insert_or_assign(key, true);
	}

	auto keys_callback = get_window_storage<keys_callback_t>(window, stringify(m_keys_callback));

	if (keys_callback) {
		keys_callback(key, (*keys_action)[key]);

		// set current action to reset after one key press
		keys_reset->insert_or_assign(key, true);
	}

	auto key_callback = get_window_storage<std::vector<key_callback_t>>(window, stringify(m_key_callback));

	for (const auto& i : key_callback) {
		if (key != i.key || i.release || !(*keys_action)[key]) {
			continue;
		}

		keys_reset->insert_or_assign(key, true);

		if (i.function) {
			i.function();
		}
	}

#elif defined(FAN_PLATFORM_LINUX)
	auto found = m_keys_reset.find(key);
			
	if (found ==  m_keys_reset.end()) {
		m_keys_action.insert_or_assign(key, true);
	}

	auto keys_callback = get_window_storage<keys_callback_t>(window, "m_keys_callback");

	if (keys_callback) {
		keys_callback(key, m_keys_action[key]);

		// set current action to reset after one key press
		m_keys_reset.insert_or_assign(key, true);
	}

	for (const auto& i : m_key_callback) {
		if (key != i.key || i.release || !m_keys_action[key]) {
			continue;
		}

		m_keys_reset.insert_or_assign(key, true);

		if (i.function) {
			i.function();
		}
	}

#endif
}

void fan::window::window_input_mouse_action(fan::window_t window, uint16_t key)
{
#ifdef FAN_PLATFORM_WINDOWS
	keymap_t* keys_down = get_window_storage<keymap_t*>(window, "m_keys_down");

	if (keys_down) {
		fan::window_input::get_keys(keys_down, key, true);
	}
	auto keys_callback = get_window_storage<keys_callback_t>(window, "m_keys_callback");

	if (keys_callback) {
		keys_callback(key, false);
	}

	auto key_callback = get_window_storage<std::vector<key_callback_t>>(window, "m_key_callback");

	for (const auto& i : key_callback) {
		if (key != i.key || i.release) {
			continue;
		}
		if (i.function) {
			i.function();
		}
	}

#elif defined(FAN_PLATFORM_LINUX)
	
	fan::window_input::get_keys(&m_keys_down, key, true);

	if (m_keys_callback) {
		m_keys_callback(key, false);
	}


	for (const auto& i : m_key_callback) {
		if (key != i.key || i.release) {
			continue;
		}
		if (i.function) {
			i.function();
		}
	}

#endif
}

void fan::window::window_input_up(fan::window_t window, uint16_t key)
{

#ifdef FAN_PLATFORM_WINDOWS

	keymap_t* keys_down = get_window_storage<keymap_t*>(window, "m_keys_down");

	if (keys_down) {
		fan::window_input::get_keys(keys_down, key, false);
	}

	if (key <= fan::input::key_menu) {
		fan::window::window_input_action_reset(window, key);
	}

	auto key_callback = get_window_storage<std::vector<key_callback_t>>(window, "m_key_callback");

	for (const auto& i : key_callback) {
		if (key != i.key || !i.release) {
			continue;
		}
		if (i.function) {
			i.function();
		}
	}

#elif defined(FAN_PLATFORM_LINUX)

	fan::window_input::get_keys(&m_keys_down, key, false);

	if (key <= fan::input::key_menu) {
		fan::window::window_input_action_reset(window, key);
	}

	for (const auto& i : m_key_callback) {
		if (key != i.key || !i.release) {
			continue;
		}
		if (i.function) {
			i.function();
		}
	}

#endif

}

void fan::window::window_input_action_reset(fan::window_t window, uint16_t key)
{

#ifdef FAN_PLATFORM_WINDOWS

	keymap_t* keys_reset = get_window_storage<keymap_t*>(window, "m_keys_reset");
	keymap_t* keys_action = get_window_storage<keymap_t*>(window, "m_keys_action");

	for (const auto& i : *keys_reset) {
		(*keys_action)[i.first] = false;
	}

	keys_reset->clear();

#elif defined(FAN_PLATFORM_LINUX)

	for (const auto& i : m_keys_reset) {
		m_keys_action[i.first] = false;
	}

	m_keys_reset.clear();

#endif

}

#ifdef FAN_PLATFORM_WINDOWS
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
#endif

#ifdef FAN_PLATFORM_WINDOWS
LRESULT CALLBACK fan::window::window_proc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {

	switch (msg)
	{

		case WM_MOVE:
		{

			fan::vec2i* position = get_window_storage<fan::vec2i*>(hwnd, stringify(m_position));

			if (position) {
				*position = fan::vec2i(
					static_cast<int>(static_cast<short>(LOWORD(lparam))), 
					static_cast<int>(static_cast<short>(HIWORD(lparam)))
				);
			}
			
			const auto move_callback = get_window_storage<std::vector<std::function<void()>>>(hwnd, stringify(m_move_callback));

			for (const auto& i : move_callback) {
				if (i) {
					i();
				}
			}
			
			break;
		}
		case WM_SIZE:
		{
			RECT rect;
			RECT rect2;
			GetWindowRect(hwnd, &rect2);
			GetClientRect(hwnd, &rect);

			auto window_size = get_window_storage<fan::vec2i*>(hwnd, stringify(m_size));

			auto previous_size = get_window_storage<fan::vec2i*>(hwnd, stringify(m_previous_size));

			if (!window_size || !previous_size) {
				break;
			}
			
			*previous_size = *window_size;
			*window_size = fan::vec2i(rect.right - rect.left, rect.bottom - rect.top);
			
			glViewport(0, 0, window_size->x, window_size->y);


			const auto resize_callback = get_window_storage<std::vector<std::function<void()>>>(hwnd, stringify(m_resize_callback));

			for (const auto& i : resize_callback) {
				if (i) {
					i();
				}
			}

			break;
		}
		case WM_KEYDOWN:
		{
			uint16_t key;

			handle_special(wparam, lparam, key, true);

			keymap_t* keys_down = get_window_storage<keymap_t*>(hwnd, stringify(m_keys_down));

			if (keys_down) {
				fan::window_input::get_keys(keys_down, key, true);
			}

			fan::window::window_input_action(hwnd, key);

			break;
		}
		case WM_LBUTTONDOWN:
		{
			const uint16_t key = fan::input::mouse_left;

			fan::window::window_input_mouse_action(hwnd, key);

			break;
		}
		case WM_RBUTTONDOWN:
		{
			const uint16_t key = fan::input::mouse_right;

			fan::window::window_input_mouse_action(hwnd, key);

			break;
		}
		case WM_MBUTTONDOWN:
		{
			const uint16_t key = fan::input::mouse_middle;

			fan::window::window_input_mouse_action(hwnd, key);

			break;
		}

		case WM_LBUTTONUP:
		{
			const uint16_t key = fan::input::mouse_left;

			window_input_up(hwnd, key);

			break;
		}
		case WM_RBUTTONUP:
		{
			const uint16_t key = fan::input::mouse_right;

			window_input_up(hwnd, key);

			break;
		}
		case WM_MBUTTONUP:
		{
			const uint16_t key = fan::input::mouse_middle;

			window_input_up(hwnd, key);

			break;
		}

		case WM_KEYUP:
		{
			uint16_t key = 0;

			handle_special(wparam, lparam, key, false);

			window_input_up(hwnd, key);

			break;
		}
		case WM_MOUSEMOVE:
		{

			const fan::vec2i position(GET_X_LPARAM(lparam), GET_Y_LPARAM(lparam));

			static auto offset = 0;

			fan::vec2i* mouse_position = get_window_storage<fan::vec2i*>(hwnd, stringify(m_mouse_position));

			auto window_position = *get_window_storage<fan::vec2i*>(hwnd, stringify(m_position));
			auto window_size = *get_window_storage<fan::vec2i*>(hwnd, stringify(m_size));

			if (mouse_position) {

				if (fan::window::flag_values::m_no_mouse) {
					POINT p = { window_position.x + window_size.x / 2,  window_position.y + window_size.y / 2 };
					ScreenToClient(hwnd, &p);

					*mouse_position += -(fan::vec2i(p.x, p.y) - position);
				}
				else {
					*mouse_position = position;
				}

				const auto& mouse_move_position_callback = get_window_storage<std::vector<mouse_move_position_callback_t>>(hwnd, stringify(m_mouse_move_position_callback));
				const auto& mouse_move_callback = get_window_storage<std::vector<mouse_move_callback_t>>(hwnd, stringify(m_mouse_move_callback));

				for (const auto& i : mouse_move_position_callback) {
					if (i) {
						i(*mouse_position);
					}
				}

				for (const auto& i : mouse_move_callback) {
					if (i) {
						i();
					}
				}

			}

			break;
		}
		case WM_MOUSEWHEEL:
		{
			auto fwKeys = GET_KEYSTATE_WPARAM(wparam);
			auto zDelta = GET_WHEEL_DELTA_WPARAM(wparam);

			keymap_t* keys_down = get_window_storage<keymap_t*>(hwnd, stringify(m_keys_down));

			if (keys_down) {
				fan::window_input::get_keys(keys_down, zDelta < 0 ? fan::input::mouse_scroll_down : fan::input::mouse_scroll_up, true);
			}

			fan::window::window_input_mouse_action(hwnd, zDelta < 0 ? fan::input::mouse_scroll_down : fan::input::mouse_scroll_up);

			auto scroll_callback = get_window_storage<std::vector<scroll_callback_t>>(hwnd, stringify(m_scroll_callback));

			for (const auto& i : scroll_callback) {
				if (i) {
					i(fwKeys < 0 ? fan::input::mouse_scroll_up : fan::input::mouse_scroll_down);
				}
			}

			break;
		}
		case WM_ACTIVATE:
		{
			bool* closing = get_window_storage<bool*>(hwnd, stringify(m_close));

			if (closing && *closing) {
				break;
			}

			enum window_activation_state{
				wa_inactive,
				wa_active,
				wa_click_active
			};

			switch (wparam) {
				case window_activation_state::wa_inactive:
				{
					keymap_t* keys_down = get_window_storage<keymap_t*>(hwnd, stringify(m_keys_down));

					for (auto& i : *keys_down) {
						i.second = false;
					}

					break;
				}
			}
			
			break;
		}
		case WM_SYSCOMMAND:
		{
			// disable alt action for window
			if (wparam == SC_KEYMENU && (lparam >> 16) <= 0) {
				return 0;
			}

			break;
		}
		case WM_CLOSE:
		{
			auto close_callback = get_window_storage<std::vector<std::function<void()>>>(hwnd, stringify(m_close_callback));
			for (int i = 0; i < close_callback.size(); i++) {
				if (close_callback[i]) {
					close_callback[i]();
				}
			}

		}
		case WM_DESTROY:
		{
			PostQuitMessage(0);
			break;
		}
	};

	return DefWindowProc(hwnd, msg, wparam, lparam);
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

std::string random_string( size_t length )
{
    auto randchar = []() -> char
    {
        const char charset[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
        const size_t max_index = (sizeof(charset) - 1);
        return charset[ rand() % max_index ];
    };
    std::string str(length,0);
    std::generate_n( str.begin(), length, randchar );
    return str;
}

#ifdef FAN_PLATFORM_WINDOWS
void init_windows_opengl_extensions()
{

	auto str = random_string(10);
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

void fan::window::initialize_window(const std::string& name, const fan::vec2i& window_size, uint64_t flags)
{

#ifdef FAN_PLATFORM_WINDOWS

	auto instance = GetModuleHandle(NULL);

	WNDCLASS wc = {0};

	auto str = random_string(20);

	const char* class_name = str.c_str();
	wc.lpszClassName = class_name;
	wc.lpfnWndProc = window_proc;

	wc.hCursor = LoadCursor(NULL, IDC_ARROW);

	wc.hInstance = GetModuleHandle(NULL);
    
	RegisterClass(&wc);

	const bool full_screen = flag_values::m_size_mode == fan::window::mode::full_screen;
	const bool borderless = flag_values::m_size_mode == fan::window::mode::borderless;


	RECT rect = {0, 0, window_size.x, window_size.y};
    AdjustWindowRect(&rect, full_screen || borderless ? WS_POPUP : WS_OVERLAPPEDWINDOW, FALSE);

	const fan::vec2i position = fan::get_resolution() / 2 - window_size / 2;

	if (full_screen) {
		this->set_resolution(window_size, fan::window::mode::full_screen);
	}

	m_window = CreateWindow(class_name, name.c_str(),
							(flag_values::m_no_resize ? ((full_screen || borderless ? WS_POPUP : WS_OVERLAPPEDWINDOW) | WS_SYSMENU) :
							(full_screen || borderless ? WS_POPUP : WS_OVERLAPPEDWINDOW) | (flag_values::m_no_resize ? SWP_NOSIZE : 0)) | WS_VISIBLE,
							position.x, position.y,
							rect.right - rect.left, rect.bottom - rect.top,
							0, 0, instance, 0);
	
	if (!m_window) {
		fan::print("fan window error: failed to initialize window", GetLastError());
		exit(1);
	}

	if (flag_values::m_no_mouse) {
		ShowCursor(false);
	}
	else {
		ShowCursor(true);
	}

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
		fan::OPENGL_SAMPLE_BUFFER, true, // Number of buffers (must be 1 at time of writing)
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
    DescribePixelFormat(m_hdc, pixel_format, sizeof(pfd), &pfd);
    if (!SetPixelFormat(m_hdc, pixel_format, &pfd)) {
		fan::print("failed to set pixel format");
		exit(1);
    }

    const int gl_attributes[] = {
        fan::OPENGL_MINOR_VERSION, flag_values::m_minor_version,
		fan::OPENGL_MAJOR_VERSION, flag_values::m_major_version,
        WGL_CONTEXT_PROFILE_MASK_ARB,  WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
        0,
    };

    m_context = wglCreateContextAttribsARB(m_hdc, 0, gl_attributes);
    if (!m_context) {
		fan::print("failed to create context");
		exit(1);
    }

    if (!wglMakeCurrent(m_hdc, m_context)) {
		fan::print("failed to make current");
		exit(1);
    }

	glewExperimental = true;

	static bool initialized = false;

	if (!initialized) {
		if (glewInit() != GLEW_OK) { // maybe
			fan::print("failed to initialize glew");
			exit(1);
		}
		initialized = true;
	}

	m_position = position;

	set_window_storage(m_window, stringify(m_size), &m_size);
	set_window_storage(m_window, stringify(m_previous_size), &m_previous_size);
	set_window_storage(m_window, stringify(m_position), &m_position);
	set_window_storage(m_window, stringify(m_mouse_position), &m_mouse_position);
	set_window_storage(m_window, stringify(m_keys_action), &m_keys_action);
	set_window_storage(m_window, stringify(m_keys_reset), &m_keys_reset);
	set_window_storage(m_window, stringify(m_keys_down), &m_keys_down);
	set_window_storage(m_window, stringify(m_close), &m_close);

	m_previous_size = m_size;

#elif defined(FAN_PLATFORM_LINUX)


	m_display = XOpenDisplay(NULL);

	if (!m_display) {
		fan::print("fan window error: failed to initialize window");
		exit(1);
	}

	m_screen = DefaultScreen(m_display);

	int minor_glx = 0, major_glx = 0;
	glXQueryVersion(m_display, &major_glx, &minor_glx);

	//if (minor_glx < flag_values::m_minor_version && major_glx <= flag_values::m_major_version) {
	//	fan::print("fan window error: too low glx version");
	//	XCloseDisplay(m_display);
	//	exit(1);
	//}

	static int pixel_format_attribs[] = {
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
		  fan::OPENGL_SAMPLE_BUFFER  , 1,
		  fan::OPENGL_SAMPLES        , fan::window::flag_values::m_samples,
		  None
	};


	if (!fan::window::flag_values::m_samples) {
		// set back to zero to disable antialising
		for (int i = 0; i < 4; i++) {
			pixel_format_attribs[22 + i] = 0;
		}
	}

	static int gl_attribs[] = {
		fan::OPENGL_MINOR_VERSION, flag_values::m_minor_version,
		fan::OPENGL_MAJOR_VERSION, flag_values::m_major_version,
		GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
		0
	};


	auto ChooseFBConfigSGIX =
		(PFNGLXCHOOSEFBCONFIGPROC)
		glXGetProcAddress((GLubyte*)"glXChooseFBConfig");

	glXGetVisualFromFBConfig =
		(PFNGLXGETVISUALFROMFBCONFIGPROC)
		glXGetProcAddress((GLubyte*)"glXGetVisualFromFBConfig");
	glXCreateContextAttribsARB =
		(PFNGLXCREATECONTEXTATTRIBSARBPROC)
		glXGetProcAddress((GLubyte*)"glXCreateContextAttribsARB");

	int fbcount;

	auto fbcs = ((GLXFBConfig*)ChooseFBConfigSGIX(m_display, m_screen, pixel_format_attribs, &fbcount));
	if (!fbcs) {
		fan::print("fan window error: failed to retreive framebuffer");
		XCloseDisplay(m_display);
		exit(1);
	}

	m_visual = glXGetVisualFromFBConfig(m_display, fbcs[0]);

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

	m_atom_delete_window = XInternAtom(m_display, "WM_DELETE_WINDOW", False);
	XSetWMProtocols(m_display, m_window, &m_atom_delete_window, 1);

	m_context = glXCreateContextAttribsARB(m_display, fbcs[0], 0, True, gl_attribs);

	XSync(m_display, True);

	glXMakeCurrent(m_display, m_window, m_context);

	XClearWindow(m_display, m_window);
	XMapRaised(m_display, m_window);
	XAutoRepeatOn(m_display);

	XFree(fbcs);

	glewExperimental = true;

	static bool initialized = false;

	if (!initialized) {
		if (glewInit() != GLEW_OK) { // maybe
			fan::print("failed to initialize glew");
			exit(1);
		}
		initialized = true;
	}

#endif

	for (int i = 0; i != fan::input::last; ++i) {
		m_keys_down[i] = false;
	}

	glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glViewport(0, 0, window_size.x, window_size.y);

}

void fan::window::handle_events() {

#ifdef FAN_PLATFORM_WINDOWS

	static MSG msg{};

	while (PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
	{
		if (msg.message == WM_QUIT)
		{
			fan::window::close();
			return;
		}

		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}

#elif defined(FAN_PLATFORM_LINUX)

	int nevents = XEventsQueued(m_display, QueuedAfterReading);

	while (nevents--) {
		XNextEvent(m_display, &m_event);

		switch (m_event.type) {

			case Expose:
			{

				XWindowAttributes attribs;
				XGetWindowAttributes(m_display, m_window, &attribs);

				glViewport(0, 0, attribs.width, attribs.height);

				m_previous_size = m_size;
				m_size = fan::vec2i(attribs.width, attribs.height);

				for (const auto& i : m_resize_callback) {
					if (i) {
						i();
					}
				}

				break;
			}
			case ConfigureNotify:
			{

				for (const auto& i : m_move_callback) {
					if (i) {
						i();
					}
				}

				break;
			}
			case DestroyNotify:
			{

				for (uint_t i = 0; i < fan::window::m_close_callback.size(); i++) {
					if (m_close_callback[i]) {
						m_close_callback[i]();
					}
				}

				fan::window::close();

				break;
			}
			case ClientMessage:
			{
				fan::print("here");
				if (m_event.xclient.data.l[0] != (long)m_atom_delete_window) {
					break;
				}
				fan::print("here2", m_close_callback.size());
				for (uint_t i = 0; i < fan::window::m_close_callback.size(); i++) {
					if (m_close_callback[i]) {
						m_close_callback[i]();
					}
				}

				fan::window::close();

				break;
			}
			case KeyPress:
			{

				uint16_t key = fan::window_input::convert_keys_to_fan(m_event.xkey.keycode);

				fan::window_input::get_keys(&m_keys_down, key, true);

				fan::window::window_input_action(m_window, key);

				break;
			}
			case KeyRelease:
			{

				if (XEventsQueued(m_display, QueuedAfterReading)) {
					XEvent nev;
					XPeekEvent(m_display, &nev);

					if (nev.type == KeyPress && nev.xkey.time == m_event.xkey.time &&
						nev.xkey.keycode == m_event.xkey.keycode) {
						break;
					}
				}

				const uint16_t key = fan::window_input::convert_keys_to_fan(m_event.xkey.keycode);

				window_input_up(m_window, key);

				break;
			}
			case MotionNotify:
			{

				const fan::vec2i position(m_event.xmotion.x, m_event.xmotion.y);

				auto mouse_move_position_callback = this->m_mouse_move_position_callback;

				for (const auto& i : mouse_move_position_callback) {
					if (i) {
						i(position);
					}
				}

				auto mouse_move_callback = this->m_mouse_move_callback;

				for (const auto& i : mouse_move_callback) {
					if (i) {
						i();
					}
				}

				m_mouse_position = position;

				break;
			}
			case ButtonPress:
			{

				uint16_t key = fan::window_input::convert_keys_to_fan(m_event.xbutton.button);

				switch (key) {
					case fan::input::mouse_scroll_up:
					case fan::input::mouse_scroll_down:
					{

						for (const auto& i : m_scroll_callback) {
							if (i) {
								i(key);
							}
						}

						fan::window::window_input_mouse_action(m_window, key);

						break;
					}
					default:
					{

						fan::window::window_input_mouse_action(m_window, key);

						break;
					}
				}


				break;
			}
			case ButtonRelease:
			{

				if (XEventsQueued(m_display, QueuedAfterReading)) {
					XEvent nev;
					XPeekEvent(m_display, &nev);

					if (nev.type == ButtonPress && nev.xbutton.time == m_event.xbutton.time &&
						nev.xbutton.button == m_event.xbutton.button) {
						break;
					}
				}

				window_input_up(m_window, fan::window_input::convert_keys_to_fan(m_event.xbutton.button));

				break;
			}
		}
	}

#endif

}