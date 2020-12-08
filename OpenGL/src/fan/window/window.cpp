#include <fan/window/window.hpp>

#ifdef FAN_PLATFORM_WINDOWS
// mouse move event position getter
#include <windowsx.h>

#elif defined(FAN_PLATFORM_LINUX)

#endif

#include <string>

fan::vec2i fan::get_resolution() {
#ifdef FAN_PLATFORM_WINDOWS

	return fan::vec2i(GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN));

#elif defined(FAN_PLATFORM_LINUX)

	Display* display = XOpenDisplay(0);

	if (!display) {
		fan::print("failed to open display");
	}

	int screen = DefaultScreen(display);

	return fan::vec2i(DisplayWidth(display, screen), DisplayHeight(display, screen));

#endif
}

fan::window::window(const std::string& name, const fan::vec2i& window_size, uint64_t flags)
	: m_size(window_size), m_fps(0), m_close(false), m_max_fps(0), m_vsync(false), 
	  m_minor_version(default_opengl_version.x), m_major_version(default_opengl_version.y),
	  m_position(-1), 
	  m_size_mode(
		flags & flags::windowed_full_screen ? size_mode::windowed_full_screen :
		flags & flags::full_screen ? size_mode::full_screen : size_mode::windowed
	  )
{
	initialize_window(name, window_size, flags);
}

fan::window::~window()
{
#ifdef FAN_PLATFORM_WINDOWS


#elif defined(FAN_PLATFORM_LINUX)

	glXDestroyContext(m_display, m_context);

	XFree(m_visual);
	XFreeColormap(m_display, m_window_attribs.colormap);
	XDestroyWindow(m_display, m_window);
	XCloseDisplay(m_display);

#endif
}

void fan::window::loop(const fan::color& background_color, const std::function<void()>& function) {

	using timer_interval_t = fan::milliseconds;

	f_t next_tick = fan::timer<timer_interval_t>::get_time();

	while (fan::window::open()) {

		this->calculate_delta_time();

		this->handle_events();
		
		glClearColor(
			background_color.r,
			background_color.g,
			background_color.b,
			background_color.a
		);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		function();

		if (f_t fps = static_cast<f_t>(timer_interval_t::period::den) / get_max_fps()) {
			next_tick += fps;
			auto time = timer_interval_t(static_cast<uint_t>(std::ceil(next_tick - fan::timer<timer_interval_t>::get_time())));
			fan::delay(timer_interval_t(std::max(static_cast<decltype(time.count())>(0), time.count())));
		}

		this->swap_buffers();

		this->reset_keys();
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

void fan::window::set_window_title(const std::string& title) const
{

#ifdef FAN_PLATFORM_WINDOWS

#elif defined(FAN_PLATFORM_LINUX)

	XStoreName(m_display, m_window, title.c_str());

#endif

}

void fan::window::calculate_delta_time()
{
	m_current_frame = fan::timer<>::get_time();
	m_delta_time = f_t(m_current_frame - m_last_frame) / 1000;
	m_last_frame = m_current_frame;
}

f_t fan::window::get_delta_time() const
{
	return m_delta_time;
}

fan::vec2i fan::window::get_cursor_position() const
{
	return m_cursor_position;
}

fan::vec2i fan::window::get_size() const
{
	return m_size;
}

fan::vec2i fan::window::get_previous_size() const 
{
	return m_previous_size;
}

fan::vec2i fan::window::get_position() const
{
	return m_position;
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

void fan::window::set_opengl_version(int major, int minor)
{
	this->m_major_version = major;
	this->m_minor_version = minor;
}

void fan::window::set_full_screen(const fan::vec2i& size)
{
	fan::window::set_size_mode(fan::window::size_mode::full_screen);

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

	fan::window::set_size_mode(fan::window::size_mode::windowed_full_screen);

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
	fan::window::set_size_mode(fan::window::size_mode::windowed);

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

void fan::window::set_resolution(const fan::vec2i& size, const size_mode& mode) const
{
#ifdef FAN_PLATFORM_WINDOWS

	if (mode == size_mode::full_screen) {
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

fan::window::size_mode fan::window::get_size_mode() const
{
	return m_size_mode;
}

void fan::window::set_size_mode(const size_mode& mode)
{
	this->m_size_mode = mode;
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

fan::window::mouse_move_callback_t fan::window::get_mouse_move_callback(uint_t i) const
{
	return this->m_mouse_move_callback[i];
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

fan::window::resize_callback_t fan::window::get_resize_callback(uint_t i) const
{
	return this->m_resize_callback[i];
}

void fan::window::add_resize_callback(const fan::window::resize_callback_t& function)
{
	this->m_resize_callback.emplace_back(function);
	set_window_storage(m_window, stringify(m_resize_callback), this->m_resize_callback);
}

fan::window::move_callback_t fan::window::get_move_callback(uint_t i) const
{
	return this->m_move_callback[i];
}

void fan::window::add_move_callback(const move_callback_t& function)
{
	this->m_move_callback.push_back(function);
	set_window_storage(m_window, stringify(m_move_callback), this->m_move_callback);
}

fan::window_t fan::window::get_handle() const
{
	return m_window;
}

uint_t fan::window::get_fps(bool window_title, bool print)
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
		if (window_title || print) {
			fps_info.append(
				std::string("FPS: ") +
				std::to_string(m_fps) +
				std::string(" frame time: ") +
				std::to_string(1.0 / m_fps * 1000) +
				std::string(" ms")
			).c_str();
		}
		if (window_title) {
			this->set_window_title(fps_info.c_str());
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
			
			const auto move_callback = get_window_storage<std::vector<move_callback_t>>(hwnd, stringify(m_move_callback));

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
			GetClientRect(hwnd, &rect);
			
			glViewport(0, 0, rect.right, rect.bottom);

			auto window_size = get_window_storage<fan::vec2i*>(hwnd, stringify(m_size));

			auto previous_size = get_window_storage<fan::vec2i*>(hwnd, stringify(m_previous_size));

			if (!window_size || !previous_size) {
				break;
			}
			
			*previous_size = *window_size;
			*window_size = fan::vec2i(rect.right, rect.bottom);

			const auto resize_callback = get_window_storage<std::vector<resize_callback_t>>(hwnd, stringify(m_resize_callback));

			for (const auto& i : resize_callback) {
				if (i) {
					i();
				}
			}

			break;
		}
		case WM_KEYDOWN:
		{
			uint16_t key = fan::window_input::convert_keys_to_fan(wparam);

			keymap_t* keys_down = get_window_storage<keymap_t*>(hwnd, stringify(m_keys_down));

			if (keys_down) {
				fan::window_input::get_keys(keys_down, key, true);
			}

			fan::window::window_input_action(hwnd, key);

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
			const uint16_t key = fan::window_input::convert_keys_to_fan(wparam);

			window_input_up(hwnd, key);

			break;
		}
		case WM_MOUSEMOVE:
		{

			const fan::vec2i position(GET_X_LPARAM(lparam), GET_Y_LPARAM(lparam));

			auto mouse_move_callback = get_window_storage<std::vector<mouse_move_callback_t>>(hwnd, stringify(m_mouse_move_callback));

			for (const auto& i : mouse_move_callback) {
				if (i) {
					i(position);
				}
			}

			fan::vec2i* cursor_position = get_window_storage<fan::vec2i*>(hwnd, stringify(m_cursor_position));

			if (cursor_position) {
				*cursor_position = position;
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
			enum window_activation_state{
				wa_inactive,
				wa_active,
				wa_click_active
			};

			switch (wparam) {
				case window_activation_state::wa_inactive:
				{
					keymap_t* keys_down = get_window_storage<keymap_t*>(hwnd, stringify(m_keys_down));
					if (keys_down) {
						for (auto& i : *keys_down) {
							i.second = false;
						}
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
		case WM_DESTROY:
		{
			HDC hdc = get_window_storage<HDC>(hwnd, stringify(m_hdc));
			HGLRC rc = get_window_storage<HGLRC>(hwnd, "rc");

			ReleaseDC(hwnd, hdc);
			wglDeleteContext(rc);
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

//temp
#include <cstring>

void fan::window::initialize_window(const std::string& name, const fan::vec2i& window_size, uint64_t flags)
{

#ifdef FAN_PLATFORM_WINDOWS

	auto instance = GetModuleHandle(NULL);

	WNDCLASS wc = {0};

	const char CLASS_NAME[] = "test";
	wc.lpszClassName = CLASS_NAME;
	wc.lpfnWndProc = window_proc;

	wc.hCursor = LoadCursor(NULL, IDC_ARROW);

	wc.hInstance = GetModuleHandle(NULL);
    
	RegisterClass(&wc);

	RECT rect = {0, 0, window_size.x, window_size.y};
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);

	m_window = CreateWindow("test", name.c_str(),
							WS_OVERLAPPEDWINDOW | WS_VISIBLE,
							CW_USEDEFAULT, CW_USEDEFAULT,
							rect.right - rect.left, rect.bottom - rect.top,
							0, 0, instance, 0);
		
	if (!m_window) {
		fan::print("fan window error: failed to initialize window", GetLastError());
		exit(1);
	}

	PIXELFORMATDESCRIPTOR pfd = {
		sizeof(PIXELFORMATDESCRIPTOR),  //  size of this pfd  
		1,                     // version number  
		PFD_DRAW_TO_WINDOW |   // support window  
		PFD_SUPPORT_OPENGL |   // support OpenGL  
		PFD_DOUBLEBUFFER,      // double buffered  
		PFD_TYPE_RGBA,         // RGBA type  
		24,                    // 24-bit color depth  
		0, 0, 0, 0, 0, 0,      // color bits ignored  
		0,                     // no alpha buffer  
		0,                     // shift bit ignored  
		0,                     // no accumulation buffer  
		0, 0, 0, 0,            // accum bits ignored  
		32,                    // 32-bit z-buffer      
		0,                     // no stencil buffer  
		0,                     // no auxiliary buffer  
		PFD_MAIN_PLANE,        // main layer  
		0,                     // reserved  
		0, 0, 0                // layer masks ignored  
	};

	m_hdc = GetDC(m_window);

	static const int attributes[] = {
		fan::OPENGL_MAJOR_VERSION, m_major_version,
		fan::OPENGL_MINOR_VERSION, m_minor_version,
		0
	};

	int pixel_format = ChoosePixelFormat(m_hdc, &pfd);
	SetPixelFormat(m_hdc, pixel_format, &pfd);
	HGLRC temp_rc = wglCreateContext(m_hdc);
	wglMakeCurrent(m_hdc, temp_rc);

	glewExperimental = true;

	static bool initialized = false;

	if (!initialized) {
		if (glewInit() != GLEW_OK) { // maybe
			fan::print("failed to initialize glew");
			exit(1);
		}
		initialized = true;
	}

	HGLRC rc = wglCreateContextAttribsARB(m_hdc, temp_rc, attributes);

	wglMakeCurrent(0, 0);
	wglDeleteContext(temp_rc);
	wglCreateContext(m_hdc);
	wglMakeCurrent(m_hdc, rc);
	
	rect = {};
    if(GetWindowRect(GetConsoleWindow(), &rect)) {
        m_position = fan::vec2i(
			rect.left, 
			rect.top
		);
    }

	set_window_storage(m_window, stringify(rc), rc);
	set_window_storage(m_window, stringify(m_size), &m_size);
	set_window_storage(m_window, stringify(m_previous_size), &m_previous_size);
	set_window_storage(m_window, stringify(m_position), &m_position);

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

	if (minor_glx < 2 && major_glx <= 1) {
		fan::print("fan window error: glx 1.2 or greater is required");
		XCloseDisplay(m_display);
		exit(1);
	}
	
	static constexpr int fb_attribs[] = {
		GLX_RENDER_TYPE, GLX_RGBA_BIT,
		GLX_X_RENDERABLE, True,
		GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
		GLX_DOUBLEBUFFER, True,
		GLX_RED_SIZE, 8,
		GLX_BLUE_SIZE, 8,
		GLX_GREEN_SIZE, 8,
		0
	};

	static int gl_attribs[] = {
		fan::OPENGL_MINOR_VERSION, 2,
		fan::OPENGL_MAJOR_VERSION, 1,
		GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
		0
	};

	int fbcount;
	fan::print(m_display, m_screen, &fbcount);
	
	auto ChooseFBConfigSGIX =
		(PFNGLXCHOOSEFBCONFIGPROC)
		glXGetProcAddress((GLubyte*)"glXChooseFBConfig");

	glXGetVisualFromFBConfig =
		(PFNGLXGETVISUALFROMFBCONFIGPROC)
		glXGetProcAddress((GLubyte*)"glXGetVisualFromFBConfig");
	glXCreateContextAttribsARB =
		(PFNGLXCREATECONTEXTATTRIBSARBPROC)
		glXGetProcAddress((GLubyte*)"glXCreateContextAttribsARB");

	auto fbcs = ((GLXFBConfig*)ChooseFBConfigSGIX(m_display, m_screen, fb_attribs, &fbcount));
	if (!fbcs) {
		fan::print("fan window error: failed to retreive framebuffer");
		XCloseDisplay(m_display);
		exit(1);
	}

	m_visual = glXGetVisualFromFBConfig(m_display, fbcs[0]);

	if (m_visual == 0) {
		std::cout << "Could not create correct visual window.\n";
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

	m_window = XCreateWindow(
		m_display, 
		RootWindow(m_display, m_screen), 
		0, 
		0,
		window_size.x, 
		window_size.y, 
		0,
		m_visual->depth, 
		InputOutput, 
		m_visual->visual, 
		CWBackPixel | CWColormap | CWBorderPixel | CWEventMask | CWCursor, 
		&m_window_attribs
	);

	this->set_window_title(name);

	m_atom_delete_window = XInternAtom(m_display, "WM_DELETE_WINDOW", False);
	XSetWMProtocols(m_display, m_window, &m_atom_delete_window, 1);

	constexpr int context_attribs[] = {
		GLX_CONTEXT_MAJOR_VERSION_ARB, 3,
		GLX_CONTEXT_MINOR_VERSION_ARB, 2,
		GLX_CONTEXT_FLAGS_ARB, GLX_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB,
		None
	};

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

	for (int i = fan::input::first + 1; i != fan::input::last; ++i) {
		m_keys_down.insert(std::make_pair(i, false));
	}

	if (flags & fan::window::flags::full_screen) {
		this->set_full_screen();
	}

	glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	set_window_storage(m_window, stringify(m_keys_action), &m_keys_action);
	set_window_storage(m_window, stringify(m_keys_reset), &m_keys_reset);
	set_window_storage(m_window, stringify(m_keys_down), &m_keys_down);

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

				fan::window::close();

				break;
			}
			case ClientMessage:
			{

				if (m_event.xclient.data.l[0] != m_atom_delete_window) {
					break;
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

				auto mouse_move_callback = this->m_mouse_move_callback;

				for (const auto& i : mouse_move_callback) {
					if (i) {
						i(position);
					}
				}

				m_cursor_position = position;

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