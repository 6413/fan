#include <fan/window.hpp>

auto& fan::default_callback::get_function(uint64_t i)
{
	return functions[i];
}

uint64_t fan::default_callback::size() const
{
	return functions.size();
}

int fan::KeyCallback::get_action(uint64_t i) const
{
	return action[i];
}

int fan::KeyCallback::get_key(uint64_t i) const
{
	return key[i];
}

uint64_t fan::KeyCallback::size() const
{
	return functions.size();
}

auto fan::KeyCallback::get_function(uint64_t i) const
{
	return functions[i];
}

fan::window::window(GLFWwindow* window) : m_window(window) { }

fan::window::window(uint64_t flags, const fan::vec2i& size, const std::string& name) : m_size(size)
{
	glfwSetErrorCallback(ErrorCallback);
	if (!glfwInit()) {
		fan::print("GLFW ded");
		exit(1);
	}

	m_window = glfwCreateWindow(size.x, size.y, name.c_str(),
		flags & fan::window_flags::FULL_SCREEN ? glfwGetPrimaryMonitor() : 0, 0);

	if (!fan::window::m_window) {
		fan::print("failed to create a window");
		glfwTerminate();
		exit(1);
	}

	glfwMakeContextCurrent(fan::window::m_window);
	if ( glewInit() != GLEW_OK) {
		fan::print("failed to initialize glew");
		glfwTerminate();
		exit(1);
	}

	glViewport(0, 0, fan::window::m_size.x,fan::window::m_size.y);

	glfwSetKeyCallback(fan::window::m_window, KeyCallback);
	//glfwSetCharCallback(fan::window::m_window, CharacterCallback);
	glfwSetScrollCallback(fan::window::m_window, ScrollCallback);
	glfwSetMouseButtonCallback(fan::window::m_window, MouseButtonCallback);
	glfwSetCursorPosCallback(fan::window::m_window, CursorPositionCallback);
	glfwSetFramebufferSizeCallback(fan::window::m_window, FrameSizeCallback);
	//glfwSetDropCallback(fan::window::m_window, DropCallback);
	
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_DECORATED, !(flags & fan::window_flags::NO_DECORATE));
	glfwWindowHint(GLFW_RESIZABLE, !(flags & fan::window_flags::NO_RESIZE));
	glfwSetInputMode(this->m_window, GLFW_CURSOR, 
		!(flags & fan::window_flags::NO_MOUSE) ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);

	if (!(flags & fan::window_flags::ANITIALISING)) {
		glfwWindowHint(GLFW_SAMPLES, 16);
		glEnable(GL_MULTISAMPLE);
	}

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	//fan::get_fps(fan::window::m_window);
	//glfwSwapBuffers(fan::window::m_window);
	//glfwPollEvents();

	m_callback_key.add(GLFW_KEY_ESCAPE, true, [&] {
		glfwSetWindowShouldClose(fan::window::m_window, true);
	});

	glfwSetWindowUserPointer(fan::window::m_window, this);
}

fan::window::~window()
{
	if (fan::window::close()) {
		glfwDestroyWindow(fan::window::m_window);
	}
}

bool fan::window::close() const
{
	return glfwWindowShouldClose(fan::window::m_window);
}

int fan::window::key_press(int key, bool action) {
	if (!m_input_previous_action[key] && m_input_action[key]) {
		m_input_previous_action[key] = true;
	}
	else {
		m_input_action[key] = false;
		m_input_previous_action[key] = false;
	}
	return get_key(m_window, key) & (action ? m_input_action[key] : 1);
}

int fan::window::get_key(const fan::window& window, int key) const
{
	if (key <= GLFW_MOUSE_BUTTON_8) {
		return glfwGetMouseButton(window.m_window, key);
	}
	return glfwGetKey(window.m_window, key);
}

void fan::window::set_window_title(const std::string& title) const
{
	glfwSetWindowTitle(fan::window::m_window, title.c_str());
}

void fan::window::calculate_delta_time()
{
	m_current_frame = glfwGetTime();
	m_delta_time = m_current_frame - m_last_frame;
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

fan::mat4 fan::window::get_projection_2d(const fan::mat4& projection) const
{
	return fan::ortho((f_t)m_size.x / 2, m_size.x + (f_t)m_size.x * 0.5f, m_size.y + (f_t)m_size.y * 0.5f, (f_t)m_size.y / 2.f, 0.1f, 1000.0f);
}

fan::mat4 fan::window::get_view_translation_2d(const fan::mat4& view) const
{
	return fan::translate(view, fan::vec3((f_t)m_size.x / 2, (f_t)m_size.y / 2, -700.0f));
}

void fan::window::swap_buffers() const
{
	glfwPollEvents();
	glfwSwapBuffers(fan::window::m_window);
}

void fan::window::ErrorCallback(int id, const char* error) {
	printf("GLFW Error %d : %s\n", id, error);
}

void fan::window::KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	fan::window* fwindow = (fan::window*)glfwGetWindowUserPointer(window);

	if (action == GLFW_PRESS) {
		fwindow->m_input_action[key] = action;
	}

	for (uint_t i = 0; i < fwindow->m_callback_key.size(); i++) {
		if (key != fwindow->m_callback_key.get_key(i)) {
			continue;
		}
		if (fwindow->m_callback_key.get_action(i) != action && fwindow->m_callback_key.get_action(i)) {
			continue;
		}
		fwindow->m_callback_key.get_function(i)();
	}

	static bool release = 0;

	if (release) {
		for (uint_t i = 0; i < fwindow->m_callback_key_release.size(); i++) {
			if (key != fwindow->m_callback_key_release.get_key(i)) {
				continue;
			}
			if (fwindow->m_callback_key_release.get_action(i) != GLFW_RELEASE) {
				continue;
			}
			fwindow->m_callback_key_release.get_function(i)();
		}
	}
	
	release = !release;
}

void fan::window::MouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	fan::window* fwindow = (fan::window*)glfwGetWindowUserPointer(window);

	fwindow->m_input_action[button] = action;
	for (uint_t i = 0; i < fwindow->m_callback_key.size(); i++) {
		if (button != fwindow->m_callback_key.get_key(i)) {
			continue;
		}
		if (fwindow->m_callback_key.get_action(i) && fwindow->m_callback_key.get_action(i) != action) {
			continue;
		}
		fwindow->m_callback_key.get_function(i)();
	}

	static bool release = 0;
	if (release) {
		for (uint_t i = 0; i < fwindow->m_callback_key_release.size(); i++) {
			if (button != fwindow->m_callback_key_release.get_key(i)) {
				continue;
			}
			if (fwindow->m_callback_key_release.get_action(i) != GLFW_RELEASE) {
				continue;
			}
			fwindow->m_callback_key_release.get_function(i)();
		}
	}
	release = !release;
}

void fan::window::CursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {
	fan::window* fwindow = (fan::window*)glfwGetWindowUserPointer(window);

	fwindow->m_cursor_position = fan::vec2i(xpos, ypos);
	for (uint_t i = 0; i < fwindow->m_callback_cursor_move.size(); i++) {
		fwindow->m_callback_cursor_move.get_function(i)();
	}
}

void fan::window::ScrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
	fan::window* fwindow = (fan::window*)glfwGetWindowUserPointer(window);

	for (uint_t i = 0; i < fwindow->m_callback_scroll.size(); i++) {
		if (fwindow->m_callback_scroll.get_key(i) != GLFW_MOUSE_SCROLL_UP && yoffset == 1) {
			continue;
		}
		else if (fwindow->m_callback_scroll.get_key(i) != GLFW_MOUSE_SCROLL_DOWN && yoffset == -1) {
			continue;
		}
		fwindow->m_callback_scroll.get_function(i)();
	}
}

void fan::window::FrameSizeCallback(GLFWwindow* window, int width, int height) {
	fan::window* fwindow = (fan::window*)glfwGetWindowUserPointer(window);

	glViewport(0, 0, width, height);
	fwindow->m_previous_size = fwindow->m_size;
	fwindow->m_size = fan::vec2i(width, height);
	for (uint_t i = 0; i < fwindow->m_callback_window_resize.size(); i++) {
		fwindow->m_callback_window_resize.get_function(i)();
	}
}