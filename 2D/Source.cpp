#include <iostream>
#include <FAN/Graphics.hpp>
#include <thread>
#include <iomanip>
#include <ctime>

constexpr auto PORT = 43254;
constexpr auto SERVER_IP = "192.168.1.143";

//#define FAN_CLIENT

#ifdef FAN_CLIENT
void process_handler(client& client, message_t& their_messages) {
	while (1) {
		packet_type job = *(packet_type*)client.get_message().data();
		printf("packet type: %s\n", packet_type_str[enum_to_int(job)]);

		switch (job) {
		case packet_type::send_file: {
			client.get_file();
			break;
		}
		case packet_type::send_message: {
			client.get_message();
			break;
		}

		case packet_type::send_message_user: {
			Message_Info info = client.get_message_user();
			std::ostringstream oss;
			auto t = std::time(nullptr);
			auto tm = *std::localtime(&t);
			oss << std::put_time(&tm, "%H:%M : ");
			std::string message = oss.str() += info.message;
			message.push_back('\b');
			their_messages[info.get_username()].insert(0, message);
			printf("received message from %s: %s\n", info.get_username().c_str(), info.message.c_str());
			break;
		}
		}
	}
}
#endif

int main() {
#ifdef FAN_WINDOWS
	init_winsock();
#endif
#ifdef FAN_CLIENT
	client client(SERVER_IP, PORT, "Jokke");
#endif

	glfwSetErrorCallback(GlfwErrorCallback);
	if (!glfwInit()) {
		printf("GLFW ded\n");
		return 0;
	}
	WindowInit();
	Camera camera;
	glfwSetKeyCallback(window, KeyCallback);
	glfwSetCharCallback(window, CharacterCallback);
	glfwSetScrollCallback(window, ScrollCallback);
	glfwSetWindowUserPointer(window, &camera);
	glfwSetCursorPosCallback(window, CursorPositionCallback);
	glfwSetMouseButtonCallback(window, MouseButtonCallback);
	glfwSetFramebufferSizeCallback(window, FrameSizeCallback);

	File f_oldpos("oldpos");

	if (!f_oldpos.read()) {
		glfwSetWindowPos(window, window_size.x / 2, window_size.y / 2);
	}
	else {
		_vec2<int> oldpos = *(_vec2<int>*)f_oldpos.data.data();
		glfwSetWindowPos(window, oldpos.x, oldpos.y);
	}
	
	//glfwSetCursorEnterCallback(window, CursorEnterCallback);

	using namespace fan_gui;

	vec2 text_send_size(window_size.x, type_box_height);
	Square type_box(
		vec2(user_divider_x, window_size.y - text_send_size.y),
		text_send_size,
		user_box_color
	);
	Timer char_erase_speed(Timer::start(), erase_speedlimit);

	TextRenderer tr;
	message_t my_messages, their_messages;

	//std::map<std::string, std::deque<std::string>> d_my_messages, d_their_messages;

	std::string my_text;
	float text_begin = window_size.y - chat_begin_height;

	Line blinker(mat2x2(
		vec2(),
		vec2()
	), Color(1));

	bool blink_now = false;

	Users users("Muhde", their_messages);
	users.add("some random for select test", their_messages);

#ifdef FAN_CLIENT
	std::thread t(process_handler, std::ref(client), std::ref(their_messages), std::ref(d_their_messages));
	t.detach();
#endif

	auto blinker_reposition = [&] {
		vec2 length = tr.get_length(my_text, font_size);
		if (my_text.empty()) {
			length = 0;
		}
		blinker.set_position(
			0,
			mat2x2(
				vec2(text_position_x + length.x, window_size.y - type_box_height / 2 +
					tr.get_length(my_text, font_size).y / 2 - blinker_height),
				vec2(text_position_x + length.x, window_size.y - type_box_height / 2 + 
					tr.get_length(my_text, font_size).y / 2)
			)
		);
	};

	Titlebar title_bar;

	key_callback.add(GLFW_KEY_BACKSPACE, false, [&] {
		if (!my_text.empty() && char_erase_speed.finished()) {
			my_text.erase(my_text.size() - 1);
			char_erase_speed.restart();
		}
		blinker_reposition();
	}); 

	key_callback.add(GLFW_KEY_ESCAPE, false, [&] {
		users.reset();
		//glfwSetWindowShouldClose(window, true);
	});

	key_callback.add(GLFW_KEY_UP, false, [] {
		glfwSetWindowSize(window, window_size.x + 10, window_size.y + 10);
	});

	key_callback.add(GLFW_KEY_DOWN, false, [] {
		glfwSetWindowSize(window, window_size.x - 10, window_size.y - 10);
	});

	key_callback.add(GLFW_KEY_ENTER, false, [&] {
		if (!users.selected()) {
			return;
		}
		if (!my_text.empty()) {
			auto t = std::time(nullptr);
			auto tm = *std::localtime(&t);
			std::ostringstream oss;
			oss << std::put_time(&tm, " -> %H:%M");
			std::string message = my_text += oss.str();
#ifdef FAN_CLIENT
			client.send_message(my_text, users.get_user());
			my_messages[users.get_user()].insert(0, std::string(client.get_username()) += std::string(": ") += my_text += oss.str());
#endif
			my_messages[users.get_user()].push_front(message);
			my_text.clear();
		}
		blinker_reposition();
	});

	key_callback.add(GLFW_MOUSE_BUTTON_LEFT, true, [&] {
		title_bar.callbacks();
		users.select();
	});

	key_release_callback.add(GLFW_MOUSE_BUTTON_LEFT, GLFW_RELEASE, [&] {
		title_bar.move_window(false);
	});

	window_resize_callback.add([&] {
		text_send_size = vec2(window_size.x, type_box_height);

		type_box.set_position(0, vec2(0, window_size.y - text_send_size.y));
		type_box.set_size(0, text_send_size);

		blinker_reposition();
		text_begin = window_size.y - chat_begin_height;

		title_bar.resize_update();
	});

	cursor_move_callback.add([&] {
		title_bar.cursor_update();
		users.color_callback();
	});

	character_callback.add([&](int key) {
		if (!users.selected()) {
			return;
		}
		my_text.push_back(key);
		blinker_reposition();
	});

	Timer timer(Timer::start(), blink_rate, Timer::mode::WAIT_FINISH, [&] {
		blink_now = !blink_now;
	});
	timer.add_function(Timer::mode::EVERY_OTHER, [&] {
		blinker.draw();
	});

	// init
	type_box.set_position(0, vec2(0, window_size.y - text_send_size.y));
	blinker_reposition();
	text_begin = window_size.y - chat_begin_height;
	title_bar.resize_update();

	while (!glfwWindowShouldClose(window)) {

		glfwPollEvents();
		glClearColor(
			background_color.r,
			background_color.g,
			background_color.b,
			background_color.a
		);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		if (users.selected()) {
			type_box.draw();
			timer.run_functions();
		}

		tr.render(my_text, vec2(text_position_x, window_size.y - type_box_height / 2 + tr.get_length(my_text, font_size).y / 2), font_size, white_color);
		for (int i = 0; i < my_messages[users.get_user()].size(); i++) {
			tr.render(my_messages[users.get_user()][i], vec2(window_size.x - tr.get_length(my_messages[users.get_user()][i], font_size).x - 10, text_begin - i * text_gap), font_size, white_color);
		}
		for (int i = 0; i < their_messages[users.get_user()].size(); i++) {
			tr.render(their_messages[users.get_user()][i], vec2(user_box_size.x + 10, text_begin - i * text_gap), font_size, white_color);
		}
		users.draw();

		title_bar.draw();
		users.render_text(tr);

		GetFps();
		glfwSwapBuffers(window);
		Sleep(10);
	}
	_vec2<int> window_position;
	glfwGetWindowPos(window, &window_position.x, &window_position.y);
	File::write("oldpos", std::string((const char*)&window_position, sizeof(window_position)));
}