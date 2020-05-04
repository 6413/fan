#include <iostream>
#include <FAN/Graphics.hpp>
#include <iomanip>

using namespace fan_gui;
#define FAN_CLIENT

void save_chat_history(message_t messages, const char* path) {
	std::string l_chat;
	for (auto user : messages) {
		for (int j = 0; j < messages[user.first].size(); j++) {
			l_chat.append(std::string(user.first.c_str()) += std::string(":") += messages[user.first][j] += std::string("\n"));
		}
	}
	File::write(path, l_chat, std::ios_base::binary);
}

void load_messages(const char* path, message_t& messages) {
	if (!File::file_exists(path)) {
		return;
	}
	std::ifstream file(path, std::ios_base::binary);

	message_t r_chat;
	std::vector<std::string> l_users;
	std::string current;
	while (std::getline(file, current)) {
		auto found = current.find_first_of(':');
		if (found != std::string::npos) {
			auto find_user = std::find(l_users.begin(), l_users.end(), current.substr(0, found));
			if (find_user == l_users.end()) {
				l_users.push_back(current.substr(0, found));
			}
			r_chat[*(l_users.end() - 1)].push_back(
				current.substr(
					found + 1,
					current.size() - (found + 1)
				)
			);
		}
		else {
			puts("failed to read chat");
		}
	}
	file.close();
	for (int i = 0; i < l_users.size(); i++) {
		for (int j = 0; j < r_chat[l_users[i]].size(); j++) {
			messages[l_users[i]].push_back(r_chat[l_users[i]][j]);
		}
	}
}

constexpr auto PORT = 43254;
constexpr auto SERVER_IP = "192.168.1.143";

float max_height = 0;

bool do_refresh = false;

unsigned int EndlineCount(const std::string& str) {
	int count = 0;
	for (int i = 0; i < str.size(); i++) {
		if (str[i] == '\n') {
			count++;
		}
	}
	return count;
}

void process_handler(
	client& client,
	message_t& messages,
	chat_box_t& chat_box,
	const Users& users,
	TextRenderer* tr,
	message_t& my_messages,
	chat_box_t& my_chat_box
) {
	while (1) {
		packet_type job = *(packet_type*)client.get_message().data();
		printf("packet type: %s\n", packet_type_str[eti(job)]);

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

			for (int i = 0; i < EndlineCount(text_box::get_finished_string(tr, info.message)) + 1; i++) {
				printf("%d\n", i);
				my_messages[info.get_username()].push_front("\n");
				my_chat_box[info.get_username()].push_back(
					fan_gui::text_box(
						tr,
						"\n",
						vec2(),
						select_color
					)
				);
			}
			messages[info.get_username()].push_front(info.message);
			chat_box[info.get_username()].push_back(
				fan_gui::text_box(
					tr,
					info.message,
					vec2(),
					select_color
				)
			);
			max_height = window_size.y - type_box_height;
			for (int i = 0; i < my_messages[users.get_user()].size(); i++) {
				if (my_messages[users.get_user()][i].size() == 1 && my_messages[users.get_user()][i][0] == '\n') {
					continue;
				}
				max_height -= text_box::get_size_all(tr, my_messages[users.get_user()][i]).y + chat_boxes_gap;
			}
			for (int i = 0; i < messages[users.get_user()].size(); i++) {
				if (messages[users.get_user()][i].size() == 1 && messages[users.get_user()][i][0] == '\n') {
					continue;
				}
				max_height -= text_box::get_size_all(tr, messages[users.get_user()][i]).y + chat_boxes_gap;
			}
			max_height -= scroll_max_gap;

			do_refresh = true;
			break;
		}
		}
	}
}

bool add_users = false;
std::string new_user;
void add_user(Users& users, message_t& their_messages) {
	std::cout << "enter username: ";
	std::cin >> new_user;
	add_users = true;
	std::cout << std::endl;
}

void console(Users& users, message_t& their_messages) {
	std::string command;
	while (1) {
		std::cin >> command;
		if (command == "add_user") {
			add_user(users, their_messages);
		}
		command.clear();
	}
}

//#define ADD_USER

int main() {
#ifdef ADD_USER
	std::string username;
	std::cout << "enter username: ";
	std::cin >> username;
	std::cout << std::endl;
#endif
#ifdef FAN_WINDOWS
	ShowWindow(GetActiveWindow(), SW_HIDE);
	init_winsock();
#endif

#ifdef FAN_CLIENT
#ifdef ADD_USER
	client client(SERVER_IP, PORT, username);
#else
	client client(SERVER_IP, PORT, "2");
#endif
#endif
	glfwSetErrorCallback(GlfwErrorCallback);
	if (!glfwInit()) {
		printf("GLFW ded\n");
		return 0;
	}

	WindowInit();
	Camera camera;
	glfwSetKeyCallback(window, KeyCallback);
	glfwSetWindowUserPointer(window, &camera);
	glfwSetScrollCallback(window, ScrollCallback);
	glfwSetCharCallback(window, CharacterCallback);
	glfwSetWindowFocusCallback(window, FocusCallback);
	glfwSetMouseButtonCallback(window, MouseButtonCallback);
	glfwSetCursorPosCallback(window, CursorPositionCallback);
	glfwSetFramebufferSizeCallback(window, FrameSizeCallback);
	glfwSetDropCallback(window, DropCallback);

	File f_oldpos("oldpos");

	if (!f_oldpos.read()) {
		glfwSetWindowPos(window, window_size.x / 2, window_size.y / 2);
	}
	else {
		_vec2<int> oldpos = *(_vec2<int>*)f_oldpos.data.data();
		glfwSetWindowPos(window, oldpos.x, oldpos.y);
	}

	vec2 text_box_size(window_size.x, type_box_height);
	button_single type_box(
		vec2(user_divider_x, window_size.y - text_box_size.y),
		text_box_size,
		user_box_color
	);
	Timer char_erase_speed(Timer::start(), erase_speedlimit);

	message_t my_messages, their_messages;

	std::string my_text;
	float text_begin = window_size.y - chat_begin_height;

	Line blinker(mat2x2(
		vec2(),
		vec2()
	), Color(1));

	bool blink_now = false;

	Users users("1", their_messages);

#ifdef ADD_USER
	std::thread c(console, std::ref(users), std::ref(their_messages));
	c.detach();
#endif

	chat_box_t my_chatboxes, their_chatboxes;
	TextRenderer tr;

	auto blinker_reposition = [&] {
		vec2 length = tr.get_length(my_text, font_size);
		if (my_text.empty()) {
			length = 0;
		}
		length.y = tr.get_length("Å", font_size).y / 2;
		blinker.set_position(
			mat2x2(
				vec2(text_position_x + length.x, window_size.y - type_box_height / 2 +
					length.y - blinker_height),
				vec2(text_position_x + length.x, window_size.y - type_box_height / 2 +
					length.y + blinker_height / 2)
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
		});

	key_callback.add(GLFW_KEY_UP, false, [] {
		glfwSetWindowSize(window, window_size.x + 10, window_size.y + 10);
		});

	key_callback.add(GLFW_KEY_DOWN, false, [] {
		glfwSetWindowSize(window, window_size.x - 10, window_size.y - 10);
		});

	key_callback.add(GLFW_KEY_V, false, [&] {
		if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL)) {
			if (!users.selected()) {
				return;
			}
			const char* verify = glfwGetClipboardString(window);
			if (!verify) {
				return;
			}
			std::string clip_board(verify);
			for (int i = 0; i < clip_board.size(); i++) {
				if (std::isspace(clip_board[i])) {
					clip_board[i] = ' ';
				}
				my_text.push_back(clip_board[i]);
			}
			blinker_reposition();
		}
		});

	int iScroll = 0;

	key_callback.add(GLFW_KEY_ENTER, false, [&] {
		if (!users.selected()) {
			return;
		}
		if (!my_text.empty()) {
			auto t = std::time(nullptr);
			auto tm = *std::localtime(&t);
			std::ostringstream oss;
			oss << std::put_time(&tm, " - %H:%M");
			std::string message;
			my_text += oss.str();
#ifdef FAN_CLIENT
			client.send_message(my_text, users.get_user());
#endif

			for (int i = 0; i < EndlineCount(text_box::get_finished_string(&tr, my_text)) + 1; i++) {
				their_messages[users.get_user()].push_front("\n");
				their_chatboxes[users.get_user()].push_back(
					fan_gui::text_box(
						&tr,
						"\n",
						vec2(),
						select_color
					)
				);
			}

			my_messages[users.get_user()].push_front(my_text);
			my_chatboxes[users.get_user()].push_back(
				fan_gui::text_box(
					&tr,
					my_text,
					vec2(),
					select_color
				)
			);

			Timer timer(Timer::start(), 10000);
			do_refresh = true;
			printf("reload time %llu\n", timer.elapsed());
			my_text.clear();
			max_height = window_size.y - type_box_height;
			for (int i = 0; i < my_messages[users.get_user()].size(); i++) {
				if (my_messages[users.get_user()][i].size() == 1 && my_messages[users.get_user()][i][0] == '\n') {
					continue;
				}
				max_height -= text_box::get_size_all(&tr, my_messages[users.get_user()][i]).y + chat_boxes_gap;
			}
			for (int i = 0; i < their_messages[users.get_user()].size(); i++) {
				if (their_messages[users.get_user()][i].size() == 1 && their_messages[users.get_user()][i][0] == '\n') {
					continue;
				}
				max_height -= text_box::get_size_all(&tr, their_messages[users.get_user()][i]).y + chat_boxes_gap;
			}
			max_height -= scroll_max_gap;
			blinker_reposition();
			iScroll = 0;
		}
		});

	key_callback.add(GLFW_MOUSE_BUTTON_LEFT, true, [&] {
		title_bar.move_window();
		users.select();
		});

	Square text_highlight(vec2(0), vec2(0), select_color);

	key_release_callback.add(GLFW_MOUSE_BUTTON_LEFT, GLFW_RELEASE, [&] {
		title_bar.callbacks();
		title_bar.move_window(false);
		for (int i = 0; i < my_chatboxes[users.get_user()].size(); i++) {
			if (my_chatboxes[users.get_user()][i].inside()) {
				my_chatboxes[users.get_user()][i].set_text(
					"replace test", my_messages[users.get_user()],
					my_chatboxes[users.get_user()].size() - i - 1
				);
				text_box::refresh(my_chatboxes[users.get_user()], my_messages[users.get_user()], &tr);
			}
		}
		});

	window_resize_callback.add([&] {
		text_box_size = vec2(window_size.x, type_box_height);

		blinker_reposition();
		text_begin = window_size.y - chat_begin_height;

		title_bar.resize_update();
		type_box.set_position(vec2(0, window_size.y - text_box_size.y));
		type_box.set_size(vec2(window_size.x, 100));

		float highest(
			text_begin -
			(my_messages[users.get_user()].size() * text_gap +
				their_messages[users.get_user()].size() * text_gap)
		);

		for (int i = 0; i < users.size(); i++) {
			text_box::refresh(my_chatboxes[users.get_username(i)], my_messages[users.get_username(i)], &tr);
		}
		max_height = window_size.y - type_box_height;
		for (int i = 0; i < my_messages[users.get_user()].size(); i++) {
			max_height -= text_box::get_size_all(&tr, my_messages[users.get_user()][i]).y + chat_boxes_gap;
		}
		max_height -= scroll_max_gap;
		users.resize_update();
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

	load_messages(my_chat_path, my_messages);
	load_messages(their_chat_path, their_messages);

	scroll_callback.add(GLFW_MOUSE_SCROLL_UP, false, [&] {
		iScroll += scroll_sensitivity;
		if (iScroll >= -max_height) {
			iScroll = -max_height;
		}
		if (-max_height < 0) {
			iScroll = 0;
		}
		text_box::refresh(
			my_chatboxes[users.get_user()],
			my_messages[users.get_user()],
			their_chatboxes[users.get_user()],
			their_messages[users.get_user()],
			&tr,
			iScroll
		);
		});

	scroll_callback.add(GLFW_MOUSE_SCROLL_DOWN, false, [&] {
		iScroll -= scroll_sensitivity;
		if (iScroll < 0) {
			iScroll = 0;
		}
		text_box::refresh(
			my_chatboxes[users.get_user()],
			my_messages[users.get_user()],
			their_chatboxes[users.get_user()],
			their_messages[users.get_user()],
			&tr,
			iScroll
		);
		});

	drop_callback.add([&](int path_count, const char* path[]) {});

	// init
	type_box.set_position(vec2(user_divider_x, window_size.y - text_box_size.y));
	blinker_reposition();
	text_begin = window_size.y - chat_begin_height;
	title_bar.resize_update();

	for (int i = 0; i < users.size(); i++) {
		text_box::refresh(
			my_chatboxes[users.get_username(i)],
			my_messages[users.get_username(i)],
			their_chatboxes[users.get_username(i)],
			their_messages[users.get_username(i)],
			&tr,
			iScroll
		);
	}
	max_height = window_size.y - type_box_height;
	for (int i = 0; i < users.size(); i++) {
		for (int j = 0; j < my_messages[users.get_username(i)].size(); j++) {
			max_height -= text_box::get_size_all(&tr, my_messages[users.get_username(i)][j]).y + chat_boxes_gap;
		}
	}
	max_height -= scroll_max_gap;

#ifdef FAN_CLIENT
	std::thread t(
		process_handler,
		std::ref(client),
		std::ref(their_messages),
		std::ref(their_chatboxes),
		std::ref(users),
		&tr,
		std::ref(my_messages),
		std::ref(my_chatboxes)
	);
	t.detach();
#endif

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		glClearColor(
			background_color.r,
			background_color.g,
			background_color.b,
			background_color.a
		);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		for (int i = my_chatboxes[users.get_user()].size(); i--; ) {
			auto chat_box = my_chatboxes[users.get_user()][i];
			if (chat_box.get_text().empty() || (chat_box.get_text().size() == 1 && chat_box.get_text()[0] == '\n')) {
				continue;
			}
			chat_box.draw();
		}
		for (int i = their_chatboxes[users.get_user()].size(); i--; ) {
			auto chat_box = their_chatboxes[users.get_user()][i];
			if (chat_box.get_text().empty() || (chat_box.get_text().size() == 1 && chat_box.get_text()[0] == '\n')) {
				continue;
			}
			chat_box.draw();
		}

		if (users.selected()) {
			type_box.draw();
			if (window_focused()) {
				timer.run_functions();
			}
			else {
				timer.restart();
			}
		}

		text_highlight.draw();

		tr.render(
			my_text,
			vec2(
				text_position_x,
				window_size.y - type_box_height / 2 +
				tr.get_length("Å", font_size).y / 2
			),
			font_size,
			white_color
		);

		if (do_refresh) {
			text_box::refresh(
				my_chatboxes[users.get_user()],
				my_messages[users.get_user()],
				their_chatboxes[users.get_user()],
				their_messages[users.get_user()],
				&tr,
				iScroll
			);
			do_refresh = false;
		}

		if (add_users) {
			users.add(new_user, their_messages);
			load_messages(my_chat_path, my_messages);
			load_messages(their_chat_path, their_messages);
			add_users = false;
			new_user.clear();
		}

		users.draw();

		title_bar.draw();
		users.render_text(tr);

		GetFps();
		glfwSwapBuffers(window);
	}
	_vec2<int> window_position;
	glfwGetWindowPos(window, &window_position.x, &window_position.y);
	File::write(
		"oldpos",
		std::string((const char*)&window_position,
			sizeof(window_position)),
		std::ios_base::binary
	);

	save_chat_history(my_messages, my_chat_path);
	save_chat_history(their_messages, their_chat_path);
	glfwTerminate();
	return 0;
}