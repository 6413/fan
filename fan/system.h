#pragma once
//
#include <fan/utility.h>
//
//#if defined(fan_platform_windows)
//#include <DXGI.h>
//#include <DXGI1_2.h>
//#include <D3D11.h>
//
//#pragma comment(lib, "Dxgi.lib")
//#pragma comment(lib, "D3D11.lib")
//
//#elif defined(fan_platform_unix)
//
//#include <X11/extensions/Xrandr.h>
//#include <X11/Xlib.h>
//#include <X11/Xutil.h>
//#include <X11/Xos.h>
//#include <X11/Xatom.h>
//#include <X11/keysym.h>
//#include <X11/XKBlib.h>
//
//#include <dlfcn.h>
//
//#undef index_t // xos.h
//
//#include <sys/time.h>
//#include <unistd.h>
//
//#endif
//
//import fan.types.vector;
//import fan.window.input;
//
//namespace fan {
//  namespace sys {
//
//      #if defined(fan_platform_windows)
//
//      struct input {
//
//        static inline std::unordered_map<uint16_t, bool> key_down;
//        static inline std::unordered_map<uint16_t, bool> reset_keys;
//
//								fan::vec2i get_mouse_position() {
//					POINT p;
//					GetCursorPos(&p);
//
//					return fan::vec2i(p.x, p.y);
//				}
//
//        auto get_key_state(int key) {
//					return GetAsyncKeyState(fan::window::input::convert_fan_to_scancode(key));
//				}
//
//        void set_mouse_position(const fan::vec2i& position) {
//					SetCursorPos(position.x, position.y);
//				}
//
//        void send_mouse_event(int key, fan::mouse_state state) {
//					constexpr auto get_key = [](int key, fan::mouse_state state) {
//
//						auto press = state == fan::mouse_state::press;
//
//						switch (key) {
//							case fan::mouse_left: {
//								if (press) {
//									return MOUSEEVENTF_LEFTDOWN;
//								}
//
//								return MOUSEEVENTF_LEFTUP;
//							}
//
//							case fan::mouse_middle: {
//								if (press) {
//									return MOUSEEVENTF_MIDDLEDOWN;
//								}
//
//								return MOUSEEVENTF_MIDDLEUP;
//							}
//
//							case fan::mouse_right: {
//								if (press) {
//									return MOUSEEVENTF_RIGHTDOWN;
//								}
//
//								return MOUSEEVENTF_RIGHTUP;
//							}
//
//						}
//						};
//
//					INPUT input = {};
//
//					input.type = INPUT_MOUSE;
//					input.mi.dwFlags = get_key(key, state);
//
//					if (SendInput(1, &input, sizeof(input)) != 1) {
//						fan::throw_error("");
//					}
//				}
//
//        static void send_keyboard_event(int key, fan::keyboard_state_t state) {
//					INPUT input;
//
//					input.type = INPUT_KEYBOARD;
//					//input.ki.wScan = 0;
//					input.ki.time = 0;
//					input.ki.dwExtraInfo = 0;
//					input.ki.wVk = 0;
//
//					input.ki.wScan = fan::window::input::convert_fan_to_scancode(key);
//
//					input.ki.dwFlags = (state == fan::keyboard_state::press ? 0 : KEYEVENTF_KEYUP) | KEYEVENTF_SCANCODE;
//
//					if (SendInput(1, &input, sizeof(input)) != 1) {
//						fan::throw_error("");
//					}
//				}
//
//        void send_string(const std::string & str, uint32_t delay_between) {
//					for (int i = 0; i < str.size(); i++) {
//
//						if (str[i] >= '0' && str[i] <= '9') {
//							int key = static_cast<int>(fan::key_0 + (str[i] - '0'));
//							fan::sys::input::send_keyboard_event(key, fan::keyboard_state::press);
//							Sleep(delay_between);
//							fan::sys::input::send_keyboard_event(key, fan::keyboard_state::release);
//							continue;
//						}
//
//						uint32_t offset = 0;
//
//						if (str[i] >= 97) {
//							offset -= 32;
//						}
//						auto send_press = [](uint32_t key, uint32_t delay) {
//							fan::sys::input::send_keyboard_event(key, fan::keyboard_state::press);
//							Sleep(delay);
//							fan::sys::input::send_keyboard_event(key, fan::keyboard_state::release);
//							};
//
//						switch (str[i] + offset) {
//							case ' ': {
//								send_press(fan::key_space, delay_between);
//								break;
//							}
//							case '\n': {
//								send_press(fan::key_enter, delay_between);
//								break;
//							}
//							case '?': {
//								fan::sys::input::send_keyboard_event(fan::key_left_shift, fan::keyboard_state::press);
//								send_press(fan::key_plus, delay_between);
//								fan::sys::input::send_keyboard_event(fan::key_left_shift, fan::keyboard_state::release);
//								break;
//							}
//							case '!': {
//								fan::sys::input::send_keyboard_event(fan::key_left_shift, fan::keyboard_state::press);
//								send_press(fan::key_1, delay_between);
//								fan::sys::input::send_keyboard_event(fan::key_left_shift, fan::keyboard_state::release);
//								break;
//							}
//							case '-': {
//								send_press(fan::key_minus, delay_between);
//								break;
//							}
//							case '_': {
//								fan::sys::input::send_keyboard_event(fan::key_left_shift, fan::keyboard_state::press);
//								send_press(fan::key_minus, delay_between);
//								fan::sys::input::send_keyboard_event(fan::key_left_shift, fan::keyboard_state::release);
//								break;
//							}
//							default: {
//								send_press(str[i] + offset, delay_between);
//							}
//						}
//
//						if (str[i] >= 65 && str[i] <= 90) {
//							fan::sys::input::send_keyboard_event(fan::key_left_shift, fan::keyboard_state::release);
//						}
//					}
//				}
//
//        // creates another thread, non blocking
//        void listen_keyboard(std::function<void(int key, fan::keyboard_state_t keyboard_state, bool action)> input_callback_) {
//
//					input_callback = input_callback_;
//
//					//thread_loop();
//
//				}
//
//        // BLOCKS
//        void listen_mouse(std::function<void(const fan::vec2i& position)> mouse_move_callback_) {
//
//					mouse_move_callback = mouse_move_callback_;
//
//					// thread_loop();
//				}
//
//
//      //private:
//
//        static DWORD WINAPI thread_loop(auto l) {
//					HINSTANCE hInstance = GetModuleHandle(NULL);
//
//					if (!hInstance) {
//						fan::print("failed to initialize hinstance");
//						exit(1);
//					}
//
//					mouse_hook = SetWindowsHookEx(WH_MOUSE_LL, (HOOKPROC)mouse_callback, hInstance, NULL);
//					keyboard_hook = SetWindowsHookEx(WH_KEYBOARD_LL, (HOOKPROC)keyboard_callback, hInstance, NULL);
//
//
//					MSG message;
//
//					while (1)
//					{
//						int ret = GetMessage(&message, NULL, 0, 0);
//
//						if (ret == -1)
//						{
//							fan::print("failed to get message");
//						}
//						else
//						{
//							TranslateMessage(&message);
//							DispatchMessage(&message);
//							l();
//						}
//					}
//
//					UnhookWindowsHookEx(mouse_hook);
//					UnhookWindowsHookEx(keyboard_hook);
//				}
//
//
//        static __declspec(dllexport) LRESULT CALLBACK keyboard_callback(int nCode, WPARAM wParam, LPARAM lParam) {
//					KBDLLHOOKSTRUCT hooked_key = *((KBDLLHOOKSTRUCT*)lParam);
//
//					auto key = fan::window::input::convert_scancode_to_fan(hooked_key.scanCode);
//
//					auto state = (fan::keyboard_state_t)((nCode == HC_ACTION) && ((wParam == WM_SYSKEYDOWN) || (wParam == WM_KEYDOWN)));
//
//					if (state == fan::keyboard_state::press && !reset_keys[key]) {
//						key_down[key] = true;
//					}
//
//					input_callback(key, state, key_down[key]);
//
//					if (state == fan::keyboard_state::release) {
//						reset_keys[key] = false;
//					}
//					else {
//						key_down[key] = false;
//						reset_keys[key] = true;
//					}
//
//					return CallNextHookEx(keyboard_hook, nCode, wParam, lParam);
//				}
//
//        //WM_LBUTTONDOWN
//        inline LRESULT fan::sys::input::mouse_callback(int nCode, WPARAM wParam, LPARAM lParam) {
//					MSLLHOOKSTRUCT hooked_key = *((MSLLHOOKSTRUCT*)lParam);
//
//					if (nCode != HC_ACTION) {
//						return CallNextHookEx(mouse_hook, nCode, wParam, lParam);
//					}
//
//					constexpr auto get_mouse_key = [](int key) {
//						switch (key) {
//
//							case WM_LBUTTONDOWN: { return fan::input::mouse_left; }
//							case WM_MBUTTONDOWN: { return fan::input::mouse_middle; }
//							case WM_RBUTTONDOWN: { return fan::input::mouse_right; }
//
//							case WM_LBUTTONUP: { return fan::input::mouse_left; }
//							case WM_MBUTTONUP: { return fan::input::mouse_middle; }
//							case WM_RBUTTONUP: { return fan::input::mouse_right; }
//
//							case WM_MOUSEWHEEL: { return fan::input::mouse_scroll_up; } // ?
//							case WM_MOUSEHWHEEL: { return fan::input::mouse_scroll_down; } // ?
//							default: return fan::input::mouse_scroll_down;
//						}
//						};
//
//					auto key = get_mouse_key(wParam);
//
//					switch (wParam) {
//						case WM_MOUSEMOVE: {
//
//							mouse_move_callback(fan::vec2i(hooked_key.pt.x, hooked_key.pt.y));
//
//							break;
//						}
//						case WM_LBUTTONDOWN:
//						case WM_MBUTTONDOWN:
//						case WM_RBUTTONDOWN:
//						case WM_MOUSEWHEEL:
//						case WM_MOUSEHWHEEL:
//						{
//							if (!reset_keys[key]) {
//								key_down[key] = true;
//							}
//
//							if (input_callback) {
//								input_callback(key, fan::keyboard_state::press, key_down[key]);
//							}
//
//							key_down[key] = false;
//							reset_keys[key] = true;
//
//							break;
//						}
//						case WM_LBUTTONUP:
//						case WM_MBUTTONUP:
//						case WM_RBUTTONUP:
//						{
//							if (input_callback) {
//								input_callback(key, fan::keyboard_state::release, key_down[key]);
//							}
//
//							reset_keys[key] = false;
//
//							break;
//						}
//					}
//
//					return CallNextHookEx(mouse_hook, nCode, wParam, lParam);
//				}
//
//        static inline HHOOK keyboard_hook;
//        static inline HHOOK mouse_hook;
//
//        static inline std::function<void(uint16_t, fan::keyboard_state_t, bool action)> input_callback = [](uint16_t, fan::keyboard_state_t, bool action) {};
//        static inline std::function<void(const fan::vec2i& position)> mouse_move_callback = [](const fan::vec2i& position) {};
//
//      };
//			#endif
//    }
//  }