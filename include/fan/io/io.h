#pragma once

#include _FAN_PATH(types/types.h)
#include _FAN_PATH(types/utf_string.h)

#ifdef fan_platform_windows
	#define WIN32_LEAN_AND_MEAN
	#include <Windows.h)
	#undef min
	#undef max
#endif

namespace fan {
	namespace io {

		static std::wstring read_console_utf8(bool ignore_endline = true) {

			uint32_t prev_cp = GetConsoleCP();
			SetConsoleOutputCP(CP_UTF8);

			wchar_t buffer[0x400];
			unsigned long read = 1;
			HANDLE handle = GetStdHandle(STD_INPUT_HANDLE);
	
			if(handle == INVALID_HANDLE_VALUE) {
				fan::throw_error("invalid handle value " + std::to_string(GetLastError()));
			}

			fan::utf16_string str;
			
			// cant read more than std::size(buffer)
			if(!ReadConsoleW(handle, buffer, std::size(buffer), &read, NULL)) {
				fan::throw_error("failed to read from console " + std::to_string(GetLastError()));
			}
			str.insert(str.end(), buffer, buffer + read);

			if (ignore_endline) {
				str.pop_back();
				str.pop_back();
				read--;
			}

			read--;

			SetConsoleCP(prev_cp);

			return str;
		}

		template <typename ...Args>
		constexpr void print_utf16(const Args&... args) {
			static bool x = 0;
			if (!x) {
				SetConsoleOutputCP(CP_UTF8);
				std::wcout.imbue(std::locale("en_US.UTF-8"));
				x = true;
			}
			fan::wprint(args...);
		}

	}
}