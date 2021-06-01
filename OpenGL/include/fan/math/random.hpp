#pragma once

#include <fan/types/vector.hpp>
#include <fan/types/color.hpp>

#include <random>

namespace fan {

	namespace random {

		template <typename first, typename second>
		auto random(first min, second max) {
			static std::random_device device;
			static std::mt19937_64 random(device());
			std::uniform_int_distribution<first> distance(min, max);
			return distance(random);
		}

		static std::string string(uint32_t len) {
			std::string str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
			std::string newstr;
			int pos;
			while(newstr.size() != len) {
				pos = fan::random::random(0, str.size() - 1);
				newstr += str.substr(pos, 1);
			}
			return newstr;
		}

		template <typename T = fan::vec2>
		static T vector(f_t min, f_t max) {
			if constexpr (std::is_same_v<T, fan::vec2>) {
				return T(fan::random::random<int64_t, int64_t>(min, max), fan::random::random<int64_t, int64_t>(min, max));
			}
			else {
				return T(fan::random::random<int64_t, int64_t>(min, max), fan::random::random<int64_t, int64_t>(min, max), fan::random::random<int64_t, int64_t>(min, max));
			}
		}

		static fan::color color() {
			return fan::color::rgb(std::rand() % 255, std::rand() % 255, std::rand() % 255, 255);
		}

	}

}