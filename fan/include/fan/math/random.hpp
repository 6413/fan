#pragma once

#include <fan/types/vector.hpp>
#include <fan/types/color.hpp>
#include <fan/types/utf_string.hpp>

#include <random>

namespace fan {

	namespace random {

		template <typename first, typename second>
		auto value(first min, second max) {
			static std::random_device device;
			static std::mt19937_64 random(device());
			std::uniform_int_distribution<first> distance(min, max);
			return distance(random);
		}

		template <typename type_t>
		struct fast_rand_t {

			fast_rand_t(type_t min, type_t max) : 
				m_random(m_device()),
				m_distance(min, max)
			{ }

			type_t get() {
				return m_distance(m_random);
			}

			void set_min_max(type_t min, type_t max) {
				m_distance.param(std::uniform_int_distribution<type_t>::param_type(min, max));
			}

		protected:

			std::random_device m_device;
			std::mt19937_64 m_random;

			std::uniform_int_distribution<type_t> m_distance;

		};

		template <typename first, typename second>
		auto fast_value(first min, second max) {
			static std::random_device device;
			static std::mt19937_64 random(device());
			static first g_min;
			static second g_max;

			if (min != g_min) {
				g_min = min;
			}
			if (g_max != max) {
				g_max = max;
			}


			static std::uniform_int_distribution<first> distance(min, max);
			return distance(random);
		}

		static std::string string(uint32_t len) {
			std::string str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
			std::string newstr;
			int pos;
			while(newstr.size() != len) {
				pos = fan::random::value(0, str.size() - 1);
				newstr += str.substr(pos, 1);
			}
			return newstr;
		}

		static fan::utf16_string utf_string(uint32_t len) {
			fan::utf16_string str = L"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
			fan::utf16_string newstr;
			int pos;
			while(newstr.size() != len) {
				pos = fan::random::value(0, str.size() - 1);
				newstr += str.substr(pos, 1);
			}
			return newstr;
		}

		template <typename T = fan::vec2>
		static T vector(f_t min, f_t max) {
			if constexpr (std::is_same_v<T, fan::vec2>) {
				return T(fan::random::value<int64_t, int64_t>(min, max), fan::random::value<int64_t, int64_t>(min, max));
			}
			else {
				return T(fan::random::value<int64_t, int64_t>(min, max), fan::random::value<int64_t, int64_t>(min, max), fan::random::value<int64_t, int64_t>(min, max));
			}
		}

		static fan::color color() {
			return fan::color::rgb(std::rand() % 255, std::rand() % 255, std::rand() % 255, 255);
		}

		struct percent_output_t {
			f32_t percent;
			uint32_t output;
		};

		// percent 0-1
		static uint32_t get_output_with_percent(const std::vector<percent_output_t>& po) {

			for (int i = 0; i < po.size(); i++) {
				if (!(1.0 / fan::random::value<uint32_t, uint32_t>(0, ~0) < 1.0 / (po[i].percent * (uint32_t)~0))) {
					return po[i].output;
				}
			}

			return -1;
		}
	}
}