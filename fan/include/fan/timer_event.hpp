#pragma once

#include <fan/time/time.hpp>

#include <set>
#include <thread>

namespace fan {
	namespace event {

		struct timer_event_t {

			timer_event_t() = default;

			struct single_time_t {

				single_time_t() = default;

				uint64_t time;
				std::function<void(uint64_t* time)> cb;
				fan::time::clock passed_time;
			};
	
			void push(uint64_t time, std::function<void(uint64_t* time)> cb) {

				fan::time::clock c;
				c.start();

				single_time_t st;
				st.time = time;
				st.cb = cb;
				st.passed_time = c;

				m_times.insert(st);

			}

			std::thread th;

			void threaded_update() {
				th = std::thread(std::bind(&timer_event_t::update, this));
				th.detach();
			}

			void update() {

				while (1) {

					if (m_times.empty()) {
						continue;
					}

					auto it = m_times.begin();

					int64_t wait_time = it->time - it->passed_time.elapsed();
					
					wait_time = fan::clamp(wait_time, (int64_t)0, wait_time);

					fan::delay(fan::time::nanoseconds(wait_time));

					single_time_t s;
					s.cb = it->cb;
					s.passed_time.start();
					it->cb(&s.time);

					m_times.erase(it);

					if (s.time != (uint64_t)-1) {
						m_times.insert(s);
					}

				}

			}

			struct compare_mt_t {

				compare_mt_t() = default;

					bool operator()(const single_time_t &l, 
													const single_time_t &r) const {
							return l.time < r.time;
					}
			};

			std::multiset<single_time_t, compare_mt_t> m_times;

		};

	}
}