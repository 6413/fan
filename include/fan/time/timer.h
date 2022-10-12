#pragma once

#include "time.h"

#include <set>

namespace fan {
	struct ev_timer_t {
		ev_timer_t() {
			m_current_time = fan::time::clock::now();
		}

		struct timer_t;

		struct cb_data_t {
			ev_timer_t* ev_timer;
			timer_t* timer;
		};

		struct timer_t {
			bool operator<(const timer_t& r) const {
				return time_left < r.time_left;
			}

			timer_t(const fan::function_t<void(const cb_data_t&)>& c) {
				cb = c;
			}
			uint64_t time_left;
			fan::function_t<void(const cb_data_t&)> cb;
		};

		void start(timer_t* timer, uint64_t time_left) {
			time_left += m_current_time;
			timer->time_left = time_left;
			time_list.insert(timer);
		}
		void stop(timer_t* timer) {
			time_list.erase(timer);
		}

		void process() {
			m_current_time = fan::time::clock::now();
			for (auto it = time_list.begin(); it != time_list.end(); ++it) {
				if ((*it)->time_left > m_current_time) {
					break;
				}
				timer_t* t = (*it);
				it = time_list.erase(it);
				cb_data_t cb_data;
				cb_data.ev_timer = this;
				cb_data.timer = t;
				t->cb(cb_data);
			}
		}

		uint64_t m_current_time;
		std::multiset<timer_t*> time_list;
	};
}