#pragma once

#include "time.h"

#include <set>
#include <map>
#include <functional>

namespace fan {

	struct ev_timer_t {

    struct cb_data_t;

    struct id_t {
      id_t() = default;
      id_t(uint32_t i) : iid(i) {}
      operator uint32_t() {
        return iid;
      }
      bool is_valid() {
        return *this != (uint32_t)-1;
      }
      void invalidate() {
        *this = -1;
      }
      uint32_t iid = -1;
    };

    struct timer_t {
      timer_t() = default;
      uint64_t ns;
      uint64_t time_left;
      bool repeat;
      std::function<void()> cb = [] {};
      id_t id;
      bool operator<(const timer_t& r) const {
        return time_left < r.time_left;
      }
    };

    using nr_t = std::multiset<timer_t>::iterator;

    std::vector<nr_t> iid_list;

		ev_timer_t() {
			m_current_time = fan::time::clock::now();
		}

    struct cb_data_t {
      ev_timer_t* ev_timer;
      timer_t timer;
    };

    id_t start(uint64_t ns, auto lambda) {
      return impl_start(ns, lambda, true);
		}
    id_t start_single(uint64_t ns, auto lambda) {
      return impl_start(ns, lambda, false);
    }
    id_t impl_start(uint64_t ns, auto lambda, bool repeat) {
      timer_t timer;
      timer.cb = lambda;
      timer.repeat = repeat;
      timer.ns = ns;
      timer.time_left = ns + m_current_time;
      iid_list.resize(iid_list.size() + 1);
      timer.id = iid_list.size() - 1;
      nr_t nr = time_list.insert(timer);
      iid_list.back() = nr;
      return timer.id;
    }
    bool is_valid(id_t id) {
      return id.is_valid();
    }
		void stop(id_t& id) {
      if (!is_valid(id)) {
        return;
      }
			time_list.erase(iid_list[id]);
      iid_list.erase(iid_list.begin() + id);
      id.invalidate();
		}

		void process() {
			m_current_time = fan::time::clock::now();
			for (auto it = time_list.begin(); it != time_list.end(); ++it) {
				if (it->time_left > m_current_time) {
					break;
				}
        it->cb();
        if (it->repeat == false) {
          it = time_list.erase(it);
        }
        else {
          timer_t timer;
          timer = *it;
          timer.time_left = timer.ns + m_current_time;
          auto& node = iid_list[timer.id];
          time_list.erase(it);
          auto x = time_list.insert(timer);
          node = x;
        }

        if (it == time_list.end()) {
          break;
        }
			}
		}

		uint64_t m_current_time;
		std::multiset<timer_t> time_list;
	};
}