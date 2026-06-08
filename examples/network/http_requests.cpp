#include <coroutine>

import fan;

int main() {
  auto t = []() -> fan::event::task_t {
    {
      fan::json msg;
      msg["test"] = "hello";
      auto result = co_await fan::network::http::post(
        "https://httpbin.org/post",
        msg.dump(),
        { {"Content-Type", "application/json"} },
        { .verify_ssl = 0 }
      );

      if (result) {
        fan::print("POST Status:", result->status_code);
        fan::print("POST Body:", result->body);
      }
      else {
        fan::print("POST Error:", result.error());
      }
    }

    {
      auto get_result = co_await fan::network::http::get(
        "https://httpbin.org/get",
        { .verify_ssl = 0 }
      );

      if (get_result) {
        fan::print("GET Status:", get_result->status_code);
        fan::print("GET Body:", get_result->body);
      }
      else {
        fan::print("GET Error:", get_result.error());
      }
    }
  }();

  fan::event::loop();
}