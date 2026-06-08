#include <coroutine>
#include <expected>

import fan;

struct user_db_t {
  std::unordered_map<int, fan::json> users;
  int next_id = 1;

  fan::event::task_value_resume_t<std::expected<fan::json, fan::network::http::error_t>> get_user(int id) {
    auto it = users.find(id);
    if (it == users.end()) {
      co_return std::unexpected(fan::network::http::error_t {fan::network::http::error_t::not_found_error, "User not found"});
    }
    co_return it->second;
  }

  fan::event::task_value_resume_t<std::expected<fan::json, fan::network::http::error_t>> create_user(const fan::json& user_data) {
    if (!user_data.contains("name") || !user_data.contains("email")) {
      co_return std::unexpected(fan::network::http::error_t {fan::network::http::error_t::validation_error, "Missing required fields: name, email"});
    }

    fan::json new_user = user_data;
    new_user["id"] = next_id;
    new_user["created_at"] = "2024-01-01T00:00:00Z";

    users[next_id] = new_user;
    int created_id = next_id++;

    co_return users[created_id];
  }

  fan::event::task_value_resume_t<std::expected<fan::json, fan::network::http::error_t>> get_all_users() {
    fan::json result = fan::json::array();
    for (const auto& [id, user] : users) {
      result.push_back(user);
    }
    co_return result;
  }

  std::expected<std::vector<fan::json>, fan::network::http::error_t> search_users(const std::string& name) {
    std::vector<fan::json> results;
    for (const auto& [id, user] : users) {
      if (user.contains("name")) {
        std::string user_name = user["name"];
        if (user_name.find(name) != std::string::npos) {
          results.push_back(user);
        }
      }
    }
    return results;
  }
};

fan::event::task_t run_server() {
  user_db_t db;

  fan::network::http::server_t server;

  server.get("/users", [&db](const fan::network::http::request_t& req, fan::network::http::response_t& res) -> fan::event::task_t {
    auto users_result = co_await db.get_all_users();
    if (!users_result) {
      res.error(users_result.error());
      co_return;
    }
    res.json({{"users", *users_result}, {"count", users_result->size()}});
  });

  server.get("/users/{id}", [&db](const fan::network::http::request_t& req, fan::network::http::response_t& res) -> fan::event::task_t {
    auto id_result = req.param<int>("id");
    if (!id_result) {
      res.error(id_result.error());
      co_return;
    }

    auto user_result = co_await db.get_user(*id_result);
    if (!user_result) {
      res.error(user_result.error());
      co_return;
    }
    res.json(*user_result);
  });

  server.post("/users", [&db](const fan::network::http::request_t& req, fan::network::http::response_t& res) -> fan::event::task_t {
    auto user_data_result = req.json();
    if (!user_data_result) {
      res.error(user_data_result.error());
      co_return;
    }

    auto created_user_result = co_await db.create_user(*user_data_result);
    if (!created_user_result) {
      res.error(created_user_result.error());
      co_return;
    }
    res.created(*created_user_result);
  });

  server.get("/health", [&db](const fan::network::http::request_t& req, fan::network::http::response_t& res) -> fan::event::task_t {
    res.json({
      {"status", "healthy"},
      {"timestamp", "2024-01-01T00:00:00Z"},
      {"users_count", db.users.size()}
      });
    co_return;
  });

  server.get("/search", [&db](const fan::network::http::request_t& req, fan::network::http::response_t& res) -> fan::event::task_t {
    auto name_result = req.query_param<std::string>("name");
    if (!name_result) {
      res.error(name_result.error());
      co_return;
    }

    auto search_result = db.search_users(*name_result);
    if (!search_result) {
      res.error(search_result.error());
      co_return;
    }

    fan::json results = fan::json::array();
    for (const auto& user : *search_result) {
      results.push_back(user);
    }
    res.json({{"results", results}, {"query", *name_result}});
  });

  fan::print_color(fan::colors::cyan, "Server listening on port 8080");
  co_await server.listen({"0.0.0.0", 8080});
}

int main() {
  auto server_task = run_server();
  fan::event::loop();
}