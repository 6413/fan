#include <coroutine>
#include <expected>

import fan;

struct user_db_t {
  std::unordered_map<int, fan::json> users;
  int next_id = 1;
  
  fan::event::task_value_resume_t<std::expected<fan::json, fan::network::http_error_t>> get_user(int id) {
    auto it = users.find(id);
    if (it == users.end()) {
      co_return std::unexpected(fan::network::http_error_t{fan::network::http_error_t::not_found_error, "User not found"});
    }
    co_return it->second;
  }
  
  fan::event::task_value_resume_t<std::expected<fan::json, fan::network::http_error_t>> create_user(const fan::json& user_data) {
    if (!user_data.contains("name") || !user_data.contains("email")) {
      co_return std::unexpected(fan::network::http_error_t{fan::network::http_error_t::validation_error, "Missing required fields: name, email"});
    }
    
    fan::json new_user = user_data;
    new_user["id"] = next_id;
    new_user["created_at"] = "2024-01-01T00:00:00Z";
    
    users[next_id] = new_user;
    int created_id = next_id++;
    
    co_return users[created_id];
  }
  
  fan::event::task_value_resume_t<std::expected<fan::json, fan::network::http_error_t>> get_all_users() {
    fan::json result = fan::json::array();
    for (const auto& [id, user] : users) {
      result.push_back(user);
    }
    co_return result;
  }
  
  std::expected<std::vector<fan::json>, fan::network::http_error_t> search_users(const std::string& name) {
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

user_db_t db;

fan::event::awaitable_signal_t<> server_ready;

template<typename T>
void handle_response_error(const std::expected<T, fan::network::http_error_t>& result, fan::network::http_response_t& res) {
  if (!result) {
    res.error(result.error());
  }
}

fan::event::task_t run_server() {
  fan::network::http_server_t server;
  
  server.get("/users", [](const fan::network::http_request_t& req, fan::network::http_response_t& res) -> fan::event::task_t {
    auto users_result = co_await db.get_all_users();
    if (!users_result) {
      res.error(users_result.error());
      co_return;
    }
    res.json({{"users", *users_result}, {"count", users_result->size()}});
  });
  
  server.get("/users/{id}", [](const fan::network::http_request_t& req, fan::network::http_response_t& res) -> fan::event::task_t {
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
  
  server.post("/users", [](const fan::network::http_request_t& req, fan::network::http_response_t& res) -> fan::event::task_t {
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
  
  server.get("/health", [](const fan::network::http_request_t& req, fan::network::http_response_t& res) -> fan::event::task_t {
    res.json({
      {"status", "healthy"},
      {"timestamp", "2024-01-01T00:00:00Z"},
      {"users_count", db.users.size()}
    });
    co_return;
  });
  
  server.get("/search", [](const fan::network::http_request_t& req, fan::network::http_response_t& res) -> fan::event::task_t {
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
  server_ready.signal();
  co_await server.listen({"0.0.0.0", 8080});
}

void print_test_result(const std::string& test_name, bool success, const std::string& message = "") {
  fan::print_color(fan::colors::white, test_name);
  if (success) {
    fan::print_color(fan::colors::green, "   Success:", message.empty() ? "Test passed" : message);
  } else {
    fan::print_color(fan::colors::red, "   Error:", message);
  }
}

bool is_error_status(int status_code) {
  return status_code >= fan::network::http_status_t::bad_request;
}

fan::event::task_t run_client_tests() {
  fan::network::http_config_t config;
  config.verify_ssl = false;
  fan::network::async_http_client_t client("http://127.0.0.1:8080", config);
  fan::print_color(fan::colors::cyan, "\n--- HTTP Client Tests ---");

  auto health_result = co_await client.get("/health");
  print_test_result("1. Testing health endpoint...", health_result.has_value(), 
                   health_result ? "Health check passed" : health_result.error());

  auto users_result = co_await client.get("/users");
  print_test_result("2. Getting all users...", users_result.has_value(),
                   users_result ? "Retrieved users" : users_result.error());

  fan::json new_user = {{"name", "John Doe"}, {"email", "john@example.com"}, {"age", 30}};
  auto create_result = co_await client.post("/users", new_user);
  if (create_result) {
    print_test_result("3. Creating valid user...", true, "User created");
    fan::print_color(fan::colors::white, "   Response:", create_result->body);
  } else {
    print_test_result("3. Creating valid user...", false, create_result.error());
  }

  fan::json invalid_user = {{"name", "Jane Doe"}, {"age", 25}};
  auto invalid_create_result = co_await client.post("/users", invalid_user);
  bool validation_worked = !invalid_create_result || is_error_status(invalid_create_result->status_code);
  print_test_result("4. Creating invalid user (missing email)...", validation_worked,
                   validation_worked ? "Validation error caught correctly" : 
                   "Validation should have failed - got status: " + std::to_string(invalid_create_result->status_code));

  auto user_result = co_await client.get("/users/1");
  print_test_result("5. Getting user by ID...", user_result.has_value(),
                   user_result ? "Retrieved user by ID" : user_result.error());

  auto missing_user_result = co_await client.get("/users/999");
  bool not_found_handled = !missing_user_result || missing_user_result->status_code == 404;
  print_test_result("6. Getting non-existent user...", not_found_handled,
                   not_found_handled ? "404 error handled correctly" :
                   "Should have returned 404, got status: " + std::to_string(missing_user_result->status_code));

  auto search_result = co_await client.get("/search?name=Alice");
  print_test_result("7. Searching users...", search_result.has_value(),
                   search_result ? "Search completed" : search_result.error());

  auto invalid_search_result = co_await client.get("/search");
  bool param_validation_worked = !invalid_search_result || is_error_status(invalid_search_result->status_code);
  print_test_result("8. Search without name parameter...", param_validation_worked,
                   param_validation_worked ? "Missing parameter error handled" :
                   "Should have failed validation, got status: " + std::to_string(invalid_search_result->status_code));

  auto not_found_result = co_await client.get("/nonexistent");
  bool route_not_found = !not_found_result || not_found_result->status_code == 404;
  print_test_result("9. Testing 404 handling...", route_not_found,
                   route_not_found ? "404 handling works" : "404 handling failed");

  fan::print_color(fan::colors::cyan, "\n--- All Tests Completed ---");
}

fan::event::task_t run_demo() {
  db.users[1] = {{"id", 1}, {"name", "Alice Smith"}, {"email", "alice@example.com"}};
  db.users[2] = {{"id", 2}, {"name", "Bob Johnson"}, {"email", "bob@example.com"}};
  db.next_id = 3;
  
  fan::print_color(fan::colors::magenta, "Starting REST API Demo");
  fan::print_color(fan::colors::cyan, "Server starting on http://127.0.0.1:8080");
  
  auto server_task = run_server();
  co_await server_ready;
  co_await run_client_tests();
  
  fan::print_color(fan::colors::green, "\nDemo completed - all error handling tested!");
  fan::event::loop_stop();
}

int main() {
  auto demo_task = run_demo();
  fan::event::loop();
}