#include <coroutine>

import fan;

void print_test_result(const std::string& test_name, bool success, const std::string& message = "") {
  fan::print_color(fan::colors::white, test_name);
  if (success) {
    fan::print_color(fan::colors::green, "   Success:", message.empty() ? "Test passed" : message);
  }
  else {
    fan::print_color(fan::colors::red, "   Error:", message);
  }
}

bool is_error_status(int status_code) {
  return status_code >= fan::network::http::status_t::bad_request;
}

fan::event::task_t run_client_tests() {
  fan::network::http::config_t config;
  config.verify_ssl = false;
  fan::network::http::client_t client("http://127.0.0.1:8080", config);

  fan::print_color(fan::colors::cyan, "--- HTTP Client Tests ---");

  auto health_result = co_await client.get("/health");
  print_test_result("1. Testing health endpoint...", health_result.has_value(),
    health_result ? "Health check passed" : health_result.error());
  if (!health_result) {
    fan::print_color(fan::colors::red, "Server unreachable, stopping tests");
    co_return;
  }

  auto users_result = co_await client.get("/users");
  print_test_result("2. Getting all users...", users_result.has_value(),
    users_result ? "Retrieved users" : users_result.error());

  fan::json new_user = {{"name", "John Doe"}, {"email", "john@example.com"}, {"age", 30}};
  auto create_result = co_await client.post("/users", new_user);
  if (create_result) {
    print_test_result("3. Creating valid user...", true, "User created");
    fan::print_color(fan::colors::white, "   Response:", create_result->body);
  }
  else {
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

  fan::print_color(fan::colors::cyan, "--- All Tests Completed ---");
}

int main() {
  auto client_task = run_client_tests();
  fan::event::loop();
}