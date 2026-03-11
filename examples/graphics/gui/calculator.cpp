#include <string>
import fan;
using namespace fan::graphics;
struct calc_t {
  enum class state_t { entering, result } state = state_t::entering;
  calc_t() : e([&] { handle_input(); draw(); }) {}
  void dispatch(char c) {
    if (c >= '0' && c <= '9') display += c;
    else if (c == '.') on_dot();
    else if (c == '+' || c == '-' || c == '*' || c == '/') on_op(c);
    else if (c == '=') on_equal();
    else if (c == 'C') on_clear();
  }
  void handle_input() {
    if (uint32_t c = fan::window::get_char_pressed()) {
      if (state == state_t::result && (c >= '0' && c <= '9' || c == '.' || c == '+' || c == '-' || c == '*' || c == '/'))
        soft_reset();
      dispatch((char)c);
    }
    if (fan::window::is_key_clicked(fan::key_enter))     on_equal();
    if (fan::window::is_key_clicked(fan::key_backspace)) on_clear();
  }
  void draw() {
    gui::begin("Calculator");
    gui::text_sized(formula.empty() ? " " : formula, 28);
    gui::text_sized(display.empty() ? "0" : display, 48);
    gui::separator();
    gui::button_grid({"7","8","9","4","5","6","1","2","3","0","."}, 3, {60,60}, [&](int, const char* l) {dispatch(*l); }, 48);
    gui::button_row({"+","-","*","/","=","C"}, {60,60}, 48, [&](const char* l) {dispatch(*l); });
    gui::end();
  }
  void soft_reset() { display.clear(); formula.clear(); state = state_t::entering; }
  void on_dot() { if (display.find('.') == std::string::npos) display += display.empty() ? "0." : "."; }
  void on_op(char o) {
    if (!display.empty()) { a = std::stod(display); formula = display + " " + o; display.clear(); }
    else if (!formula.empty()) formula.back() = o;
    pending = o; state = state_t::entering;
  }
  void on_equal() {
    if (!pending || display.empty()) return;
    f64_t b = std::stod(display);
    if (pending == '/' && b == 0) { display = "ERR"; formula.clear(); pending = 0; state = state_t::result; return; }
    f64_t r = pending == '+' ? a + b : pending == '-' ? a - b : pending == '*' ? a * b : a / b;
    formula += " " + display + " =";
    display = fan::format_number(r); pending = 0; state = state_t::result;
  }
  void on_clear() { display.clear(); formula.clear(); a = 0; pending = 0; state = state_t::entering; }
  f64_t a = 0;
  char pending = 0;
  std::string display, formula;
  engine_t e;
};
int main() { 
  calc_t{};
}