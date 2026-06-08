import fan;
using namespace fan::graphics;

struct calculator_t {
  void dispatch(char c) {
    if (c >= '0' && c <= '9') { display += c; } 
    else if (c == '.') { on_dot(); }
    else if (fan::is_any_of(c, "+-*/")) { on_op(c); }
    else if (c == '=') { on_equal(); } 
    else if (c == 'C') { on_clear(); }
  }
  void handle_input() {
    if (char c = fan::window::get_char_pressed()) {
      if (state == state_t::result && fan::is_any_of(c, "0123456789.+-*/")) { soft_reset(); }
      dispatch(c);
    }
    if (fan::window::is_key_clicked(fan::key_enter)) { on_equal(); }
    if (fan::window::is_key_clicked(fan::key_backspace)) { on_clear(); }
  }
  void draw() {
    if (auto w = gui::window("Calculator")) {
      gui::text_sized(formula, 28); gui::text_sized(display.empty() ? "0" : display, 48);
      gui::separator();
      if (auto cl = gui::button_grid({"7","8","9","4","5","6","1","2","3","0","."}, 3, {60, 60}, 48)) { dispatch((*cl)[0]); }
      if (auto cl = gui::button_grid({"+","-","*","/","C","="}, 4, {60, 60}, 48)) { dispatch((*cl)[0]); }
    }
  }
  void soft_reset() { display = formula = ""; state = state_t::entering; }
  void on_dot() { if (!display.contains('.')) { display += display.empty() ? "0." : "."; } }
  void on_op(char o) {
    if (!display.empty()) { pending_lhs = display; formula = display + " " + o; display = ""; }
    else if (!formula.empty()) { formula.back() = o; }
    pending = o; state = state_t::entering;
  }
  void on_equal() {
    if (!pending || display.empty()) { return; }
    auto r = fan::math::compute_from_strings(pending_lhs, pending, display);
    if (!r) { display = "ERR"; formula = ""; pending = 0; state = state_t::result; return; }
    formula += " " + display + " ="; display = fan::format_number(*r); pending = 0; state = state_t::result;
  }
  void on_clear() { display = formula = pending_lhs = ""; pending = 0; state = state_t::entering; }

  enum class state_t { entering, result } state = state_t::entering;
  char pending = 0;
  std::string display, formula, pending_lhs;
  engine_t e{[this]() { handle_input(); draw(); }};
};

int main() {
  calculator_t{};
}