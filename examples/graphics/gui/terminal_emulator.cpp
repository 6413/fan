#include <pty.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/wait.h>

import fan;
using namespace fan::graphics;

struct terminal_t {
  terminal_t() {
    // spawn bash in a pty
    pid = forkpty(&master_fd, nullptr, nullptr, nullptr);
    if (pid == 0) {
      // child: exec bash
      setenv("TERM", "xterm-256color", 1);
      execl("/bin/bash", "bash", nullptr);
      _exit(1);
    }
    // parent: make master_fd non-blocking
    fcntl(master_fd, F_SETFL, O_NONBLOCK);
  }

  ~terminal_t() {
    close(master_fd);
    kill(pid, SIGKILL);
    waitpid(pid, nullptr, 0);
  }

  void write_input(const std::string& s) {
    ::write(master_fd, s.c_str(), s.size());
  }

  void read_output() {
    char buf[4096];
    ssize_t n;
    while ((n = read(master_fd, buf, sizeof(buf))) > 0) {
      output += std::string(buf, n);
      // keep last N lines only
      while (std::count(output.begin(), output.end(), '\n') > 200) {
        output.erase(0, output.find('\n') + 1);
      }
    }
  }

  void render() {
    read_output();

    gui::set_next_window_pos({0, 0});
    gui::set_next_window_size(fan::vec2(engine.window.get_size()));
    gui::set_next_window_bg_alpha(1.f);
    gui::begin("terminal", nullptr,
      gui::window_flags_no_title_bar |
      gui::window_flags_no_resize |
      gui::window_flags_no_move |
      gui::window_flags_no_saved_settings
    );

    // scrollable output
    gui::begin_child("output", {0, -30}, false);
    gui::text_unformatted(output.c_str());
    if (gui::get_scroll_y() >= gui::get_scroll_max_y()) {
      gui::set_scroll_here_y(1.f);
    }
    gui::end_child();

    // input line
    gui::separator();
    bool enter = false;
    gui::set_next_item_width(-1);
    if (gui::input_text("##input", input_buf, sizeof(input_buf),
          gui::input_text_flags_enter_returns_true)) {
      enter = true;
    }
    gui::set_keyboard_focus_here(-1); // keep focus on input

    if (enter) {
      std::string cmd = input_buf;
      cmd += "\n";
      write_input(cmd);
      input_buf[0] = '\0';
    }

    gui::end();
  }

  int master_fd = -1;
  pid_t pid = -1;
  std::string output;
  char input_buf[1024] = {};

  engine_t engine;
};

int main() {
  terminal_t terminal;
  terminal.engine.set_clear_color(fan::colors::black);
  terminal.engine.loop([&] {
    terminal.render();
  });
}