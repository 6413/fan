#include <pch.h>

// TEMP

#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/Module.h"

#include <string>
#include <condition_variable>
//
extern std::unique_ptr<llvm::LLVMContext> TheContext;
extern std::unique_ptr<llvm::Module> TheModule;
extern std::unique_ptr<llvm::DIBuilder> DBuilder;

///////////////
#include <llvm-ir/lexer.h>
////
#include <llvm-ir/ast.h>
#include <llvm-ir/parser.h>
#include <llvm-ir/codegen.h>
#include <llvm-ir/run.h>


#include <llvm-ir/library.h>

//===----------------------------------------------------------------------===//
// Main driver code.
//===----------------------------------------------------------------------===//
#include "llvm/MC/TargetRegistry.h"



struct pile_t {

  loco_t loco;
}pile;

static int add(int a, int b) {
  return a + b;
}

void init() {
  init_code();
}

void recompile() {
  recompile_code();
}

double test(double x) {
  fan::print(x);
  return 0;
}

int run() {

  return run_code();
}

void printDebugInfo(llvm::Module& M) {
  static std::string Str;
  llvm::raw_string_ostream OS(Str);
  M.print(OS, nullptr);
  OS.flush();
  add_task([] {
    fan::printcl(Str);
    Str.clear();
    });
}

void init_loco(TextEditor& editor, const char* file_name) {
  static int current_font = 2;

  static bool block_zoom[2]{};

  static float font_scale_factor = 1.0f;
  pile.loco.window.add_buttons_callback([&](const auto& d) {
    if (d.state != fan::mouse_state::press) {
      return;
    }
    if (pile.loco.window.key_pressed(fan::key_left_control) == false) {
      return;
    }

    auto& io = ImGui::GetIO();
    switch (d.button) {
    case fan::mouse_scroll_up: {
      if (block_zoom[0] == true) {
        break;
      }
      font_scale_factor *= 1.1;
      block_zoom[1] = false;
      break;
    }
    case fan::mouse_scroll_down: {
      if (block_zoom[1] == true) {
        break;
      }
      font_scale_factor *= 0.9;
      block_zoom[0] = false;
      break;
    }
    }

    //ImFont* selected_font = nullptr;
    //for (int i = 0; i < std::size(loco.fonts); ++i) {
    //  if (new_font_size <= font_size * (1 << i) / 2) {
    //    selected_font = loco.fonts[i];
    //    break;
    //  }
    //}

    if (font_scale_factor > 1.5) {
      current_font++;
      if (current_font > std::size(pile.loco.fonts) - 1) {
        current_font = std::size(pile.loco.fonts) - 1;
        block_zoom[0] = true;
      }
      else {
        io.FontDefault = pile.loco.fonts[current_font];
        font_scale_factor = 1;
      }
    }

    if (font_scale_factor < 0.5) {
      current_font--;
      if (current_font < 0) {
        current_font = 0;
        block_zoom[1] = true;
      }
      else {
        io.FontDefault = pile.loco.fonts[current_font];
        font_scale_factor = 1;
      }
    }


    // Set the window font scale
    io.FontGlobalScale = font_scale_factor;
    // Set the selected font for ImGui
    //io.FontDefault = selected_font;
    return;
    });

  TextEditor::LanguageDefinition lang = TextEditor::LanguageDefinition::CPlusPlus();
  // set your own known preprocessor symbols...
  static const char* ppnames[] = { "NULL" };
  // ... and their corresponding values
  static const char* ppvalues[] = {
    "#define NULL ((void*)0)",
  };

  for (int i = 0; i < sizeof(ppnames) / sizeof(ppnames[0]); ++i)
  {
    TextEditor::Identifier id;
    id.mDeclaration = ppvalues[i];
    lang.mPreprocIdentifiers.insert(std::make_pair(std::string(ppnames[i]), id));
  }

  //for (auto& i : commands.func_table) {
  //  TextEditor::Identifier id;
  //  id.mDeclaration = i.second.description;
  //  lang.mIdentifiers.insert(std::make_pair(i.first, id));
  //}

  editor.SetLanguageDefinition(lang);
  //

  auto palette = editor.GetPalette();

  palette[(int)TextEditor::PaletteIndex::Background] = 0xff202020;
  editor.SetPalette(palette);
  editor.SetPalette(editor.GetRetroBluePalette());
  editor.SetTabSize(2);
  editor.SetShowWhitespaces(false);

  fan::string str;
  fan::io::file::read(
    file_name,
    &str
  );

  editor.SetText(str);
}

std::mutex g_mutex;
std::condition_variable g_cv;
bool ready = false;
bool processed = false;

void t0() {
  std::unique_lock lk(g_mutex);
  g_cv.wait(lk, [] { return ready; });

  init();
  recompile();
  run();

  processed = true;
  ready = false;
  lk.unlock();
  t0();
}
//


struct a_t {
  virtual void f() {
    // print("A");
  }
};

struct b_t : a_t {
  double data;
  virtual void f() {
    //print("B");
  }
};

struct c_t : a_t {
  int data;
  virtual void f() {
    //print("C");
  }
};

int main() {
  //
  std::jthread t(t0);
  t.detach();

  pile.loco.console.commands.add("clear_shapes", [](const fan::commands_t::arg_t& args) {
    shapes.clear();
    }).description = "";

  /*-------------------------------------------------*/

  TextEditor editor, input;
  auto file_name = "test.fpp";

  init_loco(editor, file_name);

  /*-------------------------------------------------*/

  pile.loco.input_action.add_keycombo({ fan::key_left_control, fan::key_s }, "save_file");
  pile.loco.input_action.add_keycombo({ fan::key_f5 }, "compile_and_run");

  auto compile_and_run = [&editor] {
    code_input = editor.GetText();
    code_input.push_back(EOF);

    {
      std::lock_guard lk(g_mutex);
      ready = true;
    }
    g_cv.notify_one();
    };

  uint32_t task_id = 0;

  pile.loco.loop([&] {
    ImGui::Begin("window");
    ImGui::SameLine();



    if (ImGui::Button("compile & run")) {
      compile_and_run();
    }
    if (pile.loco.input_action.is_active("compile_and_run")) {
      compile_and_run();
    }
    editor.Render("editor");
    ImGui::End();
    if (pile.loco.input_action.is_active("save_file")) {
      std::string str = editor.GetText();
      fan::io::file::write(file_name, str.substr(0, std::max(0ull, str.size() - 1)), std::ios_base::binary);
    }
    //
    needs_frame_skip = false;
    if (processed) {
    g_step:
      task_queue[task_id++]();
      if (task_id == task_queue.size()) {
        task_id = 0;
        task_queue.clear();
        processed = false;
        needs_frame_skip = false;
      }
      if (task_queue.size() && needs_frame_skip == false) {
        goto g_step;
      }
    }

    });
}