#include _FAN_PATH(graphics/gui/fgm/fgm.h)

struct fsm_t {

  static constexpr uint32_t version_001 = 1;

  static constexpr auto modify_str = "Modify";
  static constexpr auto stages_str = "Stages";

  static constexpr auto max_depth = 0xff;
  static constexpr int max_path_input = 40;

  static constexpr int max_id_input = 10;
  static constexpr fan::vec2 default_button_size{100, 30};

  // fan stage maker
  fan::string file_name = "file.fsm";

  static constexpr const char* stage_compile_folder_name = "stages_compile";
  static constexpr const char* stage_runtime_folder_name = "stages_runtime";
  static auto get_file_fullpath(const fan::string& stage_name) {
    return fan::string(stage_compile_folder_name) + "/" +
      stage_name + ".h";
  };

  static auto get_file_fullpath_runtime(const fan::string& stage_name) {
    return fan::string(stage_runtime_folder_name) + "/" +
      stage_name + ".fgm";
  };

  static constexpr const char* stage_instance_tempalte_str = R"(void open(void* sod) {
  
}

void close() {
		
}

void update(){
	
}
)";

  static constexpr f32_t gui_size = 0.05;

  auto write_stage() {
    stage_h.write(&stage_h_str);
  };

  auto append_stage_to_file(const fan::string& stage_name) {
    if (stage_h_str.find(stage_name) != fan::string::npos) {
      return;
    }

    static constexpr const char find_end_str[]("\n};");
    auto struct_stage_end = stage_h_str.find_last_of(find_end_str);

    if (struct_stage_end == fan::string::npos) {
      fan::throw_error("corrupted stage.h");
    }

    struct_stage_end -= sizeof(find_end_str) - 2;

    auto append_struct = fmt::format(R"(
  lstd_defstruct({0}_t)
    #include _FAN_PATH(graphics/gui/stage_maker/preset.h)

    static constexpr auto stage_name = "{0}";
    #include _PATH_QUOTE(stage_loader_path/{1})
  }};)",
    stage_name,
    get_file_fullpath(stage_name));
    stage_h_str.insert(struct_stage_end, append_struct);
  };

  auto write_stage_instance(const fan::string& stage_name) {
    auto file_name = get_file_fullpath(stage_name);
    fan::io::file::write(file_name, stage_instance_tempalte_str, std::ios_base::binary);
    if (!fan::io::file::exists(get_file_fullpath_runtime(stage_name))) {
      std::ofstream _fgm(get_file_fullpath_runtime(stage_name), std::ios_base::out | std::ios::binary);
      _fgm.write((const char*)&current_version, sizeof(current_version));
    }
    append_stage_to_file(stage_name);
    write_stage();
  };

  void create_stage_file(const fan::string& stage_name) {
    write_stage_instance(stage_name);
  };

  void create_stage(const fan::string& stage_name) {
    auto it = stage_list.NewNodeLast();
    auto& node = stage_list[it];
    node = new stage_t(this, stage_name);
    create_stage_file(stage_name);
  }

  void create_stage() {
    auto it = stage_list.NewNodeLast();
    auto& node = stage_list[it];
    node = new stage_t(this, "");
    while (stage_exists("stage" + std::to_string(stage_counter))) { ++stage_counter; }
    node->stage_name = "stage" + std::to_string(stage_counter);
    create_stage_file(node->stage_name);
  }

  static constexpr uint32_t current_version = version_001;

  fan::string current_stage;
  uint32_t stage_counter = 0;
  bool render_fsm = true;

  // might need storing of stage name
  struct stage_t : fan::graphics::imgui_element_t {
    stage_t(fsm_t* fsm, const fan::string& stage_name_)
      : fan::graphics::imgui_element_t([fsm, this] {
      if (!fsm->render_fsm) {
        return;
      }
      ImGui::Begin(stages_str, nullptr, ImGuiWindowFlags_DockNodeHost);
      if (ImGui::Button(stage_name.c_str())) {
        fsm->current_stage = stage_name;
      }
      ImGui::End();
      })
    {
      stage_name = stage_name_;
    }

    fan::string stage_name;
  };

  fan::graphics::imgui_element_t main_view =
    fan::graphics::imgui_element_t(
    [&] {

      if (!render_fsm) {
        if (fgm == nullptr) {
          fgm = new fgm_t();
          fgm->open("TexturePack");
          fgm->file_name = fan::string(stage_runtime_folder_name) + "/" + current_stage + ".fgm";
          fgm->fin(fgm->file_name);
          fgm->close_cb = [this] {
            fgm->close();
            delete fgm;
            fgm = nullptr;
            render_fsm = true;
            };
        }
        return;
      }

      if (ImGui::Begin(modify_str, nullptr, ImGuiWindowFlags_DockNodeHost)) {
        if (ImGui::Button("Create new stage")) {

          create_stage();
        }
        if (ImGui::Button("Edit GUI")) {
          if (!current_stage.empty()) {
            render_fsm = false;
          }
          // todo hardcoded
        }
        if (!current_stage.empty()) {
          ImGui::Text("Rename");
          ImGui::SameLine();
          fan::string new_name;
          new_name.resize(20);
          if (ImGui::InputText("##hidden_label0" "id", new_name.data(), new_name.size())) {
            if (ImGui::IsItemDeactivatedAfterEdit()) {
              new_name = new_name.substr(0, std::strlen(new_name.c_str()));
              fan::string old_name = current_stage;
              if (rename(current_stage, new_name)) {
                auto it = stage_list.GetNodeFirst();
                while (it != stage_list.dst) {
                  if (stage_list[it]->stage_name == old_name) {
                    stage_list[it]->stage_name = new_name;
                    break;
                  }
                  it = it.Next(&stage_list);
                }
                
              }
            }
          }
        }
      }

      ImGui::End();

      fan::vec2 editor_size;

      if (ImGui::Begin(stages_str, nullptr, ImGuiWindowFlags_DockNodeHost)) {
        fan::vec2 window_size = gloco->window.get_size();
        fan::vec2 viewport_size = ImGui::GetWindowSize();
        fan::vec2 ratio = viewport_size / viewport_size.max();
        gloco->default_camera->camera.set_ortho(
          fan::vec2(0, viewport_size.x),
          fan::vec2(0, viewport_size.y)
        );
        gloco->default_camera->viewport.set(ImGui::GetWindowPos(), viewport_size, window_size);
        editor_size = ImGui::GetContentRegionAvail();
      }

      ImGui::End();
  });
  /*
  header 4 byte
  */
  void fout(const fan::string& filename) {

    fan::string ostr;
    ostr.append((char*)&current_version, sizeof(current_version));


    fan::io::file::write(filename, ostr, std::ios_base::binary);
    fan::print("file saved to:" + filename);
  }

  void fin(const fan::string& filename) {
    fan::string in;
    fan::io::file::read(filename, &in);
    uint64_t off = 0;
    uint32_t version = fan::read_data<uint32_t>(in, off);
    if (version != current_version) {
      fan::print_format("invalid file version, file:{}, current:{}", version, current_version);
      return;
    }

  }

  bool stage_exists(const fan::string& stage) {
    auto it = stage_list.GetNodeFirst();
    while (it != stage_list.dst) {
      if (stage_list[it]->stage_name == stage) {
        return true;
      }

      it = it.Next(&stage_list);
    }
    return false;
  }

  bool rename(const fan::string& old_name, const fan::string& new_name) {
    if (!stage_exists(new_name)) {
      do {
        #if defined(fan_platform_windows)
        static std::regex windows_filename_regex(R"([^\\/:*?"<>|\r\n]+(?!\\)$)");
        #elif defined(fan_platform_unix)
        static std::regex unix_filename_regex("[^/]+$");
        #endif

        if (new_name.empty() ||
          #if defined(fan_platform_windows)
            !std::regex_match(new_name, windows_filename_regex)
          #elif defined(fan_platform_unix)
            !std::regex_match(new_name, unix_filename_regex)
          #endif
          ) {
          fan::print("invalid stage name");
          break;
        }

        if (fan::io::file::rename(get_file_fullpath(old_name), get_file_fullpath(new_name)) ||
            fan::io::file::rename(get_file_fullpath_runtime(old_name), get_file_fullpath_runtime(new_name))) {

          fan::print_format("failed to rename file from:{} - to:{}", old_name, new_name);
          break;
        }
        if (old_name == new_name) {
          break;
        }
        std::regex rg(fmt::format(R"(\b{0}(_t)?\b)", old_name));
        stage_h_str = std::regex_replace(stage_h_str, rg, new_name + R"($1)");
        write_stage();
        current_stage = new_name;
        return 1;
      } while (0);
    }
    else {
      fan::print("stage name already exists");
    }
    return 0;
  }

  #define BLL_set_StoreFormat 1
  //#define BLL_set_CPP_CopyAtPointerChange
  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_AreWeInsideStruct 1
  #include _FAN_PATH(fan_bll_preset.h)
  #define BLL_set_prefix stage_list
  #define BLL_set_type_node uint32_t
  #define BLL_set_NodeDataType stage_t*
  #define BLL_set_Link 1
  #include _FAN_PATH(BLL/BLL.h)

  fsm_t() {
    fan::io::create_directory("stages_compile");
    fan::io::create_directory("stages_runtime");

    auto stage_path = fan::string(stage_compile_folder_name) + "/stage.h";
    bool data_exists = fan::io::file::exists(stage_path);

    stage_h.open(stage_path);

    if (!data_exists) {
      stage_h_str = R"(struct stage {
};)";
      write_stage();
    }
    else {
      stage_h.read(&stage_h_str);
    }


    fan::io::iterate_directory_files(stage_compile_folder_name, [this](const fan::string& path) {

      fan::string p = path;
      auto len = strlen(fan::string(fan::string(stage_compile_folder_name) + "/").c_str());
      p = p.substr(len, p.size() - len);

      if (p == "stage.h") {
        return;
      }
      p.pop_back();
      p.pop_back();
      create_stage(p);
    });
  }

  stage_list_t stage_list;
  fgm_t* fgm = nullptr;
  fan::string stage_h_str;
  fan::io::file::fstream stage_h;
};