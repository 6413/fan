namespace fan::graphics::entity {
  template<typename... enemy_types_t>
  struct enemy_container_t {
    struct enemy_t : std::variant<enemy_types_t...> {
      using std::variant<enemy_types_t...>::variant;
      void update() { std::visit([](auto& e) { e.update(); }, *this); }
      void destroy() { std::visit([](auto& e) { e.destroy(); }, *this); }
      auto& get_body() { return std::visit([](auto& e) -> auto& { return e.get_body(); }, *this); }
      bool on_hit(auto* source, const fan::vec2& dir) {
        return std::visit([source, dir](auto& e) { return e.on_hit(source, dir); }, *this);
      }
    };

    //#define bcontainer_set_StoreFormat 1
    #define BLL_set_CPP_CopyAtPointerChange 1
    //#define BLL_set_CPP_Node_ConstructDestruct 1
    #define BLL_set_SafeNext 1
    #define BLL_set_Usage 1
    #define BLL_set_AreWeInsideStruct 1 
    #define BLL_set_prefix enemies
    #define BLL_set_NodeDataType enemy_t
    #include <fan/fan_bll_preset.h>
    #include <BLL/BLL.h>

    using nr_t = enemies_t::nr_t;
    using size_type = decltype(nr_t::NRI);
    using nd_t = enemies_t::nd_t;

    enemies_t list;

    nr_t add(const nd_t& nd) {
      return list.push_back(nd);
    }
    nr_t add(nd_t&& nd) {
      return list.push_back(std::move(nd));
    }
    nr_t add() {
      return list.NewNodeLast();
    }
    void remove(nr_t nr) {
      list.unlrec(nr);
    }
    void clear() {
      list.Clear();
    }
    void update() {
      for (auto& enemy : *this) {
        enemy.update();
      }
    }
    auto size() const {
      return list.Usage();
    }

    nd_t& operator[](nr_t nr) {
      return list[nr];
    }
    auto begin() {
      return list.begin();
    }

    auto end() {
      return list.end();
    }

    auto begin() const {
      return list.begin();
    }

    auto end() const {
      return list.end();
    }
  };
}