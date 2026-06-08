module;

#if defined(FAN_OPENGL)
#endif

export module fan.texture_pack.tp0;

#if defined (FAN_WINDOW)

import std;

#if defined(FAN_OPENGL)

import fan.types.vector;

import fan.graphics.common_context;
import fan.graphics.opengl.core;

export namespace fan::graphics {
  struct texture_pack_t;

  struct texture_pack {
    struct ti_t;

    struct unique_data_t {
      std::uint32_t major;
      std::uint32_t minor;
    };

    struct unique_t {
      std::uint32_t id = static_cast<std::uint32_t>(-1);
      constexpr bool iic() const { return id == static_cast<std::uint32_t>(-1); }
      constexpr bool operator==(const unique_t& o) const { return id == o.id; }
    };

    struct ti_t {
      ti_t() = default;
      ti_t(std::string_view name, texture_pack_t* tp);
      bool qti(texture_pack_t* tp, const std::string& name);
      bool qti(texture_pack_t* tp, std::uint64_t hash);

      bool valid() const {
        return image.iic() == false;
      }

      fan::graphics::texture_pack::unique_t unique_id;
      fan::vec2 position;
      fan::vec2 size;
      fan::graphics::image_t image;
    };

    struct internal_t {
      struct open_properties_t {
        fan::vec2ui preferred_pack_size = 1024;
        std::uint32_t visual_output = fan::opengl::context_t::image_load_properties_defaults::visual_output;
        std::uint32_t min_filter = 0x2703; // GL_LINEAR_MIPMAP_LINEAR
        std::uint32_t mag_filter = fan::opengl::context_t::image_load_properties_defaults::mag_filter;
      };

      struct pack_properties_t {
        fan::vec2ui pack_size;
        std::uint32_t visual_output;
        std::uint32_t min_filter;
        std::uint32_t mag_filter;
        std::uint32_t group_id;
      };

      struct texture_properties_t {
        fan::vec2 uv_pos = 0;
        fan::vec2 uv_size = 1;
        std::string image_name;
        std::uint32_t visual_output = -1;
        std::uint32_t min_filter = -1;
        std::uint32_t mag_filter = -1;
        std::uint32_t group_id = -1;
      };

      void* internal_state = nullptr;

      internal_t();
      ~internal_t();

      void open();
      void open(const open_properties_t& op);
      void close();

      std::size_t push_pack(const pack_properties_t& p);
      std::size_t push_pack();

      bool push_texture(const std::string& image_path, const texture_properties_t& texture_properties);
      bool push_texture(fan::graphics::image_t image, const texture_properties_t& texture_properties);

      void process();
      void save_compiled(const std::string& filename);
      void load_compiled(const char* filename);

      std::size_t size() const;
      
      void* get_pixel_data_raw(std::uint32_t pack_id); 
    };
  };

  struct texture_pack_t {
    using ti_t = fan::graphics::texture_pack::ti_t;

    struct texture_minor_t {
      std::string name;
      fan::vec2i position;
      fan::vec2i size;
    };

    inline static constexpr std::uint16_t MAX_TEXTURE_MINOR = 1024;

    struct texture_minor_decoded_t {
      fan::graphics::texture_pack::unique_t unique_id;
      std::string name;
      fan::vec2i position;
      fan::vec2i size;
    };

    struct pixel_data_t {
      fan::graphics::image_t image;
    };

    void* internal_state = nullptr;
    std::string file_path;

    texture_pack_t();
    texture_pack_t(const std::string& filename, const std::source_location& callers_path = std::source_location::current());
    ~texture_pack_t();

    void open_compiled(const std::string& filename, const std::source_location& callers_path = std::source_location::current());
    void open_compiled(const std::string& filename, fan::graphics::image_load_properties_t lp, const std::source_location& callers_path = std::source_location::current());

    pixel_data_t& get_pixel_data(fan::graphics::texture_pack::unique_t unique);

    void iterate_loaded_images_raw(void* user_data, void(*cb)(const texture_minor_decoded_t&, void*));

    template<typename Lambda>
    void iterate_loaded_images(Lambda&& lambda) {
      auto cb = [](const texture_minor_decoded_t& t, void* user_data) {
        (*static_cast<std::decay_t<Lambda>*>(user_data))(t);
      };
      auto* state = new std::decay_t<Lambda>(std::forward<Lambda>(lambda));
      iterate_loaded_images_raw(state, cb);
      delete state;
    }

    explicit operator bool() const;
    std::size_t size() const;

    texture_minor_decoded_t operator[](fan::graphics::texture_pack::unique_t unique_id);
    fan::graphics::texture_pack::unique_t operator[](const std::string& name);

    bool qti(const std::string& name, ti_t* ti);
    bool qti(std::uint64_t hash, ti_t* ti);
  };

  inline texture_pack::ti_t::ti_t(std::string_view name, texture_pack_t* tp) {
    tp->qti(std::string(name), this);
  }
  inline bool texture_pack::ti_t::qti(texture_pack_t* tp, const std::string& name) {
    return tp->qti(name, this);
  }
  inline bool texture_pack::ti_t::qti(texture_pack_t* tp, std::uint64_t hash) {
    return tp->qti(hash, this);
  }
}
#endif

#endif