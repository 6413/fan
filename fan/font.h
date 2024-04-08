#pragma once

#include <fan/io/file.h>
#include <unordered_map>

namespace fan
{

  namespace font
  {

    // -y offset from top
    static constexpr auto highest_character = -2;

    struct mapping_t
    {
      uint32_t parse_index;
    };

    // physical letter info
    struct metrics_info_t
    {
      fan::vec2 size;
      fan::vec2 offset;
      f32_t advance;
    };

    // image info
    struct glyph_info_t
    {
      fan::vec2 position;
      fan::vec2 size;
      f32_t border;
    };

    struct character_info_t
    {
      mapping_t mapping;
      metrics_info_t metrics;
      glyph_info_t glyph;
      metrics_info_t original_metrics;
      uint32_t utf8_character;
    };

#define BLL_set_CPP_ConstructDestruct
#define BLL_set_CPP_Node_ConstructDestruct
#define BLL_set_AreWeInsideStruct 0
#include <fan/fan_bll_preset.h>
#define BLL_set_prefix character_info_list
#define BLL_set_type_node uint16_t
#define BLL_set_NodeDataType character_info_t
#define BLL_set_Link 0
#include _FAN_PATH(BLL/BLL.h)

    using character_info_nr_t = character_info_list_NodeReference_t;

    using characters_t = std::unordered_map<uint32_t, character_info_nr_t>;

    struct font_t {

      f32_t top;
      f32_t bottom;

      f32_t height;

      f32_t size;
      character_info_list_t character_info_list;
      characters_t characters;
      f32_t line_height;

      character_info_nr_t get_character(uint32_t c);

      uint32_t get_font_index(uint32_t character) const;
      f32_t convert_font_size(f32_t font_size) const;
      fan::font::character_info_t get_letter_info(uint32_t c, f32_t font_size);
      fan::font::character_info_t get_letter_info(character_info_nr_t char_internal_id, f32_t font_size);
      fan::font::character_info_t get_letter_info(character_info_nr_t char_internal_id);
      f32_t get_line_height(f32_t font_size) const;
    };

    enum class parse_stage_e
    {
      mapping,
      metrics_info,
      glyph_info
    };

    struct line_t
    {
      uint32_t utf8;
      character_info_t font_info;
    };

    line_t parse_line(std::unordered_multimap<uint32_t, uint32_t> *reverse_mapping, const fan::string &line, parse_stage_e stage);

    void parse_font(font_t &font, const fan::string &path);
  }
}