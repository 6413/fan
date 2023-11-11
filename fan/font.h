#pragma once

#include _FAN_PATH(io/file.h)
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
#define BLL_set_BaseLibrary 1
#define BLL_set_prefix character_info_list
#define BLL_set_type_node uint16_t
#define BLL_set_NodeDataType character_info_t
#define BLL_set_Link 0
#include _FAN_PATH(BLL/BLL.h)

    using character_info_nr_t = character_info_list_NodeReference_t;

    using characters_t = std::unordered_map<uint32_t, character_info_nr_t>;

    struct font_t
    {

      f32_t top;
      f32_t bottom;

      f32_t height;

      f32_t size;
      character_info_list_t character_info_list;
      characters_t characters;
      f32_t line_height;

      auto get_character(uint32_t c)
      {
        auto found = characters.find(c);
#if fan_debug >= fan_debug_low
        if (found == characters.end())
        {
          fan::throw_error("failed to find character from font");
        }
#endif
        return found->second;
      }

      uint32_t get_font_index(uint32_t character) const
      {
        auto found = characters.find(character);
#if fan_debug >= fan_debug_low
        if (found == characters.end())
        {
          fan::print_warning("failed to find character from font");
          return -1;
        }
#endif
        return std::distance(characters.begin(), found);
      }
      f32_t convert_font_size(f32_t font_size) const
      {
        return font_size / this->size;
      }
      fan::font::character_info_t get_letter_info(uint32_t c, f32_t font_size)
      {

        auto found = characters.find(c);

#if fan_debug >= fan_debug_low
        if (found == characters.end())
        {
          throw std::runtime_error(fmt::format("failed to find character:{:x}", c));
        }
#endif

        auto &ci = character_info_list[found->second];

        f32_t converted_size = convert_font_size(font_size);
        fan::font::character_info_t font_info;
        font_info.metrics.size = ci.metrics.size * converted_size;
        font_info.metrics.offset = ci.metrics.offset * converted_size;
        font_info.metrics.advance = ci.metrics.size.x * converted_size;
        font_info.original_metrics = ci.original_metrics;

        font_info.glyph = ci.glyph;
        font_info.mapping = ci.mapping;

        font_info.utf8_character = ci.utf8_character;

        return font_info;
      }
      fan::font::character_info_t get_letter_info(character_info_nr_t char_internal_id, f32_t font_size)
      {

        auto &ci = character_info_list[char_internal_id];

        f32_t converted_size = convert_font_size(font_size);
        fan::font::character_info_t font_info;
        font_info.metrics.size = ci.metrics.size * converted_size;
        font_info.metrics.offset = ci.metrics.offset * converted_size;
        font_info.metrics.advance = ci.metrics.size.x * converted_size;
        font_info.original_metrics = ci.original_metrics;

        font_info.glyph = ci.glyph;
        font_info.mapping = ci.mapping;

        font_info.utf8_character = ci.utf8_character;

        return font_info;
      }
      fan::font::character_info_t get_letter_info(character_info_nr_t char_internal_id)
      {
        return get_letter_info(char_internal_id, this->size);
      }
      f32_t get_line_height(f32_t font_size) const
      {
        return line_height * convert_font_size(font_size);
      }
      // fan::vec2 get_text_size(const char* str, f32_t font_size) {
      //	fan::vec2 text_size = 0;

      //	text_size.y = line_height;

      //	f32_t width = 0;

      //	for (int i = 0; str[i] != 0; i++) {

      //		switch (str[i]) {
      //		case '\n': {
      //			text_size.x = std::max(width, text_size.x);
      //			text_size.y += line_height;
      //			width = 0;
      //			continue;
      //		}
      //		}

      //		auto letter = characters[str[i]];

      //		if (str[i + 1] == 0) {
      //			width += letter.glyph.size.x;
      //		}
      //		else {
      //			width += letter.metrics.advance;
      //		}
      //	}

      //	text_size.x = std::max(width, text_size.x);

      //	return text_size * convert_font_size(font_size);
      //}
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

    static line_t parse_line(std::unordered_multimap<uint32_t, uint32_t> *reverse_mapping, const fan::string &line, parse_stage_e stage)
    {
      switch (stage)
      {
      case parse_stage_e::mapping:
      {

        line_t l;

        auto r = fan::io::file::get_string_valuei_n(line);

        fan::string str;
        fan::utf16_to_utf8((wchar_t *)&r.value, &str);

        l.utf8 = str.get_utf8(0);
        r = fan::io::file::get_string_valuei_n(line, r.end);

        l.font_info.mapping.parse_index = r.value;

        reverse_mapping->insert(std::pair(r.value, l.utf8));

        return l;
      }
      case parse_stage_e::metrics_info:
      {
        line_t l;

        auto r = fan::io::file::get_string_valuei_n(line);

        auto utf8 = reverse_mapping->find(r.value);

        if (utf8 == reverse_mapping->end())
        {
          fan::throw_error("utf was not found from map index");
        }

        l.utf8 = utf8->second;

        auto r2 = fan::io::file::get_string_valuevec2i_n(line, r.end);

        l.font_info.metrics.size = r2.value;

        r2 = fan::io::file::get_string_valuevec2i_n(line, r2.end);

        l.font_info.metrics.offset = r2.value;

        r = fan::io::file::get_string_valuei_n(line, r2.end);

        l.font_info.metrics.advance = r.value;

        return l;
      }
      case parse_stage_e::glyph_info:
      {
        line_t l;

        auto r = fan::io::file::get_string_valuei_n(line);

        auto utf8 = reverse_mapping->find(r.value);

        if (utf8 == reverse_mapping->end())
        {
          fan::throw_error("utf was not found from map index");
        }

        l.utf8 = utf8->second;

        auto r2 = fan::io::file::get_string_valuevec2i_n(line, r.end);

        l.font_info.glyph.position = r2.value;
        r2 = fan::io::file::get_string_valuevec2i_n(line, r2.end);

        l.font_info.glyph.size = r2.value;

        r = fan::io::file::get_string_valuei_n(line, r2.end);

        l.font_info.glyph.border = r.value;

        return l;
      }
      }
    }

    static void parse_font(font_t &font, const fan::string &path)
    {
      if (!fan::io::file::exists(path))
      {
        fan::throw_error(fan::string("font not found") + path);
      }

      std::ifstream file(path.c_str());

      std::vector<fan::string> lines;
      std::string line;

      while (std::getline(file, line))
      {
        lines.push_back(line.data());
      }

      f32_t flowest = -fan::math::inf;
      f32_t fhighest = fan::math::inf;

      f32_t lowest = 0, highest = 0;

      std::size_t iline = 0;

      while (lines[iline].substr(0, 4) != "font")
      {
        iline++;
      }

      auto r = fan::io::file::get_string_valuei_n(lines[iline]);

      r = fan::io::file::get_string_valuei_n(lines[iline], r.end);
      r = fan::io::file::get_string_valuei_n(lines[iline], r.end);

      font.size = r.value;

      font.line_height = font.size * 1.5;

      while (lines[iline++].find("# code index") == fan::string::npos)
      {
      }

      parse_stage_e stage = parse_stage_e::mapping;

      std::unordered_multimap<uint32_t, uint32_t> reverse_mapping;

      while (1)
      {
        if (lines[iline].empty())
        {
          stage = parse_stage_e::metrics_info;
          break;
        }

        auto line = parse_line(&reverse_mapping, lines[iline], stage);
        {
          auto cilnr = font.character_info_list.NewNode();
          font.characters[line.utf8] = cilnr;
          auto &ci = font.character_info_list[cilnr];
          ci.mapping = line.font_info.mapping;
        }

        iline++;
      }

      while (lines[iline++].find("# index width height offset_x offset_y advance") == fan::string::npos)
      {
      }

      while (1)
      {
        if (lines[iline].empty())
        {
          stage = parse_stage_e::glyph_info;
          break;
        }

        auto line = parse_line(&reverse_mapping, lines[iline], stage);
        auto &character_info = font.character_info_list[font.characters[line.utf8]];

        character_info.metrics = line.font_info.metrics;
        character_info.original_metrics = character_info.metrics;

        iline++;
      }

      while (lines[iline++].find("# index x y width height border") == fan::string::npos)
      {
      }

      while (1)
      {

        if (lines[iline].empty())
        {

          {
            auto cilnr = font.character_info_list.NewNode();
            font.characters[L'\n'] = cilnr;
            auto &ci = font.character_info_list[cilnr];
            ci.utf8_character = L'\n';
            ci.glyph.position = 0;
            ci.glyph.size = 0;
            ci.metrics.advance = 0;
            ci.metrics.offset = 0;
            ci.metrics.size = 0;
          }

          for (auto &i : font.characters)
          {
            auto &character_info = font.character_info_list[i.second];
            character_info.metrics.size = character_info.glyph.size;
          }

          f32_t uppest = 100000;
          f32_t downest = -100000;
          for (auto &i : font.characters)
          {

            auto letter_info = font.get_letter_info(i.first, font.size);
            f32_t height = (font.size - letter_info.metrics.size.y) / 2 - letter_info.metrics.offset.y;
            if (uppest > height - letter_info.metrics.size.y / 2)
            {
              uppest = height - letter_info.metrics.size.y / 2;
            }
            if (downest < height + letter_info.metrics.size.y / 2)
            {
              downest = height + letter_info.metrics.size.y / 2;
            }
          }
          font.top = uppest;
          font.bottom = downest;
          font.height = std::max(std::abs(font.top), std::abs(font.bottom)) * 2;
          return;
        }

        auto line = parse_line(&reverse_mapping, lines[iline], stage);

        auto &character_info = font.character_info_list[font.characters[line.utf8]];
        character_info.glyph = line.font_info.glyph;
        character_info.utf8_character = line.utf8;

        iline++;
      }
    }
  }
}