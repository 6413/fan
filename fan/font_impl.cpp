#include "font.h"

#include <fan/io/file.h>
#include <fan/types/fstring.h>
#include <fan/types/utf8.h>

fan::font::character_info_nr_t fan::font::font_t::get_character(uint32_t c)
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

uint32_t fan::font::font_t::get_font_index(uint32_t character) const
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

f32_t fan::font::font_t::convert_font_size(f32_t font_size) const
{
  return font_size / this->size;
}

fan::font::character_info_t fan::font::font_t::get_letter_info(uint32_t c, f32_t font_size)
{

  auto found = characters.find(c);

  #if fan_debug >= fan_debug_low
  if (found == characters.end())
  {
    fan::throw_error("failed to find character:", c);
  }
  #endif

  auto& ci = character_info_list[found->second];

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

fan::font::character_info_t fan::font::font_t::get_letter_info(character_info_nr_t char_internal_id, f32_t font_size)
{

  auto& ci = character_info_list[char_internal_id];

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

fan::font::character_info_t fan::font::font_t::get_letter_info(character_info_nr_t char_internal_id)
{
  return get_letter_info(char_internal_id, this->size);
}

f32_t fan::font::font_t::get_line_height(f32_t font_size) const
{
  return line_height * convert_font_size(font_size);
}

fan::font::line_t fan::font::parse_line(reverse_mapping_t* reverse_mapping, const std::string& line, parse_stage_e stage)
{
  switch (stage)
  {
    case parse_stage_e::mapping:
    {

      line_t l;

      auto r = fan::io::file::get_string_valuei_n(line);

      std::string str;
      fan::utf16_to_utf8((wchar_t*)&r.value, &str);

      l.utf8 = fan::get_utf8(&str, 0);
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

void fan::font::parse_font(font_t& font, const std::string& path)
{
  if (!fan::io::file::exists(path))
  {
    fan::throw_error(std::string("font not found") + path);
  }

  std::ifstream file(path.c_str());

  std::vector<std::string> lines;
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

  while (lines[iline++].find("# code index") == std::string::npos)
  {
  }

  parse_stage_e stage = parse_stage_e::mapping;

  reverse_mapping_t reverse_mapping;

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
      auto& ci = font.character_info_list[cilnr];
      ci.mapping = line.font_info.mapping;
    }

    iline++;
  }

  while (lines[iline++].find("# index width height offset_x offset_y advance") == std::string::npos)
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
    auto& character_info = font.character_info_list[font.characters[line.utf8]];

    character_info.metrics = line.font_info.metrics;
    character_info.original_metrics = character_info.metrics;

    iline++;
  }

  while (lines[iline++].find("# index x y width height border") == std::string::npos)
  {
  }

  while (1)
  {

    if (lines[iline].empty())
    {

      {
        auto cilnr = font.character_info_list.NewNode();
        font.characters[L'\n'] = cilnr;
        auto& ci = font.character_info_list[cilnr];
        ci.utf8_character = L'\n';
        ci.glyph.position = 0;
        ci.glyph.size = 0;
        ci.metrics.advance = 0;
        ci.metrics.offset = 0;
        ci.metrics.size = 0;
      }

      for (auto& i : font.characters)
      {
        auto& character_info = font.character_info_list[i.second];
        character_info.metrics.size = character_info.glyph.size;
      }

      f32_t uppest = 100000;
      f32_t downest = -100000;
      for (auto& i : font.characters)
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

    auto& character_info = font.character_info_list[font.characters[line.utf8]];
    character_info.glyph = line.font_info.glyph;
    character_info.utf8_character = line.utf8;

    iline++;
  }
}