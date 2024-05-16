#pragma once

#include <string>

namespace fan {
  uint32_t get_utf8(std::string* str, std::size_t i) {
    const std::u8string_view& sv = (char8_t*)str->c_str();
    uint32_t code = 0;
    uint32_t offset = 0;
    for (uint32_t k = 0; k <= i; k++) {
      code = 0;
      int len = 1;
      if ((sv[offset] & 0xF8) == 0xF0) { len = 4; }
      else if ((sv[offset] & 0xF0) == 0xE0) { len = 3; }
      else if ((sv[offset] & 0xE0) == 0xC0) { len = 2; }
      for (int j = 0; j < len; j++) {
        code <<= 8;
        code |= sv[offset];
        offset++;
      }
    }
    return code;
  }

  bool utf16_to_utf8(const wchar_t* utf16, std::string* out) {
    unsigned int codepoint = 0;

    for (; *utf16 != 0; ++utf16)
    {
      if (*utf16 >= 0xd800 && *utf16 <= 0xdbff)
        codepoint = ((*utf16 - 0xd800) << 10) + 0x10000;
      else
      {
        if (*utf16 >= 0xdc00 && *utf16 <= 0xdfff)
          codepoint |= *utf16 - 0xdc00;
        else
          codepoint = *utf16;

        if (codepoint <= 0x7f)
          out->append(1, static_cast<char>(codepoint));
        else if (codepoint <= 0x7ff)
        {
          out->append(1, static_cast<char>(0xc0 | ((codepoint >> 6) & 0x1f)));
          out->append(1, static_cast<char>(0x80 | (codepoint & 0x3f)));
        }
        else if (codepoint <= 0xffff)
        {
          out->append(1, static_cast<char>(0xe0 | ((codepoint >> 12) & 0x0f)));
          out->append(1, static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
          out->append(1, static_cast<char>(0x80 | (codepoint & 0x3f)));
        }
        else
        {
          out->append(1, static_cast<char>(0xf0 | ((codepoint >> 18) & 0x07)));
          out->append(1, static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
          out->append(1, static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
          out->append(1, static_cast<char>(0x80 | (codepoint & 0x3f)));
        }
        codepoint = 0;
      }
    }
    return 0;
  }
}