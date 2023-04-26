#pragma once

#include <vector>
#include <cstring>
#include <memory>
#include <algorithm>
#include <string>
#include <regex>

#define FMT_HEADER_ONLY
#include <fmt/format.h>
#include <fmt/xchar.h>


namespace fan {

	template <typename T>
	concept is_char_t = std::is_same<T, char>::value;

	template <typename type_t>
	struct basic_string {

		using char_type = type_t;

		constexpr std::size_t stringlen(const char_type* s) {
			std::size_t l = 0;
			while (*s != 0) {
				++l;
				++s;
			}
			return l;
		}

		using value_type = std::vector<char_type>;

		basic_string() {
			str.push_back(0);
		}
		basic_string(char_type c) {
			str.push_back(c);
			str.push_back(0);
		}
		basic_string(const char_type* s) : str(s, s + stringlen(s)) {
			str.push_back(0);
		}
    basic_string(const char_type* src, const char_type* dst) : str(src, dst) {
			str.push_back(0);
		}
		basic_string(typename value_type::const_iterator beg, typename value_type::const_iterator end) : str(beg, end) {
			str.push_back(0);
		}
		basic_string(std::basic_string_view<char_type> sv) : str(sv.begin(), sv.end()) {
			str.push_back(0);
		}
		basic_string(const basic_string<wchar_t>& str) requires is_char_t<char_type>
												// cheats
			: basic_string(std::string(str.data(), str.data() + str.size()).c_str()) {}

		basic_string(const basic_string&) = default;

		void push_back(char_type c) {
			str.insert(str.end() - 1, c);
		}

		bool empty() const {
			return !size();
		}

		std::size_t size() const {
			if (str.size()) {
				return str.size() - 1;
			}
			return std::size_t(0);
		}

		const char_type* c_str() const {
			return str.data();
		}

		void clear() {
			str.clear();
		}

		auto begin() const {
			return str.begin();
		}
		auto end() const {
			if (str.size()) {
				return str.end() - 1;
			}
			return str.end();
		}

		auto begin() {
			return str.begin();
		}
		auto end() {
			if (str.size()) {
				return str.end() - 1;
			}
			return str.end();
		}

		void insert(typename value_type::const_iterator where, typename value_type::const_iterator begin, typename value_type::const_iterator end) {
			str.insert(where, begin, end);
		}
		void insert(typename value_type::const_iterator iter, const basic_string& s) {
			str.insert(iter, s.begin(), s.end());
		}
		void insert(std::size_t where, const basic_string& s) {
			str.insert(begin() + where, s.begin(), s.end());
		}

		auto append(const basic_string& s) {
      insert(end(), s);
			return *this;
		}

		auto erase(std::size_t beg, std::size_t count) {
			str.erase(str.begin() + beg, str.begin() + beg + count);
			return *this;
		}
		void erase(std::size_t where) {
			str.erase(str.begin() + where);
		}
		void erase(typename value_type::const_iterator where) {
			str.erase(where);
		}
		void erase(typename value_type::const_iterator begin, typename value_type::const_iterator end) {
			str.erase(begin, end);
		}
		void pop_back() {
			str.erase(end());
		}

		basic_string substr(std::size_t beg, std::size_t n) const {
			if (beg > size() || beg + n > size()) {
				return *this;
			}
			return basic_string(begin() + beg, begin() + beg + n);
		}
		basic_string substr(typename value_type::const_iterator beg, std::size_t n) const {
			return basic_string(beg, beg + n);
		}

		std::size_t find(const basic_string& s) const {
			auto found = std::search(begin(), end(), s.begin(), s.end());
			if (found == end()) {
				return npos;
			}
			return std::distance(begin(), found);
		}
		std::size_t find(const basic_string& s, std::size_t start) const {
			auto found = std::search(begin() + start, end(), s.begin(), s.end());
			if (found == end()) {
				return npos;
			}
			return std::distance(begin(), found);
		}

		template<class Iterator> struct is_not_in_range
		{
			Iterator const begin;
			Iterator const end;
			is_not_in_range(Iterator const& b, Iterator const& e)
				: begin(b)
				, end(e) {}
			template<class Value> bool operator()(Value& v)
			{
				return std::find(begin, end, v) == end;
			}
		};

		std::size_t find_first_of(const basic_string& s, std::size_t start = 0) const {
			auto found = std::find_first_of(begin() + start, end(), s.begin(), s.end());
			if (found == end()) {
				return npos;
			}
			return std::distance(begin(), found);
		}

		std::size_t find_first_not_of(const basic_string& s, std::size_t start = 0) const {
			std::string ss = c_str();
			return ss.find_first_not_of(s.c_str(), start);
		}

		std::size_t find_last_of(const basic_string& s, std::size_t start = 0) const {
			auto found = std::find_end(begin() + start, end(), s.begin(), s.end());
			if (found == end()) {
				return npos;
			}
			return std::distance(begin(), found);
		}

		bool equals(const basic_string& s) const {
			if (size() != s.size()) {
				return false;
			}
			if (empty() && s.empty()) {
				return true;
			}
			if (empty()) {
				return false;
			}
			return std::equal(begin(), end() - 1, s.begin(), s.end() - 1);
		}

    bool operator==(const char_type* c) const {
      return std::equal(begin(), end(), c);
		}

		bool operator==(const basic_string& s) const {
			return equals(s);
		}

		bool operator!=(const basic_string& s) const {
			return !equals(s);
		}

		basic_string& operator+=(const basic_string& s) {
			append(s);
			return *this;
		}

		basic_string& operator+=(const char_type* s) {
			insert(end(), s);
			return *this;
		}

		basic_string operator+(const basic_string& s) const {
			basic_string result = *this;
			result.insert(result.end(), s);
			return result;
		}
		basic_string operator+(const char_type* c) const {
			basic_string result = *this;
			result.insert(result.end(), c);
			return result;
		}

		constexpr char_type& operator[](std::size_t i) {
			return str[i];
		}
		constexpr char_type operator[](std::size_t i) const {
			return str[i];
		}

		constexpr char_type* data() noexcept {
			return str.data();
		}

		constexpr const char_type* data() const noexcept {
			return str.data();
		}

		void resize(std::size_t c) {
			str.resize(c + 1);
		}

		void replace(std::size_t beg, std::size_t count, const basic_string& replace_str) {
      erase(beg, count);
      insert(beg, replace_str);
		}

		void replace_all(const basic_string& search, const basic_string& replace) {
			for (size_t pos = 0; ; pos += replace.size()) {
				// Locate the substring to replace
				pos = find(search, pos);
				if (pos == basic_string::npos) break;
				// Replace by erasing and inserting
				erase(pos, search.size());
				insert(pos, replace);
			}
		}

		static constexpr std::size_t npos = (std::size_t)-1;
		value_type str;
	};
	
	template <typename T>
	static std::ostream& operator<<(std::ostream& os, const fan::basic_string<T>& s) {
		return os << s.c_str();
	}
}

template <typename T>
static fan::basic_string<T> operator+(const T* left, const fan::basic_string<T>& s) {
	fan::basic_string<T> str = s;
	str.insert(str.begin(), left);
	return str;
}

namespace fan {
  struct string : public std::string {

    using type_t = std::string;
    using type_t::basic_string;

    string(const std::string& str) : type_t(str) {}

    using char_type = std::string::value_type;

    constexpr uint32_t get_utf8(std::size_t i) const {
      std::u8string_view sv = (char8_t*)c_str();
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
    constexpr auto utf8_size() const {
      std::size_t count = 0;
      for (auto i = begin(); i != end(); i++){
        if ((*i & 0xC0) != 0x80) {
          count++;
        }
      }
      return count;
    }

    void replace_all(const basic_string& search, const basic_string& replace) {
      for (size_t pos = 0; ; pos += replace.size()) {
        // Locate the substring to replace
        pos = find(search, pos);
        if (pos == basic_string::npos) break;
        // Replace by erasing and inserting
        erase(pos, search.size());
        insert(pos, replace);
      }
    }

  };

  static bool utf8_to_utf16(const uint8_t* utf8, std::wstring* out)
  {
    bool error = false;
    std::vector<unsigned long> unicode;
    size_t i = 0;
    while (*(utf8 + i))
    {
      unsigned long uni;
      size_t todo = 0;
      unsigned char ch = utf8[i++];
      if (ch <= 0x7F)
      {
        uni = ch;
        todo = 0;
      }
      else if (ch <= 0xBF)
      {
        error = true;
      }
      else if (ch <= 0xDF)
      {
        uni = ch & 0x1F;
        todo = 1;
      }
      else if (ch <= 0xEF)
      {
        uni = ch & 0x0F;
        todo = 2;
      }
      else if (ch <= 0xF7)
      {
        uni = ch & 0x07;
        todo = 3;
      }
      else
      {
        error = true;
      }
      for (size_t j = 0; j < todo; ++j)
      {
        unsigned char ch = utf8[i++];
        if (ch < 0x80 || ch > 0xBF)
          error = true;
        uni <<= 6;
        uni += ch & 0x3F;
      }
      if (uni >= 0xD800 && uni <= 0xDFFF)
        error = true;
      if (uni > 0x10FFFF)
        error = true;
      unicode.push_back(uni);
    }
    for (size_t i = 0; i < unicode.size(); ++i)
    {
      unsigned long uni = unicode[i];
      if (uni <= 0xFFFF)
      {
        *out += (wchar_t)uni;
      }
      else
      {
        uni -= 0x10000;
        *out += (wchar_t)((uni >> 10) + 0xD800);
        *out += (wchar_t)((uni & 0x3FF) + 0xDC00);
      }
    }

    return error;
  }

  static bool utf16_to_utf8(const wchar_t* utf16, fan::string* out) {
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

  template <typename T>
  T read_data(auto& f, auto& off) {
    if constexpr (std::is_same<fan::string, T>::value) {
      uint64_t len = read_data<uint64_t>(f, off);
      fan::string str;
      str.resize(len);
      memcpy(str.data(), &f[off], len);
      off += len;
      return str;
    }
    else {
      auto obj = &f[off];
      off += sizeof(T);
      return *(T*)obj;
    }
  }

  template <typename T>
  void write_to_string(auto& f, const T& o) {
    if constexpr (std::is_same<fan::string, T>::value) {
      uint64_t len = o.size();
      f.append((char*)&len, sizeof(len));
      f.append(o.data(), len);
    }
    else {
      f.append((char*)&o, sizeof(o));
    }
  }

	//struct string : fan::basic_string<char>{
	//	string(const fan::basic_string<char>& b) : fan::basic_string<char>(b) {

	//	}

	//	using basic_string::basic_string;
	//};
	//struct wstring: fan::basic_string<uint32_t> {
	//	wstring(const basic_string& b) : basic_string(b) {

	//	}
 //   wstring(const char8_t* data) /*: basic_string((wstring::char_type*)c, (wstring::char_type*)c + std::u8string(c).size())*/ {
 //     //for (auto it = data; it != ) {
 //     //
 //     //}
 //     auto str = std::u8string(data);
 //     auto x = strlen((const char*)data);
 //     for (uint32_t i = 0; i < x; ++i) {
 //      // push_back(*(uint32_t*)&c[i]);
 //     }
 //   }
	//	using basic_string::basic_string;
	//};

  //template <typename... T>
  //static FMT_INLINE auto format(fmt::wformat_string<T...> fmt, T&&... args)
  //  -> fan::wstring {
  //  return fmt::vformat(fmt::wstring_view(fmt), fmt::make_wformat_args(args...)).c_str();
  //}

  template <typename T>
  T string_to(const fan::string& fstring) {
    T out;
    std::istringstream iss(fstring);
    if constexpr (has_bracket_operator<T>::value) {
      std::size_t i = 0;
      while (iss >> out[i++]) { iss.ignore(); }
    }
    else {
      while (iss >> out) { iss.ignore(); }
    }
    return out;
  }

}