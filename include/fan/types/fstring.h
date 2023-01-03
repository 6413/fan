#pragma once

#include <vector>
#include <cstring>
#include <memory>
#include <algorithm>
#include <string>

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
	struct string : fan::basic_string<char>{
		string(const fan::basic_string<char>& b) : fan::basic_string<char>(b) {

		}

		using basic_string::basic_string;
	};
	struct wstring: fan::basic_string<wchar_t> {
		wstring(const basic_string& b) : basic_string(b) {

		}
		using basic_string::basic_string;
	};
}