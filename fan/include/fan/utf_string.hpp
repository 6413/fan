#pragma once

#include <string>
#include <array>
#include <vector>
#include <stdexcept>

namespace fan {

	struct utf8_string;
	struct utf16_string;

	static std::wstring utf8_to_utf16(const uint8_t* utf8);
	static fan::utf8_string utf16_to_utf8(const wchar_t* utf16);

	static uint8_t utf8_get_sizeof_character(uint8_t byte) {
		if (byte < 0x80) {
			return 1;
		}
		if (byte < 0xc0) {
			throw std::runtime_error("not utf8 character");
			return 1;
		}
		if (byte < 0xe0) {
			return 2;
		}
		if (byte < 0xf0) {
			return 3;
		}
		if (byte <= 0xf7) {
			return 4;
		}

		throw std::runtime_error("not utf8 character");
		return 1;
	}

	struct utf8_string : 
		public std::basic_string<uint8_t, std::char_traits<uint8_t>, std::allocator<uint8_t>> {

		using inherit_t = std::basic_string<uint8_t, std::char_traits<uint8_t>, std::allocator<uint8_t>>;

		utf8_string() {}

		//utf8_string(uint8_t* str) : inherit_t(str) {}

		utf8_string(uint32_t character) {
			this->push_back(character);
		}

		utf8_string(utf16_string str);

		void push_back(uint32_t character) {

			for (int i = 0; i < utf8_get_sizeof_character(character); i++) {

				inherit_t::push_back(character >> (i * 8));
			}
		}

		utf16_string to_utf16() const;

		uint32_t get_character(std::size_t i) const  {
			std::size_t j = 0, size = 0;

			while (j < i) {
				size += fan::utf8_get_sizeof_character(data()[size]);
				j++;
			}

			uint32_t value = 0;

			auto character_size = fan::utf8_get_sizeof_character(data()[size]);

			for (j = 0; j < character_size; j++) {
				value |= (data()[size + j] << (j * 8));
			}

			return value;
		}

		friend std::ostream& operator<<(std::ostream& os, const utf8_string& str);
	};

	inline std::ostream& operator<<(std::ostream& os, const utf8_string& str)
	{
		os << *(std::string*)&str;
		return os;
	}

	struct utf16_string :
		public std::wstring {

		using std::wstring::basic_string;

		using inherit_t = std::wstring;

		utf16_string(const std::wstring& str) : inherit_t(str) {} // visual studio magics

		utf16_string(uint32_t character) : inherit_t(fan::utf8_to_utf16((uint8_t*)&character).data()) {}

		utf16_string(uint8_t* data) : inherit_t(fan::utf8_to_utf16(data).data()) {}

		utf16_string(utf8_string str) : inherit_t(fan::utf8_to_utf16(str.data())) {}

		utf8_string to_utf8() const {
			return utf8_string(fan::utf16_to_utf8(this->data()));
		}

		friend std::wostream& operator<<(std::wostream& os, const utf16_string& str);
	};

	inline std::wostream& operator<<(std::wostream& os, const utf16_string& str)
	{
		os << (std::wstring)str;
		return os;
	}

	// utf8 struct definitions

	inline utf8_string::utf8_string(utf16_string str) : utf8_string::inherit_t(fan::utf16_to_utf8(str.data()).data()) {}

	inline utf16_string utf8_string::to_utf16() const {
		return utf16_string(fan::utf8_to_utf16(this->data()).data());
	}

	static std::wstring utf8_to_utf16(const uint8_t* utf8)
	{
		std::vector<unsigned long> unicode;
		size_t i = 0;
		while (*(utf8 + i))
		{
			unsigned long uni;
			size_t todo;
			bool error = false;
			unsigned char ch = utf8[i++];
			if (ch <= 0x7F)
			{
				uni = ch;
				todo = 0;
			}
			else if (ch <= 0xBF)
			{
				throw std::logic_error("not a utf-8 string");
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
				throw std::runtime_error("not a utf-8 string");
			}
			for (size_t j = 0; j < todo; ++j)
			{
				unsigned char ch = utf8[i++];
				if (ch < 0x80 || ch > 0xBF)
					throw std::runtime_error("not a utf-8 string");
				uni <<= 6;
				uni += ch & 0x3F;
			}
			if (uni >= 0xD800 && uni <= 0xDFFF)
				throw std::runtime_error("not a utf-8 string");
			if (uni > 0x10FFFF)
				throw std::runtime_error("not a utf-8 string");
			unicode.push_back(uni);
		}
		std::wstring utf16;
		for (size_t i = 0; i < unicode.size(); ++i)
		{
			unsigned long uni = unicode[i];
			if (uni <= 0xFFFF)
			{
				utf16 += (wchar_t)uni;
			}
			else
			{
				uni -= 0x10000;
				utf16 += (wchar_t)((uni >> 10) + 0xD800);
				utf16 += (wchar_t)((uni & 0x3FF) + 0xDC00);
			}
		}

		return utf16;
	}

	static fan::utf8_string utf16_to_utf8(const wchar_t* utf16) {

		fan::utf8_string out;
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
					out.append(1, static_cast<char>(codepoint));
				else if (codepoint <= 0x7ff)
				{
					out.append(1, static_cast<char>(0xc0 | ((codepoint >> 6) & 0x1f)));
					out.append(1, static_cast<char>(0x80 | (codepoint & 0x3f)));
				}
				else if (codepoint <= 0xffff)
				{
					out.append(1, static_cast<char>(0xe0 | ((codepoint >> 12) & 0x0f)));
					out.append(1, static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
					out.append(1, static_cast<char>(0x80 | (codepoint & 0x3f)));
				}
				else
				{
					out.append(1, static_cast<char>(0xf0 | ((codepoint >> 18) & 0x07)));
					out.append(1, static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
					out.append(1, static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
					out.append(1, static_cast<char>(0x80 | (codepoint & 0x3f)));
				}
				codepoint = 0;
			}
		}
		return out;
	}

}