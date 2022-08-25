#pragma once
//
#include <string>
//#include <array>
//#include <vector>
//#include <stdexcept>

#include _FAN_PATH(types/memory.h)

//namespace fan {
//
//	struct character_t;
//	struct utf8_string;
//	struct utf16_string;
//
//
//	struct character_t {
//		void open(wchar_t c_) {
//			c = c_;
//		}
//		bool open(uint8_t* c_, std::wstring* out) {
//			return utf8_to_utf16(c_, out);
//		}
//		wchar_t c;
//	};
//
//	static bool utf8_get_sizeof_character(uint8_t byte, uint8_t* out) {
//		if (byte < 0x80) {
//			*out = 1;
//		}
//		if (byte < 0xc0) {
//			// not utf8
//			return 1;
//		}
//		if (byte < 0xe0) {
//			*out = 2;
//		}
//		if (byte < 0xf0) {
//			*out = 3;
//		}
//		if (byte <= 0xf7) {
//			*out = 4;
//		}
//		// not utf8
//		return 1;
//	}
//
//	struct utf8_string : 
//		public std::basic_string<uint8_t, std::char_traits<uint8_t>, std::allocator<uint8_t>> {
//
//		using inherit_t = std::basic_string<uint8_t, std::char_traits<uint8_t>, std::allocator<uint8_t>>;
//
//		utf8_string() = default;
//
//		void open(uint32_t character) {
//			this->push_back(character);
//		}
//		bool open(const utf16_string& str) {
//			return utf16_to_utf8(str.data(), this);
//		}
//		//utf8_string(const utf16_string& str);
//
//		bool push_back(uint32_t character) {
//
//			uint8_t n;
//			bool err = utf8_get_sizeof_character(character, &n);
//
//			if (err) {
//				return err;
//			}
//
//			for (int i = 0; i < n; i++) {
//				inherit_t::push_back(character >> (i * 8));
//			}
//
//			return 0;
//		}
//
//		bool to_utf16(utf16_string* out) const {
//			return utf8_to_utf16(this->data(), (std::wstring*)out);
//		}
//
//		bool get_character(std::size_t i, uint32_t* character) const  {
//			std::size_t j = 0, size = 0;
//
//			while (j < i) {
//				uint8_t n;
//				bool err = fan::utf8_get_sizeof_character(data()[size], &n);
//				if (err) {
//					return err;
//				}
//				n += size;
//				j++;
//			}
//
//			uint8_t character_size;
//			bool err = fan::utf8_get_sizeof_character(data()[size], &character_size);
//			if (err) {
//				return err;
//			}
//
//			for (j = 0; j < character_size; j++) {
//				*character |= (data()[size + j] << (j * 8));
//			}
//
//			return 0;
//		}
//
//		friend std::ostream& operator<<(std::ostream& os, const utf8_string& str);
//	};
//
//	inline std::ostream& operator<<(std::ostream& os, const utf8_string& str)
//	{
//		os << *(std::string*)&str;
//		return os;
//	}
//
//	struct utf16_string :
//		public std::wstring {
//
//		using std::wstring::basic_string;
//
//		using inherit_t = std::wstring;
//
//		bool open(character_t c) {
//			return utf16_string::push_back(c);
//		}
//
//		bool open(std::string* str) {
//			this->open() = str->data();
//		}
//		bool open()
//		utf16_string(const std::wstring& str) : inherit_t(str) {} // visual studio magics
//
//		utf16_string(uint32_t character) : inherit_t(fan::utf8_to_utf16((uint8_t*)&character).data()) {}
//
//		utf16_string(uint8_t* data) : inherit_t(fan::utf8_to_utf16(data).data()) {}
//
//		utf16_string(const utf8_string& str) : inherit_t(fan::utf8_to_utf16(str.data())) {}
//		utf16_string(const char* str) : inherit_t(fan::utf8_to_utf16((uint8_t*)str)) {}
//
//		bool to_utf8(utf8_string* str) const {
//			utf8_string out;
//			return fan::utf16_to_utf8(this, str);
//		}
//
//		inline bool push_back(character_t character) {
//			inherit_t::push_back(character.c);
//			return 0;
//		}
//
//		friend std::wostream& operator<<(std::wostream& os, const utf16_string& str);
//	};
//
//	inline std::wostream& operator<<(std::wostream& os, const utf16_string& str)
//	{
//		os << (std::wstring)str;
//		return os;
//	}
//
//	// utf8 struct definitions
//
//	inline utf8_string::utf8_string(const utf16_string& str) : utf8_string::inherit_t(fan::utf16_to_utf8(str.data()).data()) {}
//
//	inline utf16_string utf8_string::to_utf16() const {
//		return utf16_string(fan::utf8_to_utf16(this->data()).data());
//	}
//
	static bool utf8_to_utf16(const uint8_t* utf8, std::wstring* out)
	{
		bool error = false;
		fan::hector_t<unsigned long> unicode;
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

	static bool utf16_to_utf8(const wchar_t* utf16, std::string* out) {
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
//
//	using utf8_string_ptr_t = fan::ptr_maker_t<utf8_string>;
//	using utf16_string_ptr_t = fan::ptr_maker_t<utf16_string>;
//
//}