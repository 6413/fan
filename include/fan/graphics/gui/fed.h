struct fed_t {
	static constexpr uint32_t character_width_multiplier = 100;

	struct properties_t {
		uint32_t line_height = -1;
		uint32_t line_width = -1;
		uint32_t line_limit = -1;
		uint32_t line_character_limit = -1;
		f32_t font_size;
		loco_t* loco = nullptr;
	};

	void open(const properties_t& p) {
		#if fan_debug >= fan_debug_medium
			if (p.loco == nullptr) {
				fan::throw_error("set loco in fed");
			}
		#endif
		m_wed.open(p.line_height, p.line_width, p.line_limit, p.line_character_limit);
		m_cr = m_wed.cursor_open();
		m_font_size = p.font_size;
		loco = p.loco;
	}

	void close() {
		m_wed.close();
	}

	void push_text(const fan::wstring& str) {
		for (const auto& i : str) {
			add_character(i);
		}
	}

	void add_character(wed_t::CharacterData_t character) {
		auto found = loco->font.decode_letter(character);

		if (found == -1) {
			return;
		}

		auto letter = loco->font.info.get_letter_info(character, m_font_size);

		m_wed.AddCharacterToCursor(m_cr, character, letter.metrics.advance * character_width_multiplier);
	}

	void freestyle_erase_character() {
		m_wed.DeleteCharacterFromCursor(m_cr);
	}
	void freestyle_erase_character_right() {
		m_wed.DeleteCharacterFromCursorRight(m_cr);
	}
	void freestyle_move_line_begin() {
		m_wed.MoveCursorFreeStyleToBeginOfLine(m_cr);
	}
	void freestyle_move_line_end() {
		m_wed.MoveCursorFreeStyleToEndOfLine(m_cr);
	}
	void freestyle_move_left() {
		m_wed.MoveCursorFreeStyleToLeft(m_cr);
	}
	void freestyle_move_right() {
		m_wed.MoveCursorFreeStyleToRight(m_cr);
	}

	fan::wstring get_text(wed_t::LineReference_t line_id) {
		wed_t::ExportLine_t el;
		m_wed.ExportLine_init(&el, line_id);
		wed_t::CharacterReference_t cr;
		fan::wstring text;
		while (m_wed.ExportLine(&el, &cr)) {
			text.push_back(*m_wed.GetDataOfCharacter(line_id, cr));
		}
		return text;
	}

	wed_t::CursorReference_t m_cr;
	wed_t m_wed;
	f32_t m_font_size;
	loco_t* loco;
};