struct fed_t {
	static constexpr uint32_t character_width_multiplier = 100;

	struct properties_t {
		uint32_t line_height = -1;
		uint32_t line_width = -1;
		uint32_t line_limit = -1;
		uint32_t line_character_limit = -1;
		f32_t font_size;
	};

	void open(const properties_t& p) {
		m_wed.open(p.line_height, p.line_width, p.line_limit, p.line_character_limit);
		m_cr = m_wed.cursor_open();
		m_font_size = p.font_size;
	}

	void close() {
		m_wed.close();
	}

	void push_text(const fan::string& str) {
		for (const auto& i : str) {
			add_character(i);
		}
	}

  void set_text(const fan::string& str) {
    wed_t::CursorInformation_t ci;
    m_wed.GetCursorInformation(m_cr, &ci);
    clear_text(ci.FreeStyle.LineReference);
		for (const auto& i : str) {
			add_character(i);
		}
	}

	void add_character(wed_t::CharacterData_t character) {
    auto found = gloco->font.info.characters.find(character);
    if (found == gloco->font.info.characters.end()) {
      return;
    }
    //m_wed.set
		auto letter = gloco->font.info.get_letter_info(character, m_font_size);
		m_wed.AddCharacterToCursor(m_cr, character, letter.metrics.advance * character_width_multiplier);
	}

  void set_font_size(wed_t::LineReference_t line_id, f32_t font_size) {
    bool smaller = font_size < m_font_size;
    m_font_size = font_size;
    for (wed_t::li_t li(&m_wed); li(); li.it()) {
      for (wed_t::ci_t ci(&m_wed, li.id); ci(); ci.it()) {
        wed_t::CharacterData_t* cd = m_wed.GetDataOfCharacter(li.id, ci.id);
        auto letter = gloco->font.info.get_letter_info(*cd, m_font_size);
        m_wed.SetCharacterWidth_Silent(li.id, ci.id, letter.metrics.advance * character_width_multiplier);
      }
    }
    m_wed.NowAllCharacterSizesAre(smaller);
  }

  void clear_text(wed_t::LineReference_t line_id) {
    auto cursor_info = freestyle_get_cursor_position().FreeStyle;
    freestyle_move_line_end();
    auto text = get_text(line_id);
    for (const auto& i : text) {
      freestyle_erase_character();
    }
    //m_wed.
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

  wed_t::CursorInformation_t freestyle_get_cursor_position() {
    wed_t::CursorInformation_t cursor_info;
    m_wed.GetCursorInformation(m_cr, &cursor_info);
    return cursor_info;
  }

	void set_mouse_position(const fan::vec2& src, const fan::vec2& dst) {
		wed_t::LineReference_t FirstLineReference = m_wed.GetFirstLineID();
		wed_t::LineReference_t LineReference0, LineReference1;
		wed_t::CharacterReference_t CharacterReference0, CharacterReference1;
		m_wed.GetLineAndCharacter(FirstLineReference, src.y, src.x * character_width_multiplier, &LineReference0, &CharacterReference0);
		m_wed.GetLineAndCharacter(FirstLineReference, dst.y, dst.x * character_width_multiplier, &LineReference1, &CharacterReference1);
		m_wed.ConvertCursorToSelection(m_cr, LineReference0, CharacterReference0, LineReference1, CharacterReference1);
	}

	fan::string get_text(wed_t::LineReference_t line_id) {
		wed_t::ExportLine_t el;
		m_wed.ExportLine_init(&el, line_id);
		wed_t::CharacterReference_t cr;
		fan::string text;
		while (m_wed.ExportLine(&el, &cr)) {
      uint32_t c = *m_wed.GetDataOfCharacter(line_id, cr);

      if (c > 0xffff) {
        text.push_back((c & 0xff0000) >> 16);
      }
      if (c > 0xff) {
        text.push_back((c & 0xff00) >> 8);
      }
      text.push_back(c & 0xff);
		}
		return text;
	}

	wed_t::CursorReference_t m_cr;
	wed_t m_wed;
	f32_t m_font_size;
};