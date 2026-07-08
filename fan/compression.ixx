module;

export module fan.compression;

import std;
import fan.types;
import fan.io.file;
import fan.io.directory;
import fan.io.types;
import fan.memory;
import fan.utility;
import fan.types.fstring;
import fan.print;
import fan.event.types;
import fan.event;

namespace file = fan::io::file;

export namespace fan::fcs {
  inline constexpr std::size_t default_chunk_size = 1uz << 26;
  inline constexpr std::size_t chunk_size = default_chunk_size;

  constexpr void bcj_transform_range(fan::bytes_t& d, std::size_t begin, std::size_t end, bool enc) {
    end = std::min(end, d.size());
    for (std::size_t i = begin; i + 5 <= end; ++i) {
      auto p = d.data() + i;
      auto b = p[0];
      int s = (b == 0xe8 || b == 0xe9) ? 1 :
        (i + 6 <= end && ((b == 0x0f && (p[1] & 0xf0) == 0x80) || (b == 0xff && (p[1] == 0x15 || p[1] == 0x25)))) ? 2 :
        (i + 7 <= end && (b == 0x48 || b == 0x4c) && p[1] == 0x8d && (p[2] & 0xc7) == 0x05) ? 3 : 0;
      if (s) {
        std::uint32_t v; std::memcpy(&v, p + s, 4);
        auto pos = std::uint32_t(i - begin);
        v += enc ? pos : -pos;
        std::memcpy(p + s, &v, 4);
        i += s + 3;
      }
    }
  }

  constexpr void bcj_transform(fan::bytes_t& d, bool enc) { bcj_transform_range(d, 0, d.size(), enc); }

  constexpr void delta_encode(fan::bytes_t& d, int stride) {
    for (std::size_t i = d.size(); i-- > std::size_t(stride);) { d[i] -= d[i - stride]; }
  }
  constexpr void delta_decode(fan::bytes_t& d, int stride) {
    for (std::size_t i = stride; i < d.size(); ++i) { d[i] += d[i - stride]; }
  }

  int detect_delta_stride(const fan::bytes_t& d);

  std::size_t archive_payload_offset(const fan::bytes_t& raw);

  bool looks_like_pe(const fan::bytes_t& d);
  bool looks_like_elf_x86(const fan::bytes_t& d);
  bool looks_like_coff_x86(const fan::bytes_t& d);
  bool looks_like_x86_binary(const fan::bytes_t& d);
  bool looks_like_bcj_data(const fan::bytes_t& d, std::size_t payload_offset);

  inline constexpr std::uint8_t flag_bcj = 1u, flag_delta = 2u, flag_text = 32u, flag_text2 = 64u, flag_rle = 128u, text_marker = 1u;

  inline constexpr std::array<bool, 256> text_char_lut = [] {
    std::array<bool, 256> t{};
    for (int i = 0; i < 256; ++i) { t[i] = (i >= 'a' && i <= 'z') || (i >= 'A' && i <= 'Z') || (i >= '0' && i <= '9') || i == '_' || i == '-'; }
    return t;
  }();
  inline constexpr std::array<bool, 256> text_first_lut = [] {
    std::array<bool, 256> t{};
    for (int i = 0; i < 256; ++i) { t[i] = (i >= 'a' && i <= 'z') || (i >= 'A' && i <= 'Z'); }
    return t;
  }();
  constexpr bool text_word_char(std::uint8_t c) { return text_char_lut[c]; }
  constexpr bool text_word_first(std::uint8_t c) { return text_first_lut[c]; }

  struct text_word_t { std::string_view word; std::uint32_t count = 0, score = 0; };

  void decode_text_head(const fan::bytes_t& in, std::size_t& p, std::uint64_t& out_size, std::vector<std::string>& dict);
  std::size_t skip_word(const fan::bytes_t& raw, std::size_t& i);
  fan::bytes_t text_encode_transform(const fan::bytes_t& raw);
  fan::bytes_t text_decode_transform(const fan::bytes_t& in);

  inline constexpr std::string_view text2_static_tokens[] = {
    "<page>", "</page>", "<title>", "</title>", "<ns>", "</ns>", "<id>", "</id>", "<revision>", "</revision>", "<timestamp>", "</timestamp>", "<contributor>", "</contributor>", "<username>", "</username>", "<ip>", "</ip>", "<comment>", "</comment>", "<text xml:space=\"preserve\">", "</text>", "<minor />", "<sha1>", "</sha1>", "<model>wikitext</model>", "<format>text/x-wiki</format>", "[[", "]]", "{{", "}}", "'''", "''", "==", "||", "|-", "|}", "{|", "&quot;", "&amp;", "&lt;", "&gt;", "&nbsp;", "&ndash;", "&mdash;", "<redirect title=\"", "#REDIRECT", "Category:", "Image:", "File:", "Template:", "http://", "https://", "www.", "ref>", "</ref>", "<br />", "which", "there", "their", "that", "this", "with", "have", "were", "from", ".com", "ing", "ion", "and", "the", " tha", "ent", "ere", " co", "e o", "e a", "e c", "e s", "e t", " th", " t", "in ", "he ", " to", "of ", " of", "for", "you", "not", "all", "was", "one", "our", "ver", "ter", "men", "ati", "ass", "ate", "div", "whi", "who", "but", "his", "her", "hat", "tha", "the ", "they", "are", "res", "com", "con", "per", "pro", "tion", "atio", "ment", "ence", "able", "ould", "ight", "ally", "ally ", " and", " the", " to ", " of ", " in ", " a ", " is ", " as ", " on ", " or ", " by ", " an ", " be ", " re", " wa", " wi", " wh", " ma", " ha", " fo", " cl", " we", "</", "=\"", "ch ", "th ", "#include", "import ", "export ", "module ", "namespace ", "template", "typename", "constexpr", "consteval", "constinit", "inline ", "static ", "struct ", "class ", "public:", "private:", "protected:", "return ", "std::", "std::size_t", "std::uint", "std::vector", "std::array", "std::string", "std::string_view", "std::span", "std::move", "std::forward", "std::min", "std::max", "std::clamp", "std::runtime_error", "std::filesystem", "std::memory_order_relaxed", "fan::", "fan::bytes_t", "std::uint8_t", "std::uint16_t", "std::uint32_t", "std::uint64_t", "std::size_t", "std::memcpy", "std::memset", "if (", "for (", "while (", "else ", "auto ", "bool ", "void ", "true", "false", "nullptr", "throw ", "continue;", "break;", "return;", "push_back", "emplace_back", ".data()", ".size()", ".begin()", ".end()", "operator", "using ", "typedef", "virtual ", "override", "final", "noexcept", "requires", "concept ", "switch (", "case ", "default:", "lambda", "->", "::", "&&", "||", "==", "!=", "<=", ">=", "<<", ">>"
  };

  std::uint8_t text2_lower(std::uint8_t c);
  std::uint8_t text2_upper(std::uint8_t c);
  std::uint8_t text2_case_kind(const std::uint8_t* p, std::size_t n);
  std::string text2_to_lower_word(const std::uint8_t* p, std::size_t n);
  void text2_emit_id(fan::bytes_t& out, std::uint8_t t8, std::uint8_t t16, std::uint32_t id);
  const std::array<std::vector<std::uint8_t>, 256>& text2_token_by_first();
  int text2_static_match(const fan::bytes_t& raw, std::size_t i);
  fan::bytes_t text2_encode_transform(const fan::bytes_t& raw);
  fan::bytes_t text2_decode_transform(const fan::bytes_t& in);

  void rle_write_var(fan::bytes_t& out, std::uint64_t v);
  std::uint64_t rle_read_var(const fan::bytes_t& in, std::size_t& p);
  fan::bytes_t rle_encode_transform(const fan::bytes_t& raw);
  fan::bytes_t rle_decode_transform(const fan::bytes_t& in);

  std::uint32_t get_match_len(const std::uint8_t* p1, const std::uint8_t* p2, std::uint32_t max_l);

  struct match_t { std::uint32_t offset = 0, length = 0; };
  struct match_cache_t { std::uint32_t count = 0; std::array<match_t, 128> matches{}; };

  struct match_finder_t {
    static constexpr std::uint32_t nil = 0xFFFFFFFFu;
    std::vector<std::uint32_t> h2, h3, h4, son;
    std::size_t base = 0;

    match_finder_t(std::size_t d_start, std::size_t c_end);
    static void hashes(const std::uint8_t* p, std::uint32_t& k2, std::uint32_t& k3, std::uint32_t& k4);
    std::uint32_t find_and_insert(const std::uint8_t* src_base, std::size_t i, std::size_t c_end, std::uint32_t max_depth, std::uint32_t nice_len, std::uint32_t max_len, match_t* out);
  };

  struct compress_params_t {
    std::uint32_t chain_main = 64, nice_len = 273;
    bool bcj = true, optimal = true;
    int lazy_depth = 2;
    std::size_t chunk_size = default_chunk_size;
    std::uint32_t opt_window = 1 << 12;
    std::size_t candidate_sample_size = 2uz << 20;
    std::uint32_t candidate_sample_count = 4, commit_limit = 0;
    bool parser_verify = false;
  };

  constexpr compress_params_t params_fast()   { return {32,   64,   true, false, 1, default_chunk_size, 1 << 10, 1uz << 20, 2, 0}; }
  constexpr compress_params_t params_normal() { return {128,  256,  true, true,  0, default_chunk_size, 1 << 11, 2uz << 20, 3, 0}; }
  constexpr compress_params_t params_high()   { return {512,  512,  true, true,  0, default_chunk_size, 1 << 12, 4uz << 20, 4, 0}; }
  constexpr compress_params_t params_max()    { return {2048, 1024, true, true,  0, default_chunk_size, 1 << 13, 8uz << 20, 6, 0}; }
  constexpr compress_params_t params_ultra()  { return {4096, 4111, true, true,  0, default_chunk_size, 1 << 14, 8uz << 20, 6, 0}; }

  inline constexpr std::uint8_t state_lit_next[12]      = {0,0,0,0,1,2,3,4,5,6,4,5};
  inline constexpr std::uint8_t state_match_next[12]    = {7,7,7,7,7,7,7,10,10,10,10,10};
  inline constexpr std::uint8_t state_rep_next[12]      = {8,8,8,8,8,8,8,11,11,11,11,11};
  inline constexpr std::uint8_t state_shortrep_next[12] = {9,9,9,9,9,9,9,11,11,11,11,11};
  inline constexpr bool state_is_lit[12]                = {true,true,true,true,true,true,true,false,false,false,false,false};

  enum class op_e : std::uint8_t { rep0=0, rep1=1, rep2=2, rep3=3, exp=4, none=5 };
  struct seq_t { std::uint32_t lit_len; op_e op; std::uint32_t match_len, offset; };
  struct chunk_payload_t { std::vector<seq_t> seqs; };

  inline constexpr std::uint32_t magic_v6 = 0x36334346;
  inline constexpr int num_pos_states = 4, lc = 8, lp = 0, num_lit_ctx = 1 << (lc + lp);

  constexpr std::uint32_t lit_ctx(std::size_t pos, std::uint8_t prev) {
    return ((std::uint32_t(pos) & ((1u << lp) - 1)) << lc) | (prev >> (8 - lc));
  }

  struct lzma_model_t {
    std::array<std::array<std::uint16_t, num_pos_states>, 12> is_match, is_rep0_short;
    std::array<std::uint16_t, 12> is_rep, is_rep0, is_rep1, is_rep2;
    std::array<std::array<std::array<std::uint16_t, 512>, num_lit_ctx>, 2> lit;
    struct len_model_t {
      std::array<std::array<std::uint16_t, num_pos_states>, 1> choice, choice2;
      std::array<std::array<std::uint16_t, 8>, num_pos_states> low, mid;
      std::array<std::uint16_t, 4096> high;
    };
    len_model_t match_len, rep_len;
    std::array<std::array<std::uint16_t, 64>, 4> pos_slot;
    std::array<std::array<std::uint16_t, 32>, 10> pos_special;
    std::array<std::uint16_t, 16> align_bits;

    lzma_model_t() {
      auto fill = [](this const auto& self, auto& arr) -> void {
        for (auto& e : arr) {
          if constexpr (std::is_same_v<std::remove_reference_t<decltype(e)>, std::uint16_t>) { e = 1024; }
          else { self(e); }
        }
      };
      fill(is_match); fill(is_rep); fill(is_rep0); fill(is_rep1); fill(is_rep2); fill(is_rep0_short); fill(lit);
      fill(match_len.choice); fill(match_len.choice2); fill(match_len.low); fill(match_len.mid); fill(match_len.high);
      fill(rep_len.choice); fill(rep_len.choice2); fill(rep_len.low); fill(rep_len.mid); fill(rep_len.high);
      fill(pos_slot); fill(pos_special); fill(align_bits);
    }
  };

  inline std::uint32_t slot_for_dist(std::uint32_t d) {
    if (d < 4) { return d; }
    std::uint32_t k = std::bit_width(d) - 1; return (k << 1) | ((d >> (k-1)) & 1);
  }

  template <typename Writer>
  struct range_enc_t {
    Writer& out;
    std::uint64_t low = 0; std::uint32_t range = -1u, cache_size = 1; std::uint8_t cache = 0;
    void shift_low() {
      if (std::uint32_t(low) < 0xFF000000 || int(low >> 32) != 0) {
        out.write_byte(cache + std::uint8_t(low >> 32));
        out.write_repeat(std::uint8_t(low >> 32) ? 0x00 : 0xFF, cache_size - 1);
        cache = std::uint8_t(low >> 24); cache_size = 1;
      } else { ++cache_size; }
      low = std::uint32_t(low << 8);
    }
    void flush() { for (int i = 0; i < 5; ++i) { shift_low(); } }
    void encode(std::uint16_t& prob, std::uint32_t bit) {
      std::uint32_t bound = (range >> 11) * prob;
      if (!bit) { range = bound; prob += (2048 - prob) >> 5; }
      else { low += bound; range -= bound; prob -= prob >> 5; }
      while (range < 0x1000000) { range <<= 8; shift_low(); }
    }
    void encode_tree(std::uint16_t* tree, std::uint32_t sym, int bits) {
      for (int ctx = 1, i = bits - 1; i >= 0; --i) { std::uint32_t bit = (sym >> i) & 1; encode(tree[ctx], bit); ctx = (ctx << 1) | bit; }
    }
    void encode_direct(std::uint32_t bit) {
      range >>= 1; if (bit) { low += range; }
      while (range < 0x1000000) { range <<= 8; shift_low(); }
    }
    void encode_direct_n(std::uint32_t val, int bits) {
      for (int i = bits - 1; i >= 0; --i) { encode_direct((val >> i) & 1); }
    }
  };

  template <typename Reader>
  struct range_dec_t {
    Reader& in;
    std::uint32_t range = -1u, code = 0;
    range_dec_t(Reader& r) : in(r) { read_byte(); for (int i = 0; i < 4; ++i) { code = (code << 8) | read_byte(); } }
    std::uint8_t read_byte() { return in.read_byte(); }
    bool decode(std::uint16_t& prob) {
      std::uint32_t bound = (range >> 11) * prob; bool bit = code >= bound;
      if (!bit) { range = bound; prob += (2048 - prob) >> 5; } else { range -= bound; code -= bound; prob -= prob >> 5; }
      while (range < 0x1000000) { range <<= 8; code = (code << 8) | read_byte(); }
      return bit;
    }
    std::uint32_t decode_tree(std::uint16_t* tree, int bits) {
      std::uint32_t ctx = 1; for (int i = 0; i < bits; ++i) { ctx = (ctx << 1) | decode(tree[ctx]); } return ctx - (1u << bits);
    }
    bool decode_direct() {
      range >>= 1; bool bit = code >= range; if (bit) { code -= range; }
      while (range < 0x1000000) { range <<= 8; code = (code << 8) | read_byte(); }
      return bit;
    }
    std::uint32_t decode_direct_n(int bits) {
      std::uint32_t v = 0; for (int i = 0; i < bits; ++i) { v = (v << 1) | decode_direct(); } return v;
    }
  };

  template <typename Writer>
  inline void encode_len(range_enc_t<Writer>& rc, lzma_model_t::len_model_t& lm, std::uint32_t len, int ps) {
    if (len < 8) { rc.encode(lm.choice[0][ps], 0); rc.encode_tree(lm.low[ps].data(), len, 3); }
    else if (len < 16) { rc.encode(lm.choice[0][ps], 1); rc.encode(lm.choice2[0][ps], 0); rc.encode_tree(lm.mid[ps].data(), len - 8, 3); }
    else { rc.encode(lm.choice[0][ps], 1); rc.encode(lm.choice2[0][ps], 1); rc.encode_tree(lm.high.data(), len - 16, 12); }
  }

  template <typename Reader>
  inline std::uint32_t decode_len(range_dec_t<Reader>& rd, lzma_model_t::len_model_t& lm, int ps) {
    if (!rd.decode(lm.choice[0][ps])) { return rd.decode_tree(lm.low[ps].data(), 3); }
    if (!rd.decode(lm.choice2[0][ps])) { return 8 + rd.decode_tree(lm.mid[ps].data(), 3); }
    return 16 + rd.decode_tree(lm.high.data(), 12);
  }

  template <typename Writer>
  inline void encode_dist(range_enc_t<Writer>& rc, lzma_model_t& m, std::uint32_t dist, std::uint32_t len_state) {
    std::uint32_t d = dist - 1, slot = slot_for_dist(d); rc.encode_tree(m.pos_slot[len_state].data(), slot, 6);
    if (slot >= 4) {
      std::uint32_t footer_bits = (slot >> 1) - 1, footer = d - ((2u | (slot & 1u)) << footer_bits);
      if (slot < 14) { rc.encode_tree(m.pos_special[slot - 4].data(), footer, footer_bits); }
      else { rc.encode_direct_n(footer >> 4, footer_bits - 4); rc.encode_tree(m.align_bits.data(), footer & 0xFu, 4); }
    }
  }

  template <typename Reader>
  inline std::uint32_t decode_dist(range_dec_t<Reader>& rd, lzma_model_t& m, std::uint32_t len_state) {
    std::uint32_t slot = rd.decode_tree(m.pos_slot[len_state].data(), 6);
    if (slot < 4) { return slot + 1; }
    std::uint32_t footer_bits = (slot >> 1) - 1, base_val = (2u | (slot & 1u)) << footer_bits, footer;
    if (slot < 14) { footer = rd.decode_tree(m.pos_special[slot - 4].data(), footer_bits); }
    else { footer = (rd.decode_direct_n(footer_bits - 4) << 4) | rd.decode_tree(m.align_bits.data(), 4); }
    return base_val + footer + 1;
  }

  template <typename Writer>
  inline void encode_literal(range_enc_t<Writer>& rc, lzma_model_t& m, std::uint8_t state, std::size_t pos, std::uint8_t prev, std::uint8_t byte, std::uint8_t match_byte) {
    int sc = state_is_lit[state] ? 0 : 1; auto* tree = m.lit[sc][lit_ctx(pos, prev)].data();
    if (sc == 0) { rc.encode_tree(tree, byte, 8); }
    else {
      for (int i = 7, sym = 1; i >= 0; --i) {
        std::uint32_t bit = (byte >> i) & 1, mb_bit = (match_byte >> 7) & 1; match_byte <<= 1;
        rc.encode(tree[sym + (mb_bit ? 256 : 0)], bit); sym = (sym << 1) | bit;
      }
    }
  }

  template <typename Reader>
  inline std::uint8_t decode_literal(range_dec_t<Reader>& rd, lzma_model_t& m, std::uint8_t state, std::size_t pos, std::uint8_t prev, std::uint8_t match_byte) {
    int sc = state_is_lit[state] ? 0 : 1; auto* tree = m.lit[sc][lit_ctx(pos, prev)].data();
    if (sc == 0) { return std::uint8_t(rd.decode_tree(tree, 8)); }
    std::uint32_t sym = 1;
    for (int i = 7; i >= 0; --i) {
      std::uint32_t mb_bit = (match_byte >> 7) & 1; match_byte <<= 1;
      sym = (sym << 1) | rd.decode(tree[sym + (mb_bit ? 256 : 0)]);
    }
    return std::uint8_t(sym & 0xFF);
  }

  inline constexpr std::uint32_t max_opt_window = 1 << 14, max_exp_len = 4111, max_rep_len = 4110, progress_step = 512, inf_price = 0xFFFFFFFFu;
  inline constexpr std::uint64_t parse_progress_weight = 80, encode_progress_weight = 20, progress_scale = 100;

  namespace detail {
    inline std::array<std::uint32_t, 2049> init_price_table() {
      std::array<std::uint32_t, 2049> t{};
      for (int i = 1; i <= 2048; ++i) { t[i] = std::uint32_t(-std::log2(double(i) / 2048.0) * 64.0 + 0.5); }
      t[0] = t[1]; return t;
    }
    inline const auto price_table = init_price_table();
  }

  inline std::uint32_t bit_price(std::uint16_t prob, std::uint32_t bit) { return detail::price_table[bit ? 2048 - prob : prob]; }

  template <int bits>
  inline std::uint32_t tree_price_v(const std::uint16_t* tree, std::uint32_t sym) {
    std::uint32_t cost = 0, ctx = 1;
    if constexpr (bits == 3) {
      std::uint32_t b2 = (sym >> 2) & 1; cost += bit_price(tree[ctx], b2); ctx = (ctx << 1) | b2;
      std::uint32_t b1 = (sym >> 1) & 1; cost += bit_price(tree[ctx], b1); ctx = (ctx << 1) | b1;
      std::uint32_t b0 = sym & 1;        cost += bit_price(tree[ctx], b0);
    } else if constexpr (bits == 4 || bits == 6 || bits == 8) {
      for (int i = bits - 1; i >= 0; --i) { std::uint32_t bit = (sym >> i) & 1; cost += bit_price(tree[ctx], bit); ctx = (ctx << 1) | bit; }
    }
    return cost;
  }

  std::uint32_t tree_price(const std::uint16_t* tree, std::uint32_t sym, int bits);
  std::uint32_t len_price(const lzma_model_t::len_model_t& lm, std::uint32_t len, int ps);
  std::uint32_t dist_price(const lzma_model_t& m, std::uint32_t dist, std::uint32_t len_state);

  inline void update_bit(std::uint16_t& prob, std::uint32_t bit) {
    std::uint32_t p = prob; prob = std::uint16_t(bit ? p - (p >> 5) : p + ((2048 - p) >> 5));
  }

  template <int bits>
  inline void update_tree_v(std::uint16_t* tree, std::uint32_t sym) {
    std::uint32_t ctx = 1;
    for (int i = bits - 1; i >= 0; --i) { std::uint32_t bit = (sym >> i) & 1; update_bit(tree[ctx], bit); ctx = (ctx << 1) | bit; }
  }

  void update_tree(std::uint16_t* tree, std::uint32_t sym, int bits);
  void update_len(lzma_model_t::len_model_t& lm, std::uint32_t len, int ps);
  void update_dist(lzma_model_t& m, std::uint32_t dist, std::uint32_t len_state);
  void update_literal_symbol(lzma_model_t& m, std::uint8_t state, std::size_t pos, std::uint8_t prev, std::uint8_t byte, std::uint8_t match_byte);
  void update_match_symbol(lzma_model_t& m, std::uint8_t state, std::size_t pos, op_e op, std::uint32_t len, std::uint32_t dist);

  struct opt_t {
    std::uint32_t price = inf_price, prev = 0, len = 1, offset = 0;
    op_e op = op_e::none; std::array<std::uint32_t, 4> rep{}; std::uint8_t state = 0;
  };

  struct parse_price_cache_t {
    std::array<std::array<std::uint32_t, max_exp_len - 1>, num_pos_states> match_len{};
    std::array<std::array<std::uint32_t, max_rep_len>, num_pos_states> rep_len{};
    std::array<std::array<std::uint32_t, 64>, 4> pos_slot{};
    std::array<std::array<std::uint32_t, 32>, 10> pos_special_price{};
    std::array<std::uint32_t, 16> align_price{};
    void refresh(lzma_model_t& m);
    std::uint32_t dist_price(lzma_model_t&, std::uint32_t dist, std::uint32_t len_state) const;
  };

  constexpr void shift_rep(int r, std::uint32_t off, std::array<std::uint32_t, 4>& rp) {
    if (r == 1) { std::swap(rp[0], rp[1]); }
    else if (r == 2) { auto t=rp[2]; rp[2]=rp[1]; rp[1]=rp[0]; rp[0]=t; }
    else if (r == 3) { auto t=rp[3]; rp[3]=rp[2]; rp[2]=rp[1]; rp[1]=rp[0]; rp[0]=t; }
    else if (r == 4) { rp[3]=rp[2]; rp[2]=rp[1]; rp[1]=rp[0]; rp[0]=off; }
  }

  chunk_payload_t parse_chunk_optimal(const fan::bytes_t& src, std::size_t c_start, std::size_t c_end, const compress_params_t& params, std::function<void(std::uint64_t)> set_prog);

  fan::bytes_t encode_stream_seq(const fan::bytes_t& src, const std::vector<chunk_payload_t>& blocks, std::size_t chunk_size, std::function<void(std::uint64_t)> add_prog);

  fan::bytes_t decode_stream_seq(const fan::bytes_t& comp, std::size_t idx, std::uint64_t total_uncomp, std::size_t chunk_size, fan::progress_t* prog);

  std::size_t run_compress_core(const fan::bytes_t& raw, compress_params_t params, fan::progress_t* user_prog, std::size_t thread_count, fan::bytes_t& out_comp);

  struct compress_candidate_t {
    fan::bytes_t data, comp; std::string_view name; std::uint8_t flags = 0;
    std::size_t chunk_size = default_chunk_size;
  };

  void add_transform_candidates(std::vector<compress_candidate_t>& candidates, fan::bytes_t&& raw, std::size_t payload_offset, bool can_bcj, bool can_text, compress_params_t params);

  std::size_t quick_test_candidate(const fan::bytes_t& data, compress_params_t params);

  struct full_param_candidate_t { std::string_view name; compress_params_t params; };

  std::vector<full_param_candidate_t> make_full_param_candidates(compress_params_t params);

  compress_candidate_t compress_best_candidate(std::vector<compress_candidate_t>& candidates, compress_params_t params, fan::progress_t* prog, std::size_t thread_count, bool verbose = false);

  bool compress_path_to_file(const std::filesystem::path& in, const std::filesystem::path& out_path, compress_params_t params = params_max(), fan::progress_t* prog = nullptr, bool verbose = false, std::size_t thread_count = 0);

  bool decompress_file_to_dir(const std::filesystem::path& in_path, const std::filesystem::path& out_dir, bool default_out, fan::progress_t* prog = nullptr);

  fan::bytes_t compress(const std::vector<fan::io::file_buffer_t>& files, compress_params_t params = params_max(), fan::progress_t* prog = nullptr);

  std::vector<fan::io::file_buffer_t> decompress(const fan::bytes_t& comp, fan::progress_t* prog = nullptr);

  fan::event::waitv_t<fan::bytes_result_t> compress_on_thread(
    std::string path,
    fan::bytes_t data,
    const std::source_location caller = std::source_location::current()
  );

  fan::event::waitv_t<fan::bytes_result_t> decompress_on_thread(
    std::string path,
    const std::source_location caller = std::source_location::current()
  );
}