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

namespace file = fan::io::file;

export namespace fan::fcs {
  inline constexpr std::size_t default_chunk_size = 1uz << 26; // 64 MiB
  inline constexpr std::size_t chunk_size = default_chunk_size;

  constexpr void bcj_transform_range(fan::bytes_t& d, std::size_t begin, std::size_t end, bool enc) {
    end = std::min(end, d.size());
    for (std::size_t i = begin; i + 5 <= end; ++i) {
      auto p = d.data() + i;
      auto b = p[0];
      int s = (b == 0xe8 || b == 0xe9) ? 1 :
        (i + 6 <= end && ((b == 0x0f && (p[1] & 0xf0) == 0x80) || (b == 0xff && (p[1] == 0x15 || p[1] == 0x25)))) ? 2 :
        (i + 7 <= end && b == 0x48 && p[1] == 0x8d && (p[2] & 0xc7) == 0x05) ? 3 : 0;
      if (s) {
        std::uint32_t v; std::memcpy(&v, p + s, 4);
        auto pos = std::uint32_t(i - begin);
        v += enc ? pos : -pos;
        std::memcpy(p + s, &v, 4);
        i += s + 3;
      }
    }
  }

  constexpr void bcj_transform(fan::bytes_t& d, bool enc) {
    bcj_transform_range(d, 0, d.size(), enc);
  }

  constexpr void delta_encode(fan::bytes_t& d, int stride) {
    for (std::size_t i = d.size(); i-- > std::size_t(stride);) { d[i] -= d[i - stride]; }
  }
  constexpr void delta_decode(fan::bytes_t& d, int stride) {
    for (std::size_t i = stride; i < d.size(); ++i) { d[i] += d[i - stride]; }
  }
  inline int detect_delta_stride(const fan::bytes_t& d) {
    std::size_t probe = std::min<std::size_t>(d.size(), 65536);
    int best_stride = 1;
    double best_ent = 1e18, base_ent = 1e18;
    for (int s : {1, 2, 4, 8}) {
      std::array<int, 256> freq {}; std::size_t n = 0;
      for (std::size_t i = s; i < probe; ++i) { ++freq[std::uint8_t(d[i] - d[i - s])]; ++n; }
      double ent = 0;
      for (int c : freq) { if (c) { double p = double(c) / n; ent -= p * std::log2(p); } }
      if (s == 1) { base_ent = ent; }
      if (ent < best_ent) { best_ent = ent; best_stride = s; }
    }
    return best_stride != 1 && best_ent + 0.05 < base_ent ? best_stride : 1;
  }

  inline std::size_t archive_payload_offset(const fan::bytes_t& raw) {
    if (raw.size() < 4) { return 0; }
    std::size_t out_idx = 4;
    std::uint32_t num_files = fan::memory::read_le32(raw.data());
    for (std::uint32_t i = 0; i < num_files; ++i) {
      if (out_idx + 2 > raw.size()) { return raw.size(); }
      std::uint16_t path_len = fan::memory::read_le16(raw.data() + out_idx); out_idx += 2;
      if (out_idx + path_len + 8 > raw.size()) { return raw.size(); }
      out_idx += path_len + 8;
    }
    out_idx += (4 - (out_idx & 3)) & 3;
    return std::min(out_idx, raw.size());
  }

  inline constexpr std::uint8_t flag_bcj = 1u;
  inline constexpr std::uint8_t flag_delta = 2u;
  inline constexpr std::uint8_t flag_text = 32u;
  inline constexpr std::uint8_t flag_text2 = 64u;
  inline constexpr std::uint8_t text_marker = 1u;

  constexpr bool text_word_char(std::uint8_t c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-';
  }

  constexpr bool text_word_first(std::uint8_t c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
  }

  inline bool looks_like_large_text(const fan::bytes_t& raw, std::size_t payload_offset) {
    std::size_t begin = std::min(payload_offset, raw.size());
    std::size_t n = std::min<std::size_t>(raw.size() - begin, 1uz << 20);
    if (n < (1uz << 20) || raw.size() < (64uz << 20)) { return false; }
    std::size_t printable = 0, letters = 0, zeros = 0;
    for (std::size_t i = 0; i < n; ++i) {
      std::uint8_t c = raw[begin + i];
      printable += (c == 9 || c == 10 || c == 13 || (c >= 32 && c < 127));
      letters += ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'));
      zeros += c == 0;
    }
    return zeros == 0 && printable * 100 >= n * 93 && letters * 100 >= n * 35;
  }

  struct text_word_t {
    std::string word;
    std::uint32_t count = 0;
    std::uint32_t score = 0;
  };

  inline void decode_text_head(const fan::bytes_t& in, std::size_t& p, std::uint64_t& out_size, std::vector<std::string>& dict) {
    if (in.size() < 10) { throw std::runtime_error("corrupt text transform"); }
    out_size = fan::memory::read_le64(in.data()); p += 8;
    std::uint16_t dict_count = fan::memory::read_le16(in.data() + p); p += 2;
    dict.reserve(dict_count);
    for (std::uint32_t i = 0; i < dict_count; ++i) {
      if (p >= in.size()) { throw std::runtime_error("corrupt text transform"); }
      std::uint8_t len = in[p++];
      if (p + len > in.size()) { throw std::runtime_error("corrupt text transform"); }
      dict.emplace_back(reinterpret_cast<const char*>(in.data() + p), len);
      p += len;
    }
  }

  inline std::size_t skip_word(const fan::bytes_t& raw, std::size_t& i) {
    std::size_t b = i++;
    while (i < raw.size() && text_word_char(raw[i]) && i - b < 31) { ++i; }
    while (i < raw.size() && text_word_char(raw[i])) { ++i; }
    return b;
  }

  inline fan::bytes_t text_encode_transform(const fan::bytes_t& raw) {
    std::unordered_map<std::string, std::uint32_t> counts;
    counts.reserve(1 << 16);
    for (std::size_t i = 0; i < raw.size();) {
      if (!text_word_first(raw[i])) { ++i; continue; }
      std::size_t b = skip_word(raw, i);
      std::size_t len = i - b;
      if (len < 4 || len > 31) { continue; }
      ++counts[std::string(reinterpret_cast<const char*>(raw.data() + b), len)];
    }

    std::vector<text_word_t> words;
    words.reserve(std::min<std::size_t>(counts.size(), 8192));
    for (auto& [w, c] : counts) {
      if (c < 4) { continue; }
      std::uint32_t score = std::uint32_t(c * (w.size() > 3 ? w.size() - 3 : 0));
      if (score > w.size() + 16) { words.push_back({std::move(w), c, score}); }
    }
    std::sort(words.begin(), words.end(), [](const auto& a, const auto& b) {
      return a.score != b.score ? a.score > b.score : a.word < b.word;
    });

    std::vector<std::string> dict;
    dict.reserve(4096);
    for (auto& w : words) {
      std::size_t id = dict.size();
      std::size_t repl = id < 256 ? 3 : 4;
      std::size_t saved = w.count * (w.word.size() - repl);
      if (w.word.size() <= repl || saved <= w.word.size() + 8) { continue; }
      dict.push_back(std::move(w.word));
      if (dict.size() == 4096) { break; }
    }
    if (dict.empty()) { return {}; }

    std::unordered_map<std::string_view, std::uint32_t> ids;
    ids.reserve(dict.size() * 2);
    for (std::uint32_t i = 0; i < dict.size(); ++i) { ids.emplace(dict[i], i); }

    fan::bytes_t out;
    out.reserve(raw.size() * 9 / 10);
    std::uint8_t buf[8], dc[2];
    fan::memory::write_le64(buf, raw.size());
    out.insert(out.end(), buf, buf + 8);
    fan::memory::write_le16(dc, std::uint16_t(dict.size()));
    out.insert(out.end(), dc, dc + 2);
    for (const auto& w : dict) {
      out.push_back(std::uint8_t(w.size()));
      out.insert(out.end(), w.begin(), w.end());
    }

    for (std::size_t i = 0; i < raw.size();) {
      if (raw[i] == text_marker) { out.push_back(text_marker); out.push_back(0); ++i; continue; }
      if (!text_word_first(raw[i])) { out.push_back(raw[i++]); continue; }
      std::size_t b = skip_word(raw, i);
      std::string_view w(reinterpret_cast<const char*>(raw.data() + b), i - b);
      auto it = ids.find(w);
      if (it == ids.end()) {
        out.insert(out.end(), raw.begin() + b, raw.begin() + i);
        continue;
      }
      out.push_back(text_marker);
      if (it->second < 256) { out.push_back(1); out.push_back(std::uint8_t(it->second)); }
      else { out.push_back(2); out.push_back(std::uint8_t(it->second)); out.push_back(std::uint8_t(it->second >> 8)); }
    }
    return out.size() + raw.size() / 64 < raw.size() ? std::move(out) : fan::bytes_t{};
  }

  inline fan::bytes_t text_decode_transform(const fan::bytes_t& in) {
    std::size_t p = 0; std::uint64_t out_size = 0; std::vector<std::string> dict;
    decode_text_head(in, p, out_size, dict);

    fan::bytes_t out;
    out.reserve(std::size_t(out_size));
    while (p < in.size()) {
      std::uint8_t c = in[p++];
      if (c != text_marker) { out.push_back(c); continue; }
      if (p >= in.size()) { throw std::runtime_error("corrupt text transform"); }
      std::uint8_t t = in[p++];
      if (t == 0) { out.push_back(text_marker); }
      else if (t == 1 || t == 2) {
        if (p + (t - 1) >= in.size()) { throw std::runtime_error("corrupt text transform"); }
        std::uint32_t id = in[p++];
        if (t == 2) { id |= std::uint32_t(in[p++]) << 8; }
        if (id >= dict.size()) { throw std::runtime_error("corrupt text transform"); }
        out.insert(out.end(), dict[id].begin(), dict[id].end());
      } else { throw std::runtime_error("corrupt text transform"); }
    }
    if (out.size() != out_size) { throw std::runtime_error("corrupt text transform"); }
    return out;
  }

  inline constexpr std::string_view text2_static_tokens[] = {
    "<page>", "</page>", "<title>", "</title>", "<ns>", "</ns>", "<id>", "</id>",
    "<revision>", "</revision>", "<timestamp>", "</timestamp>", "<contributor>", "</contributor>",
    "<username>", "</username>", "<ip>", "</ip>", "<comment>", "</comment>",
    "<text xml:space=\"preserve\">", "</text>", "<minor />", "<sha1>", "</sha1>",
    "<model>wikitext</model>", "<format>text/x-wiki</format>",
    "[[", "]]", "{{", "}}", "'''", "''", "==", "||", "|-", "|}", "{|",
    "&quot;", "&amp;", "&lt;", "&gt;", "&nbsp;", "&ndash;", "&mdash;",
    "<redirect title=\"", "#REDIRECT", "Category:", "Image:", "File:", "Template:",
    "http://", "https://", "www.", "ref>", "</ref>", "<br />",
    "which", "there", "their", "that", "this", "with", "have", "were", "from", ".com", "ing", "ion", "and", "the", " tha", "ent", "ere", " co", "e o", "e a", "e c", "e s", "e t", " th", " t", "in ", "he ", " to", "of ", " of", "for", "you", "not", "all", "was", "one", "our", "ver", "ter", "men", "ati", "ass", "ate", "div", "whi", "who", "but", "his", "her", "hat", "tha", "the ", "they", "are", "res", "com", "con", "per", "pro", "tion", "atio", "ment", "ence", "able", "ould", "ight", "ally", "ally ", " and", " the", " to ", " of ", " in ", " a ", " is ", " as ", " on ", " or ", " by ", " an ", " be ", " re", " wa", " wi", " wh", " ma", " ha", " fo", " cl", " we", "</", "=\"", "ch ", "th "
  };

  inline std::uint8_t text2_lower(std::uint8_t c) { return c >= 'A' && c <= 'Z' ? std::uint8_t(c + ('a' - 'A')) : c; }
  inline std::uint8_t text2_upper(std::uint8_t c) { return c >= 'a' && c <= 'z' ? std::uint8_t(c - ('a' - 'A')) : c; }

  inline std::uint8_t text2_case_kind(const std::uint8_t* p, std::size_t n) {
    bool has_alpha = false, has_lower = false, has_upper = false;
    for (std::size_t i = 0; i < n; ++i) {
      if (p[i] >= 'a' && p[i] <= 'z') { has_alpha = has_lower = true; }
      else if (p[i] >= 'A' && p[i] <= 'Z') { has_alpha = has_upper = true; }
    }
    if (!has_alpha || (has_lower && !has_upper)) { return 0; }
    if (has_upper && !has_lower) { return 2; }
    if (p[0] >= 'A' && p[0] <= 'Z') {
      for (std::size_t i = 1; i < n; ++i) { if (p[i] >= 'A' && p[i] <= 'Z') { return 3; } }
      return 1;
    }
    return 3;
  }

  inline std::string text2_to_lower_word(const std::uint8_t* p, std::size_t n) {
    std::string s(n, 0);
    for (std::size_t i = 0; i < n; ++i) { s[i] = char(text2_lower(p[i])); }
    return s;
  }

  inline void text2_emit_id(fan::bytes_t& out, std::uint8_t t8, std::uint8_t t16, std::uint32_t id) {
    out.push_back(text_marker);
    if (id < 256) { out.push_back(t8); out.push_back(std::uint8_t(id)); }
    else { out.push_back(t16); out.push_back(std::uint8_t(id)); out.push_back(std::uint8_t(id >> 8)); }
  }

  inline int text2_static_match(const fan::bytes_t& raw, std::size_t i) {
    switch (raw[i]) {
      case '<': case '[': case '{': case '&': case '#': case 'C': case 'I': case 'F': case 'T':
      case 'a': case 'b': case 'c': case 'd': case 'e': case 'f': case 'g': case 'h':
      case 'i': case 'm': case 'n': case 'o': case 'p': case 'r': case 's': case 't':
      case 'u': case 'v': case 'w': case 'y': case '|': case '=': case '\'': break;
      default: return -1;
    }
    int best = -1; std::size_t best_len = 0;
    for (std::size_t k = 0; k < std::size(text2_static_tokens); ++k) {
      auto tok = text2_static_tokens[k];
      if (tok.empty() || raw[i] != std::uint8_t(tok[0]) || i + tok.size() > raw.size()) { continue; }
      if (tok.size() >= (k < 256 ? 3 : 4) && tok.size() > best_len && std::memcmp(raw.data() + i, tok.data(), tok.size()) == 0) {
        best = int(k); best_len = tok.size();
      }
    }
    return best;
  }

  inline fan::bytes_t text2_encode_transform(const fan::bytes_t& raw) {
    std::unordered_map<std::string, std::uint32_t> counts;
    counts.reserve(1 << 17);
    for (std::size_t i = 0; i < raw.size();) {
      if (!text_word_first(raw[i])) { ++i; continue; }
      std::size_t b = skip_word(raw, i);
      std::size_t len = i - b;
      if (len < 4 || len > 31) { continue; }
      if (text2_case_kind(raw.data() + b, len) == 3) { continue; }
      ++counts[text2_to_lower_word(raw.data() + b, len)];
    }

    std::vector<text_word_t> words;
    words.reserve(std::min<std::size_t>(counts.size(), 32768));
    for (auto& [w, c] : counts) {
      if (c < 4) { continue; }
      std::uint32_t score = std::uint32_t(c * (w.size() > 3 ? w.size() - 3 : 0));
      if (score > w.size() + 10) { words.push_back({std::move(w), c, score}); }
    }
    std::sort(words.begin(), words.end(), [](const auto& a, const auto& b) {
      return a.score != b.score ? a.score > b.score : a.word < b.word;
    });

    std::vector<std::string> dict;
    dict.reserve(16384);
    for (auto& w : words) {
      std::size_t id = dict.size();
      std::size_t repl = id < 256 ? 3 : 4;
      std::size_t saved = w.count * (w.word.size() - repl);
      if (w.word.size() <= repl || saved <= w.word.size() + 4) { continue; }
      dict.push_back(std::move(w.word));
      if (dict.size() == 16384) { break; }
    }
    if (dict.empty()) { return {}; }

    std::unordered_map<std::string_view, std::uint32_t> ids;
    ids.reserve(dict.size() * 2);
    for (std::uint32_t i = 0; i < dict.size(); ++i) { ids.emplace(dict[i], i); }

    fan::bytes_t out;
    out.reserve(raw.size() * 4 / 5);
    std::uint8_t buf[8], dc[2];
    fan::memory::write_le64(buf, raw.size());
    out.insert(out.end(), buf, buf + 8);
    fan::memory::write_le16(dc, std::uint16_t(dict.size()));
    out.insert(out.end(), dc, dc + 2);
    for (const auto& w : dict) {
      out.push_back(std::uint8_t(w.size()));
      out.insert(out.end(), w.begin(), w.end());
    }

    for (std::size_t i = 0; i < raw.size();) {
      if (raw[i] == text_marker) { out.push_back(text_marker); out.push_back(0); ++i; continue; }
      int st = text2_static_match(raw, i);
      if (st >= 0) { text2_emit_id(out, 7, 8, std::uint32_t(st)); i += text2_static_tokens[st].size(); continue; }
      if (!text_word_first(raw[i])) { out.push_back(raw[i++]); continue; }
      std::size_t b = skip_word(raw, i);
      std::size_t len = i - b;
      std::uint8_t ck = len <= 31 ? text2_case_kind(raw.data() + b, len) : 3;
      if (len < 4 || len > 31 || ck == 3) {
        out.insert(out.end(), raw.begin() + b, raw.begin() + i); continue;
      }
      auto lw = text2_to_lower_word(raw.data() + b, len);
      auto it = ids.find(std::string_view(lw.data(), lw.size()));
      if (it == ids.end()) { out.insert(out.end(), raw.begin() + b, raw.begin() + i); continue; }
      std::uint8_t t8 = ck == 0 ? 1 : ck == 1 ? 3 : 5;
      text2_emit_id(out, t8, std::uint8_t(t8 + 1), it->second);
    }
    return out.size() < raw.size() + raw.size() / 64 ? std::move(out) : fan::bytes_t{};
  }

  inline fan::bytes_t text2_decode_transform(const fan::bytes_t& in) {
    std::size_t p = 0; std::uint64_t out_size = 0; std::vector<std::string> dict;
    decode_text_head(in, p, out_size, dict);

    fan::bytes_t out;
    out.reserve(std::size_t(out_size));
    while (p < in.size()) {
      std::uint8_t c = in[p++];
      if (c != text_marker) { out.push_back(c); continue; }
      if (p >= in.size()) { throw std::runtime_error("corrupt text2 transform"); }
      std::uint8_t t = in[p++];
      if (t == 0) { out.push_back(text_marker); continue; }
      std::uint32_t id = 0;
      if (t == 1 || t == 3 || t == 5 || t == 7) {
        if (p >= in.size()) { throw std::runtime_error("corrupt text2 transform"); }
        id = in[p++];
      } else if (t == 2 || t == 4 || t == 6 || t == 8) {
        if (p + 2 > in.size()) { throw std::runtime_error("corrupt text2 transform"); }
        id = std::uint32_t(in[p]) | (std::uint32_t(in[p + 1]) << 8); p += 2;
      } else { throw std::runtime_error("corrupt text2 transform"); }

      if (t == 7 || t == 8) {
        if (id >= std::size(text2_static_tokens)) { throw std::runtime_error("corrupt text2 transform"); }
        auto tok = text2_static_tokens[id];
        out.insert(out.end(), tok.begin(), tok.end());
      } else {
        if (id >= dict.size()) { throw std::runtime_error("corrupt text2 transform"); }
        std::uint8_t ck = (t == 1 || t == 2) ? 0 : (t == 3 || t == 4) ? 1 : 2;
        for (std::size_t i = 0; i < dict[id].size(); ++i) {
          std::uint8_t dc = std::uint8_t(dict[id][i]);
          if ((ck == 1 && i == 0) || ck == 2) { dc = text2_upper(dc); }
          out.push_back(dc);
        }
      }
    }
    if (out.size() != out_size) { throw std::runtime_error("corrupt text2 transform"); }
    return out;
  }

  inline std::uint32_t get_match_len(const std::uint8_t* p1, const std::uint8_t* p2, std::uint32_t max_l) {
    std::uint32_t l = 0;
    while (l + 8 <= max_l) {
      std::uint64_t x, y; std::memcpy(&x, p1 + l, 8); std::memcpy(&y, p2 + l, 8);
      if (auto d = x ^ y) { return l + (std::uint32_t(std::countr_zero(d)) >> 3); }
      l += 8;
    }
    if (l + 4 <= max_l) {
      std::uint32_t x, y; std::memcpy(&x, p1 + l, 4); std::memcpy(&y, p2 + l, 4);
      if (auto d = x ^ y) { return l + (std::uint32_t(std::countr_zero(d)) >> 3); }
      l += 4;
    }
    while (l < max_l && p1[l] == p2[l]) { ++l; }
    return l;
  }

  struct match_t { std::uint32_t offset = 0, length = 0; };
  struct match_heads_t { std::uint32_t h2 = -1u, h3 = -1u, h4 = -1u; };
  struct match_cache_t { std::uint32_t count = 0; std::array<match_t, 64> matches{}; };

  struct match_finder_t {
    static constexpr std::uint32_t nil = 0xFFFFFFFFu;
    std::vector<std::uint32_t> h2, h3, h4, chain;
    std::vector<std::uint64_t> inserted_bits;
    std::size_t base = 0;

    match_finder_t(std::size_t d_start, std::size_t c_end)
      : h2(1 << 16, nil), h3(1 << 20, nil), h4(1 << 24, nil),
        chain(c_end - d_start, nil), inserted_bits(((c_end - d_start) + 63) / 64), base(d_start) {}

    static void hashes(const std::uint8_t* p, std::uint32_t& k2, std::uint32_t& k3, std::uint32_t& k4) {
      std::uint32_t v; std::memcpy(&v, p, 4);
      k2 = ((v & 0xFFFFu) * 0x9E3779B9u) >> 16;
      k3 = ((v & 0xFFFFFFu) * 0x1E35A7BDu) >> 12;
      k4 = (v * 0x9E3779B9u) >> 8;
    }
    std::uint32_t to_rel(std::size_t p) const { return std::uint32_t(p - base); }

    void warm_up(const fan::bytes_t& d, std::size_t ws, std::size_t we) {
      for (std::size_t i = ws; i + 4 <= we; ++i) { insert(d, i); }
    }

    void insert(const fan::bytes_t& d, std::size_t i) {
      if (i + 4 > d.size()) { return; }
      std::uint32_t k2, k3, k4; hashes(d.data() + i, k2, k3, k4);
      insert_known_hashes(i, k2, k3, k4);
    }

    match_heads_t get_heads(std::uint32_t k2, std::uint32_t k3, std::uint32_t k4) const {
      return {h2[k2], h3[k3], h4[k4]};
    }

    void insert_known_hashes(std::size_t i, std::uint32_t k2, std::uint32_t k3, std::uint32_t k4) {
      std::uint32_t r = to_rel(i);
      std::uint64_t bit = 1ull << (r & 63);
      if (!(inserted_bits[r >> 6] & bit)) {
        inserted_bits[r >> 6] |= bit;
        chain[r] = h4[k4];
        h2[k2] = h3[k3] = h4[k4] = r;
      }
    }

    std::uint32_t find_from_heads(const std::uint8_t* src_base, std::size_t i, std::size_t c_end,
                                  match_heads_t heads, std::uint32_t chain_limit, std::uint32_t nice_len, match_t* out) const {
      const std::uint32_t max_avail = std::uint32_t(c_end - i);
      std::uint32_t n_out = 0;
      if (max_avail < 4) { return 0; }

      const auto* po = src_base + i;
      const std::uint32_t* p_chain = chain.data();

      auto try_quick = [&](std::uint32_t cur, std::uint32_t min_len) {
        if (cur == nil || n_out >= 64 || max_avail < min_len) { return; }
        std::uint32_t len = get_match_len(po, src_base + base + cur, max_avail);
        if (len >= min_len && (n_out == 0 || len > out[n_out - 1].length)) {
          out[n_out++] = {std::uint32_t(i - (base + cur)), len};
        }
      };

      try_quick(heads.h2, 2);
      try_quick(heads.h3, 3);

      std::uint32_t best_len = (n_out > 0) ? out[n_out - 1].length : 0;
      std::uint32_t cur = heads.h4, iters = 0, po4; std::memcpy(&po4, po, 4);

      while (cur != nil && iters++ < chain_limit) {
        std::size_t cur_abs = base + cur;
        if (best_len > 0 && best_len < max_avail && po[best_len] != src_base[cur_abs + best_len]) {
          cur = p_chain[cur]; continue;
        }
        const auto* pc = src_base + cur_abs;
        std::uint32_t pc4; std::memcpy(&pc4, pc, 4);
        if (po4 == pc4) {
          std::uint32_t len = get_match_len(po, pc, max_avail);
          if (len > best_len) {
            best_len = len;
            out[n_out < 64 ? n_out++ : n_out - 1] = {std::uint32_t(i - cur_abs), len};
            if (len >= nice_len) { break; }
          }
        }
        cur = p_chain[cur];
      }
      return n_out;
    }

    std::uint32_t find(const std::uint8_t* src_base, std::size_t i, std::size_t c_end,
                       std::uint32_t chain_limit, std::uint32_t nice_len, bool push, match_t* out) {
      if (i + 4 > c_end) { return 0; }
      std::uint32_t k2, k3, k4; hashes(src_base + i, k2, k3, k4);
      auto heads = get_heads(k2, k3, k4);
      if (push) { insert_known_hashes(i, k2, k3, k4); }
      return find_from_heads(src_base, i, c_end, heads, chain_limit, nice_len, out);
    }
  };

  struct compress_params_t {
    std::uint32_t chain_main = 64;
    std::uint32_t nice_len = 273;
    bool bcj = true;
    bool optimal = true;
    int lazy_depth = 2;
    std::size_t chunk_size = default_chunk_size;
  };

  constexpr compress_params_t params_fast()   { return {16, 32, true, false, 1, default_chunk_size}; }
  constexpr compress_params_t params_normal() { return {32, 64, true, false, 2, default_chunk_size}; }
  constexpr compress_params_t params_max()    { return {64, 273, true, true,  0, default_chunk_size}; }

  inline constexpr std::uint8_t state_lit_next[12]      = {0,0,0,0,1,2,3,4,5,6,4,5};
  inline constexpr std::uint8_t state_match_next[12]    = {7,7,7,7,7,7,7,10,10,10,10,10};
  inline constexpr std::uint8_t state_rep_next[12]      = {8,8,8,8,8,8,8,11,11,11,11,11};
  inline constexpr std::uint8_t state_shortrep_next[12] = {9,9,9,9,9,9,9,11,11,11,11,11};
  inline constexpr bool state_is_lit[12]                = {true,true,true,true,true,true,true,false,false,false,false,false};

  enum class op_e : std::uint8_t { rep0=0, rep1=1, rep2=2, rep3=3, exp=4, none=5 };
  struct seq_t { std::uint32_t lit_len; op_e op; std::uint32_t match_len, offset; };
  struct chunk_payload_t { std::vector<seq_t> seqs; };

  inline constexpr std::uint32_t magic_v4 = 0x34334346;
  inline constexpr int num_pos_states = 4;
  inline constexpr int lc = 3, lp = 0, num_lit_ctx = 1 << (lc + lp);

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
      std::array<std::uint16_t, 256> high;
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
    std::uint32_t k = std::bit_width(d) - 1;
    return (k << 1) | ((d >> (k-1)) & 1);
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
      std::uint32_t ctx = 1;
      for (int i = 0; i < bits; ++i) { ctx = (ctx << 1) | decode(tree[ctx]); }
      return ctx - (1u << bits);
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
    else { rc.encode(lm.choice[0][ps], 1); rc.encode(lm.choice2[0][ps], 1); rc.encode_tree(lm.high.data(), len - 16, 8); }
  }

  template <typename Reader>
  inline std::uint32_t decode_len(range_dec_t<Reader>& rd, lzma_model_t::len_model_t& lm, int ps) {
    if (!rd.decode(lm.choice[0][ps])) { return rd.decode_tree(lm.low[ps].data(), 3); }
    if (!rd.decode(lm.choice2[0][ps])) { return 8 + rd.decode_tree(lm.mid[ps].data(), 3); }
    return 16 + rd.decode_tree(lm.high.data(), 8);
  }

  template <typename Writer>
  inline void encode_dist(range_enc_t<Writer>& rc, lzma_model_t& m, std::uint32_t dist, std::uint32_t len_state) {
    std::uint32_t d = dist - 1, slot = slot_for_dist(d);
    rc.encode_tree(m.pos_slot[len_state].data(), slot, 6);
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

  inline constexpr std::uint32_t kNumOpts = 1 << 11, max_exp_len = 273, max_rep_len = 272;
  inline constexpr std::uint32_t progress_step = 4096;
  inline constexpr std::uint64_t parse_progress_weight = 80, encode_progress_weight = 20, progress_scale = 100;
  inline constexpr std::uint32_t inf_price = 0xFFFFFFFFu;

  namespace detail {
    inline std::array<std::uint32_t, 2049> init_price_table() {
      std::array<std::uint32_t, 2049> t{};
      for (int i = 1; i <= 2048; ++i) { t[i] = std::uint32_t(-std::log2(double(i) / 2048.0) * 64.0 + 0.5); }
      t[0] = t[1];
      return t;
    }
    inline const auto price_table = init_price_table();
  }

  inline std::uint32_t bit_price(std::uint16_t prob, std::uint32_t bit) {
    return detail::price_table[bit ? 2048 - prob : prob];
  }

  template <int bits>
  inline std::uint32_t tree_price_v(const std::uint16_t* tree, std::uint32_t sym) {
    std::uint32_t cost = 0, ctx = 1;
    if constexpr (bits == 3) {
      std::uint32_t b2 = (sym >> 2) & 1; cost += bit_price(tree[ctx], b2); ctx = (ctx << 1) | b2;
      std::uint32_t b1 = (sym >> 1) & 1; cost += bit_price(tree[ctx], b1); ctx = (ctx << 1) | b1;
      std::uint32_t b0 = sym & 1;        cost += bit_price(tree[ctx], b0);
    } else if constexpr (bits == 4) {
      for (int i = 3; i >= 0; --i) { std::uint32_t bit = (sym >> i) & 1; cost += bit_price(tree[ctx], bit); ctx = (ctx << 1) | bit; }
    } else if constexpr (bits == 6) {
      for (int i = 5; i >= 0; --i) { std::uint32_t bit = (sym >> i) & 1; cost += bit_price(tree[ctx], bit); ctx = (ctx << 1) | bit; }
    } else if constexpr (bits == 8) {
      for (int i = 7; i >= 0; --i) { std::uint32_t bit = (sym >> i) & 1; cost += bit_price(tree[ctx], bit); ctx = (ctx << 1) | bit; }
    }
    return cost;
  }

  inline std::uint32_t tree_price(const std::uint16_t* tree, std::uint32_t sym, int bits) {
    std::uint32_t cost = 0, ctx = 1;
    for (int i = bits - 1; i >= 0; --i) { std::uint32_t bit = (sym >> i) & 1; cost += bit_price(tree[ctx], bit); ctx = (ctx << 1) | bit; }
    return cost;
  }

  inline std::uint32_t len_price(const lzma_model_t::len_model_t& lm, std::uint32_t len, int ps) {
    if (len < 8) { return bit_price(lm.choice[0][ps], 0) + tree_price_v<3>(lm.low[ps].data(), len); }
    if (len < 16) { return bit_price(lm.choice[0][ps], 1) + bit_price(lm.choice2[0][ps], 0) + tree_price_v<3>(lm.mid[ps].data(), len - 8); }
    return bit_price(lm.choice[0][ps], 1) + bit_price(lm.choice2[0][ps], 1) + tree_price_v<8>(lm.high.data(), len - 16);
  }

  inline std::uint32_t dist_price(const lzma_model_t& m, std::uint32_t dist, std::uint32_t len_state) {
    std::uint32_t d = dist - 1, slot = slot_for_dist(d), cost = tree_price_v<6>(m.pos_slot[len_state].data(), slot);
    if (slot >= 4) {
      std::uint32_t footer_bits = (slot >> 1) - 1, footer = d - ((2u | (slot & 1u)) << footer_bits);
      if (slot < 14) { cost += tree_price(m.pos_special[slot - 4].data(), footer, footer_bits); }
      else { cost += (footer_bits - 4) * 64 + tree_price_v<4>(m.align_bits.data(), footer & 0xFu); }
    }
    return cost;
  }

  inline void update_bit(std::uint16_t& prob, std::uint32_t bit) {
    std::uint32_t p = prob;
    prob = std::uint16_t(bit ? p - (p >> 5) : p + ((2048 - p) >> 5));
  }

  template <int bits>
  inline void update_tree_v(std::uint16_t* tree, std::uint32_t sym) {
    std::uint32_t ctx = 1;
    for (int i = bits - 1; i >= 0; --i) {
      std::uint32_t bit = (sym >> i) & 1; update_bit(tree[ctx], bit); ctx = (ctx << 1) | bit;
    }
  }

  inline void update_tree(std::uint16_t* tree, std::uint32_t sym, int bits) {
    std::uint32_t ctx = 1;
    for (int i = bits - 1; i >= 0; --i) { std::uint32_t bit = (sym >> i) & 1; update_bit(tree[ctx], bit); ctx = (ctx << 1) | bit; }
  }

  inline void update_len(lzma_model_t::len_model_t& lm, std::uint32_t len, int ps) {
    if (len < 8) { update_bit(lm.choice[0][ps], 0); update_tree_v<3>(lm.low[ps].data(), len); }
    else if (len < 16) { update_bit(lm.choice[0][ps], 1); update_bit(lm.choice2[0][ps], 0); update_tree_v<3>(lm.mid[ps].data(), len - 8); }
    else { update_bit(lm.choice[0][ps], 1); update_bit(lm.choice2[0][ps], 1); update_tree_v<8>(lm.high.data(), len - 16); }
  }

  inline void update_dist(lzma_model_t& m, std::uint32_t dist, std::uint32_t len_state) {
    std::uint32_t d = dist - 1, slot = slot_for_dist(d); update_tree_v<6>(m.pos_slot[len_state].data(), slot);
    if (slot >= 4) {
      std::uint32_t footer_bits = (slot >> 1) - 1, footer = d - ((2u | (slot & 1u)) << footer_bits);
      if (slot < 14) { update_tree(m.pos_special[slot - 4].data(), footer, footer_bits); }
      else { update_tree_v<4>(m.align_bits.data(), footer & 0xFu); }
    }
  }

  inline void update_literal_symbol(lzma_model_t& m, std::uint8_t state, std::size_t pos, std::uint8_t prev, std::uint8_t byte, std::uint8_t match_byte) {
    update_bit(m.is_match[state][pos & (num_pos_states - 1)], 0);
    int sc = state_is_lit[state] ? 0 : 1; auto* tree = m.lit[sc][lit_ctx(pos, prev)].data();
    if (sc == 0) { update_tree_v<8>(tree, byte); }
    else {
      for (int i = 7, sym = 1; i >= 0; --i) {
        std::uint32_t bit = (byte >> i) & 1, mb_bit = (match_byte >> 7) & 1; match_byte <<= 1;
        update_bit(tree[sym + (mb_bit << 8)], bit); sym = (sym << 1) | bit;
      }
    }
  }

  inline void update_match_symbol(lzma_model_t& m, std::uint8_t state, std::size_t pos, op_e op, std::uint32_t len, std::uint32_t dist) {
    int pst = int(pos) & (num_pos_states - 1); update_bit(m.is_match[state][pst], 1);
    if (op == op_e::exp) { update_bit(m.is_rep[state], 0); update_len(m.match_len, len - 2, pst); update_dist(m, dist, std::min<std::uint32_t>(len - 2, 3)); return; }
    update_bit(m.is_rep[state], 1);
    if (op == op_e::rep0) {
      update_bit(m.is_rep0[state], 1); update_bit(m.is_rep0_short[state][pst], len != 1);
      if (len > 1) { update_len(m.rep_len, len - 1, pst); } return;
    }
    update_bit(m.is_rep0[state], 0);
    if (op == op_e::rep1) { update_bit(m.is_rep1[state], 0); }
    else { update_bit(m.is_rep1[state], 1); update_bit(m.is_rep2[state], op != op_e::rep2); }
    update_len(m.rep_len, len - 1, pst);
  }

  struct opt_t {
    std::uint32_t price = inf_price, prev = 0, len = 1, offset = 0;
    op_e op = op_e::none; std::array<std::uint32_t, 4> rep{}; std::uint8_t state = 0;
  };

  struct parse_price_cache_t {
    std::array<std::array<std::uint32_t, max_exp_len - 1>, num_pos_states> match_len{};
    std::array<std::array<std::uint32_t, max_rep_len>, num_pos_states> rep_len{};
    std::array<std::array<std::uint32_t, 64>, 4> pos_slot{};
    void refresh(lzma_model_t& m) {
      for (int ps = 0; ps < num_pos_states; ++ps) {
        for (std::uint32_t i = 0; i < max_exp_len - 1; ++i) { match_len[ps][i] = len_price(m.match_len, i, ps); }
        for (std::uint32_t i = 0; i < max_rep_len; ++i) { rep_len[ps][i] = len_price(m.rep_len, i, ps); }
      }
      for (std::uint32_t ls = 0; ls < 4; ++ls) {
        for (std::uint32_t slot = 0; slot < 64; ++slot) { pos_slot[ls][slot] = tree_price_v<6>(m.pos_slot[ls].data(), slot); }
      }
    }
    std::uint32_t dist_price(lzma_model_t& m, std::uint32_t dist, std::uint32_t len_state) const {
      std::uint32_t d = dist - 1, slot = slot_for_dist(d), cost = pos_slot[len_state][slot];
      if (slot >= 4) {
        std::uint32_t footer_bits = (slot >> 1) - 1, footer = d - ((2u | (slot & 1u)) << footer_bits);
        if (slot < 14) { cost += tree_price(m.pos_special[slot - 4].data(), footer, footer_bits); }
        else { cost += (footer_bits - 4) * 64 + tree_price_v<4>(m.align_bits.data(), footer & 0xFu); }
      }
      return cost;
    }
  };

  template <typename AddProg>
  chunk_payload_t parse_chunk_optimal(const fan::bytes_t& src, std::size_t c_start, std::size_t c_end, const compress_params_t& params, std::size_t thread_count, AddProg&& add_prog) {
    chunk_payload_t out; out.seqs.reserve((c_end - c_start) / 5);
    std::size_t dict_size = std::max<std::size_t>(1, params.chunk_size), d_start = c_start >= dict_size ? c_start - dict_size : 0;
    std::size_t ci = c_start, cend = c_end;

    match_finder_t finder(d_start, cend); finder.warm_up(src, d_start, ci);
    auto m_ptr = std::make_unique<lzma_model_t>(); lzma_model_t& model = *m_ptr;

    std::array<std::uint32_t, 4> g_rep{1, 1, 1, 1};
    std::uint8_t g_state = 0; std::uint32_t lit_cnt = 0; std::size_t last_report = c_start;

    std::vector<opt_t> opts(kNumOpts + max_exp_len + 1);
    std::vector<match_heads_t> match_heads(kNumOpts);
    std::vector<match_cache_t> match_cache(kNumOpts);
    parse_price_cache_t price_cache;

    auto ps = [](std::size_t pos) { return int(pos & (num_pos_states - 1)); };
    auto shift_rep = [](int r, std::uint32_t off, auto& rep) {
      if (r == 1) std::swap(rep[0], rep[1]);
      else if (r == 2) { auto t=rep[2]; rep[2]=rep[1]; rep[1]=rep[0]; rep[0]=t; }
      else if (r == 3) { auto t=rep[3]; rep[3]=rep[2]; rep[2]=rep[1]; rep[1]=rep[0]; rep[0]=t; }
      else if (r == 4) { rep[3]=rep[2]; rep[2]=rep[1]; rep[1]=rep[0]; rep[0]=off; }
    };

    if (!params.optimal) {
      auto best_cand = [&](std::size_t p, bool push, std::uint32_t cl) {
        struct cand_t { op_e op; std::uint32_t len; std::uint32_t off; std::uint32_t price; } b{op_e::none, 0, 0, inf_price};
        std::uint32_t avail = std::uint32_t(cend - p); if (!avail) { return b; }
        for (int r = 0; r < 4; ++r) {
          if (p < d_start + g_rep[r]) { continue; }
          std::uint32_t max_l = std::min(avail, max_rep_len);
          std::uint32_t l = get_match_len(src.data() + p, src.data() + p - g_rep[r], max_l);
          if (l == 0 || (l == 1 && r > 0)) { continue; }
          std::uint32_t pft = bit_price(model.is_match[g_state][ps(p)], 1) + bit_price(model.is_rep[g_state], 1);
          if (r == 0) { pft += bit_price(model.is_rep0[g_state], 1) + bit_price(model.is_rep0_short[g_state][ps(p)], l != 1) + (l > 1 ? len_price(model.rep_len, l - 1, ps(p)) : 0); }
          else { pft += bit_price(model.is_rep0[g_state], 0) + (r == 1 ? bit_price(model.is_rep1[g_state], 0) : bit_price(model.is_rep1[g_state], 1) + bit_price(model.is_rep2[g_state], r != 2)) + len_price(model.rep_len, l - 1, ps(p)); }
          if (pft < b.price || (pft == b.price && l > b.len)) { b = {static_cast<op_e>(r), l, g_rep[r], pft}; }
        }
        if (avail >= 4) {
          match_t l_matches[64];
          std::uint32_t n = finder.find(src.data(), p, cend, cl, params.nice_len, push, l_matches);
          for (std::uint32_t mi = 0; mi < n; ++mi) {
            if (l_matches[mi].length < 2) { continue; }
            std::uint32_t len = std::min<std::uint32_t>(l_matches[mi].length, max_exp_len);
            std::uint32_t pft = bit_price(model.is_match[g_state][ps(p)], 1) + bit_price(model.is_rep[g_state], 0) + len_price(model.match_len, len - 2, ps(p)) + dist_price(model, l_matches[mi].offset, std::min<std::uint32_t>(len - 2, 3));
            if (pft < b.price || (pft == b.price && len > b.len)) { b = {op_e::exp, len, l_matches[mi].offset, pft}; }
          }
        } else if (push) { finder.insert(src, p); }
        return b;
      };

      while (ci < cend) {
        auto M = best_cand(ci, true, params.chain_main);
        if (M.len == 0 || M.op == op_e::none) {
          update_literal_symbol(model, g_state, ci, ci > 0 ? src[ci - 1] : 0, src[ci], ci >= g_rep[0] ? src[ci - g_rep[0]] : 0);
          g_state = state_lit_next[g_state]; ++lit_cnt; ++ci;
        } else {
          bool took_lazy = false;
          if (ci + 4 < cend && M.len < 128) {
            for (int skip = 1; skip <= params.lazy_depth && ci + skip < cend; ++skip) {
              if (best_cand(ci + skip, false, params.chain_main >> 1).price < M.price + std::uint32_t(skip) * 8 * 64) {
                for (int k = 0; k < skip; ++k) {
                  update_literal_symbol(model, g_state, ci, ci > 0 ? src[ci - 1] : 0, src[ci], ci >= g_rep[0] ? src[ci - g_rep[0]] : 0);
                  g_state = state_lit_next[g_state]; ++lit_cnt; ++ci;
                }
                took_lazy = true; break;
              }
            }
          }
          if (took_lazy) { continue; }
          out.seqs.push_back({lit_cnt, M.op, M.len, M.off});
          update_match_symbol(model, g_state, ci, M.op, M.len, M.off);
          if (M.op == op_e::rep0) { g_state = (M.len == 1) ? state_shortrep_next[g_state] : state_rep_next[g_state]; }
          else { shift_rep(int(M.op), M.off, g_rep); g_state = M.op == op_e::exp ? state_match_next[g_state] : state_rep_next[g_state]; }
          lit_cnt = 0;
          for (std::uint32_t step = M.len < 32 ? 1 : M.len >> 4, k = 1; k < M.len; k += step) { if (ci + k + 2 < cend) finder.insert(src, ci + k); }
          ci += M.len;
        }
        if (ci - last_report >= progress_step) { add_prog((ci - last_report) * parse_progress_weight); last_report = ci; }
      }
      add_prog((cend - last_report) * parse_progress_weight);
      if (lit_cnt) { out.seqs.push_back({lit_cnt, op_e::none, 0, 0}); }
      return out;
    }

    std::size_t cache_thread_count = std::max<std::size_t>(1, thread_count);
    std::size_t cache_worker_count = cache_thread_count > 1 ? cache_thread_count - 1 : 0;
    std::atomic<std::uint32_t> next_match = 0; std::mutex cache_mutex; std::condition_variable cache_cv, cache_done_cv;
    std::uint64_t cache_generation = 0; std::size_t cache_finished = 0, cache_ci = 0; std::uint32_t cache_window = 0; bool cache_stop = false;

    auto fill_match_cache = [&](std::size_t base_pos, std::uint32_t begin, std::uint32_t end) {
      for (std::uint32_t j = begin; j < end; ++j) {
        if (base_pos + j + 4 > cend) { continue; }
        match_cache[j].count = finder.find_from_heads(src.data(), base_pos + j, cend, match_heads[j], params.chain_main, params.nice_len, match_cache[j].matches.data());
      }
    };

    std::vector<std::jthread> cache_workers;
    for (std::size_t w = 0; w < cache_worker_count; ++w) {
      cache_workers.emplace_back([&] {
        std::uint64_t seen = 0;
        for (;;) {
          std::unique_lock lock(cache_mutex);
          cache_cv.wait(lock, [&] { return cache_stop || cache_generation != seen; });
          if (cache_stop) { break; } seen = cache_generation;
          std::size_t base_pos = cache_ci; std::uint32_t window = cache_window; lock.unlock();
          for (std::uint32_t begin; (begin = next_match.fetch_add(64, std::memory_order_relaxed)) < window;) {
            fill_match_cache(base_pos, begin, std::min<std::uint32_t>(begin + 64, window));
          }
          lock.lock(); if (++cache_finished == cache_worker_count) { cache_done_cv.notify_one(); }
        }
      });
    }

    while (ci < cend) {
      price_cache.refresh(model);
      std::uint32_t window = std::min(std::uint32_t(cend - ci), kNumOpts - 1), len_end = 1;
      opts[0] = {0, 0, 1, 0, op_e::none, g_rep, g_state};
      for (std::uint32_t k = 1; k <= window + max_exp_len; ++k) { opts[k].price = inf_price; }

      for (std::uint32_t j = 0; j < window; ++j) {
        match_cache[j].count = 0; if (ci + j + 4 > cend) { continue; }
        std::uint32_t k2, k3, k4; match_finder_t::hashes(src.data() + ci + j, k2, k3, k4);
        match_heads[j] = finder.get_heads(k2, k3, k4); finder.insert_known_hashes(ci + j, k2, k3, k4);
      }

      if (cache_worker_count > 0 && window >= 512) {
        { std::lock_guard lock(cache_mutex); cache_ci = ci; cache_window = window; cache_finished = 0; next_match.store(0, std::memory_order_relaxed); ++cache_generation; }
        cache_cv.notify_all();
        for (std::uint32_t begin; (begin = next_match.fetch_add(64, std::memory_order_relaxed)) < window;) { fill_match_cache(ci, begin, std::min<std::uint32_t>(begin + 64, window)); }
        std::unique_lock lock(cache_mutex); cache_done_cv.wait(lock, [&] { return cache_finished == cache_worker_count; });
      } else {
        fill_match_cache(ci, 0, window);
      }

      for (std::uint32_t j = 0; j < window; ++j) {
        if (opts[j].price == inf_price) { continue; }
        std::uint8_t st = opts[j].state; std::array<std::uint32_t,4> rep = opts[j].rep; std::size_t pos = ci + j; int pst = ps(pos);

        auto try_price = [&](std::uint32_t k, std::uint32_t p, op_e op, std::uint32_t l, std::uint32_t o, std::uint8_t nst, const std::array<std::uint32_t,4>& nrep) {
          if (p < opts[k].price) { opts[k] = {p, j, l, o, op, nrep, nst}; if (k > len_end) len_end = k; }
        };

        if (pos < cend) {
          std::uint32_t lp = bit_price(model.is_match[st][pst], 0);
          int sc = state_is_lit[st] ? 0 : 1; const std::uint16_t* tree = model.lit[sc][lit_ctx(pos, pos > 0 ? src[pos - 1] : 0)].data();
          if (sc == 0) { lp += tree_price_v<8>(tree, src[pos]); }
          else {
            std::uint32_t sym = 1, mb = pos >= rep[0] ? src[pos - rep[0]] : 0, byte = src[pos];
            for (int i = 7; i >= 0; --i) {
              std::uint32_t bit = (byte >> i) & 1, mb_bit = (mb >> 7) & 1; mb <<= 1;
              lp += bit_price(tree[sym + (mb_bit << 8)], bit); sym = (sym << 1) | bit;
            }
          }
          try_price(j+1, opts[j].price + lp, op_e::none, 1, 0, state_lit_next[st], rep);
        }

        for (int r = 0; r < 4 && pos < cend; ++r) {
          if (pos < d_start + rep[r]) { continue; }
          std::uint32_t max_l = std::min<std::uint32_t>(std::uint32_t(cend - pos), max_rep_len);
          std::uint32_t l = get_match_len(src.data() + pos, src.data() + pos - rep[r], max_l);
          if (l == 0) { continue; }
          std::uint32_t bp = opts[j].price + bit_price(model.is_match[st][pst], 1) + bit_price(model.is_rep[st], 1);
          if (r == 0) {
            std::uint32_t r0p = bp + bit_price(model.is_rep0[st], 1);
            try_price(j+1, r0p + bit_price(model.is_rep0_short[st][pst], 0), op_e::rep0, 1, rep[0], state_shortrep_next[st], rep);
            if (l >= 2) {
              std::uint32_t r0p_long = r0p + bit_price(model.is_rep0_short[st][pst], 1);
              auto put = [&](std::uint32_t ml) { try_price(j + ml, r0p_long + price_cache.rep_len[pst][ml - 1], op_e::rep0, ml, rep[0], state_rep_next[st], rep); };
              for (std::uint32_t ml = 2, e = std::min(l, 16u); ml <= e; ++ml) { put(ml); }
              if (l > 16) { for (std::uint32_t ml = 20; ml < l; ml += 8) { put(ml); } put(l); }
            }
          } else {
            std::uint32_t rp = bp + bit_price(model.is_rep0[st], 0) + (r == 1 ? bit_price(model.is_rep1[st], 0) : bit_price(model.is_rep1[st], 1) + bit_price(model.is_rep2[st], r != 2));
            std::array<std::uint32_t,4> nr = rep; shift_rep(r, rep[r], nr);
            if (l >= 2) {
              auto put = [&](std::uint32_t ml) { try_price(j + ml, rp + price_cache.rep_len[pst][ml - 1], static_cast<op_e>(r), ml, rep[r], state_rep_next[st], nr); };
              for (std::uint32_t ml = 2, e = std::min(l, 16u); ml <= e; ++ml) { put(ml); }
              if (l > 16) { for (std::uint32_t ml = 20; ml < l; ml += 8) { put(ml); } put(l); }
            }
          }
        }

        if (pos + 4 <= cend) {
          std::uint32_t bp = opts[j].price + bit_price(model.is_match[st][pst], 1) + bit_price(model.is_rep[st], 0);
          std::uint32_t prev_len = 1;
          for (std::uint32_t mi = 0; mi < match_cache[j].count; ++mi) {
            auto& mc = match_cache[j].matches[mi]; if (mc.length < 2) { continue; }
            std::uint32_t limit = std::min({mc.length, std::uint32_t(cend - pos), max_exp_len});
            if (limit <= prev_len) { continue; }
            std::array<std::uint32_t, 4> dc; for (std::uint32_t ls = 0; ls < 4; ++ls) { dc[ls] = price_cache.dist_price(model, mc.offset, ls); }
            
            std::uint32_t start_len = std::max(2u, prev_len + 1);
            auto put = [&](std::uint32_t ml) {
              std::array<std::uint32_t,4> nr = rep; shift_rep(4, mc.offset, nr);
              try_price(j + ml, bp + price_cache.match_len[pst][ml - 2] + dc[std::min<std::uint32_t>(ml - 2, 3)], op_e::exp, ml, mc.offset, state_match_next[st], nr);
            };

            for (std::uint32_t ml = start_len, e = std::min(limit, 16u); ml <= e; ++ml) { put(ml); }
            if (limit > 16) {
              for (std::uint32_t ml = std::max(20u, (start_len + 7) & ~7u); ml < limit; ml += 8) { put(ml); }
              put(limit);
            }
            prev_len = limit;
          }
        }
      }

      std::vector<std::uint32_t> path;
      for (std::uint32_t cur = len_end; cur > 0; cur = opts[cur].prev) { path.push_back(cur); }
      for (auto it = path.rbegin(); it != path.rend(); ++it) {
        auto& d = opts[*it];
        if (d.op == op_e::none) {
          update_literal_symbol(model, g_state, ci, ci > 0 ? src[ci - 1] : 0, src[ci], ci >= g_rep[0] ? src[ci - g_rep[0]] : 0);
          ++lit_cnt; g_state = state_lit_next[g_state]; ++ci;
        } else {
          out.seqs.push_back({lit_cnt, d.op, d.len, d.offset});
          update_match_symbol(model, g_state, ci, d.op, d.len, d.offset);
          lit_cnt = 0;
          if (d.op == op_e::rep0) { g_state = (d.len == 1) ? state_shortrep_next[g_state] : state_rep_next[g_state]; }
          else { shift_rep(int(d.op), d.offset, g_rep); g_state = (d.op == op_e::exp) ? state_match_next[g_state] : state_rep_next[g_state]; }
          ci += d.len;
        }
      }
      if (ci - last_report >= progress_step) { add_prog((ci - last_report) * parse_progress_weight); last_report = ci; }
    }

    if (cache_worker_count != 0) { { std::lock_guard lock(cache_mutex); cache_stop = true; } cache_cv.notify_all(); cache_workers.clear(); }
    add_prog((cend - last_report) * parse_progress_weight);
    if (lit_cnt) { out.seqs.push_back({lit_cnt, op_e::none, 0, 0}); }
    return out;
  }

  template <typename AddProg>
  fan::bytes_t encode_stream_seq(const fan::bytes_t& src, const std::vector<chunk_payload_t>& blocks, std::size_t chunk_size, AddProg&& add_prog) {
    fan::bytes_t out; out.reserve(src.size() / 5); fan::io::bytes_writer_t bw{out}; range_enc_t<fan::io::bytes_writer_t> rc{bw};
    auto m_ptr = std::make_unique<lzma_model_t>(); lzma_model_t& model = *m_ptr;

    std::size_t src_ptr = 0, last_report = 0, next_boundary = chunk_size;
    std::array<std::uint32_t,4> rep{1,1,1,1}; std::uint8_t state = 0;
    auto shift_rep = [](int r, std::uint32_t off, auto& rp) {
      if (r == 1) std::swap(rp[0], rp[1]);
      else if (r == 2) { auto t=rp[2]; rp[2]=rp[1]; rp[1]=rp[0]; rp[0]=t; }
      else if (r == 3) { auto t=rp[3]; rp[3]=rp[2]; rp[2]=rp[1]; rp[1]=rp[0]; rp[0]=t; }
      else if (r == 4) { rp[3]=rp[2]; rp[2]=rp[1]; rp[1]=rp[0]; rp[0]=off; }
    };
    auto report = [&] { if (src_ptr - last_report >= progress_step) { add_prog((src_ptr - last_report) * encode_progress_weight); last_report = src_ptr; } };

    for (const auto& block : blocks) {
      if (src_ptr == next_boundary) { rep = {1,1,1,1}; state = 0; model = lzma_model_t(); next_boundary += chunk_size; }
      for (const auto& s : block.seqs) {
        int pst = int(src_ptr) & (num_pos_states - 1);
        for (std::uint32_t j = 0; j < s.lit_len; ++j) {
          rc.encode(model.is_match[state][pst], 0);
          encode_literal(rc, model, state, src_ptr, src_ptr > 0 ? src[src_ptr - 1] : 0, src[src_ptr], src_ptr >= rep[0] ? src[src_ptr - rep[0]] : 0);
          state = state_lit_next[state]; ++src_ptr; pst = int(src_ptr) & (num_pos_states - 1); report();
        }
        if (s.op == op_e::none) { continue; }
        rc.encode(model.is_match[state][pst], 1);
        if (s.op != op_e::exp) {
          rc.encode(model.is_rep[state], 1);
          if (s.op == op_e::rep0) {
            rc.encode(model.is_rep0[state], 1);
            if (s.match_len == 1) { rc.encode(model.is_rep0_short[state][pst], 0); state = state_shortrep_next[state]; }
            else { rc.encode(model.is_rep0_short[state][pst], 1); encode_len(rc, model.rep_len, s.match_len - 1, pst); state = state_rep_next[state]; }
          } else {
            rc.encode(model.is_rep0[state], 0);
            if (s.op == op_e::rep1) { rc.encode(model.is_rep1[state], 0); }
            else { rc.encode(model.is_rep1[state], 1); rc.encode(model.is_rep2[state], s.op != op_e::rep2); }
            encode_len(rc, model.rep_len, s.match_len - 1, pst); state = state_rep_next[state];
          }
          if (s.op != op_e::rep0) { shift_rep(int(s.op), s.offset, rep); }
        } else {
          rc.encode(model.is_rep[state], 0); encode_len(rc, model.match_len, s.match_len - 2, pst); encode_dist(rc, model, s.offset, std::min<std::uint32_t>(s.match_len - 2, 3));
          shift_rep(4, s.offset, rep); state = state_match_next[state];
        }
        src_ptr += s.match_len; pst = int(src_ptr) & (num_pos_states - 1); report();
      }
    }
    rc.flush(); add_prog((src_ptr - last_report) * encode_progress_weight);
    return out;
  }

  fan::bytes_t decode_stream_seq(const fan::bytes_t& comp, std::size_t idx, std::uint64_t total_uncomp, std::size_t chunk_size, fan::progress_t* prog) {
    fan::bytes_t out; out.reserve(total_uncomp); fan::io::bytes_reader_t br{comp, idx}; range_dec_t<fan::io::bytes_reader_t> rd{br};
    auto m_ptr = std::make_unique<lzma_model_t>(); lzma_model_t& model = *m_ptr;

    std::array<std::uint32_t,4> rep{1,1,1,1}; std::uint8_t state = 0;
    std::size_t next_boundary = chunk_size, last_report = 0;

    while (out.size() < total_uncomp) {
      if (out.size() == next_boundary) { rep = {1,1,1,1}; state = 0; model = lzma_model_t(); next_boundary += chunk_size; }
      std::size_t pos = out.size(); int pst = int(pos) & (num_pos_states - 1);

      if (!rd.decode(model.is_match[state][pst])) {
        out.push_back(decode_literal(rd, model, state, pos, pos > 0 ? out[pos-1] : 0, pos >= rep[0] ? out[pos - rep[0]] : 0));
        state = state_lit_next[state];
        if (prog && out.size() - last_report >= 65536) { prog->done.store(out.size(), std::memory_order_relaxed); last_report = out.size(); }
        continue;
      }

      std::uint32_t mlen = 0, off = 0;
      if (rd.decode(model.is_rep[state])) {
        if (rd.decode(model.is_rep0[state])) {
          if (!rd.decode(model.is_rep0_short[state][pst])) { mlen = 1; off = rep[0]; state = state_shortrep_next[state]; }
          else { mlen = 1 + decode_len(rd, model.rep_len, pst); off = rep[0]; state = state_rep_next[state]; }
        } else {
          int r = !rd.decode(model.is_rep1[state]) ? 1 : rd.decode(model.is_rep2[state]) ? 3 : 2;
          mlen = 1 + decode_len(rd, model.rep_len, pst);
          if (r == 1) { off = rep[1]; std::swap(rep[0], rep[1]); }
          else if (r == 2) { off = rep[2]; auto t=rep[2]; rep[2]=rep[1]; rep[1]=rep[0]; rep[0]=t; }
          else { off = rep[3]; auto t=rep[3]; rep[3]=rep[2]; rep[2]=rep[1]; rep[1]=rep[0]; rep[0]=t; }
          state = state_rep_next[state];
        }
      } else {
        mlen = 2 + decode_len(rd, model.match_len, pst); off = decode_dist(rd, model, std::min<std::uint32_t>(mlen - 2, 3));
        rep[3]=rep[2]; rep[2]=rep[1]; rep[1]=rep[0]; rep[0]=off; state = state_match_next[state];
      }

      if (off == 0 || off > out.size()) { throw std::runtime_error("corrupt stream"); }
      std::size_t old = out.size(); out.resize(old + mlen);
      if (off >= mlen) { std::memcpy(&out[old], &out[old - off], mlen); }
      else { for (std::uint32_t j = 0; j < mlen; ++j) { out[old + j] = out[old - off + j]; } }
      if (prog && out.size() - last_report >= 65536) { prog->done.store(out.size(), std::memory_order_relaxed); last_report = out.size(); }
    }
    if (prog) { prog->done.store(total_uncomp, std::memory_order_relaxed); }
    return out;
  }

  inline void run_compress_core(fan::bytes_t& raw, compress_params_t params, std::atomic<std::uint64_t>* prog_done, fan::progress_t* user_prog, std::size_t thread_count, fan::bytes_t& out_comp) {
    std::size_t actual_threads = thread_count ? thread_count : std::max<std::size_t>(1, std::thread::hardware_concurrency());
    std::size_t ch_size = std::min<std::size_t>(std::max<std::size_t>(1, params.chunk_size), 1uz << 30);
    std::size_t nc = (raw.size() + ch_size - 1) / ch_size;
    std::size_t block_threads = std::min(actual_threads, std::max<std::size_t>(1, nc));
    std::size_t parse_threads = std::max<std::size_t>(1, actual_threads / block_threads);

    auto add_prog = [&](std::uint64_t w) {
      if (prog_done && user_prog) { user_prog->done.store(prog_done->fetch_add(w, std::memory_order_relaxed) + w, std::memory_order_relaxed); }
    };

    std::vector<chunk_payload_t> blocks(nc); std::atomic<std::size_t> next = 0; std::vector<std::jthread> workers;
    for (std::size_t w = 0; w < block_threads; ++w) {
      workers.emplace_back([&] {
        for (std::size_t k; (k = next.fetch_add(1, std::memory_order_relaxed)) < nc;) {
          blocks[k] = parse_chunk_optimal(raw, k * ch_size, std::min(raw.size(), (k + 1) * ch_size), params, parse_threads, add_prog);
        }
      });
    }
    workers.clear(); out_comp = encode_stream_seq(raw, blocks, ch_size, add_prog);
  }

  struct compress_candidate_t {
    fan::bytes_t data;
    fan::bytes_t comp;
    std::string_view name;
    std::uint8_t flags = 0;
    std::size_t chunk_size = default_chunk_size;
  };

  inline std::size_t safe_parallel_chunk_size(std::size_t size, std::size_t threads) {
    if (threads <= 1) { return size; }
    std::size_t mib = 1uz << 20;
    std::size_t cs = ((size + threads - 1) / threads + mib - 1) & ~(mib - 1);
    return std::clamp<std::size_t>(cs, mib, std::min<std::size_t>(size, 1uz << 30));
  }

  inline compress_candidate_t compress_best_candidate(std::vector<compress_candidate_t>& candidates, compress_params_t params, fan::progress_t* prog, std::size_t thread_count, bool verbose = false) {
    std::size_t actual_threads = thread_count ? thread_count : std::max<std::size_t>(1, std::thread::hardware_concurrency());
    std::size_t best_idx = 0; bool best_par = false; std::size_t best_size = -1uz; std::size_t best_chunk_size = params.chunk_size;

    std::uint64_t total_weight = 0;
    for (const auto& c : candidates) {
      total_weight += c.data.size() * progress_scale;
      if (actual_threads >= 2 && c.data.size() >= (2uz << 20)) { total_weight += c.data.size() * progress_scale; }
    }
    if (prog) { prog->total.store(total_weight, std::memory_order_relaxed); prog->done.store(0, std::memory_order_relaxed); }
    std::atomic<std::uint64_t> prog_done = 0;

    if (verbose) { fan::print("testing candidates..."); }
    for (std::size_t i = 0; i < candidates.size(); ++i) {
      auto& c = candidates[i];
      fan::bytes_t comp_solid;
      run_compress_core(c.data, params, prog ? &prog_done : nullptr, prog, actual_threads, comp_solid);
      if (verbose) { fan::print("candidate:", c.name, "solid", comp_solid.size(), "bytes"); }
      if (comp_solid.size() < best_size) {
        best_size = comp_solid.size();
        best_idx = i;
        best_par = false;
        best_chunk_size = params.chunk_size;
        c.comp = std::move(comp_solid);
      }

      if (actual_threads >= 2 && c.data.size() >= (2uz << 20)) {
        fan::bytes_t comp_par; auto p_params = params; p_params.chunk_size = safe_parallel_chunk_size(c.data.size(), actual_threads);
        run_compress_core(c.data, p_params, prog ? &prog_done : nullptr, prog, actual_threads, comp_par);
        if (verbose) { fan::print("candidate:", c.name, "parallel", comp_par.size(), "bytes"); }
        if (comp_par.size() < best_size) {
          best_size = comp_par.size();
          best_idx = i;
          best_par = true;
          best_chunk_size = p_params.chunk_size;
          c.comp = std::move(comp_par);
        }
      }
    }
    if (verbose) { fan::print("selected:", candidates[best_idx].name, best_par ? "parallel" : "solid", best_size, "bytes"); }
    if (prog) { prog->done.store(total_weight, std::memory_order_relaxed); }
    candidates[best_idx].chunk_size = best_chunk_size;
    return std::move(candidates[best_idx]);
  }

  bool compress_path_to_file(const std::filesystem::path& in, const std::filesystem::path& out_path, compress_params_t params = params_max(), fan::progress_t* prog = nullptr, bool verbose = false, std::size_t thread_count = 0) {
    std::vector<fan::io::file_info_t> files;
    if (std::filesystem::is_directory(in)) {
      fan::io::iterate_files_recursive(in, [&](const auto& full, const auto& rel) {
        if (verbose) { fan::print("found:", rel.generic_string()); }
        files.push_back({full, rel.generic_string(), std::filesystem::file_size(full)});
      });
    } else {
      if (verbose) { fan::print("found:", in.filename().string()); }
      files.push_back({in, in.filename().string(), std::filesystem::file_size(in)});
    }

    fan::io::vfs_provider_t provider;
    std::uint8_t u32_buf[4]; fan::memory::write_le32(u32_buf, std::uint32_t(files.size()));
    provider.append_bytes(std::span<const std::uint8_t>(u32_buf, 4));
    for (const auto& f : files) {
      if (f.archive_path.size() > std::numeric_limits<std::uint16_t>::max()) { throw std::runtime_error("path too long"); }
      fan::bytes_t meta; std::uint8_t u16_buf[2], u64_buf[8]; fan::memory::write_le16(u16_buf, std::uint16_t(f.archive_path.size()));
      meta.insert(meta.end(), u16_buf, u16_buf + 2);
      auto* p_path = reinterpret_cast<const std::uint8_t*>(f.archive_path.data());
      meta.insert(meta.end(), p_path, p_path + f.archive_path.size());
      fan::memory::write_le64(u64_buf, f.size); meta.insert(meta.end(), u64_buf, u64_buf + 8);
      provider.append_bytes(std::span<const std::uint8_t>(meta.data(), meta.size()));
    }
    std::size_t pad = (4 - (provider.size() & 3)) & 3;
    if (pad) { std::uint8_t zeros[3]{}; provider.append_bytes(std::span<const std::uint8_t>(zeros, pad)); }
    std::size_t payload_offset = provider.size();
    for (const auto& f : files) { provider.append_file(f.real_path, f.size); }

    fan::bytes_t raw_buf; provider.read_range(0, provider.size(), raw_buf);

    bool use_bcj = false, use_delta = false, use_text = false, use_text2 = false; int delta_stride = 1;
    if (params.bcj && files.size() == 1) {
      fan::bytes_t file_head(std::min<std::uint64_t>(files[0].size, 4));
      fan::io::file::file_t* fpp = nullptr;
      if (!fan::io::file::open(&fpp, files[0].real_path.string(), {"rb"})) {
        fan::io::file::file_reader_t fr{fpp}; fr.read_exact(file_head); fan::io::file::close(fpp);
        if (file_head.size() >= 2 && file_head[0] == 'M' && file_head[1] == 'Z') { use_bcj = true; }
      }
    }

    fan::bytes_t text_buf, text2_buf;
    bool try_text = !use_bcj && files.size() == 1 && looks_like_large_text(raw_buf, payload_offset);
    if (try_text) { text_buf = text_encode_transform(raw_buf); text2_buf = text2_encode_transform(raw_buf); }

    if (use_bcj) { bcj_transform_range(raw_buf, payload_offset, raw_buf.size(), true); }
    else if (!try_text) {
      delta_stride = detect_delta_stride(raw_buf); use_delta = delta_stride > 1;
      if (use_delta) { delta_encode(raw_buf, delta_stride); }
    }

    int stride_log2 = delta_stride == 8 ? 3 : delta_stride == 4 ? 2 : delta_stride == 2 ? 1 : 0;
    std::uint8_t flags = std::uint8_t(use_bcj ? flag_bcj : 0) | std::uint8_t(use_delta ? flag_delta : 0) | std::uint8_t(stride_log2 << 2);

    fan::io::file::file_t* fp = nullptr;
    if (fan::io::file::open(&fp, out_path.string(), {"wb"})) { return false; }
    try {
      fan::bytes_t comp; std::size_t selected_chunk_size = std::min<std::size_t>(std::max<std::size_t>(1, params.chunk_size), 1uz << 30);
      if (try_text) {
        std::vector<compress_candidate_t> candidates; candidates.push_back({std::move(raw_buf), {}, "raw", flags, params.chunk_size});
        if (!text_buf.empty()) { candidates.push_back({std::move(text_buf), {}, "text", std::uint8_t(flags | flag_text), params.chunk_size}); }
        if (!text2_buf.empty()) { candidates.push_back({std::move(text2_buf), {}, "text2", std::uint8_t(flags | flag_text2), params.chunk_size}); }
        auto best = compress_best_candidate(candidates, params, prog, thread_count, verbose);
        raw_buf = std::move(best.data); comp = std::move(best.comp); flags = best.flags; selected_chunk_size = best.chunk_size;
      } else {
        std::vector<compress_candidate_t> candidates; candidates.push_back({std::move(raw_buf), {}, "raw", flags, params.chunk_size});
        auto best = compress_best_candidate(candidates, params, prog, thread_count, verbose);
        raw_buf = std::move(best.data); comp = std::move(best.comp); flags = best.flags; selected_chunk_size = best.chunk_size;
      }

      std::uint8_t header[17]; fan::memory::write_le32(header, magic_v4); fan::memory::write_le64(header + 4, raw_buf.size());
      fan::memory::write_le32(header + 12, std::uint32_t(selected_chunk_size)); header[16] = flags;

      fan::io::file::file_writer_t sink{fp}; sink.write_bytes(std::span<const std::uint8_t>(header, 17)); sink.write_bytes(comp);
      fan::io::file::close(fp); return true;
    } catch (...) { fan::io::file::close(fp); throw; }
  }

  bool decompress_file_to_dir(const std::filesystem::path& in_path, const std::filesystem::path& out_dir, bool default_out, fan::progress_t* prog = nullptr) {
    fan::io::file::file_t* fp = nullptr;
    if (fan::io::file::open(&fp, in_path.string(), {"rb"})) { return false; }
    fan::io::file::file_reader_t src{fp};
    try {
      std::uint8_t header[17]; src.read_exact(std::span<std::uint8_t>(header, 17));
      if (fan::memory::read_le32(header) != magic_v4) { throw std::runtime_error("needs FCS4"); }
      std::uint64_t total_uncomp = fan::memory::read_le64(header + 4);
      std::size_t stored_chunk_size = fan::memory::read_le32(header + 12); if (stored_chunk_size == 0) { stored_chunk_size = default_chunk_size; }
      std::uint8_t flags = header[16]; bool use_bcj = flags & flag_bcj, use_delta = flags & flag_delta, use_text = flags & flag_text, use_text2 = flags & flag_text2;
      int delta_stride = 1 << ((flags >> 2) & 7);

      fan::bytes_t comp; std::uint64_t fsz = fan::io::file::file_size(in_path.string());
      if (fsz > 17) { comp.resize(fsz - 17); src.read_exact(comp); }
      fan::io::file::close(fp); fp = nullptr;
      if (prog) { prog->done.store(0, std::memory_order_relaxed); prog->total.store(total_uncomp, std::memory_order_relaxed); }

      fan::bytes_t raw = decode_stream_seq(comp, 0, total_uncomp, stored_chunk_size, prog);
      if (use_delta) { delta_decode(raw, delta_stride); }
      if (use_text2) { raw = text2_decode_transform(raw); }
      if (use_text)  { raw = text_decode_transform(raw); }
      if (use_bcj)   { bcj_transform_range(raw, archive_payload_offset(raw), raw.size(), false); }

      fan::io::file::archive_extractor_t writer(out_dir, default_out);
      for (std::uint8_t b : raw) { writer.put(b); }
      writer.finish(); return true;
    } catch (...) { if (fp) { fan::io::file::close(fp); } throw; }
  }

  fan::bytes_t compress(const std::vector<fan::io::file_buffer_t>& files, compress_params_t params = params_max(), fan::progress_t* prog = nullptr) {
    if (files.size() > std::numeric_limits<std::uint32_t>::max()) { throw std::runtime_error("too many files"); }
    std::size_t total_sz = 4;
    for (const auto& f : files) {
      if (f.path.size() > std::numeric_limits<std::uint16_t>::max()) { throw std::runtime_error("path too long"); }
      total_sz += 2 + f.path.size() + 8;
    }
    std::size_t meta_pad = (4 - (total_sz & 3)) & 3; total_sz += meta_pad;
    for (const auto& f : files) { total_sz += f.data.size(); }

    fan::bytes_t raw; raw.reserve(total_sz); std::uint8_t u32_buf[4], u64_buf[8];
    fan::memory::write_le32(u32_buf, std::uint32_t(files.size())); raw.insert(raw.end(), u32_buf, u32_buf + 4);
    for (const auto& f : files) {
      std::uint8_t u16_buf[2]; fan::memory::write_le16(u16_buf, std::uint16_t(f.path.size())); raw.insert(raw.end(), u16_buf, u16_buf + 2);
      auto* p_path = reinterpret_cast<const std::uint8_t*>(f.path.data()); raw.insert(raw.end(), p_path, p_path + f.path.size());
      fan::memory::write_le64(u64_buf, f.data.size()); raw.insert(raw.end(), u64_buf, u64_buf + 8);
    }
    if (meta_pad) { raw.insert(raw.end(), meta_pad, 0); }
    std::size_t payload_offset = raw.size();
    for (const auto& f : files) { raw.insert(raw.end(), f.data.begin(), f.data.end()); }

    bool use_bcj = params.bcj && files.size() == 1 && files[0].data.size() >= 2 && files[0].data[0] == 'M' && files[0].data[1] == 'Z';
    if (use_bcj) { bcj_transform_range(raw, payload_offset, raw.size(), true); }

    int delta_stride = 1; bool use_delta = false, use_text = false, use_text2 = false; fan::bytes_t text_buf, text2_buf;
    bool try_text = !use_bcj && files.size() == 1 && looks_like_large_text(raw, payload_offset);
    if (try_text) { text_buf = text_encode_transform(raw); text2_buf = text2_encode_transform(raw); }
    if (!use_bcj && !try_text) {
      delta_stride = detect_delta_stride(raw); use_delta = delta_stride > 1;
      if (use_delta) { delta_encode(raw, delta_stride); }
    }
    int stride_log2 = delta_stride == 8 ? 3 : delta_stride == 4 ? 2 : delta_stride == 2 ? 1 : 0;
    std::uint8_t flags = std::uint8_t(use_bcj ? flag_bcj : 0) | std::uint8_t(use_delta ? flag_delta : 0) | std::uint8_t(stride_log2 << 2);

    fan::bytes_t comp; std::size_t selected_chunk_size = std::min<std::size_t>(std::max<std::size_t>(1, params.chunk_size), 1uz << 30);
    if (try_text) {
      std::vector<compress_candidate_t> candidates; candidates.push_back({std::move(raw), {}, "raw", flags, params.chunk_size});
      if (!text_buf.empty()) { candidates.push_back({std::move(text_buf), {}, "text", std::uint8_t(flags | flag_text), params.chunk_size}); }
      if (!text2_buf.empty()) { candidates.push_back({std::move(text2_buf), {}, "text2", std::uint8_t(flags | flag_text2), params.chunk_size}); }
      auto best = compress_best_candidate(candidates, params, prog, 0);
      raw = std::move(best.data); comp = std::move(best.comp); flags = best.flags; selected_chunk_size = best.chunk_size;
    } else {
      std::vector<compress_candidate_t> candidates; candidates.push_back({std::move(raw), {}, "raw", flags, params.chunk_size});
      auto best = compress_best_candidate(candidates, params, prog, 0);
      raw = std::move(best.data); comp = std::move(best.comp); flags = best.flags; selected_chunk_size = best.chunk_size;
    }

    fan::bytes_t result; result.reserve(comp.size() + 17); std::uint8_t header[17];
    fan::memory::write_le32(header, magic_v4); fan::memory::write_le64(header + 4, raw.size());
    fan::memory::write_le32(header + 12, std::uint32_t(selected_chunk_size)); header[16] = flags;
    result.insert(result.end(), header, header + sizeof(header)); result.insert(result.end(), comp.begin(), comp.end());
    return result;
  }

  std::vector<fan::io::file_buffer_t> decompress(const fan::bytes_t& comp, fan::progress_t* prog = nullptr) {
    std::vector<fan::io::file_buffer_t> files;
    if (comp.size() < 17 || fan::memory::read_le32(comp.data()) != magic_v4) { throw std::runtime_error("needs FCS4"); }
    std::uint64_t total_uncomp = fan::memory::read_le64(comp.data() + 4);
    std::size_t stored_chunk_size = fan::memory::read_le32(comp.data() + 12); if (stored_chunk_size == 0) { stored_chunk_size = default_chunk_size; }
    std::uint8_t flags = comp[16]; bool use_bcj = flags & flag_bcj, use_delta = flags & flag_delta, use_text = flags & flag_text, use_text2 = flags & flag_text2;
    int delta_stride = 1 << ((flags >> 2) & 7);
    if (prog) { prog->total.store(total_uncomp, std::memory_order_relaxed); }

    fan::bytes_t raw = decode_stream_seq(comp, 17, total_uncomp, stored_chunk_size, prog);
    if (use_delta) { delta_decode(raw, delta_stride); }
    if (use_text2) { raw = text2_decode_transform(raw); }
    if (use_text)  { raw = text_decode_transform(raw); }
    if (use_bcj)   { bcj_transform_range(raw, archive_payload_offset(raw), raw.size(), false); }
    if (raw.size() < 4) { return files; }

    std::size_t out_idx = 0; std::uint32_t num_files = fan::memory::read_le32(raw.data() + out_idx); out_idx += 4;
    struct meta_t { std::string path; std::uint64_t size; }; std::vector<meta_t> metas(num_files);
    for (auto& m : metas) {
      if (out_idx + 2 > raw.size()) { break; }
      std::uint16_t path_len = fan::memory::read_le16(raw.data() + out_idx); out_idx += 2;
      if (out_idx + path_len > raw.size()) { break; }
      m.path.assign(reinterpret_cast<const char*>(raw.data() + out_idx), path_len); out_idx += path_len;
      if (out_idx + 8 > raw.size()) { break; }
      m.size = fan::memory::read_le64(raw.data() + out_idx); out_idx += 8;
    }
    out_idx += (4 - (out_idx & 3)) & 3;
    for (auto& m : metas) {
      if (out_idx + m.size > raw.size()) { break; }
      fan::bytes_t data(raw.begin() + out_idx, raw.begin() + out_idx + std::size_t(m.size)); out_idx += std::size_t(m.size);
      files.push_back({std::move(m.path), std::move(data)});
    }
    return files;
  }
}