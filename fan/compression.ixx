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
  inline constexpr std::size_t chunk_size = 1uz << 26; // 64 MiB

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

  inline fan::bytes_t text_encode_transform(const fan::bytes_t& raw) {
    std::unordered_map<std::string, std::uint32_t> counts;
    counts.reserve(1 << 16);
    for (std::size_t i = 0; i < raw.size();) {
      std::uint8_t c = raw[i];
      if (!text_word_first(c)) { ++i; continue; }
      std::size_t b = i++;
      while (i < raw.size() && text_word_char(raw[i]) && i - b < 31) { ++i; }
      while (i < raw.size() && text_word_char(raw[i])) { ++i; }
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
      if (a.score != b.score) { return a.score > b.score; }
      return a.word < b.word;
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
    std::uint8_t buf[8];
    fan::memory::write_le64(buf, raw.size());
    out.insert(out.end(), buf, buf + 8);
    std::uint8_t dc[2];
    fan::memory::write_le16(dc, std::uint16_t(dict.size()));
    out.insert(out.end(), dc, dc + 2);
    for (const auto& w : dict) {
      out.push_back(std::uint8_t(w.size()));
      out.insert(out.end(), w.begin(), w.end());
    }

    for (std::size_t i = 0; i < raw.size();) {
      std::uint8_t c = raw[i];
      if (c == text_marker) {
        out.push_back(text_marker);
        out.push_back(0);
        ++i;
        continue;
      }
      if (!text_word_first(c)) {
        out.push_back(c);
        ++i;
        continue;
      }
      std::size_t b = i++;
      while (i < raw.size() && text_word_char(raw[i]) && i - b < 31) { ++i; }
      while (i < raw.size() && text_word_char(raw[i])) { ++i; }
      std::string_view w(reinterpret_cast<const char*>(raw.data() + b), i - b);
      auto it = ids.find(w);
      if (it == ids.end()) {
        out.insert(out.end(), raw.begin() + b, raw.begin() + i);
        continue;
      }
      std::uint32_t id = it->second;
      out.push_back(text_marker);
      if (id < 256) {
        out.push_back(1);
        out.push_back(std::uint8_t(id));
      } else {
        out.push_back(2);
        out.push_back(std::uint8_t(id));
        out.push_back(std::uint8_t(id >> 8));
      }
    }
    return out.size() + raw.size() / 64 < raw.size() ? std::move(out) : fan::bytes_t{};
  }

  inline fan::bytes_t text_decode_transform(const fan::bytes_t& in) {
    if (in.size() < 10) { throw std::runtime_error("corrupt text transform"); }
    std::size_t p = 0;
    std::uint64_t out_size = fan::memory::read_le64(in.data()); p += 8;
    std::uint16_t dict_count = fan::memory::read_le16(in.data() + p); p += 2;
    std::vector<std::string> dict;
    dict.reserve(dict_count);
    for (std::uint32_t i = 0; i < dict_count; ++i) {
      if (p >= in.size()) { throw std::runtime_error("corrupt text transform"); }
      std::uint8_t len = in[p++];
      if (p + len > in.size()) { throw std::runtime_error("corrupt text transform"); }
      dict.emplace_back(reinterpret_cast<const char*>(in.data() + p), len);
      p += len;
    }

    fan::bytes_t out;
    out.reserve(std::size_t(out_size));
    while (p < in.size()) {
      std::uint8_t c = in[p++];
      if (c != text_marker) {
        out.push_back(c);
        continue;
      }
      if (p >= in.size()) { throw std::runtime_error("corrupt text transform"); }
      std::uint8_t t = in[p++];
      if (t == 0) { out.push_back(text_marker); }
      else if (t == 1) {
        if (p >= in.size()) { throw std::runtime_error("corrupt text transform"); }
        std::uint32_t id = in[p++];
        if (id >= dict.size()) { throw std::runtime_error("corrupt text transform"); }
        out.insert(out.end(), dict[id].begin(), dict[id].end());
      } else if (t == 2) {
        if (p + 2 > in.size()) { throw std::runtime_error("corrupt text transform"); }
        std::uint32_t id = std::uint32_t(in[p]) | (std::uint32_t(in[p + 1]) << 8); p += 2;
        if (id >= dict.size()) { throw std::runtime_error("corrupt text transform"); }
        out.insert(out.end(), dict[id].begin(), dict[id].end());
      } else { throw std::runtime_error("corrupt text transform"); }
    }
    if (out.size() != out_size) { throw std::runtime_error("corrupt text transform"); }
    return out;
  }

  // --- match finder: h2/h3/h4 + hash chain, returns up to max_out candidates ---

  struct match_t { std::uint32_t offset = 0, length = 0; };

  struct match_finder_t {
    static constexpr std::uint32_t nil = 0xFFFFFFFFu;
    static constexpr std::uint32_t h2_size = 1 << 16;
    static constexpr std::uint32_t h3_size = 1 << 20;
    static constexpr std::uint32_t h4_size = 1 << 24;

    std::vector<std::uint32_t> h2, h3, h4, chain;
    std::size_t base = 0;

    match_finder_t(std::size_t d_start, std::size_t c_end)
      : h2(h2_size, nil), h3(h3_size, nil), h4(h4_size, nil),
        chain(c_end - d_start, nil), base(d_start) {
      if (c_end - d_start > std::numeric_limits<std::uint32_t>::max() - 1) {
        throw std::runtime_error("match window too large");
      }
    }

    static void hashes(const std::uint8_t* p, std::uint32_t& k2, std::uint32_t& k3, std::uint32_t& k4) {
      std::uint32_t v; std::memcpy(&v, p, 4);
      k2 = ((v & 0xFFFFu) * 0x9E3779B9u) >> 16;
      k3 = ((v & 0xFFFFFFu) * 0x1E35A7BDu) >> 12;
      k4 = (v * 0x9E3779B9u) >> 8;
    }
    std::uint32_t to_rel(std::size_t p) const { return std::uint32_t(p - base); }
    std::size_t to_abs(std::uint32_t p) const { return base + p; }
    std::uint32_t& link(std::size_t p) { return chain[p - base]; }

    void warm_up(const fan::bytes_t& d, std::size_t ws, std::size_t we) {
      for (std::size_t i = ws; i + 4 <= we; ++i) { insert(d, i); }
    }

    void insert(const fan::bytes_t& d, std::size_t i) {
      if (i + 4 > d.size()) { return; }
      std::uint32_t k2, k3, k4;
      hashes(d.data() + i, k2, k3, k4);
      std::uint32_t r = to_rel(i);
      link(i) = h4[k4];
      h2[k2] = h3[k3] = h4[k4] = r;
    }

    // returns number of matches written into out[], sorted ascending by length
    std::uint32_t find(const fan::bytes_t& src, std::size_t i, std::size_t c_end,
                       const std::array<std::uint32_t,4>& rep, std::uint32_t chain_limit,
                       bool push, std::span<match_t> out) {
      const std::uint32_t max_avail = std::uint32_t(c_end - i);
      std::uint32_t n_out = 0;
      if (i + 4 > c_end) { return 0; }

      const auto* po = src.data() + i;
      std::uint32_t k2, k3, k4;
      hashes(po, k2, k3, k4);

      // 2-byte quick hit
      std::uint32_t c2r = h2[k2];
      if (c2r != nil && max_avail >= 2) {
        std::size_t c2 = to_abs(c2r);
        if (c2 < i) {
          const auto* pc = src.data() + c2;
          if (pc[0] == po[0] && pc[1] == po[1]) {
            std::uint32_t len = 2;
            while (len < max_avail && po[len] == pc[len]) { ++len; }
            out[n_out++] = {std::uint32_t(i - c2), len};
          }
        }
      }
      // 3-byte quick hit
      std::uint32_t c3r = h3[k3];
      if (c3r != nil && max_avail >= 3) {
        std::size_t c3 = to_abs(c3r);
        if (c3 < i) {
          const auto* pc = src.data() + c3;
          if (pc[0] == po[0] && pc[1] == po[1] && pc[2] == po[2]) {
            std::uint32_t len = 3;
            while (len < max_avail && po[len] == pc[len]) { ++len; }
            if (n_out == 0 || len > out[n_out-1].length) { out[n_out++] = {std::uint32_t(i - c3), len}; }
          }
        }
      }

      // main chain walk
      std::uint32_t cur = h4[k4];
      if (push) {
        link(i) = cur;
        h2[k2] = h3[k3] = h4[k4] = to_rel(i);
      }

      std::uint32_t best_len = (n_out > 0) ? out[n_out-1].length : 0;
      std::uint32_t iters = 0;
      while (cur != nil && iters++ < chain_limit) {
        std::size_t cur_abs = to_abs(cur);
        if (cur_abs >= i) { break; }
        const auto* pc = src.data() + cur_abs;
        if (best_len > 0 && best_len < max_avail && po[best_len] != pc[best_len]) { cur = link(cur_abs); continue; }
        std::uint32_t v0, v1; std::memcpy(&v0, po, 4); std::memcpy(&v1, pc, 4);
        if (v0 == v1) {
          std::uint32_t len = 4;
          while (len + 8 <= max_avail) {
            std::uint64_t x, y; std::memcpy(&x, po+len, 8); std::memcpy(&y, pc+len, 8);
            if (auto d = x ^ y) { len += std::uint32_t(std::countr_zero(d)) >> 3; break; }
            len += 8;
          }
          while (len < max_avail && po[len] == pc[len]) { ++len; }
          if (len > best_len) {
            best_len = len;
            if (n_out < out.size()) { out[n_out++] = {std::uint32_t(i - cur_abs), len}; }
            else { out[n_out-1] = {std::uint32_t(i - cur_abs), len}; }
            if (len >= max_avail) { break; }
          }
        }
        cur = link(cur_abs);
      }
      return n_out;
    }
  };

  struct compress_params_t {
    std::uint32_t chain_main = 48;
    bool bcj = true;
    bool optimal = true; // true = DP optimal parser, false = lazy
    int lazy_depth = 2;  // only used when optimal=false
  };

  constexpr compress_params_t params_fast()   { return {16, true, false, 1}; }
  constexpr compress_params_t params_normal() { return {32, true, false, 2}; }
  constexpr compress_params_t params_max()    { return {48, true, true,  0}; }

  // --- LZMA 12-state model ---
  // states: 0-3 literal, 4 match-after-lit, 5 rep-after-lit, 6 shortrep-after-lit,
  //         7-9 lit-after-match, 10 match-after-match, 11 rep-after-match
  inline constexpr std::uint8_t state_lit_next[12]      = {0,0,0,0,1,2,3,4,5,6,4,5};
  inline constexpr std::uint8_t state_match_next[12]    = {7,7,7,7,7,7,7,10,10,10,10,10};
  inline constexpr std::uint8_t state_rep_next[12]      = {8,8,8,8,8,8,8,11,11,11,11,11};
  inline constexpr std::uint8_t state_shortrep_next[12] = {9,9,9,9,9,9,9,11,11,11,11,11};
  inline constexpr bool state_is_lit[12] = {true,true,true,true,true,true,true,false,false,false,false,false};

  enum class op_e : std::uint8_t { rep0, rep1, rep2, rep3, exp, none };
  struct seq_t { std::uint32_t lit_len; op_e op; std::uint32_t match_len, offset; };
  struct chunk_payload_t { std::vector<seq_t> seqs; };

  inline constexpr int num_dist_ctx = 16;
  inline constexpr std::uint32_t magic_v4 = 0x34334346;
  inline constexpr int pb = 2; // pos bits (0..3 = 4 states)
  inline constexpr int num_pos_states = 1 << pb;

  inline constexpr int lc = 3;
  inline constexpr int lp = 0;
  inline constexpr int num_lit_ctx = 1 << (lc + lp);

  constexpr std::uint32_t lit_ctx(std::size_t pos, std::uint8_t prev) {
    return ((std::uint32_t(pos) & ((1u << lp) - 1)) << lc) | (prev >> (8 - lc));
  }

  struct lzma_model_t {
    // per-state, per-pos_state: is_match, is_rep0, is_rep0_short
    std::array<std::array<std::uint16_t, num_pos_states>, 12> is_match;
    std::array<std::uint16_t, 12> is_rep;
    std::array<std::uint16_t, 12> is_rep0;
    std::array<std::uint16_t, 12> is_rep1;
    std::array<std::uint16_t, 12> is_rep2;
    std::array<std::array<std::uint16_t, num_pos_states>, 12> is_rep0_short;

    std::array<std::array<std::array<std::uint16_t, 512>, num_lit_ctx>, 2> lit;

    // length models: match and rep, each has low (pos_state, 8 syms), mid (8 syms), high (256 syms)
    struct len_model_t {
      std::array<std::array<std::uint16_t, num_pos_states>, 1> choice;   // [0]=low vs mid/high
      std::array<std::array<std::uint16_t, num_pos_states>, 1> choice2;  // [0]=mid vs high
      std::array<std::array<std::uint16_t, 8>, num_pos_states> low;
      std::array<std::array<std::uint16_t, 8>, num_pos_states> mid;
      std::array<std::uint16_t, 256> high;
    };
    len_model_t match_len, rep_len;

    // distance: pos_slot[len_state=min(len-2,3)][64 slots]
    //           pos_special trees for slots 4..13
    //           align[16]
    std::array<std::array<std::uint16_t, 64>, 4> pos_slot;
    std::array<std::array<std::uint16_t, 32>, 10> pos_special; // for slots 4..13 (index = slot-4)
    std::array<std::uint16_t, 16> align_bits;

    lzma_model_t() {
      auto fill = [](this const auto& self, auto& arr) -> void {
        for (auto& e : arr) {
          if constexpr (std::is_same_v<std::remove_reference_t<decltype(e)>, std::uint16_t>) { e = 1024; }
          else { self(e); }
        }
      };
      fill(is_match); fill(is_rep); fill(is_rep0); fill(is_rep1); fill(is_rep2); fill(is_rep0_short);
      fill(lit);
      fill(match_len.choice); fill(match_len.choice2); fill(match_len.low); fill(match_len.mid); fill(match_len.high);
      fill(rep_len.choice);   fill(rep_len.choice2);   fill(rep_len.low);   fill(rep_len.mid);   fill(rep_len.high);
      fill(pos_slot); fill(pos_special); fill(align_bits);
    }
  };

  // LZMA distance coding:
  // slot 0..3 -> offset = slot
  // slot >= 4: footer_bits = (slot>>1)-1; offset = ((2|(slot&1)) << footer_bits) + footer
  //   for slot 4..13: footer coded as tree in pos_special[slot-4]
  //   for slot >= 14: low bits direct, low 4 as align tree
  inline std::uint32_t slot_for_dist(std::uint32_t d) {
    // d is 0-based offset-1 (i.e. offset-1 for LZMA style)
    if (d < 4) { return d; }
    std::uint32_t k = std::bit_width(d) - 1; // highest bit pos
    return (k << 1) | ((d >> (k-1)) & 1);
  }
  inline std::uint32_t dist_from_slot_and_footer(std::uint32_t slot, std::uint32_t footer) {
    if (slot < 4) { return slot; }
    std::uint32_t footer_bits = (slot >> 1) - 1;
    return ((2u | (slot & 1u)) << footer_bits) | footer;
  }

  // --- range coder (unchanged) ---

  template <typename Writer>
  struct range_enc_t {
    Writer& out;
    std::uint64_t low = 0;
    std::uint32_t range = -1u, cache_size = 1;
    std::uint8_t cache = 0;

    void shift_low() {
      if (std::uint32_t(low) < 0xFF000000 || int(low >> 32) != 0) {
        out.write_byte(cache + std::uint8_t(low >> 32));
        out.write_repeat(std::uint8_t(low >> 32) ? 0x00 : 0xFF, cache_size - 1);
        cache = std::uint8_t(low >> 24); cache_size = 1;
      } else { ++cache_size; }
      low = std::uint32_t(low << 8);
    }
    void flush() { for (int i = 0; i < 5; ++i) { shift_low(); } }
    void encode(std::uint16_t& prob, bool bit) {
      std::uint32_t bound = (range >> 11) * prob;
      if (!bit) { range = bound; prob += (2048 - prob) >> 5; }
      else { low += bound; range -= bound; prob -= prob >> 5; }
      while (range < 0x1000000) { range <<= 8; shift_low(); }
    }
    void encode_tree(std::span<std::uint16_t> tree, std::uint32_t sym, int bits) {
      for (int ctx = 1, i = bits - 1; i >= 0; --i) {
        bool bit = (sym >> i) & 1; encode(tree[ctx], bit); ctx = (ctx << 1) | bit;
      }
    }
    void encode_direct(bool bit) {
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

    range_dec_t(Reader& r) : in(r) {
      read_byte();
      for (int i = 0; i < 4; ++i) { code = (code << 8) | read_byte(); }
    }
    std::uint8_t read_byte() { return in.read_byte(); }
    bool decode(std::uint16_t& prob) {
      std::uint32_t bound = (range >> 11) * prob;
      bool bit = code >= bound;
      if (!bit) { range = bound; prob += (2048 - prob) >> 5; }
      else { range -= bound; code -= bound; prob -= prob >> 5; }
      while (range < 0x1000000) { range <<= 8; code = (code << 8) | read_byte(); }
      return bit;
    }
    std::uint32_t decode_tree(std::span<std::uint16_t> tree, int bits) {
      std::uint32_t ctx = 1;
      for (int i = 0; i < bits; ++i) { ctx = (ctx << 1) | decode(tree[ctx]); }
      return ctx - (1u << bits);
    }
    bool decode_direct() {
      range >>= 1; bool bit = code >= range;
      if (bit) { code -= range; }
      while (range < 0x1000000) { range <<= 8; code = (code << 8) | read_byte(); }
      return bit;
    }
    std::uint32_t decode_direct_n(int bits) {
      std::uint32_t v = 0;
      for (int i = 0; i < bits; ++i) { v = (v << 1) | decode_direct(); }
      return v;
    }
  };

  // --- length encoding (LZMA style: low 3 bits, mid 3 bits, high 8 bits) ---
  // encoded value: 0..7 -> low, 8..15 -> mid, 16..271 -> high
  // match length offset: 2 (min match = 2, encode len-2)
  // rep length offset: 1 (min rep = 1, encode len-1) but we encode rep0 as len-1

  template <typename Writer>
  inline void encode_len(range_enc_t<Writer>& rc, lzma_model_t::len_model_t& lm, std::uint32_t len, int ps) {
    if (len < 8) {
      rc.encode(lm.choice[0][ps], false);
      rc.encode_tree(std::span<std::uint16_t>{lm.low[ps]}, len, 3);
    } else if (len < 16) {
      rc.encode(lm.choice[0][ps], true);
      rc.encode(lm.choice2[0][ps], false);
      rc.encode_tree(std::span<std::uint16_t>{lm.mid[ps]}, len - 8, 3);
    } else {
      rc.encode(lm.choice[0][ps], true);
      rc.encode(lm.choice2[0][ps], true);
      rc.encode_tree(std::span<std::uint16_t>{lm.high}, len - 16, 8);
    }
  }

  template <typename Reader>
  inline std::uint32_t decode_len(range_dec_t<Reader>& rd, lzma_model_t::len_model_t& lm, int ps) {
    if (!rd.decode(lm.choice[0][ps])) { return rd.decode_tree(std::span<std::uint16_t>{lm.low[ps]}, 3); }
    if (!rd.decode(lm.choice2[0][ps])) { return 8 + rd.decode_tree(std::span<std::uint16_t>{lm.mid[ps]}, 3); }
    return 16 + rd.decode_tree(std::span<std::uint16_t>{lm.high}, 8);
  }

  // --- distance encoding ---

  template <typename Writer>
  inline void encode_dist(range_enc_t<Writer>& rc, lzma_model_t& m, std::uint32_t dist, std::uint32_t len_state) {
    std::uint32_t d = dist - 1; // 0-based
    std::uint32_t slot = slot_for_dist(d);
    rc.encode_tree(std::span<std::uint16_t>{m.pos_slot[len_state]}, slot, 6);
    if (slot >= 4) {
      std::uint32_t footer_bits = (slot >> 1) - 1;
      std::uint32_t base_val = (2u | (slot & 1u)) << footer_bits;
      std::uint32_t footer = d - base_val;
      if (slot < 14) {
        rc.encode_tree(std::span<std::uint16_t>{m.pos_special[slot - 4]}, footer, footer_bits);
      } else {
        std::uint32_t direct_bits = footer_bits - 4;
        rc.encode_direct_n(footer >> 4, direct_bits);
        rc.encode_tree(std::span<std::uint16_t>{m.align_bits}, footer & 0xFu, 4);
      }
    }
  }

  template <typename Reader>
  inline std::uint32_t decode_dist(range_dec_t<Reader>& rd, lzma_model_t& m, std::uint32_t len_state) {
    std::uint32_t slot = rd.decode_tree(std::span<std::uint16_t>{m.pos_slot[len_state]}, 6);
    if (slot < 4) { return slot + 1; }
    std::uint32_t footer_bits = (slot >> 1) - 1;
    std::uint32_t base_val = (2u | (slot & 1u)) << footer_bits;
    std::uint32_t footer;
    if (slot < 14) {
      footer = rd.decode_tree(std::span<std::uint16_t>{m.pos_special[slot - 4]}, footer_bits);
    } else {
      std::uint32_t direct_bits = footer_bits - 4;
      footer = (rd.decode_direct_n(direct_bits) << 4) | rd.decode_tree(std::span<std::uint16_t>{m.align_bits}, 4);
    }
    return base_val + footer + 1;
  }

  // --- literal encoding (with matched literal support) ---

  template <typename Writer>
  inline void encode_literal(range_enc_t<Writer>& rc, lzma_model_t& m, std::uint8_t state,
                              std::size_t pos, std::uint8_t prev, std::uint8_t byte, std::uint8_t match_byte) {
    int sc = state_is_lit[state] ? 0 : 1;
    int ctx = lit_ctx(pos, prev);
    auto& tree = m.lit[sc][ctx];
    if (sc == 0) {
      // normal literal: tree of 256 bits in positions 1..255
      rc.encode_tree(std::span<std::uint16_t>{tree}.subspan(0, 256), byte, 8);
    } else {
      // matched literal: bit-by-bit, prediction from match_byte
      std::uint32_t sym = 1, mb = match_byte;
      for (int i = 7; i >= 0; --i) {
        bool bit = (byte >> i) & 1;
        bool mb_bit = (mb >> 7) & 1; mb <<= 1;
        std::uint32_t idx = sym + (mb_bit ? 256 : 0);
        rc.encode(tree[idx], bit);
        sym = (sym << 1) | bit;
      }
    }
  }

  template <typename Reader>
  inline std::uint8_t decode_literal(range_dec_t<Reader>& rd, lzma_model_t& m, std::uint8_t state,
                                     std::size_t pos, std::uint8_t prev, std::uint8_t match_byte) {
    int sc = state_is_lit[state] ? 0 : 1;
    int ctx = lit_ctx(pos, prev);
    auto& tree = m.lit[sc][ctx];
    if (sc == 0) {
      return std::uint8_t(rd.decode_tree(std::span<std::uint16_t>{tree}.subspan(0, 256), 8));
    } else {
      std::uint32_t sym = 1, mb = match_byte;
      for (int i = 7; i >= 0; --i) {
        bool mb_bit = (mb >> 7) & 1; mb <<= 1;
        sym = (sym << 1) | rd.decode(tree[sym + (mb_bit ? 256 : 0)]);
      }
      return std::uint8_t(sym & 0xFF);
    }
  }

  // --- price estimation for optimal parser ---
  inline constexpr std::uint32_t price_scale = 64;
  inline constexpr std::uint32_t price_bit0 = 0, price_bit1 = 0;

  // Fast bit price: returns scaled cost in units of 1/price_scale bits
  inline std::uint32_t bit_price(std::uint16_t prob, bool bit) {
    // prob in [0,2048]. Use log2 approximation.
    // cost0 = -log2(prob/2048), cost1 = -log2((2048-prob)/2048)
    static std::array<std::uint32_t, 2049> price_table = [] {
      std::array<std::uint32_t, 2049> t{};
      for (int i = 1; i <= 2048; ++i) {
        double p = double(i) / 2048.0;
        t[i] = std::uint32_t(-std::log2(p) * price_scale + 0.5);
      }
      t[0] = t[1]; // guard
      return t;
    }();
    return bit ? price_table[2048 - prob] : price_table[prob];
  }

  inline std::uint32_t tree_price(std::span<const std::uint16_t> tree, std::uint32_t sym, int bits) {
    std::uint32_t cost = 0;
    for (int ctx = 1, i = bits - 1; i >= 0; --i) {
      bool bit = (sym >> i) & 1;
      cost += bit_price(tree[ctx], bit);
      ctx = (ctx << 1) | bit;
    }
    return cost;
  }

  inline std::uint32_t len_price(lzma_model_t::len_model_t& lm, std::uint32_t len, int ps) {
    std::uint32_t cost = 0;
    if (len < 8) {
      cost += bit_price(lm.choice[0][ps], false);
      cost += tree_price(std::span<const std::uint16_t>{lm.low[ps]}, len, 3);
    } else if (len < 16) {
      cost += bit_price(lm.choice[0][ps], true);
      cost += bit_price(lm.choice2[0][ps], false);
      cost += tree_price(std::span<const std::uint16_t>{lm.mid[ps]}, len - 8, 3);
    } else {
      cost += bit_price(lm.choice[0][ps], true);
      cost += bit_price(lm.choice2[0][ps], true);
      cost += tree_price(std::span<const std::uint16_t>{lm.high}, len - 16, 8);
    }
    return cost;
  }

  inline std::uint32_t dist_price(lzma_model_t& m, std::uint32_t dist, std::uint32_t len_state) {
    std::uint32_t d = dist - 1;
    std::uint32_t slot = slot_for_dist(d);
    std::uint32_t cost = tree_price(std::span<const std::uint16_t>{m.pos_slot[len_state]}, slot, 6);
    if (slot >= 4) {
      std::uint32_t footer_bits = (slot >> 1) - 1;
      std::uint32_t base_val = (2u | (slot & 1u)) << footer_bits;
      std::uint32_t footer = d - base_val;
      if (slot < 14) {
        cost += tree_price(std::span<const std::uint16_t>{m.pos_special[slot - 4]}, footer, footer_bits);
      } else {
        cost += (footer_bits - 4) * price_scale; // direct bits each cost 1 bit
        cost += tree_price(std::span<const std::uint16_t>{m.align_bits}, footer & 0xFu, 4);
      }
    }
    return cost;
  }

  inline void update_bit(std::uint16_t& prob, bool bit) {
    if (!bit) { prob += (2048 - prob) >> 5; }
    else { prob -= prob >> 5; }
  }

  inline void update_tree(std::span<std::uint16_t> tree, std::uint32_t sym, int bits) {
    for (int ctx = 1, i = bits - 1; i >= 0; --i) {
      bool bit = (sym >> i) & 1;
      update_bit(tree[ctx], bit);
      ctx = (ctx << 1) | bit;
    }
  }

  inline void update_len(lzma_model_t::len_model_t& lm, std::uint32_t len, int ps) {
    if (len < 8) {
      update_bit(lm.choice[0][ps], false);
      update_tree(std::span<std::uint16_t>{lm.low[ps]}, len, 3);
    } else if (len < 16) {
      update_bit(lm.choice[0][ps], true);
      update_bit(lm.choice2[0][ps], false);
      update_tree(std::span<std::uint16_t>{lm.mid[ps]}, len - 8, 3);
    } else {
      update_bit(lm.choice[0][ps], true);
      update_bit(lm.choice2[0][ps], true);
      update_tree(std::span<std::uint16_t>{lm.high}, len - 16, 8);
    }
  }

  inline void update_dist(lzma_model_t& m, std::uint32_t dist, std::uint32_t len_state) {
    std::uint32_t d = dist - 1;
    std::uint32_t slot = slot_for_dist(d);
    update_tree(std::span<std::uint16_t>{m.pos_slot[len_state]}, slot, 6);
    if (slot >= 4) {
      std::uint32_t footer_bits = (slot >> 1) - 1;
      std::uint32_t base_val = (2u | (slot & 1u)) << footer_bits;
      std::uint32_t footer = d - base_val;
      if (slot < 14) {
        update_tree(std::span<std::uint16_t>{m.pos_special[slot - 4]}, footer, footer_bits);
      } else {
        update_tree(std::span<std::uint16_t>{m.align_bits}, footer & 0xFu, 4);
      }
    }
  }

  inline void update_literal(lzma_model_t& m, std::uint8_t state, std::size_t pos, std::uint8_t prev, std::uint8_t byte, std::uint8_t match_byte) {
    int sc = state_is_lit[state] ? 0 : 1;
    int ctx = lit_ctx(pos, prev);
    auto& tree = m.lit[sc][ctx];
    if (sc == 0) {
      update_tree(std::span<std::uint16_t>{tree}.subspan(0, 256), byte, 8);
    } else {
      std::uint32_t sym = 1, mb = match_byte;
      for (int i = 7; i >= 0; --i) {
        bool bit = (byte >> i) & 1;
        bool mb_bit = (mb >> 7) & 1; mb <<= 1;
        std::uint32_t idx = sym + (mb_bit ? 256 : 0);
        update_bit(tree[idx], bit);
        sym = (sym << 1) | bit;
      }
    }
  }

  inline void update_literal_symbol(lzma_model_t& m, std::uint8_t state, std::size_t pos, std::uint8_t prev, std::uint8_t byte, std::uint8_t match_byte) {
    update_bit(m.is_match[state][pos & (num_pos_states - 1)], false);
    update_literal(m, state, pos, prev, byte, match_byte);
  }

  inline void update_match_symbol(lzma_model_t& m, std::uint8_t state, std::size_t pos, op_e op, std::uint32_t len, std::uint32_t dist) {
    int pst = int(pos) & (num_pos_states - 1);
    update_bit(m.is_match[state][pst], true);
    if (op == op_e::exp) {
      update_bit(m.is_rep[state], false);
      update_len(m.match_len, len - 2, pst);
      update_dist(m, dist, std::min<std::uint32_t>(len - 2, 3));
      return;
    }
    update_bit(m.is_rep[state], true);
    int r = int(op);
    if (r == 0) {
      update_bit(m.is_rep0[state], true);
      if (len == 1) {
        update_bit(m.is_rep0_short[state][pst], false);
      } else {
        update_bit(m.is_rep0_short[state][pst], true);
        update_len(m.rep_len, len - 1, pst);
      }
      return;
    }
    update_bit(m.is_rep0[state], false);
    if (r == 1) { update_bit(m.is_rep1[state], false); }
    else { update_bit(m.is_rep1[state], true); update_bit(m.is_rep2[state], r == 2 ? false : true); }
    update_len(m.rep_len, len - 1, pst);
  }

  // --- DP optimal parser ---

  inline constexpr std::uint32_t kNumOpts = 1 << 11;
  inline constexpr std::uint32_t max_exp_len = 273;
  inline constexpr std::uint32_t max_rep_len = 272;
  inline constexpr std::uint32_t progress_step = 4096;
  inline constexpr std::uint64_t progress_scale = 100;
  inline constexpr std::uint64_t parse_progress_weight = 80;
  inline constexpr std::uint64_t encode_progress_weight = 20;
  inline constexpr std::uint32_t inf_price = 0xFFFFFFFFu;

  struct opt_t {
    std::uint32_t price = inf_price;
    std::uint32_t prev = 0;
    std::uint32_t len = 1;
    std::uint32_t offset = 0;
    op_e op = op_e::none;
    std::array<std::uint32_t, 4> rep{};
    std::uint8_t state = 0;
  };

  chunk_payload_t parse_chunk_optimal(const fan::bytes_t& src, std::size_t c_start, std::size_t c_end,
                                      const compress_params_t& params, fan::progress_t* prog) {
    chunk_payload_t out;
    out.seqs.reserve((c_end - c_start) / 5);
    std::size_t d_start = c_start >= chunk_size ? c_start - chunk_size : 0;
    std::size_t ci = c_start, cend = c_end;

    match_finder_t finder(d_start, cend);
    finder.warm_up(src, d_start, ci);

    auto m_ptr = std::make_unique<lzma_model_t>();
    lzma_model_t& model = *m_ptr;

    std::array<std::uint32_t, 4> g_rep{1, 1, 1, 1};
    std::uint8_t g_state = 0;
    std::uint32_t lit_cnt = 0;
    std::size_t last_report = c_start;

    std::vector<opt_t> opts(kNumOpts + 1);
    std::array<match_t, 64> matches;

    auto ps = [](std::size_t pos) { return int(pos & (num_pos_states - 1)); };
    auto len_state = [](std::uint32_t len) { return std::min<std::uint32_t>(len - 2, 3); };

    // price a literal at position pos, given current state
    auto lit_price = [&](std::size_t pos, std::uint8_t state, std::uint8_t prev, std::uint8_t byte, std::uint32_t rep0) -> std::uint32_t {
      int sc = state_is_lit[state] ? 0 : 1;
      std::uint32_t cost = bit_price(model.is_match[state][ps(pos)], false);
      int ctx = lit_ctx(pos, prev);
      auto& tree = model.lit[sc][ctx];
      if (sc == 0) {
        cost += tree_price(std::span<const std::uint16_t>{tree}.subspan(0, 256), byte, 8);
      } else {
        std::uint8_t mb = (pos >= rep0) ? src[pos - rep0] : 0;
        std::uint32_t sym = 1, mbb = mb;
        for (int i = 7; i >= 0; --i) {
          bool mb_bit = (mbb >> 7) & 1; mbb <<= 1;
          bool bit = (byte >> i) & 1;
          std::uint32_t idx = sym + (mb_bit ? 256 : 0);
          cost += bit_price(tree[idx], bit);
          sym = (sym << 1) | bit;
        }
      }
      return cost;
    };

    // price a rep match
    auto rep_match_price = [&](std::size_t pos, std::uint8_t state, int r, std::uint32_t len) -> std::uint32_t {
      std::uint32_t cost = bit_price(model.is_match[state][ps(pos)], true);
      cost += bit_price(model.is_rep[state], true);
      if (r == 0) {
        if (len == 1) {
          cost += bit_price(model.is_rep0[state], true);
          cost += bit_price(model.is_rep0_short[state][ps(pos)], false);
        } else {
          cost += bit_price(model.is_rep0[state], true);
          cost += bit_price(model.is_rep0_short[state][ps(pos)], true);
          cost += len_price(model.rep_len, len - 1, ps(pos));
        }
      } else {
        cost += bit_price(model.is_rep0[state], false);
        if (r == 1) { cost += bit_price(model.is_rep1[state], false); }
        else {
          cost += bit_price(model.is_rep1[state], true);
          cost += bit_price(model.is_rep2[state], r == 2 ? false : true);
        }
        cost += len_price(model.rep_len, len - 1, ps(pos));
      }
      return cost;
    };

    // price an explicit match
    auto exp_match_price = [&](std::size_t pos, std::uint8_t state, std::uint32_t dist, std::uint32_t len) -> std::uint32_t {
      std::uint32_t cost = bit_price(model.is_match[state][ps(pos)], true);
      cost += bit_price(model.is_rep[state], false);
      cost += len_price(model.match_len, len - 2, ps(pos));
      cost += dist_price(model, dist, len_state(len));
      return cost;
    };

    auto try_lens = [](std::uint32_t min_l, std::uint32_t max_l, auto&& fn) {
      if (max_l < min_l) { return; }
      std::uint32_t last = 0;
      auto put = [&](std::uint32_t v) {
        if (v < min_l || v > max_l || v == last) { return; }
        fn(v);
        last = v;
      };
      for (std::uint32_t v = min_l, e = std::min(max_l, 16u); v <= e; ++v) { put(v); }
      for (std::uint32_t v : {24u, 32u, 48u, 64u, 96u, 128u, 192u, max_exp_len}) { put(v); }
      put(max_l);
    };

    if (!params.optimal) {
      // --- lazy parser fallback ---
      auto best_cand = [&](std::size_t p, bool push, std::uint32_t cl) {
        struct cand_t { op_e op; std::uint32_t len; std::uint32_t off; std::uint32_t price; } b{op_e::none, 0, 0, inf_price};
        std::uint32_t avail = std::uint32_t(cend - p);
        if (!avail) { return b; }
        for (int r = 0; r < 4; ++r) {
          if (p < d_start + g_rep[r]) { continue; }
          std::uint32_t max_l = std::min(avail, max_rep_len);
          std::uint32_t l = 0;
          while (l < max_l && src[p+l] == src[p - g_rep[r] + l]) { ++l; }
          if (l == 0 || (l == 1 && r > 0)) { continue; }
          std::uint32_t pft = rep_match_price(p, g_state, r, l);
          if (pft < b.price || (pft == b.price && l > b.len)) { b = {static_cast<op_e>(r), l, g_rep[r], pft}; }
        }
        if (avail >= 4) {
          std::uint32_t n = finder.find(src, p, cend, g_rep, cl, push, std::span<match_t>{matches});
          for (std::uint32_t mi = 0; mi < n; ++mi) {
            auto& mc = matches[mi];
            if (mc.length < 2) { continue; }
            std::uint32_t len = std::min<std::uint32_t>(mc.length, max_exp_len);
            std::uint32_t pft = exp_match_price(p, g_state, mc.offset, len);
            if (pft < b.price || (pft == b.price && len > b.len)) { b = {op_e::exp, len, mc.offset, pft}; }
          }
        } else if (push) { finder.insert(src, p); }
        return b;
      };

      while (ci < cend) {
        auto M = best_cand(ci, true, params.chain_main);
        if (M.len == 0 || M.op == op_e::none) {
          std::uint8_t prev = ci > 0 ? src[ci - 1] : 0;
          std::uint8_t mb = (ci >= g_rep[0]) ? src[ci - g_rep[0]] : 0;
          update_literal_symbol(model, g_state, ci, prev, src[ci], mb);
          g_state = state_lit_next[g_state];
          ++lit_cnt;
          ++ci;
        }
        else {
          bool took_lazy = false;
          if (ci + 4 < cend && M.len < 128) {
            for (int skip = 1; skip <= params.lazy_depth && ci + skip < cend; ++skip) {
              auto N = best_cand(ci + skip, false, params.chain_main >> 1);
              if (N.price < M.price + std::uint32_t(skip) * 8 * price_scale) {
                for (int k = 0; k < skip; ++k) {
                  std::uint8_t prev = ci > 0 ? src[ci - 1] : 0;
                  std::uint8_t mb = (ci >= g_rep[0]) ? src[ci - g_rep[0]] : 0;
                  update_literal_symbol(model, g_state, ci, prev, src[ci], mb);
                  g_state = state_lit_next[g_state];
                  ++lit_cnt;
                  ++ci;
                }
                took_lazy = true;
                break;
              }
            }
          }
          if (took_lazy) { continue; }
          out.seqs.push_back({lit_cnt, M.op, M.len, M.off});
          update_match_symbol(model, g_state, ci, M.op, M.len, M.off);
          if      (M.op == op_e::rep0) { g_state = (M.len == 1) ? state_shortrep_next[g_state] : state_rep_next[g_state]; }
          else if (M.op == op_e::rep1) { std::swap(g_rep[0], g_rep[1]); g_state = state_rep_next[g_state]; }
          else if (M.op == op_e::rep2) { auto t=g_rep[2]; g_rep[2]=g_rep[1]; g_rep[1]=g_rep[0]; g_rep[0]=t; g_state = state_rep_next[g_state]; }
          else if (M.op == op_e::rep3) { auto t=g_rep[3]; g_rep[3]=g_rep[2]; g_rep[2]=g_rep[1]; g_rep[1]=g_rep[0]; g_rep[0]=t; g_state = state_rep_next[g_state]; }
          else { g_rep[3]=g_rep[2]; g_rep[2]=g_rep[1]; g_rep[1]=g_rep[0]; g_rep[0]=M.off; g_state = state_match_next[g_state]; }
          lit_cnt = 0;
          std::uint32_t step = M.len < 32 ? 1 : M.len >> 4;
          for (std::uint32_t k = 1; k < M.len; k += step) {
            if (ci + k + 2 < cend) { finder.insert(src, ci + k); }
          }
          ci += M.len;
        }
        if (prog && ci - last_report >= progress_step) {
          prog->done.fetch_add((ci - last_report) * parse_progress_weight, std::memory_order_relaxed);
          last_report = ci;
        }
      }
      if (prog) { prog->done.fetch_add((cend - last_report) * parse_progress_weight, std::memory_order_relaxed); }
      if (lit_cnt) { out.seqs.push_back({lit_cnt, op_e::none, 0, 0}); }
      return out;
    }

    // --- DP optimal parser ---
    while (ci < cend) {
      std::uint32_t avail = std::uint32_t(cend - ci);
      std::uint32_t window = std::min(avail, kNumOpts - 1);

      opts[0].price = 0;
      opts[0].state = g_state;
      opts[0].rep = g_rep;
      for (std::uint32_t k = 1; k <= window; ++k) { opts[k].price = inf_price; }

      std::uint32_t len_end = 1;

      for (std::uint32_t j = 0; j < window; ++j) {
        if (opts[j].price == inf_price) { continue; }
        std::uint8_t st = opts[j].state;
        std::array<std::uint32_t,4> rep = opts[j].rep;
        std::size_t pos = ci + j;

        auto try_price = [&](std::uint32_t k, std::uint32_t price, op_e op, std::uint32_t len, std::uint32_t offset, std::uint8_t nst, std::array<std::uint32_t,4> nrep) {
          if (k > window) { return; }
          if (price < opts[k].price) {
            opts[k].price = price;
            opts[k].prev = j;
            opts[k].len = len;
            opts[k].op = op;
            opts[k].offset = offset;
            opts[k].state = nst;
            opts[k].rep = nrep;
            if (k > len_end) { len_end = k; }
          }
        };

        // literal
        if (pos < cend) {
          std::uint8_t prev = pos > 0 ? src[pos - 1] : 0;
          std::uint32_t lp = lit_price(pos, st, prev, src[pos], rep[0]);
          std::array<std::uint32_t,4> nrep = rep;
          try_price(j+1, opts[j].price + lp, op_e::none, 1, 0, state_lit_next[st], nrep);
        }

        // rep matches
        for (int r = 0; r < 4 && pos < cend; ++r) {
          if (pos < d_start + rep[r]) { continue; }
          std::uint32_t max_l = std::min<std::uint32_t>(std::uint32_t(cend - pos), max_rep_len);
          std::uint32_t l = 0;
          while (l < max_l && src[pos+l] == src[pos - rep[r] + l]) { ++l; }
          if (l == 0) { continue; }
          // short rep (rep0, len=1)
          if (r == 0 && l >= 1) {
            std::uint32_t cost = opts[j].price + bit_price(model.is_match[st][ps(pos)], true)
              + bit_price(model.is_rep[st], true)
              + bit_price(model.is_rep0[st], true)
              + bit_price(model.is_rep0_short[st][ps(pos)], false);
            std::array<std::uint32_t,4> nrep = rep;
            try_price(j+1, cost, op_e::rep0, 1, rep[0], state_shortrep_next[st], nrep);
          }
          try_lens(2, l, [&](std::uint32_t ml) {
            std::uint32_t cost = opts[j].price + rep_match_price(pos, st, r, ml);
            std::array<std::uint32_t,4> nrep = rep;
            if      (r == 1) { std::swap(nrep[0], nrep[1]); }
            else if (r == 2) { auto t=nrep[2]; nrep[2]=nrep[1]; nrep[1]=nrep[0]; nrep[0]=t; }
            else if (r == 3) { auto t=nrep[3]; nrep[3]=nrep[2]; nrep[2]=nrep[1]; nrep[1]=nrep[0]; nrep[0]=t; }
            op_e op = static_cast<op_e>(r);
            try_price(j + ml, cost, op, ml, rep[r], state_rep_next[st], nrep);
          });
        }

        // explicit matches
        if (pos + 4 <= cend) {
          std::uint32_t chain = j == 0 ? params.chain_main : std::min<std::uint32_t>(params.chain_main, 8);
          std::uint32_t n = finder.find(src, pos, cend, rep, chain, true, std::span<match_t>{matches});
          for (std::uint32_t mi = 0; mi < n; ++mi) {
            auto& mc = matches[mi];
            if (mc.length < 2) { continue; }
            std::uint32_t max_l = std::min({mc.length, std::uint32_t(cend - pos), max_exp_len});
            try_lens(2, max_l, [&](std::uint32_t ml) {
              std::uint32_t cost = opts[j].price + exp_match_price(pos, st, mc.offset, ml);
              std::array<std::uint32_t,4> nrep = rep;
              nrep[3]=nrep[2]; nrep[2]=nrep[1]; nrep[1]=nrep[0]; nrep[0]=mc.offset;
              try_price(j + ml, cost, op_e::exp, ml, mc.offset, state_match_next[st], nrep);
            });
          }
        }
      }

      // trace back optimal path from len_end to 0
      // collect decisions in reverse, then emit
      struct decision_t { std::size_t pos; op_e op; std::uint32_t len; std::uint32_t offset; };
      std::vector<decision_t> path;
      std::uint32_t cur = len_end;
      while (cur > 0) {
        path.push_back({ci + opts[cur].prev, opts[cur].op, opts[cur].len, opts[cur].offset});
        cur = opts[cur].prev;
      }
      std::reverse(path.begin(), path.end());

      for (auto& d : path) {
        std::uint32_t advance = d.len;
        if (d.op == op_e::none) {
          std::uint8_t prev = ci > 0 ? src[ci - 1] : 0;
          std::uint8_t mb = (ci >= g_rep[0]) ? src[ci - g_rep[0]] : 0;
          update_literal_symbol(model, g_state, ci, prev, src[ci], mb);
          ++lit_cnt;
          g_state = state_lit_next[g_state];
          ci++;
        } else {
          out.seqs.push_back({lit_cnt, d.op, d.len, d.offset});
          update_match_symbol(model, g_state, ci, d.op, d.len, d.offset);
          lit_cnt = 0;
          if      (d.op == op_e::rep0) { g_state = (d.len == 1) ? state_shortrep_next[g_state] : state_rep_next[g_state]; }
          else if (d.op == op_e::rep1) { std::swap(g_rep[0], g_rep[1]); g_state = state_rep_next[g_state]; }
          else if (d.op == op_e::rep2) { auto t=g_rep[2]; g_rep[2]=g_rep[1]; g_rep[1]=g_rep[0]; g_rep[0]=t; g_state = state_rep_next[g_state]; }
          else if (d.op == op_e::rep3) { auto t=g_rep[3]; g_rep[3]=g_rep[2]; g_rep[2]=g_rep[1]; g_rep[1]=g_rep[0]; g_rep[0]=t; g_state = state_rep_next[g_state]; }
          else { g_rep[3]=g_rep[2]; g_rep[2]=g_rep[1]; g_rep[1]=g_rep[0]; g_rep[0]=d.offset; g_state = state_match_next[g_state]; }
          ci += advance;
        }
      }

      if (prog && ci - last_report >= progress_step) {
        prog->done.fetch_add((ci - last_report) * parse_progress_weight, std::memory_order_relaxed);
        last_report = ci;
      }
    }

    if (prog) { prog->done.fetch_add((cend - last_report) * parse_progress_weight, std::memory_order_relaxed); }
    if (lit_cnt) { out.seqs.push_back({lit_cnt, op_e::none, 0, 0}); }
    return out;
  }

  // --- stream encode/decode using new LZMA model ---

  fan::bytes_t encode_stream_seq(const fan::bytes_t& src, const std::vector<chunk_payload_t>& blocks, fan::progress_t* prog) {
    fan::bytes_t out; out.reserve(src.size() / 5);
    fan::io::bytes_writer_t bw{out};
    range_enc_t<fan::io::bytes_writer_t> rc{bw};
    auto m_ptr = std::make_unique<lzma_model_t>();
    lzma_model_t& model = *m_ptr;

    std::size_t src_ptr = 0;
    std::size_t last_report = 0;
    std::array<std::uint32_t,4> rep{1,1,1,1};
    std::uint8_t state = 0;
    std::size_t next_boundary = chunk_size;

    auto report = [&] {
      if (prog && src_ptr - last_report >= progress_step) {
        prog->done.fetch_add((src_ptr - last_report) * encode_progress_weight, std::memory_order_relaxed);
        last_report = src_ptr;
      }
    };

    for (std::size_t k = 0; k < blocks.size(); ++k) {
      if (src_ptr == next_boundary) { rep = {1,1,1,1}; state = 0; model = lzma_model_t(); next_boundary += chunk_size; }
      for (const auto& s : blocks[k].seqs) {
        int pst = int(src_ptr) & (num_pos_states - 1);

        // encode literals
        for (std::uint32_t j = 0; j < s.lit_len; ++j) {
          rc.encode(model.is_match[state][pst], false);
          std::uint8_t prev = src_ptr > 0 ? src[src_ptr - 1] : 0;
          std::uint8_t mb = (src_ptr >= rep[0]) ? src[src_ptr - rep[0]] : 0;
          encode_literal(rc, model, state, src_ptr, prev, src[src_ptr], mb);
          state = state_lit_next[state];
          ++src_ptr;
          pst = int(src_ptr) & (num_pos_states - 1);
          report();
        }

        if (s.op == op_e::none) { continue; }

        // encode op
        rc.encode(model.is_match[state][pst], true);
        if (s.op == op_e::rep0 || s.op == op_e::rep1 || s.op == op_e::rep2 || s.op == op_e::rep3) {
          rc.encode(model.is_rep[state], true);
          int r = int(s.op);
          if (r == 0) {
            rc.encode(model.is_rep0[state], true);
            if (s.match_len == 1) {
              rc.encode(model.is_rep0_short[state][pst], false);
              state = state_shortrep_next[state];
            } else {
              rc.encode(model.is_rep0_short[state][pst], true);
              encode_len(rc, model.rep_len, s.match_len - 1, pst);
              state = state_rep_next[state];
            }
          } else {
            rc.encode(model.is_rep0[state], false);
            if (r == 1) { rc.encode(model.is_rep1[state], false); }
            else { rc.encode(model.is_rep1[state], true); rc.encode(model.is_rep2[state], r == 2 ? false : true); }
            encode_len(rc, model.rep_len, s.match_len - 1, pst);
            state = state_rep_next[state];
          }
          if      (r == 1) { std::swap(rep[0], rep[1]); }
          else if (r == 2) { auto t=rep[2]; rep[2]=rep[1]; rep[1]=rep[0]; rep[0]=t; }
          else if (r == 3) { auto t=rep[3]; rep[3]=rep[2]; rep[2]=rep[1]; rep[1]=rep[0]; rep[0]=t; }
        } else {
          rc.encode(model.is_rep[state], false);
          encode_len(rc, model.match_len, s.match_len - 2, pst);
          encode_dist(rc, model, s.offset, std::min<std::uint32_t>(s.match_len - 2, 3));
          rep[3]=rep[2]; rep[2]=rep[1]; rep[1]=rep[0]; rep[0]=s.offset;
          state = state_match_next[state];
        }
        src_ptr += s.match_len;
        pst = int(src_ptr) & (num_pos_states - 1);
        report();
      }
    }
    rc.flush();
    if (prog) { prog->done.fetch_add((src_ptr - last_report) * encode_progress_weight, std::memory_order_relaxed); }
    return out;
  }

  fan::bytes_t decode_stream_seq(const fan::bytes_t& comp, std::size_t idx, std::uint64_t total_uncomp, fan::progress_t* prog) {
    fan::bytes_t out; out.reserve(total_uncomp);
    fan::io::bytes_reader_t br{comp, idx};
    range_dec_t<fan::io::bytes_reader_t> rd{br};
    auto m_ptr = std::make_unique<lzma_model_t>();
    lzma_model_t& model = *m_ptr;

    std::array<std::uint32_t,4> rep{1,1,1,1};
    std::uint8_t state = 0;
    std::size_t next_boundary = chunk_size;
    std::size_t last_report = 0;

    while (out.size() < total_uncomp) {
      if (out.size() == next_boundary) { rep = {1,1,1,1}; state = 0; model = lzma_model_t(); next_boundary += chunk_size; }
      std::size_t pos = out.size();
      int pst = int(pos) & (num_pos_states - 1);

      if (!rd.decode(model.is_match[state][pst])) {
        // literal
        std::uint8_t prev = pos > 0 ? out[pos-1] : 0;
        std::uint8_t mb = (pos >= rep[0]) ? out[pos - rep[0]] : 0;
        out.push_back(decode_literal(rd, model, state, pos, prev, mb));
        state = state_lit_next[state];
        if (prog && out.size() - last_report >= 65536) { prog->done.store(out.size(), std::memory_order_relaxed); last_report = out.size(); }
        continue;
      }

      std::uint32_t mlen = 0, off = 0;
      if (rd.decode(model.is_rep[state])) {
        // rep match
        if (rd.decode(model.is_rep0[state])) {
          // rep0
          if (!rd.decode(model.is_rep0_short[state][pst])) {
            // short rep, len=1
            mlen = 1; off = rep[0]; state = state_shortrep_next[state];
          } else {
            mlen = 1 + decode_len(rd, model.rep_len, pst); off = rep[0]; state = state_rep_next[state];
          }
        } else {
          int r;
          if (!rd.decode(model.is_rep1[state])) { r = 1; }
          else { r = rd.decode(model.is_rep2[state]) ? 3 : 2; }
          mlen = 1 + decode_len(rd, model.rep_len, pst);
          if      (r == 1) { off = rep[1]; std::swap(rep[0], rep[1]); }
          else if (r == 2) { off = rep[2]; auto t=rep[2]; rep[2]=rep[1]; rep[1]=rep[0]; rep[0]=t; }
          else             { off = rep[3]; auto t=rep[3]; rep[3]=rep[2]; rep[2]=rep[1]; rep[1]=rep[0]; rep[0]=t; }
          state = state_rep_next[state];
        }
      } else {
        // explicit match
        mlen = 2 + decode_len(rd, model.match_len, pst);
        off = decode_dist(rd, model, std::min<std::uint32_t>(mlen - 2, 3));
        rep[3]=rep[2]; rep[2]=rep[1]; rep[1]=rep[0]; rep[0]=off;
        state = state_match_next[state];
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

  // --- shared compress core ---
  inline void run_compress_core(fan::bytes_t& raw, compress_params_t params,
                                 fan::progress_t* prog, std::size_t thread_count,
                                 fan::bytes_t& out_comp) {
    std::size_t actual_threads = thread_count ? thread_count : std::max<std::size_t>(1, std::thread::hardware_concurrency());
    if (prog) { prog->done.store(0, std::memory_order_relaxed); prog->total.store(raw.size() * progress_scale, std::memory_order_relaxed); }
    std::size_t nc = (raw.size() + chunk_size - 1) / chunk_size;
    std::vector<chunk_payload_t> blocks(nc);
    std::atomic<std::size_t> next = 0;
    std::vector<std::jthread> workers;
    for (std::size_t w = 0; w < actual_threads; ++w) {
      workers.emplace_back([&] {
        for (std::size_t k; (k = next.fetch_add(1, std::memory_order_relaxed)) < nc;) {
          std::size_t c_start = k * chunk_size;
          std::size_t c_end = std::min(raw.size(), c_start + chunk_size);
          blocks[k] = parse_chunk_optimal(raw, c_start, c_end, params, prog);
        }
      });
    }
    workers.clear();
    out_comp = encode_stream_seq(raw, blocks, prog);
  }

  bool compress_path_to_file(const std::filesystem::path& in, const std::filesystem::path& out_path,
                              compress_params_t params = params_max(), fan::progress_t* prog = nullptr,
                              bool verbose = false, std::size_t thread_count = 0) {
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
      fan::bytes_t meta;
      std::uint8_t u16_buf[2], u64_buf[8]; fan::memory::write_le16(u16_buf, std::uint16_t(f.archive_path.size()));
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

    bool use_bcj = false;
    bool use_delta = false;
    int delta_stride = 1;

    if (params.bcj && files.size() == 1) {
      // check PE on the actual file, not the archive stream
      fan::bytes_t file_head(std::min<std::uint64_t>(files[0].size, 4));
      fan::io::file::file_t* fpp = nullptr;
      if (!fan::io::file::open(&fpp, files[0].real_path.string(), {"rb"})) {
        fan::io::file::file_reader_t fr{fpp};
        fr.read_exact(file_head);
        fan::io::file::close(fpp);
        if (file_head.size() >= 2 && file_head[0] == 'M' && file_head[1] == 'Z') { use_bcj = true; }
      }
    }

    fan::bytes_t raw_buf;
    provider.read_range(0, provider.size(), raw_buf);

    bool use_text = false;
    if (!use_bcj && files.size() == 1 && looks_like_large_text(raw_buf, payload_offset)) {
      fan::bytes_t text_buf = text_encode_transform(raw_buf);
      if (!text_buf.empty()) {
        raw_buf = std::move(text_buf);
        use_text = true;
      }
    }

    if (use_bcj) {
      bcj_transform_range(raw_buf, payload_offset, raw_buf.size(), true);
    } else if (!use_text) {
      delta_stride = detect_delta_stride(raw_buf);
      use_delta = delta_stride > 1;
      if (use_delta) { delta_encode(raw_buf, delta_stride); }
    }

    int stride_log2 = delta_stride == 8 ? 3 : delta_stride == 4 ? 2 : delta_stride == 2 ? 1 : 0;
    std::uint8_t flags = std::uint8_t(use_bcj ? flag_bcj : 0) | std::uint8_t(use_delta ? flag_delta : 0) | std::uint8_t(use_text ? flag_text : 0) | std::uint8_t(stride_log2 << 2);

    fan::io::file::file_t* fp = nullptr;
    if (fan::io::file::open(&fp, out_path.string(), {"wb"})) { return false; }
    try {
      fan::bytes_t comp;
      run_compress_core(raw_buf, params, prog, thread_count, comp);

      std::uint8_t header[17];
      fan::memory::write_le32(header, magic_v4);
      fan::memory::write_le64(header + 4, raw_buf.size());
      fan::memory::write_le32(header + 12, chunk_size);
      header[16] = flags;

      fan::io::file::file_writer_t sink{fp};
      sink.write_bytes(std::span<const std::uint8_t>(header, 17));
      sink.write_bytes(comp);
      fan::io::file::close(fp);
      if (prog) { prog->done.store(raw_buf.size() * progress_scale, std::memory_order_relaxed); }
      return true;
    } catch (...) { fan::io::file::close(fp); throw; }
  }

  bool decompress_file_to_dir(const std::filesystem::path& in_path, const std::filesystem::path& out_dir,
                               bool default_out, fan::progress_t* prog = nullptr) {
    fan::io::file::file_t* fp = nullptr;
    if (fan::io::file::open(&fp, in_path.string(), {"rb"})) { return false; }
    fan::io::file::file_reader_t src{fp};
    try {
      std::uint8_t header[17]; src.read_exact(std::span<std::uint8_t>(header, 17));
      if (fan::memory::read_le32(header) != magic_v4) { throw std::runtime_error("needs FCS4"); }
      std::uint64_t total_uncomp = fan::memory::read_le64(header + 4);
      std::uint8_t flags = header[16];
      bool use_bcj = flags & flag_bcj;
      bool use_delta = flags & flag_delta;
      bool use_text = flags & flag_text;
      int stride_log2 = (flags >> 2) & 7;
      int delta_stride = 1 << stride_log2;

      fan::bytes_t comp;
      std::uint64_t fsz = fan::io::file::file_size(in_path.string());
      if (fsz > 17) { comp.resize(fsz - 17); src.read_exact(comp); }
      fan::io::file::close(fp); fp = nullptr;

      if (prog) { prog->done.store(0, std::memory_order_relaxed); prog->total.store(total_uncomp, std::memory_order_relaxed); }

      fan::bytes_t raw = decode_stream_seq(comp, 0, total_uncomp, prog);
      if (use_text)  { raw = text_decode_transform(raw); }
      if (use_delta) { delta_decode(raw, delta_stride); }
      if (use_bcj)   { bcj_transform_range(raw, archive_payload_offset(raw), raw.size(), false); }

      fan::io::file::archive_extractor_t writer(out_dir, default_out);
      for (std::uint8_t b : raw) { writer.put(b); }
      writer.finish();
      return true;
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

    fan::bytes_t raw; raw.reserve(total_sz);
    std::uint8_t u32_buf[4], u64_buf[8];
    fan::memory::write_le32(u32_buf, std::uint32_t(files.size())); raw.insert(raw.end(), u32_buf, u32_buf + 4);
    for (const auto& f : files) {
      std::uint8_t u16_buf[2]; fan::memory::write_le16(u16_buf, std::uint16_t(f.path.size())); raw.insert(raw.end(), u16_buf, u16_buf + 2);
      auto* p_path = reinterpret_cast<const std::uint8_t*>(f.path.data()); raw.insert(raw.end(), p_path, p_path + f.path.size());
      fan::memory::write_le64(u64_buf, f.data.size()); raw.insert(raw.end(), u64_buf, u64_buf + 8);
    }
    if (meta_pad) { raw.insert(raw.end(), meta_pad, 0); }
    std::size_t payload_offset = raw.size();
    for (const auto& f : files) { raw.insert(raw.end(), f.data.begin(), f.data.end()); }

    bool use_bcj = params.bcj && files.size() == 1 && file::is_pe(files[0].data);
    if (use_bcj) { bcj_transform_range(raw, payload_offset, raw.size(), true); }

    int delta_stride = 1;
    bool use_delta = false;
    bool use_text = false;
    if (!use_bcj && files.size() == 1 && looks_like_large_text(raw, payload_offset)) {
      fan::bytes_t text_buf = text_encode_transform(raw);
      if (!text_buf.empty()) {
        raw = std::move(text_buf);
        use_text = true;
      }
    }
    if (!use_bcj && !use_text) {
      delta_stride = detect_delta_stride(raw);
      use_delta = delta_stride > 1;
      if (use_delta) { delta_encode(raw, delta_stride); }
    }
    int stride_log2 = delta_stride == 8 ? 3 : delta_stride == 4 ? 2 : delta_stride == 2 ? 1 : 0;
    std::uint8_t flags = std::uint8_t(use_bcj ? flag_bcj : 0) | std::uint8_t(use_delta ? flag_delta : 0) | std::uint8_t(use_text ? flag_text : 0) | std::uint8_t(stride_log2 << 2);

    fan::bytes_t comp;
    run_compress_core(raw, params, prog, 0, comp);

    fan::bytes_t result; result.reserve(comp.size() + 17);
    std::uint8_t header[17];
    fan::memory::write_le32(header, magic_v4); fan::memory::write_le64(header + 4, raw.size()); fan::memory::write_le32(header + 12, chunk_size); header[16] = flags;
    result.insert(result.end(), header, header + sizeof(header));
    result.insert(result.end(), comp.begin(), comp.end());
    if (prog) { prog->done.store(raw.size() * progress_scale, std::memory_order_relaxed); }
    return result;
  }

  std::vector<fan::io::file_buffer_t> decompress(const fan::bytes_t& comp, fan::progress_t* prog = nullptr) {
    std::vector<fan::io::file_buffer_t> files;
    if (comp.size() < 17 || fan::memory::read_le32(comp.data()) != magic_v4) { throw std::runtime_error("needs FCS4"); }
    std::uint64_t total_uncomp = fan::memory::read_le64(comp.data() + 4);
    std::uint8_t flags = comp[16];
    bool use_bcj = flags & flag_bcj;
    bool use_delta = flags & flag_delta;
    bool use_text = flags & flag_text;
    int stride_log2 = (flags >> 2) & 7;
    int delta_stride = 1 << stride_log2;
    if (prog) { prog->total.store(total_uncomp, std::memory_order_relaxed); }

    fan::bytes_t raw = decode_stream_seq(comp, 17, total_uncomp, prog);
    if (use_text)  { raw = text_decode_transform(raw); }
    if (use_delta) { delta_decode(raw, delta_stride); }
    if (use_bcj)   { bcj_transform_range(raw, archive_payload_offset(raw), raw.size(), false); }
    if (raw.size() < 4) { return files; }

    std::size_t out_idx = 0;
    std::uint32_t num_files = fan::memory::read_le32(raw.data() + out_idx); out_idx += 4;
    struct meta_t { std::string path; std::uint64_t size; };
    std::vector<meta_t> metas(num_files);
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