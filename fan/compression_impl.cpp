module fan.compression;

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

namespace fan::fcs {

  chunk_payload_t parse_chunk_optimal(const fan::bytes_t& src, std::size_t c_start, std::size_t c_end, const compress_params_t& params, std::function<void(std::uint64_t)> set_prog) {
    chunk_payload_t out; out.seqs.reserve((c_end - c_start) / 5);
    std::size_t dict_size = std::max<std::size_t>(1, params.chunk_size), d_start = c_start >= dict_size ? c_start - dict_size : 0;
    match_finder_t finder(d_start, c_end);
    for (std::size_t p = d_start; p < c_start; ++p) { finder.find_and_insert(src.data(), p, c_end, 0, 0, 0, nullptr); }
    std::size_t mf_pos = c_start; auto m_ptr = std::make_unique<lzma_model_t>(); lzma_model_t& model = *m_ptr;
    std::array<std::uint32_t, 4> g_rep{1, 1, 1, 1}; std::uint8_t g_state = 0; std::uint32_t lit_cnt = 0; std::size_t last_report = c_start;
    auto report_prog = [&](std::size_t pos) {
      pos = std::min(pos, c_end);
      if (pos > last_report && (pos - last_report >= progress_step || pos == c_end)) { set_prog((pos - c_start) * parse_progress_weight); last_report = pos; }
    };
    std::vector<opt_t> opts(max_opt_window + max_exp_len + 1); std::vector<match_cache_t> match_cache(max_opt_window);
    parse_price_cache_t price_cache; auto ps = [](std::size_t pos) { return int(pos & (num_pos_states - 1)); };
    std::size_t ci = c_start;

    if (!params.optimal) {
      while (ci < c_end) {
        match_t matches[128]; std::uint32_t n = finder.find_and_insert(src.data(), ci, c_end, params.chain_main, params.nice_len, max_exp_len, matches); ++mf_pos;
        struct cand_t { op_e op; std::uint32_t len, off, price; } b{op_e::none, 0, 0, inf_price};
        if (std::uint32_t avail = std::uint32_t(c_end - ci); avail > 0) {
          for (int r = 0; r < 4; ++r) {
            if (ci < d_start + g_rep[r]) { continue; }
            std::uint32_t l = get_match_len(src.data() + ci, src.data() + ci - g_rep[r], std::min(avail, max_rep_len));
            if (l == 0 || (l == 1 && r > 0)) { continue; }
            std::uint32_t pft = bit_price(model.is_match[g_state][ps(ci)], 1) + bit_price(model.is_rep[g_state], 1);
            if (r == 0) { pft += bit_price(model.is_rep0[g_state], 1) + bit_price(model.is_rep0_short[g_state][ps(ci)], l != 1) + (l > 1 ? len_price(model.rep_len, l - 1, ps(ci)) : 0); }
            else { pft += bit_price(model.is_rep0[g_state], 0) + (r == 1 ? bit_price(model.is_rep1[g_state], 0) : bit_price(model.is_rep1[g_state], 1) + bit_price(model.is_rep2[g_state], r != 2)) + len_price(model.rep_len, l - 1, ps(ci)); }
            if (pft < b.price || (pft == b.price && l > b.len)) { b = {static_cast<op_e>(r), l, g_rep[r], pft}; }
          }
          for (std::uint32_t mi = 0; mi < n; ++mi) {
            if (matches[mi].length < 2 || matches[mi].offset == 0 || matches[mi].offset > ci) { continue; }
            std::uint32_t len = std::min<std::uint32_t>(matches[mi].length, max_exp_len), pft = bit_price(model.is_match[g_state][ps(ci)], 1) + bit_price(model.is_rep[g_state], 0) + len_price(model.match_len, len - 2, ps(ci)) + dist_price(model, matches[mi].offset, std::min<std::uint32_t>(len - 2, 3));
            if (pft < b.price || (pft == b.price && len > b.len)) { b = {op_e::exp, len, matches[mi].offset, pft}; }
          }
        }
        if (b.len == 0 || b.op == op_e::none) {
          update_literal_symbol(model, g_state, ci, ci > 0 ? src[ci - 1] : 0, src[ci], ci >= g_rep[0] ? src[ci - g_rep[0]] : 0);
          g_state = state_lit_next[g_state]; ++lit_cnt; ++ci;
        } else {
          out.seqs.push_back({lit_cnt, b.op, b.len, b.off}); update_match_symbol(model, g_state, ci, b.op, b.len, b.off);
          if (b.op == op_e::rep0) { g_state = (b.len == 1) ? state_shortrep_next[g_state] : state_rep_next[g_state]; }
          else { shift_rep(int(b.op), b.off, g_rep); g_state = b.op == op_e::exp ? state_match_next[g_state] : state_rep_next[g_state]; }
          lit_cnt = 0; for (std::uint32_t k = 1; k < b.len; ++k) { finder.find_and_insert(src.data(), ci + k, c_end, 0, 0, 0, nullptr); }
          mf_pos += b.len - 1; ci += b.len;
        }
        report_prog(ci);
      }
      report_prog(c_end); if (lit_cnt) { out.seqs.push_back({lit_cnt, op_e::none, 0, 0}); }
      return out;
    }

    while (ci < c_end) {
      price_cache.refresh(model);
      std::uint32_t opt_win = std::clamp(params.opt_window, 2u, max_opt_window), window = std::min(std::uint32_t(c_end - ci), opt_win - 1), len_end = 1;
      opts[0] = {0, 0, 1, 0, op_e::none, g_rep, g_state};
      for (std::uint32_t k = 1; k <= window + max_exp_len; ++k) { opts[k].price = inf_price; }
      for (std::size_t target_mf = std::min(c_end, ci + window); mf_pos < target_mf; ++mf_pos) {
        auto& mc = match_cache[mf_pos & (max_opt_window - 1)];
        mc.count = finder.find_and_insert(src.data(), mf_pos, c_end, params.chain_main, params.nice_len, max_exp_len, mc.matches.data());
      }
      for (std::uint32_t j = 0; j < window; ++j) {
        if (opts[j].price == inf_price) { continue; }
        std::uint8_t st = opts[j].state; std::array<std::uint32_t,4> rep = opts[j].rep; std::size_t pos = ci + j; int pst = ps(pos);
        std::uint32_t lp = bit_price(model.is_match[st][pst], 0); int sc = state_is_lit[st] ? 0 : 1;
        const std::uint16_t* tree = model.lit[sc][lit_ctx(pos, pos > 0 ? src[pos - 1] : 0)].data();
        if (sc == 0) { lp += tree_price_v<8>(tree, src[pos]); }
        else {
          std::uint32_t sym = 1, mb = pos >= rep[0] ? src[pos - rep[0]] : 0, byte = src[pos];
          for (int i = 7; i >= 0; --i) { std::uint32_t bit = (byte >> i) & 1, mb_bit = (mb >> 7) & 1; mb <<= 1; lp += bit_price(tree[sym + (mb_bit << 8)], bit); sym = (sym << 1) | bit; }
        }
        if (opts[j].price + lp < opts[j + 1].price) { opts[j + 1] = {opts[j].price + lp, j, 1, 0, op_e::none, rep, state_lit_next[st]}; len_end = std::max(len_end, j + 1); }
        std::uint32_t rep0_len = 0;
        for (int r = 0; r < 4; ++r) {
          if (pos < d_start + rep[r]) { continue; }
          std::uint32_t l = get_match_len(src.data() + pos, src.data() + pos - rep[r], std::min<std::uint32_t>(std::uint32_t(c_end - pos), max_rep_len));
          if (l == 0) { continue; }
          if (r == 0) { rep0_len = l; }
          std::uint32_t bp = opts[j].price + bit_price(model.is_match[st][pst], 1) + bit_price(model.is_rep[st], 1);
          if (r == 0) {
            std::uint32_t r0p = bp + bit_price(model.is_rep0[st], 1);
            if (r0p + bit_price(model.is_rep0_short[st][pst], 0) < opts[j + 1].price) { opts[j + 1] = {r0p + bit_price(model.is_rep0_short[st][pst], 0), j, 1, rep[0], op_e::rep0, rep, state_shortrep_next[st]}; len_end = std::max(len_end, j + 1); }
            if (l >= 2) {
              std::uint32_t r0p_long = r0p + bit_price(model.is_rep0_short[st][pst], 1), step = l < 32 ? 1 : l < 128 ? 2 : 4;
              for (std::uint32_t ml = 2; ml <= l; ml += (ml < 32 ? 1 : step)) {
                if (r0p_long + price_cache.rep_len[pst][ml - 1] < opts[j + ml].price) { opts[j + ml] = {r0p_long + price_cache.rep_len[pst][ml - 1], j, ml, rep[0], op_e::rep0, rep, state_rep_next[st]}; len_end = std::max(len_end, j + ml); }
              }
              if (opts[j + l].len != l && r0p_long + price_cache.rep_len[pst][l - 1] < opts[j + l].price) { opts[j + l] = {r0p_long + price_cache.rep_len[pst][l - 1], j, l, rep[0], op_e::rep0, rep, state_rep_next[st]}; len_end = std::max(len_end, j + l); }
            }
          } else if (l > rep0_len) {
            std::uint32_t rp = bp + bit_price(model.is_rep0[st], 0) + (r == 1 ? bit_price(model.is_rep1[st], 0) : bit_price(model.is_rep1[st], 1) + bit_price(model.is_rep2[st], r != 2));
            std::array<std::uint32_t,4> nr = rep; shift_rep(r, rep[r], nr);
            if (l >= 2) {
              std::uint32_t step = l < 32 ? 1 : l < 128 ? 2 : 4;
              for (std::uint32_t ml = 2; ml <= l; ml += (ml < 32 ? 1 : step)) {
                if (rp + price_cache.rep_len[pst][ml - 1] < opts[j + ml].price) { opts[j + ml] = {rp + price_cache.rep_len[pst][ml - 1], j, ml, rep[r], static_cast<op_e>(r), nr, state_rep_next[st]}; len_end = std::max(len_end, j + ml); }
              }
              if (opts[j + l].len != l && rp + price_cache.rep_len[pst][l - 1] < opts[j + l].price) { opts[j + l] = {rp + price_cache.rep_len[pst][l - 1], j, l, rep[r], static_cast<op_e>(r), nr, state_rep_next[st]}; len_end = std::max(len_end, j + l); }
            }
          }
        }
        if (pos + 4 <= c_end) {
          auto& mc = match_cache[pos & (max_opt_window - 1)]; std::uint32_t bp = opts[j].price + bit_price(model.is_match[st][pst], 1) + bit_price(model.is_rep[st], 0), prev_len = std::max(1u, rep0_len);
          for (std::uint32_t mi = 0; mi < mc.count; ++mi) {
            auto& cand = mc.matches[mi]; if (cand.length < 2 || cand.offset == 0 || cand.offset > pos) { continue; }
            if (std::uint32_t limit = std::min({cand.length, std::uint32_t(c_end - pos), max_exp_len}); limit > prev_len) {
              std::array<std::uint32_t, 4> dc; for (std::uint32_t ls = 0; ls < 4; ++ls) { dc[ls] = price_cache.dist_price(model, cand.offset, ls); }
              std::uint32_t start_len = std::max(2u, prev_len + 1), step = limit < 32 ? 1 : limit < 128 ? 2 : 4;
              std::array<std::uint32_t,4> nr = rep; shift_rep(4, cand.offset, nr);
              for (std::uint32_t ml = start_len; ml <= limit; ml += (ml < 32 ? 1 : step)) {
                if (bp + price_cache.match_len[pst][ml - 2] + dc[std::min<std::uint32_t>(ml - 2, 3)] < opts[j + ml].price) { opts[j + ml] = {bp + price_cache.match_len[pst][ml - 2] + dc[std::min<std::uint32_t>(ml - 2, 3)], j, ml, cand.offset, op_e::exp, nr, state_match_next[st]}; len_end = std::max(len_end, j + ml); }
              }
              if (opts[j + limit].len != limit && bp + price_cache.match_len[pst][limit - 2] + dc[std::min<std::uint32_t>(limit - 2, 3)] < opts[j + limit].price) { opts[j + limit] = {bp + price_cache.match_len[pst][limit - 2] + dc[std::min<std::uint32_t>(limit - 2, 3)], j, limit, cand.offset, op_e::exp, nr, state_match_next[st]}; len_end = std::max(len_end, j + limit); }
              prev_len = limit;
            }
          }
        }
        if ((j & 63u) == 0) { report_prog(pos); }
      }
      std::vector<std::uint32_t> path; for (std::uint32_t cur = len_end; cur > 0; cur = opts[cur].prev) { path.push_back(cur); }
      std::uint32_t committed = 0;
      for (auto it = path.rbegin(); it != path.rend(); ++it) {
        if (auto& d = opts[*it]; d.op == op_e::none) {
          update_literal_symbol(model, g_state, ci, ci > 0 ? src[ci - 1] : 0, src[ci], ci >= g_rep[0] ? src[ci - g_rep[0]] : 0);
          ++lit_cnt; g_state = state_lit_next[g_state]; ++ci; ++committed;
        } else {
          out.seqs.push_back({lit_cnt, d.op, d.len, d.offset}); update_match_symbol(model, g_state, ci, d.op, d.len, d.offset); lit_cnt = 0;
          if (d.op == op_e::rep0) { g_state = (d.len == 1) ? state_shortrep_next[g_state] : state_rep_next[g_state]; }
          else { shift_rep(int(d.op), d.offset, g_rep); g_state = (d.op == op_e::exp) ? state_match_next[g_state] : state_rep_next[g_state]; }
          ci += d.len; committed += d.len;
        }
        if (params.commit_limit && committed >= params.commit_limit) { break; }
      }
      report_prog(ci);
    }
    report_prog(c_end); if (lit_cnt) { out.seqs.push_back({lit_cnt, op_e::none, 0, 0}); }
    return out;
  }

  fan::bytes_t encode_stream_seq(const fan::bytes_t& src, const std::vector<chunk_payload_t>& blocks, std::size_t chunk_size, std::function<void(std::uint64_t)> add_prog) {
    fan::bytes_t out; out.reserve(src.size() / 5); fan::io::bytes_writer_t bw{out}; range_enc_t<fan::io::bytes_writer_t> rc{bw};
    auto m_ptr = std::make_unique<lzma_model_t>(); lzma_model_t& model = *m_ptr;
    std::size_t src_ptr = 0, last_report = 0, next_boundary = chunk_size; std::array<std::uint32_t,4> rep{1,1,1,1}; std::uint8_t state = 0;
    auto report = [&] { if (src_ptr - last_report >= progress_step) { add_prog((src_ptr - last_report) * encode_progress_weight); last_report = src_ptr; } };

    for (const auto& block : blocks) {
      if (src_ptr == next_boundary) { rep = {1,1,1,1}; state = 0; model = lzma_model_t(); next_boundary += chunk_size; }
      for (const auto& s : block.seqs) {
        int pst = int(src_ptr) & (num_pos_states - 1);
        for (std::uint32_t j = 0; j < s.lit_len; ++j) {
          rc.encode(model.is_match[state][pst], 0); encode_literal(rc, model, state, src_ptr, src_ptr > 0 ? src[src_ptr - 1] : 0, src[src_ptr], src_ptr >= rep[0] ? src[src_ptr - rep[0]] : 0);
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
        src_ptr += s.match_len; report();
      }
    }
    rc.flush(); add_prog((src_ptr - last_report) * encode_progress_weight);
    return out;
  }
  
  int detect_delta_stride(const fan::bytes_t& d) {
    std::size_t probe = std::min<std::size_t>(d.size(), 131072);
    int best_stride = 1; double best_ent = 1e18, base_ent = 1e18;
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

  std::size_t archive_payload_offset(const fan::bytes_t& raw) {
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

  bool looks_like_pe(const fan::bytes_t& d) {
    if (d.size() < 0x40 || d[0] != 'M' || d[1] != 'Z') { return false; }
    std::uint32_t pe = fan::memory::read_le32(d.data() + 0x3c);
    return pe + 4 <= d.size() && d[pe] == 'P' && d[pe + 1] == 'E' && d[pe + 2] == 0 && d[pe + 3] == 0;
  }

  bool looks_like_elf_x86(const fan::bytes_t& d) {
    if (d.size() < 20 || d[0] != 0x7f || d[1] != 'E' || d[2] != 'L' || d[3] != 'F' || d[5] != 1) { return false; }
    std::uint16_t machine = fan::memory::read_le16(d.data() + 18);
    return machine == 3 || machine == 62;
  }

  bool looks_like_coff_x86(const fan::bytes_t& d) {
    if (d.size() < 20) { return false; }
    std::uint16_t machine = fan::memory::read_le16(d.data()), sections = fan::memory::read_le16(d.data() + 2), opt_size = fan::memory::read_le16(d.data() + 16);
    return (machine == 0x14c || machine == 0x8664) && sections != 0 && sections < 128 && opt_size < 4096;
  }

  bool looks_like_x86_binary(const fan::bytes_t& d) {
    return looks_like_pe(d) || looks_like_elf_x86(d) || looks_like_coff_x86(d) || (d.size() >= 8 && std::memcmp(d.data(), "!<arch>\n", 8) == 0);
  }

  bool looks_like_bcj_data(const fan::bytes_t& d, std::size_t payload_offset) {
    std::size_t begin = std::min(payload_offset, d.size()), n = std::min<std::size_t>(d.size() - begin, 4uz << 20);
    if (n < (64uz << 10)) { return false; }
    std::size_t hits = 0, good = 0;
    for (std::size_t i = begin; i + 5 <= begin + n; ++i) {
      std::uint8_t b = d[i];
      int s = (b == 0xe8 || b == 0xe9) ? 1 :
        (i + 6 <= begin + n && ((b == 0x0f && (d[i + 1] & 0xf0) == 0x80) || (b == 0xff && (d[i + 1] == 0x15 || d[i + 1] == 0x25)))) ? 2 :
        (i + 7 <= begin + n && (b == 0x48 || b == 0x4c) && d[i + 1] == 0x8d && (d[i + 2] & 0xc7) == 0x05) ? 3 : 0;
      if (!s) { continue; }
      ++hits; std::uint32_t v; std::memcpy(&v, d.data() + i + s, 4);
      good += (v + std::uint32_t(i - begin)) < d.size(); i += s + 3;
    }
    return hits >= n / 4096 && good * 3 >= hits;
  }

  void decode_text_head(const fan::bytes_t& in, std::size_t& p, std::uint64_t& out_size, std::vector<std::string>& dict) {
    if (in.size() < 10) { throw std::runtime_error("corrupt text transform"); }
    out_size = fan::memory::read_le64(in.data()); p += 8;
    std::uint16_t dict_count = fan::memory::read_le16(in.data() + p); p += 2;
    dict.reserve(dict_count);
    for (std::uint32_t i = 0; i < dict_count; ++i) {
      if (p >= in.size()) { throw std::runtime_error("corrupt text transform"); }
      std::uint8_t len = in[p++];
      if (p + len > in.size()) { throw std::runtime_error("corrupt text transform"); }
      dict.emplace_back(reinterpret_cast<const char*>(in.data() + p), len); p += len;
    }
  }

  std::size_t skip_word(const fan::bytes_t& raw, std::size_t& i) {
    std::size_t b = i++;
    while (i < raw.size() && text_word_char(raw[i]) && i - b < 31) { ++i; }
    while (i < raw.size() && text_word_char(raw[i])) { ++i; }
    return b;
  }

  fan::bytes_t text_encode_transform(const fan::bytes_t& raw) {
    std::unordered_map<std::string_view, std::uint32_t> counts; counts.reserve(1 << 16);
    for (std::size_t i = 0; i < raw.size();) {
      if (!text_word_first(raw[i])) { ++i; continue; }
      std::size_t b = skip_word(raw, i), len = i - b;
      if (len >= 4 && len <= 31) { ++counts[std::string_view(reinterpret_cast<const char*>(raw.data() + b), len)]; }
    }
    std::vector<text_word_t> words; words.reserve(std::min<std::size_t>(counts.size(), 65530));
    for (auto& [w, c] : counts) {
      if (c < 4) { continue; }
      std::uint32_t score = std::uint32_t(c * (w.size() > 3 ? w.size() - 3 : 0));
      if (score > w.size() + 16) { words.push_back({w, c, score}); }
    }
    std::sort(words.begin(), words.end(), [](const auto& a, const auto& b) { return a.score != b.score ? a.score > b.score : a.word < b.word; });
    std::vector<std::string> dict; dict.reserve(65530);
    for (auto& w : words) {
      std::size_t repl = dict.size() < 256 ? 3 : 4, saved = w.count * (w.word.size() - repl);
      if (w.word.size() > repl && saved > w.word.size() + 8) { dict.emplace_back(w.word); if (dict.size() == 65530) { break; } }
    }
    if (dict.empty()) { return {}; }
    std::unordered_map<std::string_view, std::uint32_t> ids; ids.reserve(dict.size() * 2);
    for (std::uint32_t i = 0; i < dict.size(); ++i) { ids.emplace(dict[i], i); }
    fan::bytes_t out; out.reserve(raw.size() * 9 / 10);
    std::uint8_t buf[8], dc[2]; fan::memory::write_le64(buf, raw.size()); out.insert(out.end(), buf, buf + 8);
    fan::memory::write_le16(dc, std::uint16_t(dict.size())); out.insert(out.end(), dc, dc + 2);
    for (const auto& w : dict) { out.push_back(std::uint8_t(w.size())); out.insert(out.end(), w.begin(), w.end()); }
    for (std::size_t i = 0; i < raw.size();) {
      if (raw[i] == text_marker) { out.push_back(text_marker); out.push_back(0); ++i; continue; }
      if (!text_word_first(raw[i])) { out.push_back(raw[i++]); continue; }
      std::size_t b = skip_word(raw, i); auto it = ids.find(std::string_view(reinterpret_cast<const char*>(raw.data() + b), i - b));
      if (it == ids.end()) { out.insert(out.end(), raw.begin() + b, raw.begin() + i); continue; }
      out.push_back(text_marker);
      if (it->second < 256) { out.push_back(1); out.push_back(std::uint8_t(it->second)); }
      else { out.push_back(2); out.push_back(std::uint8_t(it->second)); out.push_back(std::uint8_t(it->second >> 8)); }
    }
    return out.size() + raw.size() / 64 < raw.size() ? std::move(out) : fan::bytes_t{};
  }

  fan::bytes_t text_decode_transform(const fan::bytes_t& in) {
    std::size_t p = 0; std::uint64_t out_size = 0; std::vector<std::string> dict; decode_text_head(in, p, out_size, dict);
    fan::bytes_t out; out.reserve(std::size_t(out_size));
    while (p < in.size()) {
      std::uint8_t c = in[p++];
      if (c != text_marker) { out.push_back(c); continue; }
      if (p >= in.size()) { throw std::runtime_error("corrupt text transform"); }
      std::uint8_t t = in[p++];
      if (t == 0) { out.push_back(text_marker); }
      else if (t == 1 || t == 2) {
        if (p + (t - 1) >= in.size()) { throw std::runtime_error("corrupt text transform"); }
        std::uint32_t id = in[p++]; if (t == 2) { id |= std::uint32_t(in[p++]) << 8; }
        if (id >= dict.size()) { throw std::runtime_error("corrupt text transform"); }
        out.insert(out.end(), dict[id].begin(), dict[id].end());
      } else { throw std::runtime_error("corrupt text transform"); }
    }
    if (out.size() != out_size) { throw std::runtime_error("corrupt text transform"); }
    return out;
  }

  std::uint8_t text2_lower(std::uint8_t c) { return c >= 'A' && c <= 'Z' ? std::uint8_t(c + ('a' - 'A')) : c; }
  std::uint8_t text2_upper(std::uint8_t c) { return c >= 'a' && c <= 'z' ? std::uint8_t(c - ('a' - 'A')) : c; }

  std::uint8_t text2_case_kind(const std::uint8_t* p, std::size_t n) {
    bool has_lower = false, has_upper = false;
    for (std::size_t i = 0; i < n; ++i) { has_lower |= (p[i] >= 'a' && p[i] <= 'z'); has_upper |= (p[i] >= 'A' && p[i] <= 'Z'); }
    if (!has_lower && !has_upper) { return 0; }
    if (has_upper && !has_lower) { return 2; }
    if (p[0] >= 'A' && p[0] <= 'Z') {
      for (std::size_t i = 1; i < n; ++i) { if (p[i] >= 'A' && p[i] <= 'Z') { return 3; } }
      return 1;
    }
    return 3;
  }

  std::string text2_to_lower_word(const std::uint8_t* p, std::size_t n) {
    std::string s(n, 0); for (std::size_t i = 0; i < n; ++i) { s[i] = char(text2_lower(p[i])); } return s;
  }

  void text2_emit_id(fan::bytes_t& out, std::uint8_t t8, std::uint8_t t16, std::uint32_t id) {
    out.push_back(text_marker);
    if (id < 256) { out.push_back(t8); out.push_back(std::uint8_t(id)); }
    else { out.push_back(t16); out.push_back(std::uint8_t(id)); out.push_back(std::uint8_t(id >> 8)); }
  }

  const std::array<std::vector<std::uint8_t>, 256>& text2_token_by_first() {
    static const auto table = [] {
      std::array<std::vector<std::uint8_t>, 256> t;
      for (std::size_t k = 0; k < std::size(text2_static_tokens); ++k) {
        if (!text2_static_tokens[k].empty()) { t[std::uint8_t(text2_static_tokens[k][0])].push_back(std::uint8_t(k)); }
      }
      return t;
    }();
    return table;
  }

  int text2_static_match(const fan::bytes_t& raw, std::size_t i) {
    const auto& candidates = text2_token_by_first()[raw[i]]; if (candidates.empty()) { return -1; }
    int best = -1; std::size_t best_len = 0;
    for (std::uint8_t k : candidates) {
      auto tok = text2_static_tokens[k];
      if (i + tok.size() <= raw.size() && tok.size() >= (std::size_t(k) < 256 ? 3 : 4) && tok.size() > best_len && std::memcmp(raw.data() + i, tok.data(), tok.size()) == 0) {
        best = int(k); best_len = tok.size();
      }
    }
    return best;
  }

  fan::bytes_t text2_encode_transform(const fan::bytes_t& raw) {
    std::unordered_map<std::string, std::uint32_t> counts; counts.reserve(1 << 17);
    for (std::size_t i = 0; i < raw.size();) {
      if (!text_word_first(raw[i])) { ++i; continue; }
      std::size_t b = skip_word(raw, i), len = i - b;
      if (len >= 4 && len <= 31 && text2_case_kind(raw.data() + b, len) != 3) { ++counts[text2_to_lower_word(raw.data() + b, len)]; }
    }
    struct text2_word_t { std::string word; std::uint32_t count = 0, score = 0; };
    std::vector<text2_word_t> words; words.reserve(counts.size());
    for (auto& [w, c] : counts) {
      if (c < 4) { continue; }
      std::uint32_t score = std::uint32_t(c * (w.size() > 3 ? w.size() - 3 : 0));
      if (score > w.size() + 10) { words.push_back({std::move(w), c, score}); }
    }
    std::sort(words.begin(), words.end(), [](const auto& a, const auto& b) { return a.score != b.score ? a.score > b.score : a.word < b.word; });
    std::vector<std::string> dict; dict.reserve(65530);
    for (auto& w : words) {
      std::size_t repl = dict.size() < 256 ? 3 : 4, saved = w.count * (w.word.size() - repl);
      if (w.word.size() > repl && saved > w.word.size() + 4) { dict.push_back(std::move(w.word)); if (dict.size() == 65530) { break; } }
    }
    if (dict.empty()) { return {}; }
    std::unordered_map<std::string_view, std::uint32_t> ids; ids.reserve(dict.size() * 2);
    for (std::uint32_t i = 0; i < dict.size(); ++i) { ids.emplace(dict[i], i); }
    fan::bytes_t out; out.reserve(raw.size() * 4 / 5);
    std::uint8_t buf[8], dc[2]; fan::memory::write_le64(buf, raw.size()); out.insert(out.end(), buf, buf + 8);
    fan::memory::write_le16(dc, std::uint16_t(dict.size())); out.insert(out.end(), dc, dc + 2);
    for (const auto& w : dict) { out.push_back(std::uint8_t(w.size())); out.insert(out.end(), w.begin(), w.end()); }
    for (std::size_t i = 0; i < raw.size();) {
      if (raw[i] == text_marker) { out.push_back(text_marker); out.push_back(0); ++i; continue; }
      if (int st = text2_static_match(raw, i); st >= 0) { text2_emit_id(out, 7, 8, std::uint32_t(st)); i += text2_static_tokens[st].size(); continue; }
      if (!text_word_first(raw[i])) { out.push_back(raw[i++]); continue; }
      std::size_t b = skip_word(raw, i), len = i - b;
      std::uint8_t ck = len <= 31 ? text2_case_kind(raw.data() + b, len) : 3;
      if (len < 4 || len > 31 || ck == 3) { out.insert(out.end(), raw.begin() + b, raw.begin() + i); continue; }
      auto lw = text2_to_lower_word(raw.data() + b, len); auto it = ids.find(std::string_view(lw.data(), lw.size()));
      if (it == ids.end()) { out.insert(out.end(), raw.begin() + b, raw.begin() + i); continue; }
      std::uint8_t t8 = ck == 0 ? 1 : ck == 1 ? 3 : 5; text2_emit_id(out, t8, std::uint8_t(t8 + 1), it->second);
    }
    return out.size() < raw.size() + raw.size() / 64 ? std::move(out) : fan::bytes_t{};
  }

  fan::bytes_t text2_decode_transform(const fan::bytes_t& in) {
    std::size_t p = 0; std::uint64_t out_size = 0; std::vector<std::string> dict; decode_text_head(in, p, out_size, dict);
    fan::bytes_t out; out.reserve(std::size_t(out_size));
    while (p < in.size()) {
      std::uint8_t c = in[p++];
      if (c != text_marker) { out.push_back(c); continue; }
      if (p >= in.size()) { throw std::runtime_error("corrupt text2 transform"); }
      std::uint8_t t = in[p++]; if (t == 0) { out.push_back(text_marker); continue; }
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
        auto tok = text2_static_tokens[id]; out.insert(out.end(), tok.begin(), tok.end());
      } else {
        if (id >= dict.size()) { throw std::runtime_error("corrupt text2 transform"); }
        std::uint8_t ck = (t == 1 || t == 2) ? 0 : (t == 3 || t == 4) ? 1 : 2;
        for (std::size_t i = 0; i < dict[id].size(); ++i) {
          std::uint8_t dc = std::uint8_t(dict[id][i]); if ((ck == 1 && i == 0) || ck == 2) { dc = text2_upper(dc); }
          out.push_back(dc);
        }
      }
    }
    if (out.size() != out_size) { throw std::runtime_error("corrupt text2 transform"); }
    return out;
  }

  void rle_write_var(fan::bytes_t& out, std::uint64_t v) {
    while (v >= 0x80) { out.push_back(std::uint8_t(v) | 0x80); v >>= 7; }
    out.push_back(std::uint8_t(v));
  }

  std::uint64_t rle_read_var(const fan::bytes_t& in, std::size_t& p) {
    std::uint64_t v = 0;
    for (std::uint32_t shift = 0; shift < 64; shift += 7) {
      if (p >= in.size()) { throw std::runtime_error("corrupt rle transform"); }
      std::uint8_t c = in[p++]; v |= std::uint64_t(c & 0x7f) << shift;
      if ((c & 0x80) == 0) { return v; }
    }
    throw std::runtime_error("corrupt rle transform");
  }

  fan::bytes_t rle_encode_transform(const fan::bytes_t& raw) {
    if (raw.size() < (64uz << 10)) { return {}; }
    fan::bytes_t out; out.reserve(raw.size() * 7 / 8);
    std::uint8_t sz[8]; fan::memory::write_le64(sz, raw.size()); out.insert(out.end(), sz, sz + 8);
    std::size_t lit_begin = 0;
    auto flush_lit = [&](std::size_t lit_end) {
      while (lit_begin < lit_end) {
        std::size_t n = std::min<std::size_t>(lit_end - lit_begin, 1uz << 20);
        out.push_back(0); rle_write_var(out, n); out.insert(out.end(), raw.begin() + lit_begin, raw.begin() + lit_begin + n); lit_begin += n;
      }
    };
    for (std::size_t i = 0; i < raw.size();) {
      std::uint8_t c = raw[i]; std::size_t run = 1;
      while (i + run < raw.size() && raw[i + run] == c) { ++run; }
      if (run >= 4) {
        flush_lit(i); out.push_back(1); out.push_back(c); rle_write_var(out, run); i += run; lit_begin = i;
      } else { i += run; }
    }
    flush_lit(raw.size());
    return out.size() + raw.size() / 128 < raw.size() ? std::move(out) : fan::bytes_t{};
  }

  fan::bytes_t rle_decode_transform(const fan::bytes_t& in) {
    if (in.size() < 8) { throw std::runtime_error("corrupt rle transform"); }
    std::size_t p = 8; std::uint64_t out_size = fan::memory::read_le64(in.data());
    fan::bytes_t out; out.reserve(std::size_t(out_size));
    while (p < in.size()) {
      std::uint8_t t = in[p++];
      if (t == 0) {
        std::uint64_t n = rle_read_var(in, p);
        if (n > in.size() - p || out.size() + n > out_size) { throw std::runtime_error("corrupt rle transform"); }
        out.insert(out.end(), in.begin() + p, in.begin() + p + std::size_t(n)); p += std::size_t(n);
      } else if (t == 1) {
        if (p >= in.size()) { throw std::runtime_error("corrupt rle transform"); }
        std::uint8_t v = in[p++]; std::uint64_t run = rle_read_var(in, p);
        if (out.size() + run > out_size) { throw std::runtime_error("corrupt rle transform"); }
        out.insert(out.end(), std::size_t(run), v);
      } else { throw std::runtime_error("corrupt rle transform"); }
    }
    if (out.size() != out_size) { throw std::runtime_error("corrupt rle transform"); }
    return out;
  }

  std::uint32_t get_match_len(const std::uint8_t* p1, const std::uint8_t* p2, std::uint32_t max_l) {
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

  match_finder_t::match_finder_t(std::size_t d_start, std::size_t c_end)
    : h2(1 << 16, nil), h3(1 << 19, nil), h4(1 << 22, nil), son(2 * (c_end - d_start), nil), base(d_start) {}

  void match_finder_t::hashes(const std::uint8_t* p, std::uint32_t& k2, std::uint32_t& k3, std::uint32_t& k4) {
    std::uint32_t v; std::memcpy(&v, p, 4);
    k2 = ((v & 0xFFFFu) * 0x9E3779B9u) >> 16;
    k3 = ((v & 0xFFFFFFu) * 0x1E35A7BDu) >> 13;
    k4 = (v * 0x9E3779B9u) >> 10;
  }

  std::uint32_t match_finder_t::find_and_insert(const std::uint8_t* src_base, std::size_t i, std::size_t c_end, std::uint32_t max_depth, std::uint32_t nice_len, std::uint32_t max_len, match_t* out) {
    const std::uint32_t max_avail = std::min<std::uint32_t>(std::uint32_t(c_end - i), max_len);
    if (max_avail < 4 || i + 4 > c_end) { return 0; }
    std::uint32_t k2, k3, k4; hashes(src_base + i, k2, k3, k4);
    std::uint32_t r = std::uint32_t(i - base), n_out = 0;
    auto try_q = [&](std::uint32_t h, std::uint32_t ml) {
      if (!out || h == nil || n_out >= 128) { return; }
      std::size_t abs_p = base + h; if (abs_p >= i) { return; }
      std::uint32_t l = get_match_len(src_base + i, src_base + abs_p, max_avail);
      if (l >= ml && (n_out == 0 || l > out[n_out - 1].length)) { out[n_out++] = {std::uint32_t(i - abs_p), l}; }
    };
    try_q(h2[k2], 2); try_q(h3[k3], 3);
    h2[k2] = r; h3[k3] = r;
    std::uint32_t cur = h4[k4]; h4[k4] = r;
    std::uint32_t best_l = n_out ? out[n_out - 1].length : 0;
    std::uint32_t* ptr0 = &son[(r << 1) + 1]; std::uint32_t* ptr1 = &son[r << 1];
    std::uint32_t len0 = 0, len1 = 0; const std::uint8_t* po = src_base + i;
    while (cur != nil && max_depth-- > 0) {
      std::size_t abs_p = base + cur; if (abs_p >= i) { break; }
      std::uint32_t delta = std::uint32_t(i - abs_p), l = std::min(len0, len1); const std::uint8_t* pb = src_base + abs_p;
      if (pb[l] == po[l]) {
        l += get_match_len(po + l, pb + l, max_avail - l);
        if (l > best_l) {
          best_l = l; if (out) { out[n_out < 128 ? n_out++ : n_out - 1] = {delta, l}; }
          if (l >= nice_len || l >= max_avail) { *ptr1 = son[cur << 1]; *ptr0 = son[(cur << 1) + 1]; return n_out; }
        }
      }
      if (pb[l] < po[l]) { *ptr1 = cur; ptr1 = &son[(cur << 1) + 1]; cur = *ptr1; len1 = l; }
      else { *ptr0 = cur; ptr0 = &son[cur << 1]; cur = *ptr0; len0 = l; }
    }
    *ptr0 = nil; *ptr1 = nil;
    return n_out;
  }

  std::uint32_t tree_price(const std::uint16_t* tree, std::uint32_t sym, int bits) {
    std::uint32_t cost = 0, ctx = 1;
    for (int i = bits - 1; i >= 0; --i) { std::uint32_t bit = (sym >> i) & 1; cost += bit_price(tree[ctx], bit); ctx = (ctx << 1) | bit; }
    return cost;
  }

  std::uint32_t len_price(const lzma_model_t::len_model_t& lm, std::uint32_t len, int ps) {
    if (len < 8) { return bit_price(lm.choice[0][ps], 0) + tree_price_v<3>(lm.low[ps].data(), len); }
    if (len < 16) { return bit_price(lm.choice[0][ps], 1) + bit_price(lm.choice2[0][ps], 0) + tree_price_v<3>(lm.mid[ps].data(), len - 8); }
    return bit_price(lm.choice[0][ps], 1) + bit_price(lm.choice2[0][ps], 1) + tree_price(lm.high.data(), len - 16, 12);
  }

  std::uint32_t dist_price(const lzma_model_t& m, std::uint32_t dist, std::uint32_t len_state) {
    std::uint32_t d = dist - 1, slot = slot_for_dist(d), cost = tree_price_v<6>(m.pos_slot[len_state].data(), slot);
    if (slot >= 4) {
      std::uint32_t footer_bits = (slot >> 1) - 1, footer = d - ((2u | (slot & 1u)) << footer_bits);
      if (slot < 14) { cost += tree_price(m.pos_special[slot - 4].data(), footer, footer_bits); }
      else { cost += (footer_bits - 4) * 64 + tree_price_v<4>(m.align_bits.data(), footer & 0xFu); }
    }
    return cost;
  }

  void update_tree(std::uint16_t* tree, std::uint32_t sym, int bits) {
    std::uint32_t ctx = 1;
    for (int i = bits - 1; i >= 0; --i) { std::uint32_t bit = (sym >> i) & 1; update_bit(tree[ctx], bit); ctx = (ctx << 1) | bit; }
  }

  void update_len(lzma_model_t::len_model_t& lm, std::uint32_t len, int ps) {
    if (len < 8) { update_bit(lm.choice[0][ps], 0); update_tree_v<3>(lm.low[ps].data(), len); }
    else if (len < 16) { update_bit(lm.choice[0][ps], 1); update_bit(lm.choice2[0][ps], 0); update_tree_v<3>(lm.mid[ps].data(), len - 8); }
    else { update_bit(lm.choice[0][ps], 1); update_bit(lm.choice2[0][ps], 1); update_tree(lm.high.data(), len - 16, 12); }
  }

  void update_dist(lzma_model_t& m, std::uint32_t dist, std::uint32_t len_state) {
    std::uint32_t d = dist - 1, slot = slot_for_dist(d); update_tree_v<6>(m.pos_slot[len_state].data(), slot);
    if (slot >= 4) {
      std::uint32_t footer_bits = (slot >> 1) - 1, footer = d - ((2u | (slot & 1u)) << footer_bits);
      if (slot < 14) { update_tree(m.pos_special[slot - 4].data(), footer, footer_bits); }
      else { update_tree_v<4>(m.align_bits.data(), footer & 0xFu); }
    }
  }

  void update_literal_symbol(lzma_model_t& m, std::uint8_t state, std::size_t pos, std::uint8_t prev, std::uint8_t byte, std::uint8_t match_byte) {
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

  void update_match_symbol(lzma_model_t& m, std::uint8_t state, std::size_t pos, op_e op, std::uint32_t len, std::uint32_t dist) {
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

  void parse_price_cache_t::refresh(lzma_model_t& m) {
    for (int ps = 0; ps < num_pos_states; ++ps) {
      for (std::uint32_t i = 0; i < max_exp_len - 1; ++i) { match_len[ps][i] = len_price(m.match_len, i, ps); }
      for (std::uint32_t i = 0; i < max_rep_len; ++i) { rep_len[ps][i] = len_price(m.rep_len, i, ps); }
    }
    for (std::uint32_t ls = 0; ls < 4; ++ls) {
      for (std::uint32_t slot = 0; slot < 64; ++slot) { pos_slot[ls][slot] = tree_price_v<6>(m.pos_slot[ls].data(), slot); }
    }
    for (std::uint32_t s = 0; s < 10; ++s) {
      std::uint32_t slot = s + 4, footer_bits = (slot >> 1) - 1, n = 1u << footer_bits;
      for (std::uint32_t f = 0; f < n; ++f) { pos_special_price[s][f] = tree_price(m.pos_special[s].data(), f, footer_bits); }
    }
    for (std::uint32_t f = 0; f < 16; ++f) { align_price[f] = tree_price_v<4>(m.align_bits.data(), f); }
  }

  std::uint32_t parse_price_cache_t::dist_price(lzma_model_t&, std::uint32_t dist, std::uint32_t len_state) const {
    std::uint32_t d = dist - 1, slot = slot_for_dist(d), cost = pos_slot[len_state][slot];
    if (slot >= 4) {
      std::uint32_t footer_bits = (slot >> 1) - 1, footer = d - ((2u | (slot & 1u)) << footer_bits);
      if (slot < 14) { cost += pos_special_price[slot - 4][footer]; }
      else { cost += (footer_bits - 4) * 64 + align_price[footer & 0xFu]; }
    }
    return cost;
  }

  fan::bytes_t decode_stream_seq(const fan::bytes_t& comp, std::size_t idx, std::uint64_t total_uncomp, std::size_t chunk_size, fan::progress_t* prog) {
    fan::bytes_t out; out.resize(total_uncomp);
    std::uint8_t* out_data = out.data(); std::size_t out_pos = 0, next_boundary = chunk_size, last_report = 0;
    fan::io::bytes_reader_t br{comp, idx}; range_dec_t<fan::io::bytes_reader_t> rd{br};
    auto m_ptr = std::make_unique<lzma_model_t>(); lzma_model_t& model = *m_ptr;
    std::array<std::uint32_t,4> rep{1,1,1,1}; std::uint8_t state = 0;

    while (out_pos < total_uncomp) {
      if (out_pos == next_boundary) { rep = {1,1,1,1}; state = 0; model = lzma_model_t(); next_boundary += chunk_size; }
      int pst = int(out_pos) & (num_pos_states - 1);

      if (!rd.decode(model.is_match[state][pst])) {
        out_data[out_pos] = decode_literal(rd, model, state, out_pos, out_pos > 0 ? out_data[out_pos-1] : 0, out_pos >= rep[0] ? out_data[out_pos - rep[0]] : 0);
        state = state_lit_next[state]; ++out_pos;
        if (prog && out_pos - last_report >= 65536) { prog->done.store(out_pos, std::memory_order_relaxed); last_report = out_pos; }
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

      if (off == 0 || off > out_pos || out_pos + mlen > total_uncomp) { throw std::runtime_error("corrupt stream"); }
      if (off == 1) { std::memset(out_data + out_pos, out_data[out_pos - 1], mlen); }
      else if (off >= mlen) { std::memcpy(out_data + out_pos, out_data + out_pos - off, mlen); }
      else {
        for (std::uint32_t k = 0; k < mlen; ++k) {
          out_data[out_pos + k] = out_data[out_pos + k - off];
        }
      }
      out_pos += mlen;
      if (prog && out_pos - last_report >= 65536) { prog->done.store(out_pos, std::memory_order_relaxed); last_report = out_pos; }
    }
    if (prog) { prog->done.store(total_uncomp, std::memory_order_relaxed); }
    return out;
  }

  std::size_t run_compress_core(const fan::bytes_t& raw, compress_params_t params, fan::progress_t* user_prog, std::size_t thread_count, fan::bytes_t& out_comp) {
    std::size_t actual_threads = thread_count ? thread_count : std::max<std::size_t>(1, std::thread::hardware_concurrency()), ch_size = params.chunk_size;
    if (actual_threads > 1 && raw.size() >= (32uz << 20)) {
      ch_size = std::clamp(raw.size() / actual_threads, std::min(params.optimal ? 32uz << 20 : 16uz << 20, params.chunk_size), params.chunk_size);
    }
    std::size_t nc = (raw.size() + ch_size - 1) / ch_size, block_threads = std::min(actual_threads, std::max<std::size_t>(1, nc));
    std::vector<std::atomic<std::uint64_t>> parse_prog(nc); std::atomic<std::uint64_t> parse_done = 0, encode_done = 0;
    auto publish_prog = [&] { if (user_prog) { user_prog->done.store(parse_done.load(std::memory_order_relaxed) + encode_done.load(std::memory_order_relaxed), std::memory_order_relaxed); } };
    auto set_parse_prog = [&](std::size_t k, std::uint64_t v) {
      if (!user_prog) { return; }
      std::size_t begin = k * ch_size, end = std::min(raw.size(), begin + ch_size);
      v = std::min<std::uint64_t>(v, (end - begin) * parse_progress_weight); std::uint64_t old = parse_prog[k].load(std::memory_order_relaxed);
      while (v > old && !parse_prog[k].compare_exchange_weak(old, v, std::memory_order_relaxed)) {}
      if (v > old) { parse_done.fetch_add(v - old, std::memory_order_relaxed); publish_prog(); }
    };
    auto add_encode_prog = [&](std::uint64_t w) { if (user_prog) { encode_done.fetch_add(w, std::memory_order_relaxed); publish_prog(); } };
    std::vector<chunk_payload_t> blocks(nc); std::atomic<std::size_t> next = 0; std::vector<std::jthread> workers;
    for (std::size_t w = 0; w < block_threads; ++w) {
      workers.emplace_back([&] {
        for (std::size_t k; (k = next.fetch_add(1, std::memory_order_relaxed)) < nc;) {
          blocks[k] = parse_chunk_optimal(raw, k * ch_size, std::min(raw.size(), (k + 1) * ch_size), params, [&](std::uint64_t v) { set_parse_prog(k, v); });
          set_parse_prog(k, (std::min(raw.size(), (k + 1) * ch_size) - k * ch_size) * parse_progress_weight);
        }
      });
    }
    workers.clear(); out_comp = encode_stream_seq(raw, blocks, ch_size, add_encode_prog);
    return ch_size;
  }

  void add_transform_candidates(std::vector<compress_candidate_t>& candidates, fan::bytes_t&& raw, std::size_t payload_offset, bool can_bcj, bool can_text, compress_params_t params) {
    auto push_rle = [&](const fan::bytes_t& src, std::string_view name, std::uint8_t flags) {
      if (fan::bytes_t rle = rle_encode_transform(src); !rle.empty()) { candidates.push_back({std::move(rle), {}, name, std::uint8_t(flags | flag_rle), params.chunk_size}); }
    };
    fan::bytes_t bcj_buf, delta_buf, text_buf, text2_buf;
    if (can_bcj) { bcj_buf = raw; bcj_transform_range(bcj_buf, payload_offset, bcj_buf.size(), true); }
    if (can_text && raw.size() >= (256uz << 10)) { text_buf = text_encode_transform(raw); text2_buf = text2_encode_transform(raw); }
    else if (!can_bcj) {
      if (int delta_stride = detect_delta_stride(raw); delta_stride > 1) {
        int stride_log2 = delta_stride == 8 ? 3 : delta_stride == 4 ? 2 : 1; std::uint8_t flags = std::uint8_t(flag_delta | (stride_log2 << 2));
        delta_buf = raw; delta_encode(delta_buf, delta_stride); push_rle(delta_buf, "delta+rle", flags);
        candidates.push_back({std::move(delta_buf), {}, "delta", flags, params.chunk_size});
      }
    }
    push_rle(raw, "rle", 0); candidates.push_back({std::move(raw), {}, "raw", 0, params.chunk_size});
    if (!bcj_buf.empty()) { push_rle(bcj_buf, "bcj+rle", flag_bcj); candidates.push_back({std::move(bcj_buf), {}, "bcj", flag_bcj, params.chunk_size}); }
    if (!text_buf.empty()) { candidates.push_back({std::move(text_buf), {}, "text", flag_text, params.chunk_size}); }
    if (!text2_buf.empty()) { candidates.push_back({std::move(text2_buf), {}, "text2", flag_text2, params.chunk_size}); }
  }

  std::size_t quick_test_candidate(const fan::bytes_t& data, compress_params_t params) {
    params.chain_main = std::min(params.chain_main, 32u); params.nice_len = std::min(params.nice_len, 64u);
    params.opt_window = std::min(params.opt_window, 1u << 10); params.optimal = false;
    if (data.size() <= params.candidate_sample_size || params.candidate_sample_count <= 1) {
      fan::bytes_t comp; params.chunk_size = data.size(); run_compress_core(data, params, nullptr, 1, comp); return comp.size();
    }
    std::uint32_t n = std::max<std::uint32_t>(1, params.candidate_sample_count); std::size_t each = std::max<std::size_t>(1, params.candidate_sample_size / n), total = 0;
    for (std::uint32_t i = 0; i < n; ++i) {
      std::size_t max_pos = data.size() > each ? data.size() - each : 0, pos = n == 1 ? 0 : max_pos * i / (n - 1);
      fan::bytes_t sample(data.begin() + pos, data.begin() + pos + std::min(each, data.size() - pos)), comp;
      params.chunk_size = sample.size(); run_compress_core(sample, params, nullptr, 1, comp); total += comp.size();
    }
    return total;
  }

  std::vector<full_param_candidate_t> make_full_param_candidates(compress_params_t params) {
    std::vector<full_param_candidate_t> out;
    auto same = [](const compress_params_t& a, const compress_params_t& b) {
      return a.chain_main == b.chain_main && a.nice_len == b.nice_len && a.optimal == b.optimal && a.lazy_depth == b.lazy_depth && a.chunk_size == b.chunk_size && a.opt_window == b.opt_window && a.commit_limit == b.commit_limit;
    };
    auto push = [&](std::string_view name, compress_params_t p) {
      for (const auto& v : out) { if (same(v.params, p)) { return; } }
      out.push_back({name, p});
    };
    push("base", params); if (!params.optimal) { return out; }
    auto p = params; p.commit_limit = 1024; push("commit1024", p);
    p = params; p.commit_limit = 256; push("commit256", p);
    p = params; p.commit_limit = 64; push("commit64", p);
    p = params; p.chain_main = std::max(p.chain_main, 3072u); p.nice_len = std::max(p.nice_len, 2048u); p.opt_window = std::max(p.opt_window, 1u << 14); p.commit_limit = 256; push("deep256", p);
    p = params; p.chain_main = std::min(p.chain_main, 1024u); p.nice_len = std::min(p.nice_len, 768u); p.opt_window = std::min(p.opt_window, 1u << 12); p.commit_limit = 256; push("tight256", p);
    return out;
  }

  compress_candidate_t compress_best_candidate(std::vector<compress_candidate_t>& candidates, compress_params_t params, fan::progress_t* prog, std::size_t thread_count, bool verbose) {
    std::size_t actual_threads = thread_count ? thread_count : std::max<std::size_t>(1, std::thread::hardware_concurrency()), best_idx = 0;
    if (candidates.size() > 1) {
      if (verbose) { fan::print("quick testing candidates..."); }
      std::vector<std::size_t> scores(candidates.size(), -1uz); std::atomic<std::size_t> next = 0;
      std::vector<std::jthread> workers;
      for (std::size_t w = 0; w < std::min<std::size_t>(actual_threads, candidates.size()); ++w) {
        workers.emplace_back([&] { for (std::size_t i; (i = next.fetch_add(1, std::memory_order_relaxed)) < candidates.size();) { scores[i] = quick_test_candidate(candidates[i].data, params); } });
      }
      workers.clear(); std::size_t best_s = -1uz;
      for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (verbose) { fan::print("candidate:", candidates[i].name, "sample size:", scores[i]); }
        if (scores[i] < best_s) { best_s = scores[i]; best_idx = i; }
      }
    }
    auto& c = candidates[best_idx]; if (verbose) { fan::print("selected:", c.name, "for full compression"); }
    if (!params.parser_verify) {
      if (prog) { prog->total.store(c.data.size() * progress_scale, std::memory_order_relaxed); prog->done.store(0, std::memory_order_relaxed); }
      c.chunk_size = run_compress_core(c.data, params, prog, actual_threads, c.comp);
      if (prog) { prog->done.store(c.data.size() * progress_scale, std::memory_order_relaxed); }
      return std::move(c);
    }
    auto full_params = make_full_param_candidates(params); if (verbose && full_params.size() > 1) { fan::print("full testing parser settings..."); }
    std::size_t best_full = -1uz, best_chunk = 0; fan::bytes_t best_comp;
    std::uint64_t full_unit = c.data.size() * progress_scale, done_base = 0;
    if (prog) { prog->total.store(full_unit * full_params.size(), std::memory_order_relaxed); prog->done.store(0, std::memory_order_relaxed); }
    for (const auto& fp : full_params) {
      fan::bytes_t comp; fan::progress_t local_prog; std::atomic_bool pump_done = false; std::jthread pump;
      if (prog) {
        local_prog.total.store(full_unit, std::memory_order_relaxed); local_prog.done.store(0, std::memory_order_relaxed);
        pump = std::jthread([&] { while (!pump_done.load(std::memory_order_relaxed)) { prog->done.store(done_base + std::min<std::uint64_t>(local_prog.done.load(std::memory_order_relaxed), full_unit), std::memory_order_relaxed); std::this_thread::sleep_for(std::chrono::milliseconds(32)); } });
      }
      std::size_t ch = 0;
      try { ch = run_compress_core(c.data, fp.params, prog ? &local_prog : nullptr, actual_threads, comp); }
      catch (...) { if (prog) { pump_done.store(true, std::memory_order_relaxed); } throw; }
      if (prog) { pump_done.store(true, std::memory_order_relaxed); }
      if (verbose && full_params.size() > 1) { fan::print("parser:", fp.name, "size:", comp.size()); }
      if (comp.size() < best_full) { best_full = comp.size(); best_chunk = ch; best_comp = std::move(comp); }
      if (prog) { done_base += full_unit; prog->done.store(done_base, std::memory_order_relaxed); }
    }
    c.chunk_size = best_chunk; c.comp = std::move(best_comp);
    if (prog) { prog->total.store(full_unit, std::memory_order_relaxed); prog->done.store(full_unit, std::memory_order_relaxed); }
    return std::move(c);
  }

  bool compress_path_to_file(const std::filesystem::path& in, const std::filesystem::path& out_path, compress_params_t params, fan::progress_t* prog, bool verbose, std::size_t thread_count) {
    std::vector<fan::io::file_info_t> files;
    if (std::filesystem::is_directory(in)) {
      fan::io::iterate_files_recursive(in, [&](const auto& full, const auto& rel) { if (verbose) { fan::print("found:", rel.generic_string()); } files.push_back({full, rel.generic_string(), std::filesystem::file_size(full)}); });
    } else { if (verbose) { fan::print("found:", in.filename().string()); } files.push_back({in, in.filename().string(), std::filesystem::file_size(in)}); }
    fan::io::vfs_provider_t provider; std::uint8_t u32_buf[4]; fan::memory::write_le32(u32_buf, std::uint32_t(files.size())); provider.append_bytes(std::span<const std::uint8_t>(u32_buf, 4));
    for (const auto& f : files) {
      if (f.archive_path.size() > std::numeric_limits<std::uint16_t>::max()) { throw std::runtime_error("path too long"); }
      fan::bytes_t meta; std::uint8_t u16_buf[2], u64_buf[8]; fan::memory::write_le16(u16_buf, std::uint16_t(f.archive_path.size())); meta.insert(meta.end(), u16_buf, u16_buf + 2);
      auto* p_path = reinterpret_cast<const std::uint8_t*>(f.archive_path.data()); meta.insert(meta.end(), p_path, p_path + f.archive_path.size());
      fan::memory::write_le64(u64_buf, f.size); meta.insert(meta.end(), u64_buf, u64_buf + 8); provider.append_bytes(std::span<const std::uint8_t>(meta.data(), meta.size()));
    }
    if (std::size_t pad = (4 - (provider.size() & 3)) & 3; pad) { std::uint8_t zeros[3]{}; provider.append_bytes(std::span<const std::uint8_t>(zeros, pad)); }
    std::size_t payload_offset = provider.size();
    for (const auto& f : files) { provider.append_file(f.real_path, f.size); }
    fan::bytes_t raw_buf; provider.read_range(0, provider.size(), raw_buf); bool can_bcj = false;
    if (params.bcj && files.size() == 1) {
      fan::bytes_t file_head(std::min<std::uint64_t>(files[0].size, 4096)); fan::io::file::file_t* fpp = nullptr;
      if (!fan::io::file::open(&fpp, files[0].real_path.string(), {"rb"})) {
        fan::io::file::file_reader_t fr{fpp}; fr.read_exact(file_head); fan::io::file::close(fpp);
        can_bcj = looks_like_x86_binary(file_head) || looks_like_bcj_data(raw_buf, payload_offset);
      }
    }
    fan::io::file::file_t* fp = nullptr;
    if (fan::io::file::open(&fp, out_path.string(), {"wb"})) { return false; }
    try {
      std::vector<compress_candidate_t> candidates; add_transform_candidates(candidates, std::move(raw_buf), payload_offset, can_bcj, files.size() == 1, params);
      auto best = compress_best_candidate(candidates, params, prog, thread_count, verbose);
      raw_buf = std::move(best.data); fan::bytes_t comp = std::move(best.comp); std::uint8_t flags = best.flags;
      std::uint8_t header[17]; fan::memory::write_le32(header, magic_v6); fan::memory::write_le64(header + 4, raw_buf.size()); fan::memory::write_le32(header + 12, std::uint32_t(best.chunk_size)); header[16] = flags;
      fan::io::file::file_writer_t sink{fp}; sink.write_bytes(std::span<const std::uint8_t>(header, 17)); sink.write_bytes(comp);
      fan::io::file::close(fp); return true;
    } catch (...) { fan::io::file::close(fp); throw; }
  }

  bool decompress_file_to_dir(const std::filesystem::path& in_path, const std::filesystem::path& out_dir, bool default_out, fan::progress_t* prog) {
    fan::io::file::file_t* fp = nullptr;
    if (fan::io::file::open(&fp, in_path.string(), {"rb"})) { return false; }
    fan::io::file::file_reader_t src{fp};
    try {
      std::uint8_t header[17]; src.read_exact(std::span<std::uint8_t>(header, 17));
      if (fan::memory::read_le32(header) != magic_v6) { throw std::runtime_error("needs FCS6"); }
      std::uint64_t total_uncomp = fan::memory::read_le64(header + 4); std::size_t stored_chunk_size = fan::memory::read_le32(header + 12);
      if (stored_chunk_size == 0) { stored_chunk_size = default_chunk_size; }
      std::uint8_t flags = header[16]; bool use_bcj = flags & flag_bcj, use_delta = flags & flag_delta, use_text = flags & flag_text, use_text2 = flags & flag_text2, use_rle = flags & flag_rle;
      int delta_stride = 1 << ((flags >> 2) & 7); fan::bytes_t comp; std::uint64_t fsz = fan::io::file::file_size(in_path.string());
      if (fsz > 17) { comp.resize(fsz - 17); src.read_exact(comp); }
      fan::io::file::close(fp); fp = nullptr;
      if (prog) { prog->done.store(0, std::memory_order_relaxed); prog->total.store(total_uncomp, std::memory_order_relaxed); }
      fan::bytes_t raw = decode_stream_seq(comp, 0, total_uncomp, stored_chunk_size, prog);
      if (use_rle)   { raw = rle_decode_transform(raw); }
      if (use_delta) { delta_decode(raw, delta_stride); }
      if (use_text2) { raw = text2_decode_transform(raw); }
      if (use_text)  { raw = text_decode_transform(raw); }
      if (use_bcj)   { bcj_transform_range(raw, archive_payload_offset(raw), raw.size(), false); }
      fan::io::file::archive_extractor_t writer(out_dir, default_out);
      for (std::uint8_t b : raw) { writer.put(b); }
      writer.finish(); return true;
    } catch (...) { if (fp) { fan::io::file::close(fp); } throw; }
  }

  fan::bytes_t compress(const std::vector<fan::io::file_buffer_t>& files, compress_params_t params, fan::progress_t* prog) {
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
    bool can_bcj = params.bcj && files.size() == 1 && (looks_like_x86_binary(files[0].data) || looks_like_bcj_data(raw, payload_offset));
    std::vector<compress_candidate_t> candidates; add_transform_candidates(candidates, std::move(raw), payload_offset, can_bcj, files.size() == 1, params);
    auto best = compress_best_candidate(candidates, params, prog, 0);
    raw = std::move(best.data); fan::bytes_t comp = std::move(best.comp); std::uint8_t flags = best.flags;
    fan::bytes_t result; result.reserve(comp.size() + 17); std::uint8_t header[17];
    fan::memory::write_le32(header, magic_v6); fan::memory::write_le64(header + 4, raw.size()); fan::memory::write_le32(header + 12, std::uint32_t(best.chunk_size)); header[16] = flags;
    result.insert(result.end(), header, header + sizeof(header)); result.insert(result.end(), comp.begin(), comp.end());
    return result;
  }

  std::vector<fan::io::file_buffer_t> decompress(const fan::bytes_t& comp, fan::progress_t* prog) {
    std::vector<fan::io::file_buffer_t> files;
    if (comp.size() < 17 || fan::memory::read_le32(comp.data()) != magic_v6) { throw std::runtime_error("needs FCS6"); }
    std::uint64_t total_uncomp = fan::memory::read_le64(comp.data() + 4); std::size_t stored_chunk_size = fan::memory::read_le32(comp.data() + 12);
    if (stored_chunk_size == 0) { stored_chunk_size = default_chunk_size; }
    std::uint8_t flags = comp[16]; bool use_bcj = flags & flag_bcj, use_delta = flags & flag_delta, use_text = flags & flag_text, use_text2 = flags & flag_text2, use_rle = flags & flag_rle;
    int delta_stride = 1 << ((flags >> 2) & 7);
    if (prog) { prog->total.store(total_uncomp, std::memory_order_relaxed); }
    fan::bytes_t raw = decode_stream_seq(comp, 17, total_uncomp, stored_chunk_size, prog);
    if (use_rle)   { raw = rle_decode_transform(raw); }
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

  fan::event::waitv_t<fan::bytes_result_t> compress_on_thread(
    std::string path,
    fan::bytes_t data,
    const std::source_location caller
  ) {
    return fan::event::run_on_thread([path = std::move(path), data = std::move(data), caller] mutable -> fan::bytes_result_t {
      try {
        auto p = fan::io::file::find_relative_path(path, caller);

        std::vector<fan::io::file_buffer_t> files;
        files.push_back({
          p.filename().string(),
          std::move(data)
        });

        return fan::fcs::compress(files);
      }
      catch (...) {
        return std::unexpected(exception_message());
      }
    });
  }

  fan::event::waitv_t<fan::bytes_result_t> decompress_on_thread(
    std::string path,
    const std::source_location caller
  ) {
    return fan::event::run_on_thread([path = std::move(path), caller] -> bytes_result_t {
      try {
        auto p = fan::io::file::find_relative_path(path, caller);
        std::string ps = p.string();

        if (!fan::io::file::exists(p)) {
          return fan::error("file not found: " + ps);
        }

        fan::bytes_t src = fan::io::file::read_binary(p);
        if (src.empty() && fan::io::file::file_size(p) != 0) {
          return fan::error("failed to read file: " + ps);
        }

        auto files = fan::fcs::decompress(src);
        if (files.empty()) {
          return fan::error("archive contains no files: " + ps);
        }

        if (files.size() != 1) {
          return fan::error("archive contains multiple files: " + ps);
        }

        return std::move(files[0].data);
      }
      catch (...) {
        return fan::error(exception_message());
      }
    });
  }
}