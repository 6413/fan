module;

module fan.compression;

import std;

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
}
