module;

export module fan.compression;

import std;
import fan.types;
import fan.io.file;
import fan.types.fstring;
namespace file = fan::io::file;

export namespace fan::fcs {
  constexpr std::uint32_t read_le32(const std::uint8_t* p) {
    return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) | (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
  }

  constexpr void write_le32(std::uint8_t* p, std::uint32_t v) {
    p[0] = std::uint8_t(v); p[1] = std::uint8_t(v >> 8); p[2] = std::uint8_t(v >> 16); p[3] = std::uint8_t(v >> 24);
  }

  constexpr std::size_t get_match_len(const std::uint8_t* p1, const std::uint8_t* p2, std::size_t max_len) {
    std::size_t l = 0;
    while (l + 8 <= max_len) {
      std::uint64_t x, y; std::memcpy(&x, p1 + l, 8); std::memcpy(&y, p2 + l, 8);
      if (auto d = x ^ y) { return l + (std::countr_zero(d) >> 3); }
      l += 8;
    }
    while (l < max_len && p1[l] == p2[l]) { ++l; }
    return l;
  }

  constexpr void bcj_transform(fan::bytes_t& d, bool enc) {
    for (std::size_t i = 0, n = d.size(); i + 5 <= n; ++i) {
      auto* p = d.data() + i;
      auto b = p[0];
      if (b != 0xe8 && b != 0xe9 && b != 0x0f && b != 0xff && b != 0x48) { continue; }
      int s = (b == 0xe8 || b == 0xe9) ? 1 :
        (i + 6 <= n && ((b == 0x0f && (p[1] & 0xf0) == 0x80) || (b == 0xff && (p[1] == 0x15 || p[1] == 0x25)))) ? 2 :
        (i + 7 <= n && b == 0x48 && p[1] == 0x8d && (p[2] & 0xc7) == 0x05) ? 3 : 0;
      if (!s) { continue; }
      write_le32(p + s, read_le32(p + s) + (enc ? static_cast<std::uint32_t>(i) : -static_cast<std::uint32_t>(i)));
      i += s + 3;
    }
  }

  struct hybrid_sym_t { std::uint32_t cat, extra, bits; };

  constexpr hybrid_sym_t pack_sym(std::uint32_t v, std::uint32_t t, std::uint32_t base) {
    if (v < t) { return {v, 0, 0}; }
    std::uint32_t r = v - base, k = std::bit_width(r) - 1;
    return {t + k, r - (1u << k), k};
  }

  struct match_t { std::size_t offset = 0, length = 0; int profit = 0; };

  struct compress_params_t {
    std::size_t chain_main = 8192;
    std::size_t chain_lazy = 256;
    int lazy_depth = 3;
    bool bcj = true;
  };

  constexpr compress_params_t params_fast()   { return {32,   16, 0, true}; }
  constexpr compress_params_t params_normal() { return {256,  64, 1, true}; }
  constexpr compress_params_t params_max()    { return {8192, 256, 3, true}; }

  struct match_finder_t {
    match_finder_t() { head.assign(1 << 24, -1); }

    void reset(std::size_t d_start, std::size_t c_end) {
      std::fill(head.begin(), head.end(), -1);
      base = d_start;
      if (prev.size() < c_end - base) { prev.resize(c_end - base); }
    }

    constexpr std::uint32_t hash(const std::uint8_t* p) {
      return ((read_le32(p) & 0xffffffu) * 0x1e35a7bdu) & hash_mask;
    }
    std::int32_t& link(std::size_t p) { return prev[p - base]; }

    void warm_up(const fan::bytes_t& d, std::size_t ws, std::size_t we) {
      for (std::size_t i = ws; i + 2 < we; ++i) { insert(d, i); }
    }

    void insert(const fan::bytes_t& d, std::size_t i) {
      if (i + 4 > d.size() || (i >= base + 2 && i + 1 < d.size() && d[i] == d[i-1] && d[i] == d[i-2] && d[i] == d[i+1])) { return; }
      link(i) = std::exchange(head[hash(d.data() + i)], std::int32_t(i));
    }

    constexpr match_t find(const fan::bytes_t& src, std::size_t i, std::size_t c_end, const std::array<std::uint32_t, 4>& rep, std::size_t chain, bool push) {
      match_t best;
      if (i + 4 > c_end) { return best; }
      auto h = hash(src.data() + i);
      std::int32_t cur = push ? std::exchange(head[h], std::int32_t(i)) : head[h];
      if (push) { link(i) = cur; }
      const std::int32_t base_i32 = static_cast<std::int32_t>(base);
      if (cur < base_i32) { return best; }

      std::size_t max_avail = c_end - i, best_len = 0, iters = 0;
      const auto* po = src.data() + i;
      const auto* src_data = src.data();
      const auto* prev_data = prev.data() - base;

      while (cur >= base_i32 && std::size_t(cur) < i && iters++ < chain) {
        const auto* pc = src_data + cur;
        if (best_len > 0 && po[best_len] != pc[best_len]) { cur = prev_data[cur]; continue; }
        std::size_t len = 3 + get_match_len(po + 3, pc + 3, max_avail - 3);
        std::size_t off = i - std::size_t(cur);
        int ob = (off == rep[0]) ? 2 : (off == rep[1] || off == rep[2] || off == rep[3]) ? 4 : 5 + int(pack_sym(off - 1, 32, 31).bits);
        int p = int(len * 8) - ob;
        if (p > best.profit || (p == best.profit && len > best.length)) {
          best = {off, len, p}; best_len = len;
          if (len == max_avail || best_len >= 8192) { break; }
        }
        cur = prev_data[cur];
      }
      return best;
    }

    static constexpr auto hash_mask = (1 << 24) - 1;
    std::vector<std::int32_t> head, prev;
    std::size_t base = 0;
  };

  inline constexpr int num_lit_ctx = 4096;
  inline constexpr int num_dist_ctx = 16;
  inline constexpr auto chunk_size = 1uz << 26;

  enum class op_e : std::uint8_t { rep0, rep1, rep2, rep3, exp, none };
  struct seq_t { std::uint32_t lit_len; op_e op; std::uint32_t match_len, offset; };
  struct chunk_payload_t { std::vector<seq_t> seqs; };

  constexpr chunk_payload_t parse_chunk_optimal(const fan::bytes_t& src, std::size_t c_start, std::size_t c_end, match_finder_t& finder, const compress_params_t& params) {
    chunk_payload_t out;
    out.seqs.reserve((c_end - c_start) / 5);
    std::size_t d_start = c_start >= chunk_size ? c_start - chunk_size : 0, i = c_start;
    finder.reset(d_start, c_end);
    finder.warm_up(src, d_start, c_start);

    std::array<std::uint32_t, 4> rep {1, 1, 1, 1};
    std::uint32_t lit_cnt = 0;

    auto best_cand = [&](std::size_t p, bool push, std::size_t cl) {
      struct cand_t { op_e op; std::size_t len; std::uint32_t off; int profit; } b {op_e::none, 0, 0, -9999};
      std::size_t avail = c_end - p;
      if (!avail) { return b; }

      for (int r = 0; r < 4; ++r) {
        if (p < d_start + rep[r]) { continue; }
        std::size_t l = get_match_len(src.data() + p, src.data() + p - rep[r], avail);
        if (l == 0 || (l == 1 && r > 0)) { continue; }
        int pft = int(l * 8) - ((r == 0 ? 1 : 4) + 2 + int(pack_sym(l - (r ? 2 : 1), 24, 23).bits));
        if (pft > b.profit || (pft == b.profit && l > b.len)) { b = {static_cast<op_e>(r), l, rep[r], pft}; }
      }
      if (avail >= 2) {
        match_t m = finder.find(src, p, c_end, rep, cl, push);
        if (m.length >= 2) {
          int pft = int(m.length * 8) - (9 + int(pack_sym(m.length - 2, 24, 23).bits) + int(pack_sym(m.offset - 1, 32, 31).bits));
          if (pft > b.profit || (pft == b.profit && m.length > b.len)) { b = {op_e::exp, m.length, std::uint32_t(m.offset), pft}; }
        }
      } else if (push) { finder.insert(src, p); }
      return b;
    };

    while (i < c_end) {
      auto M = best_cand(i, true, params.chain_main);
      if (M.profit <= 0 || M.len == 0) { ++lit_cnt; ++i; continue; }
      if (params.lazy_depth > 0 && i + 4 < c_end && M.len < 128) {
        int skip = 1;
        for (; skip <= params.lazy_depth && i + skip < c_end; ++skip) {
          if (best_cand(i + skip, false, params.chain_lazy).profit > M.profit + (skip * 8)) { break; }
        }
        if (skip <= params.lazy_depth) { lit_cnt += skip; i += skip; continue; }
      }
      out.seqs.push_back({lit_cnt, M.op, std::uint32_t(M.len), M.off});

      if (M.op != op_e::none && M.op != op_e::rep0) {
        int r = M.op == op_e::exp ? 3 : int(M.op);
        std::uint32_t off = M.op == op_e::exp ? M.off : rep[r];
        for (int k = r; k > 0; --k) { rep[k] = rep[k - 1]; }
        rep[0] = off;
      }

      lit_cnt = 0;
      std::size_t step = M.len < 32 ? 1 : M.len >> 4;
      for (std::size_t k = 1; k < M.len; k += step) {
        if (i + k + 2 < c_end) { finder.insert(src, i + k); }
      }
      i += M.len;
    }
    if (lit_cnt) { out.seqs.push_back({lit_cnt, op_e::none, 0, 0}); }
    return out;
  }

  struct range_enc_t {
    constexpr void shift_low() {
      if (std::uint32_t(low) < 0xFF000000 || int(low >> 32) != 0) {
        out.push_back(cache + std::uint8_t(low >> 32));
        out.insert(out.end(), cache_size - 1, std::uint8_t(low >> 32) ? 0x00 : 0xFF);
        cache = std::uint8_t(low >> 24); cache_size = 1;
      } else { ++cache_size; }
      low = std::uint32_t(low << 8);
    }

    constexpr void flush() { for (int i = 0; i < 5; ++i) { shift_low(); } }

    constexpr void encode(std::uint16_t& prob, bool bit) {
      std::uint32_t bound = (range >> 11) * prob;
      if (!bit) { range = bound; prob += (2048 - prob) >> 5; }
      else { low += bound; range -= bound; prob -= prob >> 5; }
      while (range < 0x1000000) { range <<= 8; shift_low(); }
    }

    constexpr void encode_tree(std::span<std::uint16_t> tree, std::uint32_t sym, int bits) {
      for (int ctx = 1, i = bits - 1; i >= 0; --i) {
        bool bit = (sym >> i) & 1; encode(tree[ctx], bit); ctx = (ctx << 1) | bit;
      }
    }

    constexpr void encode_direct(bool bit) {
      range >>= 1; if (bit) { low += range; }
      while (range < 0x1000000) { range <<= 8; shift_low(); }
    }

    fan::bytes_t& out;
    std::uint64_t low = 0;
    std::uint32_t range = -1u, cache_size = 1;
    std::uint8_t cache = 0;
  };

  struct range_dec_t {
    constexpr range_dec_t(const fan::bytes_t& in, std::size_t s) : in(in), pos(s) {
      read_byte();
      for (int i = 0; i < 4; ++i) { code = (code << 8) | read_byte(); }
    }
    constexpr std::uint8_t read_byte() { return pos < in.size() ? in[pos++] : 0; }

    constexpr bool decode(std::uint16_t& prob) {
      std::uint32_t bound = (range >> 11) * prob;
      bool bit = code >= bound;
      if (!bit) { range = bound; prob += (2048 - prob) >> 5; }
      else { range -= bound; code -= bound; prob -= prob >> 5; }
      while (range < 0x1000000) { range <<= 8; code = (code << 8) | read_byte(); }
      return bit;
    }

    constexpr std::uint32_t decode_tree(std::span<std::uint16_t> tree, int bits) {
      std::uint32_t ctx = 1;
      for (int i = 0; i < bits; ++i) { ctx = (ctx << 1) | decode(tree[ctx]); }
      return ctx - (1u << bits);
    }

    constexpr bool decode_direct() {
      range >>= 1; bool bit = code >= range;
      if (bit) { code -= range; }
      while (range < 0x1000000) { range <<= 8; code = (code << 8) | read_byte(); }
      return bit;
    }

    const fan::bytes_t& in;
    std::size_t pos = 0;
    std::uint32_t range = -1u, code = 0;
  };

  struct stream_model_t {
    constexpr stream_model_t() {
      auto init = [](this const auto& s, auto&... arr) -> void {
        (..., [&](auto& a) {
          for (auto& e : a) {
            if constexpr (std::is_same_v<std::remove_reference_t<decltype(e)>, std::uint16_t>) { e = 1024; }
            else { s(e); }
          }
        }(arr));
      };
      init(lit, is_exp, is_rep0, is_none, rep_idx, lit_len, lit_len_extra, match_len, match_len_extra, off, off_extra);
    }

    std::array<std::array<std::uint16_t, 256>, num_lit_ctx> lit;
    std::array<std::uint16_t, 32> is_exp, is_rep0, is_none;
    std::array<std::array<std::uint16_t, 4>, 32> rep_idx;
    std::array<std::array<std::uint16_t, 64>, 4> lit_len;
    std::array<std::array<std::uint16_t, 32>, 64> lit_len_extra;
    std::array<std::array<std::array<std::uint16_t, 64>, 5>, 4> match_len;
    std::array<std::array<std::uint16_t, 32>, 64> match_len_extra;
    std::array<std::array<std::uint16_t, 64>, num_dist_ctx> off;
    std::array<std::array<std::uint16_t, 32>, 64> off_extra;
  };

  constexpr void encode_op(range_enc_t& rc, stream_model_t& m, std::uint32_t ll, op_e op, std::size_t pos) {
    int ctx = (std::min<int>(ll, 7) << 2) | (pos & 3);
    rc.encode(m.is_exp[ctx], op == op_e::exp);   if (op == op_e::exp) { return; }
    rc.encode(m.is_rep0[ctx], op == op_e::rep0); if (op == op_e::rep0) { return; }
    rc.encode(m.is_none[ctx], op == op_e::none); if (op == op_e::none) { return; }
    rc.encode_tree(m.rep_idx[ctx], int(op) - int(op_e::rep1), 2);
  }

  constexpr op_e decode_op(range_dec_t& rd, stream_model_t& m, std::uint32_t ll, std::size_t pos) {
    int ctx = (std::min<int>(ll, 7) << 2) | (pos & 3);
    if (rd.decode(m.is_exp[ctx])) { return op_e::exp; }
    if (rd.decode(m.is_rep0[ctx])) { return op_e::rep0; }
    if (rd.decode(m.is_none[ctx])) { return op_e::none; }
    return static_cast<op_e>(int(op_e::rep1) + rd.decode_tree(m.rep_idx[ctx], 2));
  }

  constexpr void encode_val(range_enc_t& rc, auto& tree, auto& xtree, hybrid_sym_t h) {
    rc.encode_tree(std::span<std::uint16_t>{tree}, h.cat, 6);
    if (h.bits > 0) {
      int mb = std::min<int>(int(h.bits), 4), rem = h.bits - mb;
      for (int i = rem - 1; i >= 0; --i) { rc.encode_direct((h.extra >> (mb + i)) & 1); }
      rc.encode_tree(std::span<std::uint16_t>{xtree[h.cat]}, h.extra & ((1 << mb) - 1), mb);
    }
  }

  constexpr std::uint32_t decode_val(range_dec_t& rd, auto& tree, auto& xtree, int t, int base) {
    std::uint32_t c = rd.decode_tree(std::span<std::uint16_t>{tree}, 6);
    if (c < t) { return c + (t == 32); }
    int b = c - t, mb = std::min(b, 4), rem = b - mb; std::uint32_t x = 0;
    for (int i = 0; i < rem; ++i) { x = (x << 1) | rd.decode_direct(); }
    return base + (1u << b) + ((x << mb) | rd.decode_tree(std::span<std::uint16_t>{xtree[c]}, mb));
  }

  fan::bytes_t compress(fan::bytes_t src, compress_params_t params = params_max()) {
    bool bcj = params.bcj && file::is_pe(src);
    if (bcj) { fan::fcs::bcj_transform(src, true); }
    std::size_t nc = (src.size() + chunk_size - 1) / chunk_size;

    std::vector<chunk_payload_t> blocks(nc);
    std::atomic<std::size_t> next = 0;
    std::size_t nw = std::min<std::size_t>(nc, std::max(1u, std::thread::hardware_concurrency()));
    std::vector<std::jthread> workers;
    for (std::size_t w = 0; w < nw; ++w) {
      workers.emplace_back([&] {
        match_finder_t finder;
        for (std::size_t k; (k = next.fetch_add(1, std::memory_order_relaxed)) < nc;) {
          blocks[k] = parse_chunk_optimal(src, k * chunk_size, std::min(src.size(), (k + 1) * chunk_size), finder, params);
        }
      });
    }
    workers.clear();

    fan::bytes_t out; out.reserve(src.size() / 5);
    std::uint32_t t_sz = src.size(); auto* p = (std::uint8_t*)&t_sz;
    out.insert(out.end(), p, p + 4); out.push_back(bcj);

    range_enc_t rc {out};
    auto m_ptr = std::make_unique<stream_model_t>(); stream_model_t& model = *m_ptr;

    std::size_t src_ptr = 0;
    for (std::size_t k = 0; k < nc; ++k) {
      for (const auto& s : blocks[k].seqs) {
        encode_val(rc, model.lit_len[src_ptr & 3], model.lit_len_extra, pack_sym(s.lit_len, 24, 23));
        for (std::uint32_t j = 0; j < s.lit_len; ++j) {
          std::uint32_t ctx = (src_ptr >= 1 ? src[src_ptr - 1] : 0) | ((src_ptr >= 2 ? src[src_ptr - 2] >> 4 : 0) << 8);
          rc.encode_tree(model.lit[ctx], src[src_ptr++], 8);
        }
        encode_op(rc, model, s.lit_len, s.op, src_ptr);
        if (s.op == op_e::none) { continue; }

        if (s.op == op_e::rep0) {
          encode_val(rc, model.match_len[src_ptr & 3][0], model.match_len_extra, pack_sym(s.match_len - 1, 24, 23));
        } else {
          int op_i = s.op == op_e::exp ? 4 : int(s.op);
          encode_val(rc, model.match_len[src_ptr & 3][op_i], model.match_len_extra, pack_sym(s.match_len - 2, 24, 23));
          if (s.op == op_e::exp) { encode_val(rc, model.off[std::min<std::uint32_t>(s.match_len - 2, num_dist_ctx - 1)], model.off_extra, pack_sym(s.offset - 1, 32, 31)); }
        }
        src_ptr += s.match_len;
      }
    }
    rc.flush();
    return out;
  }

  fan::bytes_t decompress(const fan::bytes_t& comp) {
    std::size_t idx = 0;
    std::uint32_t total_uncomp = fan::vector_read_data<std::uint32_t>(comp, idx);
    bool bcj = comp[idx++];

    fan::bytes_t out; out.reserve(total_uncomp);
    range_dec_t rd {comp, idx};
    auto m_ptr = std::make_unique<stream_model_t>(); stream_model_t& model = *m_ptr;

    std::array<std::uint32_t, 4> rep {1, 1, 1, 1};
    std::size_t next_boundary = chunk_size;

    while (out.size() < total_uncomp) {
      if (out.size() == next_boundary) { rep = {1, 1, 1, 1}; next_boundary += chunk_size; }
      std::uint32_t ll = decode_val(rd, model.lit_len[out.size() & 3], model.lit_len_extra, 24, 23);
      for (std::uint32_t j = 0; j < ll; ++j) {
        std::uint32_t ctx = (out.size() >= 1 ? out.back() : 0) | ((out.size() >= 2 ? out[out.size() - 2] >> 4 : 0) << 8);
        out.push_back(std::uint8_t(rd.decode_tree(model.lit[ctx], 8)));
      }
      if (out.size() >= total_uncomp) { break; }

      op_e op = decode_op(rd, model, ll, out.size());
      if (op == op_e::none) { continue; }

      std::uint32_t mlen = 0, off = 0;
      if (op == op_e::rep0) {
        mlen = 1 + decode_val(rd, model.match_len[out.size() & 3][0], model.match_len_extra, 24, 23);
        off = rep[0];
      } else {
        int op_i = op == op_e::exp ? 4 : int(op);
        mlen = 2 + decode_val(rd, model.match_len[out.size() & 3][op_i], model.match_len_extra, 24, 23);
        int r = op == op_e::exp ? 3 : int(op);
        off = op == op_e::exp ? decode_val(rd, model.off[std::min<std::uint32_t>(mlen - 2, num_dist_ctx - 1)], model.off_extra, 32, 32) : rep[r];
        for (int k = r; k > 0; --k) { rep[k] = rep[k - 1]; }
        rep[0] = off;
      }

      std::size_t old = out.size(); out.resize(old + mlen);
      std::uint8_t* p = out.data() + old;
      const std::uint8_t* s = p - off;

      if (off == 1) { std::memset(p, *s, mlen); }
      else if (off >= mlen) { std::memcpy(p, s, mlen); }
      else {
        std::size_t j = 0;
        for (; j + off <= mlen; j += off) { std::memcpy(p + j, s + j, off); }
        for (; j < mlen; ++j) { p[j] = s[j]; }
      }
    }
    if (bcj) { fan::fcs::bcj_transform(out, false); }
    return out;
  }
}