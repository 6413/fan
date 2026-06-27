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

  constexpr void bcj_transform(fan::bytes_t& d, bool enc) {
    for (std::size_t i = 0, n = d.size(); i + 5 <= n; ++i) {
      auto p = d.data() + i;
      auto b = p[0];
      int s = (b == 0xe8 || b == 0xe9) ? 1 :
        (i + 6 <= n && ((b == 0x0f && (p[1] & 0xf0) == 0x80) || (b == 0xff && (p[1] == 0x15 || p[1] == 0x25)))) ? 2 :
        (i + 7 <= n && b == 0x48 && p[1] == 0x8d && (p[2] & 0xc7) == 0x05) ? 3 : 0;
      if (s) {
        std::uint32_t v; std::memcpy(&v, p + s, 4);
        v += enc ? std::uint32_t(i) : -std::uint32_t(i);
        std::memcpy(p + s, &v, 4);
        i += s + 3;
      }
    }
  }

  constexpr void delta_encode(fan::bytes_t& d, int stride) {
    for (std::size_t i = d.size() - 1; i >= std::size_t(stride); --i) { d[i] -= d[i - stride]; }
  }
  constexpr void delta_decode(fan::bytes_t& d, int stride) {
    for (std::size_t i = stride; i < d.size(); ++i) { d[i] += d[i - stride]; }
  }
  inline int detect_delta_stride(const fan::bytes_t& d) {
    std::size_t probe = std::min<std::size_t>(d.size(), 65536);
    int best_stride = 1; double best_ent = 1e18;
    for (int s : {1, 2, 4, 8}) {
      std::array<int, 256> freq {}; std::size_t n = 0;
      for (std::size_t i = s; i < probe; ++i) { ++freq[std::uint8_t(d[i] - d[i - s])]; ++n; }
      double ent = 0;
      for (int c : freq) { if (c) { double p = double(c) / n; ent -= p * std::log2(p); } }
      if (ent < best_ent) { best_ent = ent; best_stride = s; }
    }
    return best_stride;
  }

  struct hybrid_sym_t { std::uint32_t cat, extra, bits; };

  constexpr hybrid_sym_t pack_hybrid(std::uint32_t v) {
    if (v < 24) { return {v, 0, 0}; }
    std::uint32_t r = v - 23, k = std::bit_width(r) - 1;
    return {24 + k, r - (1u << k), k};
  }

  constexpr hybrid_sym_t pack_offset_hybrid(std::uint32_t off) {
    std::uint32_t v = off - 1;
    if (v < 32) { return {v, 0, 0}; }
    std::uint32_t r = v - 32, k = std::bit_width(r + 1) - 1;
    return {32 + k, r - ((1u << k) - 1), k};
  }

  struct match_t { std::size_t offset = 0, length = 0; int profit = 0; };

  struct match_finder_t {
    static constexpr auto hash_mask = (1 << 24) - 1;
    std::vector<std::int32_t> head, prev;
    std::size_t base = 0;

    match_finder_t(std::size_t d_start, std::size_t c_end) : head(1 << 24, -1), base(d_start) {
      prev.assign(c_end - base, -1);
    }

    static std::uint32_t hash(const fan::bytes_t& d, std::size_t i) {
      std::uint32_t v; std::memcpy(&v, d.data() + i, 4);
      return ((v & 0xFFFFFFu) * 0x1E35A7BDu) & hash_mask;
    }
    std::int32_t& link(std::size_t p) { return prev[p - base]; }

    void warm_up(const fan::bytes_t& d, std::size_t ws, std::size_t we) {
      for (std::size_t i = ws; i + 2 < we; ++i) { insert(d, i); }
    }

    void insert(const fan::bytes_t& d, std::size_t i) {
      if (i + 4 > d.size() || (i >= base + 2 && i + 1 < d.size() && d[i] == d[i-1] && d[i] == d[i-2] && d[i] == d[i+1])) { return; }
      auto h = hash(d, i); 
      link(i) = head[h]; 
      head[h] = std::int32_t(i);
    }

    match_t find(const fan::bytes_t& src, std::size_t i, std::size_t c_end, const std::array<std::uint32_t,4>& rep, std::size_t chain, bool push) {
      match_t best;
      if (i + 4 > c_end) { return best; }

      auto h = hash(src, i);
      std::int32_t cur = head[h];
      if (push) { link(i) = cur; head[h] = std::int32_t(i); }
      if (cur < static_cast<std::int32_t>(base)) { return best; }

      std::size_t max_avail = c_end - i, best_len = 0, iters = 0;
      const auto* po = src.data() + i;
      std::uint32_t po32; std::memcpy(&po32, po, 4); po32 &= 0xFFFFFFu;

      while (cur >= static_cast<std::int32_t>(base) && std::size_t(cur) < i && iters++ < chain) {
        const auto* pc = src.data() + cur;
        if (best_len > 0 && po[best_len] != pc[best_len]) { cur = link(cur); continue; }
        std::uint32_t pc32; std::memcpy(&pc32, pc, 4);
        if ((pc32 & 0xFFFFFFu) == po32) {
          std::size_t len = 3;
          while (len + 8 <= max_avail) {
            std::uint64_t x, y; std::memcpy(&x, po+len, 8); std::memcpy(&y, pc+len, 8);
            if (auto d = x ^ y) { len += std::countr_zero(d) >> 3; break; } 
            len += 8;
          }
          while (len < max_avail && po[len] == pc[len]) { ++len; }

          std::size_t off = i - std::size_t(cur);
          int ob = (off == rep[0]) ? 2 : (off == rep[1] || off == rep[2] || off == rep[3]) ? 4 : 5 + int(pack_offset_hybrid(std::uint32_t(off)).bits);
          int p = int(len * 8) - ob;

          if (p > best.profit || (p == best.profit && len > best.length)) {
            best = {off, len, p}; best_len = len;
            if (len == max_avail || best_len >= 8192) { break; } 
          }
        }
        cur = link(cur);
      }
      return best;
    }
  };

  struct compress_params_t {
    std::size_t chain_main = 8192, chain_lazy = 256;
    int lazy_depth = 3;
    bool bcj = true;
  };

  constexpr compress_params_t params_fast()   { return {32,   16, 0, true}; }
  constexpr compress_params_t params_normal() { return {256,  64, 1, true}; }
  constexpr compress_params_t params_max()    { return {8192, 256, 3, true}; }

  enum class op_e : std::uint8_t { rep0, rep1, rep2, rep3, exp, none };
  struct seq_t { std::uint32_t lit_len; op_e op; std::uint32_t match_len, offset; };
  struct chunk_payload_t { std::vector<seq_t> seqs; };

  chunk_payload_t parse_chunk_optimal(const fan::bytes_t& src, std::size_t c_start, std::size_t c_end, const compress_params_t& params, fan::progress_t* prog) {
    chunk_payload_t out;
    out.seqs.reserve((c_end - c_start) / 5);
    std::size_t d_start = c_start >= chunk_size ? c_start - chunk_size : 0;
    std::size_t i = c_start;
    match_finder_t finder(d_start, c_end);
    finder.warm_up(src, d_start, c_start);

    std::array<std::uint32_t, 4> rep{1, 1, 1, 1};
    std::uint32_t lit_cnt = 0;
    std::size_t last_report = c_start;

    auto best_cand = [&](std::size_t p, bool push, std::size_t cl) {
      struct cand_t { op_e op; std::size_t len; std::uint32_t off; int profit; } b{op_e::none, 0, 0, -9999};
      std::size_t avail = c_end - p;
      if (!avail) { return b; }

      for (int r = 0; r < 4; ++r) {
        if (p < d_start + rep[r]) { continue; }
        std::size_t l = 0;
        while (l < avail && src[p+l] == src[p - rep[r] + l]) { ++l; }
        if (l == 0 || (l == 1 && r > 0)) { continue; }
        int pft = int(l * 8) - ((r == 0 ? 1 : 4) + 2 + int(pack_hybrid(std::uint32_t(l - (r ? 2 : 1))).bits));
        if (pft > b.profit || (pft == b.profit && l > b.len)) { b = {static_cast<op_e>(r), l, rep[r], pft}; }
      }
      if (avail >= 2) {
        match_t m = finder.find(src, p, c_end, rep, cl, push);
        if (m.length >= 2) {
          int pft = int(m.length * 8) - (9 + int(pack_hybrid(std::uint32_t(m.length - 2)).bits) + int(pack_offset_hybrid(std::uint32_t(m.offset)).bits));
          if (pft > b.profit || (pft == b.profit && m.length > b.len)) { b = {op_e::exp, m.length, std::uint32_t(m.offset), pft}; }
        }
      } else if (push) { finder.insert(src, p); }
      return b;
    };

    while (i < c_end) {
      auto M = best_cand(i, true, params.chain_main);
      if (M.profit <= 0 || M.len == 0) { ++lit_cnt; ++i; }
      else {
        if (i + 4 < c_end && M.len < 128) {
          int skip = 1;
          for (; skip <= params.lazy_depth && i + skip < c_end; ++skip) {
            if (best_cand(i + skip, false, params.chain_lazy).profit > M.profit + (skip * 8)) { break; }
          }
          if (skip <= params.lazy_depth) { lit_cnt += skip; i += skip; continue; }
        }
        out.seqs.push_back({lit_cnt, M.op, std::uint32_t(M.len), M.off});

        if      (M.op == op_e::rep1) { std::swap(rep[0], rep[1]); }
        else if (M.op == op_e::rep2) { auto t=rep[2]; rep[2]=rep[1]; rep[1]=rep[0]; rep[0]=t; }
        else if (M.op == op_e::rep3) { auto t=rep[3]; rep[3]=rep[2]; rep[2]=rep[1]; rep[1]=rep[0]; rep[0]=t; }
        else if (M.op == op_e::exp)  { rep[3]=rep[2]; rep[2]=rep[1]; rep[1]=rep[0]; rep[0]=M.off; }

        lit_cnt = 0;
        std::size_t step = M.len < 32 ? 1 : M.len >> 4;
        for (std::size_t k = 1; k < M.len; k += step) {
          if (i + k + 2 < c_end) { finder.insert(src, i + k); }
        }
        i += M.len;
      }
      if (prog && i - last_report >= 65536) {
        prog->done.fetch_add((i - last_report) * 80 / 100, std::memory_order_relaxed);
        last_report = i;
      }
    }
    if (prog) { prog->done.fetch_add((c_end - last_report) * 80 / 100, std::memory_order_relaxed); }
    if (lit_cnt) { out.seqs.push_back({lit_cnt, op_e::none, 0, 0}); }
    return out;
  }

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
  };

  struct op_model_t {
    std::array<std::uint16_t, 32> is_exp, is_rep0, is_none;
    std::array<std::array<std::uint16_t, 4>, 32> rep_idx;
  };

  inline constexpr int num_lit_ctx = 4096;
  inline constexpr int num_dist_ctx = 16;
  inline constexpr std::uint32_t magic_v4 = 0x34334346;

  struct stream_model_t {
    std::array<std::array<std::uint16_t, 256>, num_lit_ctx> lit;
    op_model_t op;
    std::array<std::array<std::uint16_t, 64>, 4> lit_len;
    std::array<std::array<std::uint16_t, 32>, 64> lit_len_extra;
    std::array<std::array<std::array<std::uint16_t, 64>, 5>, 4> match_len;
    std::array<std::array<std::uint16_t, 32>, 64> match_len_extra;
    std::array<std::array<std::uint16_t, 64>, num_dist_ctx> off;
    std::array<std::array<std::uint16_t, 32>, 64> off_extra;

    stream_model_t() {
      auto init = [](this const auto& s, auto& arr) -> void {
        for (auto& e : arr) {
          if constexpr (std::is_same_v<std::remove_reference_t<decltype(e)>, std::uint16_t>) { e = 1024; }
          else { s(e); }
        }
      };
      init(lit); init(op.is_exp); init(op.is_rep0); init(op.is_none); init(op.rep_idx);
      init(lit_len); init(lit_len_extra); init(match_len); init(match_len_extra);
      init(off); init(off_extra);
    }
  };

  template <typename Writer>
  void encode_op(range_enc_t<Writer>& rc, stream_model_t& m, std::uint32_t ll, op_e op, std::size_t pos) {
    int ctx = (std::min<int>(ll, 7) << 2) | (pos & 3);
    rc.encode(m.op.is_exp[ctx], op == op_e::exp);         if (op == op_e::exp) { return; }
    rc.encode(m.op.is_rep0[ctx], op == op_e::rep0);       if (op == op_e::rep0) { return; }
    rc.encode(m.op.is_none[ctx], op == op_e::none);       if (op == op_e::none) { return; }
    rc.encode_tree(m.op.rep_idx[ctx], int(op) - int(op_e::rep1), 2);
  }

  template <typename Reader>
  op_e decode_op(range_dec_t<Reader>& rd, stream_model_t& m, std::uint32_t ll, std::size_t pos) {
    int ctx = (std::min<int>(ll, 7) << 2) | (pos & 3);
    if (rd.decode(m.op.is_exp[ctx])) { return op_e::exp; }
    if (rd.decode(m.op.is_rep0[ctx])) { return op_e::rep0; }
    if (rd.decode(m.op.is_none[ctx])) { return op_e::none; }
    return static_cast<op_e>(int(op_e::rep1) + rd.decode_tree(m.op.rep_idx[ctx], 2));
  }

  template <typename Writer>
  void encode_hval(range_enc_t<Writer>& rc, auto& tree, auto& xtree, std::uint32_t v) {
    auto h = pack_hybrid(v);
    rc.encode_tree(std::span<std::uint16_t>{tree}, h.cat, 6);
    if (h.bits > 0) {
      int mb = std::min<int>(h.bits, 4), rem = h.bits - mb;
      for (int i = rem - 1; i >= 0; --i) { rc.encode_direct((h.extra >> (mb + i)) & 1); }
      rc.encode_tree(std::span<std::uint16_t>{xtree[h.cat]}, h.extra & ((1 << mb) - 1), mb);
    }
  }

  template <typename Reader>
  std::uint32_t decode_hval(range_dec_t<Reader>& rd, auto& tree, auto& xtree) {
    std::uint32_t cat = rd.decode_tree(std::span<std::uint16_t>{tree}, 6);
    if (cat < 24) { return cat; }
    int b = cat - 24, mb = std::min<int>(b, 4), rem = b - mb; std::uint32_t x = 0;
    for (int i = 0; i < rem; ++i) { x = (x << 1) | rd.decode_direct(); }
    return 23 + (1u << b) + ((x << mb) | rd.decode_tree(std::span<std::uint16_t>{xtree[cat]}, mb));
  }

  template <typename Writer>
  void encode_oval(range_enc_t<Writer>& rc, auto& tree, auto& xtree, std::uint32_t v) {
    auto h = pack_offset_hybrid(v);
    rc.encode_tree(std::span<std::uint16_t>{tree}, h.cat, 6);
    if (h.bits > 0) {
      int mb = std::min<int>(h.bits, 4), rem = h.bits - mb;
      for (int i = rem - 1; i >= 0; --i) { rc.encode_direct((h.extra >> (mb + i)) & 1); }
      rc.encode_tree(std::span<std::uint16_t>{xtree[h.cat]}, h.extra & ((1 << mb) - 1), mb);
    }
  }

  template <typename Reader>
  std::uint32_t decode_oval(range_dec_t<Reader>& rd, auto& tree, auto& xtree) {
    std::uint32_t cat = rd.decode_tree(std::span<std::uint16_t>{tree}, 6);
    if (cat < 32) { return cat + 1; }
    int b = cat - 32, mb = std::min<int>(b, 4), rem = b - mb; std::uint32_t x = 0;
    for (int i = 0; i < rem; ++i) { x = (x << 1) | rd.decode_direct(); }
    return 32 + (1u << b) + ((x << mb) | rd.decode_tree(std::span<std::uint16_t>{xtree[cat]}, mb));
  }

  fan::bytes_t encode_stream_seq(const fan::bytes_t& src, const std::vector<chunk_payload_t>& blocks, fan::progress_t* prog) {
    fan::bytes_t out; out.reserve(src.size() / 5);
    fan::io::bytes_writer_t bw {out}; 
    range_enc_t<fan::io::bytes_writer_t> rc{bw};
    auto m_ptr = std::make_unique<stream_model_t>(); stream_model_t& model = *m_ptr;

    std::size_t src_ptr = 0;
    std::size_t last_report = 0;

    for (std::size_t k = 0; k < blocks.size(); ++k) {
      for (const auto& s : blocks[k].seqs) {
        encode_hval(rc, model.lit_len[src_ptr & 3], model.lit_len_extra, s.lit_len);
        for (std::uint32_t j = 0; j < s.lit_len; ++j) {
          std::uint32_t ctx = (src_ptr >= 1 ? src[src_ptr - 1] : 0) | ((src_ptr >= 2 ? src[src_ptr - 2] >> 4 : 0) << 8);
          rc.encode_tree(model.lit[ctx], src[src_ptr++], 8);
        }
        encode_op(rc, model, s.lit_len, s.op, src_ptr);
        if (s.op == op_e::none) { continue; }

        if (s.op == op_e::rep0) { encode_hval(rc, model.match_len[src_ptr & 3][0], model.match_len_extra, s.match_len - 1); } 
        else {
          int op_i = s.op == op_e::exp ? 4 : int(s.op);
          encode_hval(rc, model.match_len[src_ptr & 3][op_i], model.match_len_extra, s.match_len - 2);
          if (s.op == op_e::exp) { encode_oval(rc, model.off[std::min<std::uint32_t>(s.match_len - 2, num_dist_ctx - 1)], model.off_extra, s.offset); }
        }
        src_ptr += s.match_len;
      }
      if (prog) {
        std::size_t encoded = src_ptr - last_report;
        prog->done.fetch_add(encoded * 20 / 100, std::memory_order_relaxed);
        last_report = src_ptr;
      }
    }
    rc.flush();
    if (prog) {
      std::size_t encoded = src_ptr - last_report;
      prog->done.fetch_add(encoded * 20 / 100, std::memory_order_relaxed);
    }
    return out;
  }

  fan::bytes_t decode_stream_seq(const fan::bytes_t& comp, std::size_t idx, std::uint64_t total_uncomp, fan::progress_t* prog) {
    fan::bytes_t out; out.reserve(total_uncomp);
    fan::io::bytes_reader_t br {comp, idx};
    range_dec_t<fan::io::bytes_reader_t> rd{br};
    auto m_ptr = std::make_unique<stream_model_t>(); stream_model_t& model = *m_ptr;

    std::array<std::uint32_t, 4> rep{1, 1, 1, 1};
    std::size_t next_boundary = chunk_size;
    std::size_t last_report = 0;

    while (out.size() < total_uncomp) {
      if (out.size() == next_boundary) { rep = {1, 1, 1, 1}; next_boundary += chunk_size; }
      std::uint32_t ll = decode_hval(rd, model.lit_len[out.size() & 3], model.lit_len_extra);
      for (std::uint32_t j = 0; j < ll; ++j) {
        std::uint32_t ctx = (out.size() >= 1 ? out.back() : 0) | ((out.size() >= 2 ? out[out.size() - 2] >> 4 : 0) << 8);
        out.push_back(std::uint8_t(rd.decode_tree(model.lit[ctx], 8)));
      }
      if (out.size() >= total_uncomp) { break; }

      op_e op = decode_op(rd, model, ll, out.size());
      if (op == op_e::none) { continue; }

      std::uint32_t mlen = 0, off = 0;
      if (op == op_e::rep0) { mlen = 1 + decode_hval(rd, model.match_len[out.size() & 3][0], model.match_len_extra); off = rep[0]; } 
      else {
        int op_i = op == op_e::exp ? 4 : int(op);
        mlen = 2 + decode_hval(rd, model.match_len[out.size() & 3][op_i], model.match_len_extra);
        if      (op == op_e::rep1) { off = rep[1]; std::swap(rep[0], rep[1]); }
        else if (op == op_e::rep2) { off = rep[2]; auto t=rep[2]; rep[2]=rep[1]; rep[1]=rep[0]; rep[0]=t; }
        else if (op == op_e::rep3) { off = rep[3]; auto t=rep[3]; rep[3]=rep[2]; rep[2]=rep[1]; rep[1]=rep[0]; rep[0]=t; }
        else if (op == op_e::exp)  {
          off = decode_oval(rd, model.off[std::min<std::uint32_t>(mlen - 2, num_dist_ctx - 1)], model.off_extra);
          rep[3]=rep[2]; rep[2]=rep[1]; rep[1]=rep[0]; rep[0]=off;
        }
      }
      std::size_t old = out.size(); out.resize(old + mlen);
      if (off >= mlen) { std::memcpy(&out[old], &out[old - off], mlen); }
      else { for (std::uint32_t j = 0; j < mlen; ++j) { out[old + j] = out[old - off + j]; } }
      
      if (prog && out.size() - last_report >= 65536) {
        prog->done.store(out.size(), std::memory_order_relaxed);
        last_report = out.size();
      }
    }
    if (prog) { prog->done.store(total_uncomp, std::memory_order_relaxed); }
    return out;
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
      fan::bytes_t meta;
      std::uint8_t u16_buf[2], u64_buf[8]; fan::memory::write_le16(u16_buf, std::uint16_t(f.archive_path.size()));
      meta.insert(meta.end(), u16_buf, u16_buf + 2);
      auto* p_path = reinterpret_cast<const std::uint8_t*>(f.archive_path.data());
      meta.insert(meta.end(), p_path, p_path + f.archive_path.size());
      fan::memory::write_le64(u64_buf, f.size); meta.insert(meta.end(), u64_buf, u64_buf + 8);
      provider.append_bytes(std::span<const std::uint8_t>(meta.data(), meta.size()));
    }
    std::size_t pad = (4 - (provider.size() & 3)) & 3;
    if (pad) {
      std::uint8_t zeros[3] = {0, 0, 0};
      provider.append_bytes(std::span<const std::uint8_t>(zeros, pad));
    }
    for (const auto& f : files) { provider.append_file(f.real_path, f.size); }

    fan::bytes_t raw_buf;
    bool use_delta = false;
    int delta_stride = 1;
    if (!std::filesystem::is_directory(in)) {
      provider.read_range(0, provider.size(), raw_buf);
      delta_stride = detect_delta_stride(raw_buf);
      use_delta = delta_stride > 1;
      if (use_delta) { delta_encode(raw_buf, delta_stride); }
    } else {
      provider.read_range(0, provider.size(), raw_buf);
    }
    
    int stride_log2 = delta_stride == 8 ? 3 : delta_stride == 4 ? 2 : delta_stride == 2 ? 1 : 0;
    std::uint8_t flags = std::uint8_t(use_delta ? 2 : 0) | std::uint8_t(stride_log2 << 2);

    fan::io::file::file_t* fp = nullptr;
    if (fan::io::file::open(&fp, out_path.string(), {"wb"})) { return false; }

    try {
      std::size_t actual_threads = thread_count ? thread_count : std::max<std::size_t>(1, std::thread::hardware_concurrency());
      if (prog) { prog->done.store(0, std::memory_order_relaxed); prog->total.store(raw_buf.size(), std::memory_order_relaxed); }

      std::size_t nc = (raw_buf.size() + chunk_size - 1) / chunk_size;
      std::vector<chunk_payload_t> blocks(nc);
      std::atomic<std::size_t> next = 0;
      std::vector<std::jthread> workers;
      for (std::size_t w = 0; w < actual_threads; ++w) {
        workers.emplace_back([&] {
          for (std::size_t k; (k = next.fetch_add(1, std::memory_order_relaxed)) < nc;) {
            std::size_t c_start = k * chunk_size;
            std::size_t c_end = std::min(raw_buf.size(), c_start + chunk_size);
            blocks[k] = parse_chunk_optimal(raw_buf, c_start, c_end, params, prog);
          }
        });
      }
      workers.clear();

      fan::bytes_t comp = encode_stream_seq(raw_buf, blocks, prog);

      std::uint8_t header[17];
      fan::memory::write_le32(header, magic_v4); 
      fan::memory::write_le64(header + 4, raw_buf.size());
      fan::memory::write_le32(header + 12, chunk_size);
      header[16] = flags;
      
      fan::io::file::file_writer_t sink{fp};
      sink.write_bytes(std::span<const std::uint8_t>(header, 17));
      sink.write_bytes(comp);

      fan::io::file::close(fp);
      if (prog) { prog->done.store(raw_buf.size(), std::memory_order_relaxed); }
      return true;
    } catch (...) {
      fan::io::file::close(fp); throw;
    }
  }

  bool decompress_file_to_dir(const std::filesystem::path& in_path, const std::filesystem::path& out_dir, bool default_out, fan::progress_t* prog = nullptr) {
    fan::io::file::file_t* fp = nullptr;
    if (fan::io::file::open(&fp, in_path.string(), {"rb"})) { return false; }
    
    fan::io::file::file_reader_t src {fp};

    try {
      std::uint8_t header[17]; src.read_exact(std::span<std::uint8_t>(header, 17));
      if (fan::memory::read_le32(header) != magic_v4) { throw std::runtime_error("needs FCS4"); }
      std::uint64_t total_uncomp = fan::memory::read_le64(header + 4);
      std::uint8_t flags = header[16];
      bool bcj = flags & 1;
      bool use_delta = (flags >> 1) & 1;
      int stride_log2 = (flags >> 2) & 7;
      int delta_stride = 1 << stride_log2;
      
      fan::bytes_t comp;
      std::uint64_t fsz = fan::io::file::file_size(in_path.string());
      if (fsz > 17) {
        comp.resize(fsz - 17);
        src.read_exact(comp);
      }
      fan::io::file::close(fp); fp = nullptr;

      if (prog) { prog->done.store(0, std::memory_order_relaxed); prog->total.store(total_uncomp, std::memory_order_relaxed); }

      fan::bytes_t raw = decode_stream_seq(comp, 0, total_uncomp, prog);
      if (use_delta) { delta_decode(raw, delta_stride); }
      if (bcj) { bcj_transform(raw, false); }

      fan::io::file::archive_extractor_t writer(out_dir, default_out);
      for (std::uint8_t b : raw) { writer.put(b); }
      writer.finish();
      
      return true;
    } catch (...) {
      if (fp) fan::io::file::close(fp); 
      throw;
    }
  }

  fan::bytes_t compress(const std::vector<fan::io::file_buffer_t>& files, compress_params_t params = params_max(), fan::progress_t* prog = nullptr) {
    if (files.size() > std::numeric_limits<std::uint32_t>::max()) { throw std::runtime_error("too many files"); }
    std::size_t total_sz = 4;
    for (const auto& f : files) {
      if (f.path.size() > std::numeric_limits<std::uint16_t>::max()) { throw std::runtime_error("path too long"); }
      total_sz += 2 + f.path.size() + 8;
    }
    std::size_t pad = (4 - (total_sz & 3)) & 3; total_sz += pad;
    for (const auto& f : files) { total_sz += f.data.size(); }
    
    std::size_t actual_threads = std::max<std::size_t>(1, std::thread::hardware_concurrency());

    fan::bytes_t raw; raw.reserve(total_sz); std::uint8_t u32_buf[4], u64_buf[8];
    fan::memory::write_le32(u32_buf, std::uint32_t(files.size())); raw.insert(raw.end(), u32_buf, u32_buf + 4);
    for (const auto& f : files) {
      std::uint8_t u16_buf[2]; fan::memory::write_le16(u16_buf, std::uint16_t(f.path.size())); raw.insert(raw.end(), u16_buf, u16_buf + 2);
      auto* p_path = reinterpret_cast<const std::uint8_t*>(f.path.data()); raw.insert(raw.end(), p_path, p_path + f.path.size());
      fan::memory::write_le64(u64_buf, f.data.size()); raw.insert(raw.end(), u64_buf, u64_buf + 8);
    }
    if (pad) { raw.insert(raw.end(), pad, 0); }
    for (const auto& f : files) { raw.insert(raw.end(), f.data.begin(), f.data.end()); }

    bool bcj = params.bcj && files.size() == 1 && file::is_pe(files[0].data);
    if (bcj) { bcj_transform(raw, true); }

    int delta_stride = 1;
    bool use_delta = false;
    if (!bcj) {
      delta_stride = detect_delta_stride(raw);
      use_delta = delta_stride > 1;
      if (use_delta) { delta_encode(raw, delta_stride); }
    }
    int stride_log2 = delta_stride == 8 ? 3 : delta_stride == 4 ? 2 : delta_stride == 2 ? 1 : 0;
    std::uint8_t flags = std::uint8_t(bcj ? 1 : 0) | std::uint8_t(use_delta ? 2 : 0) | std::uint8_t(stride_log2 << 2);

    if (prog) { prog->done.store(0, std::memory_order_relaxed); prog->total.store(raw.size(), std::memory_order_relaxed); }

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

    fan::bytes_t comp = encode_stream_seq(raw, blocks, prog);

    fan::bytes_t out; out.reserve(comp.size() + 17);
    std::uint8_t header[17]; fan::memory::write_le32(header, magic_v4); fan::memory::write_le64(header + 4, raw.size()); fan::memory::write_le32(header + 12, chunk_size); header[16] = flags;
    out.insert(out.end(), header, header + sizeof(header));
    out.insert(out.end(), comp.begin(), comp.end());
    
    if (prog) { prog->done.store(raw.size(), std::memory_order_relaxed); }
    return out;
  }

  std::vector<fan::io::file_buffer_t> decompress(const fan::bytes_t& comp, fan::progress_t* prog = nullptr) {
    std::vector<fan::io::file_buffer_t> files;
    if (comp.size() < 17 || fan::memory::read_le32(comp.data()) != magic_v4) { throw std::runtime_error("needs FCS4"); }

    std::uint64_t total_uncomp = fan::memory::read_le64(comp.data() + 4);
    std::uint8_t flags = comp[16];
    bool bcj = flags & 1;
    bool use_delta = (flags >> 1) & 1;
    int stride_log2 = (flags >> 2) & 7;
    int delta_stride = 1 << stride_log2;
    
    if (prog) { prog->total.store(total_uncomp, std::memory_order_relaxed); }

    fan::bytes_t raw = decode_stream_seq(comp, 17, total_uncomp, prog);

    if (use_delta) { delta_decode(raw, delta_stride); }
    if (bcj) { bcj_transform(raw, false); }
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