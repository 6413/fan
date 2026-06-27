module;

#include <cstdio>

export module fan.compression;

import std;
import fan.types;
import fan.io.file;
import fan.types.fstring;
import fan.print;
namespace file = fan::io::file;

export namespace fan::fcs {
  constexpr std::uint16_t read_le16(const std::uint8_t* p) {
    return std::uint16_t(p[0]) | (std::uint16_t(p[1]) << 8);
  }

  constexpr void write_le16(std::uint8_t* p, std::uint16_t v) {
    p[0] = std::uint8_t(v); p[1] = std::uint8_t(v >> 8);
  }

  constexpr std::uint32_t read_le32(const std::uint8_t* p) {
    return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) | (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
  }

  constexpr void write_le32(std::uint8_t* p, std::uint32_t v) {
    p[0] = std::uint8_t(v); p[1] = std::uint8_t(v >> 8); p[2] = std::uint8_t(v >> 16); p[3] = std::uint8_t(v >> 24);
  }

  constexpr std::uint64_t read_le64(const std::uint8_t* p) {
    return std::uint64_t(read_le32(p)) | (std::uint64_t(read_le32(p + 4)) << 32);
  }

  constexpr void write_le64(std::uint8_t* p, std::uint64_t v) {
    write_le32(p, std::uint32_t(v));
    write_le32(p + 4, std::uint32_t(v >> 32));
  }

  struct archive_file_t {
    std::string path;
    fan::bytes_t data;
  };

  struct progress_t {
    std::atomic<std::size_t> done = 0;
    std::atomic<std::size_t> total = 0;
  };

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
      link(i) = std::exchange(head[hash(d.data() + i)], std::int32_t(i - base));
    }

    constexpr match_t find(const fan::bytes_t& src, std::size_t i, std::size_t c_end, const std::array<std::uint32_t, 4>& rep, std::size_t chain, bool push) {
      match_t best;
      if (i + 4 > c_end) { return best; }
      auto h = hash(src.data() + i);
      std::int32_t cur = push ? std::exchange(head[h], std::int32_t(i - base)) : head[h];
      if (push) { link(i) = cur; }
      if (cur < 0) { return best; }

      std::size_t max_avail = c_end - i, best_len = 0, iters = 0;
      const auto* po = src.data() + i;
      const auto* src_data = src.data();

      while (cur >= 0 && iters++ < chain) {
        std::size_t cur_pos = base + std::size_t(cur);
        if (cur_pos >= i) { break; }
        const auto* pc = src_data + cur_pos;
        if (best_len > 0 && po[best_len] != pc[best_len]) { cur = prev[cur]; continue; }
        std::size_t len = 3 + get_match_len(po + 3, pc + 3, max_avail - 3);
        std::size_t off = i - cur_pos;
        int ob = (off == rep[0]) ? 2 : (off == rep[1] || off == rep[2] || off == rep[3]) ? 4 : 5 + int(pack_sym(off - 1, 32, 31).bits);
        int p = int(len * 8) - ob;
        if (p > best.profit || (p == best.profit && len > best.length)) {
          best = {off, len, p}; best_len = len;
          if (len == max_avail || best_len >= 8192) { break; }
        }
        cur = prev[cur];
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
  inline constexpr std::uint32_t magic_v2 = 0x32534346;

  enum class op_e : std::uint8_t { rep0, rep1, rep2, rep3, exp, none };
  struct seq_t { std::uint32_t lit_len; op_e op; std::uint32_t match_len, offset; };
  struct chunk_payload_t { std::vector<seq_t> seqs; };

  constexpr chunk_payload_t parse_chunk_optimal(const fan::bytes_t& src, std::size_t c_start, std::size_t c_end, match_finder_t& finder, const compress_params_t& params, progress_t* prog = nullptr) {
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
      if (M.profit <= 0 || M.len == 0) {
        ++lit_cnt; ++i;
        if (prog) { prog->done.fetch_add(1, std::memory_order_relaxed); }
        continue;
      }
      if (params.lazy_depth > 0 && i + 4 < c_end && M.len < 128) {
        int skip = 1;
        for (; skip <= params.lazy_depth && i + skip < c_end; ++skip) {
          if (best_cand(i + skip, false, params.chain_lazy).profit > M.profit + (skip * 8)) { break; }
        }
        if (skip <= params.lazy_depth) { lit_cnt += skip; i += skip; if (prog) { prog->done.fetch_add(skip, std::memory_order_relaxed); } continue; }
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
      if (prog) { prog->done.fetch_add(M.len, std::memory_order_relaxed); }
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
    std::uint64_t low = 0, cache_size = 1;
    std::uint32_t range = -1u;
    std::uint8_t cache = 0;
  };

  struct file_sink_t {
    void write_byte(std::uint8_t b) {
      if (std::fputc(b, fp) == EOF) { throw std::runtime_error("fcs write failed"); }
    }

    void write_repeat(std::uint8_t b, std::uint64_t n) {
      std::array<std::uint8_t, 4096> buf;
      buf.fill(b);
      while (n) {
        std::size_t w = std::size_t(std::min<std::uint64_t>(n, buf.size()));
        if (std::fwrite(buf.data(), 1, w, fp) != w) { throw std::runtime_error("fcs write failed"); }
        n -= w;
      }
    }

    void write_bytes(std::span<const std::uint8_t> bytes) {
      if (!bytes.empty() && std::fwrite(bytes.data(), 1, bytes.size(), fp) != bytes.size()) { throw std::runtime_error("fcs write failed"); }
    }

    std::FILE* fp = nullptr;
  };

  struct file_range_enc_t {
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

    file_sink_t& out;
    std::uint64_t low = 0, cache_size = 1;
    std::uint32_t range = -1u;
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

  struct file_range_dec_t {
    file_range_dec_t(std::FILE* fp) : fp(fp) {
      read_byte();
      for (int i = 0; i < 4; ++i) { code = (code << 8) | read_byte(); }
    }
    std::uint8_t read_byte() {
      int c = std::fgetc(fp);
      return c == EOF ? 0 : std::uint8_t(c);
    }

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

    std::FILE* fp = nullptr;
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

  void encode_op(auto& rc, stream_model_t& m, std::uint32_t ll, op_e op, std::size_t pos) {
    int ctx = (std::min<int>(ll, 7) << 2) | (pos & 3);
    rc.encode(m.is_exp[ctx], op == op_e::exp);   if (op == op_e::exp) { return; }
    rc.encode(m.is_rep0[ctx], op == op_e::rep0); if (op == op_e::rep0) { return; }
    rc.encode(m.is_none[ctx], op == op_e::none); if (op == op_e::none) { return; }
    rc.encode_tree(m.rep_idx[ctx], int(op) - int(op_e::rep1), 2);
  }

  op_e decode_op(auto& rd, stream_model_t& m, std::uint32_t ll, std::size_t pos) {
    int ctx = (std::min<int>(ll, 7) << 2) | (pos & 3);
    if (rd.decode(m.is_exp[ctx])) { return op_e::exp; }
    if (rd.decode(m.is_rep0[ctx])) { return op_e::rep0; }
    if (rd.decode(m.is_none[ctx])) { return op_e::none; }
    return static_cast<op_e>(int(op_e::rep1) + rd.decode_tree(m.rep_idx[ctx], 2));
  }

  void encode_val(auto& rc, auto& tree, auto& xtree, hybrid_sym_t h) {
    rc.encode_tree(std::span<std::uint16_t>{tree}, h.cat, 6);
    if (h.bits > 0) {
      int mb = std::min<int>(int(h.bits), 4), rem = h.bits - mb;
      for (int i = rem - 1; i >= 0; --i) { rc.encode_direct((h.extra >> (mb + i)) & 1); }
      rc.encode_tree(std::span<std::uint16_t>{xtree[h.cat]}, h.extra & ((1 << mb) - 1), mb);
    }
  }

  std::uint32_t decode_val(auto& rd, auto& tree, auto& xtree, int t, int base) {
    std::uint32_t c = rd.decode_tree(std::span<std::uint16_t>{tree}, 6);
    if (c < t) { return c + (t == 32); }
    int b = c - t, mb = std::min(b, 4), rem = b - mb; std::uint32_t x = 0;
    for (int i = 0; i < rem; ++i) { x = (x << 1) | rd.decode_direct(); }
    return base + (1u << b) + ((x << mb) | rd.decode_tree(std::span<std::uint16_t>{xtree[c]}, mb));
  }


  std::size_t encode_payload(auto& rc, stream_model_t& model, const fan::bytes_t& src, std::size_t src_ptr, const chunk_payload_t& block) {
    for (const auto& s : block.seqs) {
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
    return src_ptr;
  }

  struct archive_file_info_t {
    std::filesystem::path real_path;
    std::string archive_path;
    std::uint64_t size = 0;
  };

  struct archive_provider_t : fan::io::data_provider_t {
    struct segment_t {
      std::uint64_t start = 0;
      std::uint64_t size = 0;
      std::uint64_t file_offset = 0;
      std::vector<std::uint8_t> bytes;
      std::filesystem::path file_path;
    };

    archive_provider_t(const std::vector<archive_file_info_t>& files) {
      if (files.size() > std::numeric_limits<std::uint32_t>::max()) { throw std::runtime_error("too many files"); }
      std::uint8_t u32_buf[4];
      write_le32(u32_buf, std::uint32_t(files.size()));
      append_bytes(std::span<const std::uint8_t>(u32_buf, 4));
      for (const auto& f : files) {
        if (f.archive_path.size() > std::numeric_limits<std::uint16_t>::max()) { throw std::runtime_error("path too long"); }
        std::vector<std::uint8_t> meta;
        std::uint8_t u16_buf[2], u64_buf[8];
        write_le16(u16_buf, std::uint16_t(f.archive_path.size()));
        meta.insert(meta.end(), u16_buf, u16_buf + 2);
        auto* p_path = reinterpret_cast<const std::uint8_t*>(f.archive_path.data());
        meta.insert(meta.end(), p_path, p_path + f.archive_path.size());
        write_le64(u64_buf, f.size);
        meta.insert(meta.end(), u64_buf, u64_buf + 8);
        append_bytes(std::span<const std::uint8_t>(meta.data(), meta.size()));
        append_file(f.real_path, f.size);
      }
    }

    std::uint64_t size() const override {
      return total_size;
    }

    std::uint8_t read(std::uint64_t offset) const override {
      std::vector<std::uint8_t> b;
      read_range(offset, 1, b);
      return b.empty() ? 0 : b[0];
    }

    void write(std::uint64_t, std::uint8_t) override {
    }

    std::uint64_t read_range(std::uint64_t offset, std::uint64_t length, std::vector<std::uint8_t>& out_buffer) const override {
      if (offset >= total_size) {
        out_buffer.clear();
        return 0;
      }

      std::uint64_t actual = std::min(length, total_size - offset);
      out_buffer.assign(std::size_t(actual), 0);
      std::uint64_t end = offset + actual;

      for (const auto& s : segments) {
        std::uint64_t s_end = s.start + s.size;
        if (s_end <= offset) { continue; }
        if (s.start >= end) { break; }
        std::uint64_t p = std::max(offset, s.start);
        std::uint64_t n = std::min(end, s_end) - p;
        auto* dst = out_buffer.data() + std::size_t(p - offset);
        std::uint64_t local = p - s.start;

        if (!s.bytes.empty()) {
          std::copy_n(s.bytes.data() + std::size_t(local), std::size_t(n), dst);
        } else {
          std::ifstream f(s.file_path, std::ios::binary);
          if (!f) { throw std::runtime_error("read failed: " + s.file_path.string()); }
          f.seekg(std::streamoff(s.file_offset + local));
          f.read(reinterpret_cast<char*>(dst), std::streamsize(n));
          if (std::uint64_t(f.gcount()) != n) { throw std::runtime_error("short read: " + s.file_path.string()); }
        }
      }
      return actual;
    }

    void append_bytes(std::span<const std::uint8_t> bytes) {
      segment_t s;
      s.start = total_size;
      s.size = bytes.size();
      s.bytes.assign(bytes.begin(), bytes.end());
      segments.push_back(std::move(s));
      total_size += bytes.size();
    }

    void append_file(const std::filesystem::path& path, std::uint64_t size) {
      segment_t s;
      s.start = total_size;
      s.size = size;
      s.file_path = path;
      segments.push_back(std::move(s));
      total_size += size;
    }

    std::vector<segment_t> segments;
    std::uint64_t total_size = 0;
  };

  bool is_safe_archive_path(const std::filesystem::path& path) {
    if (path.empty() || path.is_absolute()) { return false; }
    for (const auto& part : path) {
      if (part == "..") { return false; }
    }
    return true;
  }

  std::vector<archive_file_info_t> collect_archive_files(const std::filesystem::path& in, bool verbose = false) {
    std::vector<archive_file_info_t> files;
    std::error_code ec;

    if (std::filesystem::is_directory(in, ec)) {
      for (const auto& entry : std::filesystem::recursive_directory_iterator(in)) {
        if (!entry.is_regular_file()) { continue; }
        auto rel = std::filesystem::relative(entry.path(), in);
        if (verbose) { fan::print("found:", rel.generic_string()); }
        files.push_back({entry.path(), rel.generic_string(), entry.file_size()});
      }
    } else {
      files.push_back({in, in.filename().string(), std::filesystem::file_size(in)});
    }
    return files;
  }

  bool compress_path_to_file(const std::filesystem::path& in, const std::filesystem::path& out_path, compress_params_t params = params_max(), progress_t* prog = nullptr, bool verbose = false) {
    auto files = collect_archive_files(in, verbose);
    archive_provider_t provider(files);
    params.bcj = false;

    if (prog) {
      prog->done.store(0, std::memory_order_relaxed);
      prog->total.store(provider.size(), std::memory_order_relaxed);
    }

    std::FILE* fp = std::fopen(out_path.string().c_str(), "wb");
    if (!fp) { return false; }

    try {
      file_sink_t sink {fp};
      std::uint8_t header[13];
      write_le32(header, magic_v2);
      write_le64(header + 4, provider.size());
      header[12] = 0;
      sink.write_bytes(std::span<const std::uint8_t>(header, sizeof(header)));

      file_range_enc_t rc {sink};
      auto m_ptr = std::make_unique<stream_model_t>(); stream_model_t& model = *m_ptr;
      match_finder_t finder;
      fan::bytes_t buf;

      for (std::uint64_t c_start = 0, total = provider.size(); c_start < total; c_start += chunk_size) {
        std::uint64_t d_start = c_start >= chunk_size ? c_start - chunk_size : 0;
        std::uint64_t c_end = std::min<std::uint64_t>(total, c_start + chunk_size);
        provider.read_range(d_start, c_end - d_start, buf);
        auto block = parse_chunk_optimal(buf, std::size_t(c_start - d_start), buf.size(), finder, params, prog);
        encode_payload(rc, model, buf, std::size_t(c_start - d_start), block);
      }
      rc.flush();
      std::fclose(fp);
      return true;
    }
    catch (...) {
      std::fclose(fp);
      throw;
    }
  }

  struct archive_extract_writer_t {
    enum class state_e { file_count, path_len, path, size, data, done };

    archive_extract_writer_t(std::filesystem::path out_dir) : out_dir(std::move(out_dir)) {
      write_buffer.reserve(1 << 20);
    }

    ~archive_extract_writer_t() {
      close_file();
    }

    void put(std::uint8_t b) {
      if (state == state_e::data) {
        write_data(b);
        return;
      }
      tmp.push_back(b);
      if (tmp.size() != need) { return; }

      if (state == state_e::file_count) {
        file_count = read_le32(tmp.data());
        file_index = 0;
        set_state(file_count ? state_e::path_len : state_e::done, file_count ? 2 : 0);
      } else if (state == state_e::path_len) {
        path_len = read_le16(tmp.data());
        set_state(state_e::path, path_len);
      } else if (state == state_e::path) {
        archive_path.assign(reinterpret_cast<const char*>(tmp.data()), tmp.size());
        set_state(state_e::size, 8);
      } else if (state == state_e::size) {
        remaining = read_le64(tmp.data());
        open_file();
        if (remaining == 0) { finish_file(); }
        else { set_state(state_e::data, 0); }
      }
    }

    void finish() {
      flush();
      close_file();
      if (state != state_e::done) { throw std::runtime_error("truncated archive"); }
    }

    void set_state(state_e s, std::size_t n) {
      state = s;
      need = n;
      tmp.clear();
    }

    void open_file() {
      std::filesystem::path rel = archive_path;
      if (!is_safe_archive_path(rel)) { throw std::runtime_error("unsafe archive path: " + archive_path); }
      std::filesystem::path p = out_dir / rel;
      std::filesystem::create_directories(p.parent_path());
      fp = std::fopen(p.string().c_str(), "wb");
      if (!fp) { throw std::runtime_error("write failed: " + p.string()); }
    }

    void write_data(std::uint8_t b) {
      write_buffer.push_back(b);
      if (write_buffer.size() == write_buffer.capacity()) { flush(); }
      if (--remaining == 0) { finish_file(); }
    }

    void finish_file() {
      flush();
      close_file();
      ++file_index;
      set_state(file_index == file_count ? state_e::done : state_e::path_len, file_index == file_count ? 0 : 2);
    }

    void flush() {
      if (!fp || write_buffer.empty()) { return; }
      if (std::fwrite(write_buffer.data(), 1, write_buffer.size(), fp) != write_buffer.size()) { throw std::runtime_error("write failed"); }
      write_buffer.clear();
    }

    void close_file() {
      if (fp) {
        std::fclose(fp);
        fp = nullptr;
      }
    }

    std::filesystem::path out_dir;
    state_e state = state_e::file_count;
    std::vector<std::uint8_t> tmp, write_buffer;
    std::string archive_path;
    std::FILE* fp = nullptr;
    std::size_t need = 4;
    std::uint64_t remaining = 0;
    std::uint32_t file_count = 0, file_index = 0;
    std::uint16_t path_len = 0;
  };

  bool decompress_file_to_dir(const std::filesystem::path& in_path, const std::filesystem::path& out_dir, progress_t* prog = nullptr) {
    std::FILE* fp = std::fopen(in_path.string().c_str(), "rb");
    if (!fp) { return false; }

    auto read_file_byte = [&]() -> std::uint8_t {
      int c = std::fgetc(fp);
      if (c == EOF) { throw std::runtime_error("unexpected eof"); }
      return std::uint8_t(c);
    };

    try {
      std::uint8_t header[12];
      if (std::fread(header, 1, 4, fp) != 4) { throw std::runtime_error("invalid archive"); }
      if (read_le32(header) != magic_v2) { throw std::runtime_error("streaming decompress needs FCS2 archive"); }
      if (std::fread(header + 4, 1, 8, fp) != 8) { throw std::runtime_error("invalid archive"); }
      std::uint64_t total_uncomp = read_le64(header + 4);
      bool bcj = read_file_byte();
      if (bcj) { throw std::runtime_error("streaming bcj decode unsupported"); }

      if (prog) {
        prog->done.store(0, std::memory_order_relaxed);
        prog->total.store(total_uncomp, std::memory_order_relaxed);
      }

      file_range_dec_t rd {fp};
      auto m_ptr = std::make_unique<stream_model_t>(); stream_model_t& model = *m_ptr;
      archive_extract_writer_t writer(out_dir);
      std::vector<std::uint8_t> window(chunk_size);
      std::array<std::uint32_t, 4> rep {1, 1, 1, 1};
      std::uint64_t pos = 0, next_boundary = chunk_size, last_rep = 0;
      std::uint64_t window_mask = chunk_size - 1;

      auto get_back = [&](std::uint32_t off) -> std::uint8_t {
        if (off == 0 || off > pos || off > chunk_size) { throw std::runtime_error("invalid match offset"); }
        return window[(pos - off) & window_mask];
      };
      auto emit = [&](std::uint8_t b) {
        writer.put(b);
        window[pos & window_mask] = b;
        ++pos;
        if (prog && (pos - last_rep >= 65536 || pos >= total_uncomp)) {
          prog->done.store(pos, std::memory_order_relaxed);
          last_rep = pos;
        }
      };

      while (pos < total_uncomp) {
        if (pos == next_boundary) { rep = {1, 1, 1, 1}; next_boundary += chunk_size; }
        std::uint32_t ll = decode_val(rd, model.lit_len[pos & 3], model.lit_len_extra, 24, 23);
        for (std::uint32_t j = 0; j < ll && pos < total_uncomp; ++j) {
          std::uint32_t ctx = (pos >= 1 ? get_back(1) : 0) | ((pos >= 2 ? get_back(2) >> 4 : 0) << 8);
          emit(std::uint8_t(rd.decode_tree(model.lit[ctx], 8)));
        }
        if (pos >= total_uncomp) { break; }

        op_e op = decode_op(rd, model, ll, pos);
        if (op == op_e::none) { continue; }

        std::uint32_t mlen = 0, off = 0;
        if (op == op_e::rep0) {
          mlen = 1 + decode_val(rd, model.match_len[pos & 3][0], model.match_len_extra, 24, 23);
          off = rep[0];
        } else {
          int op_i = op == op_e::exp ? 4 : int(op);
          mlen = 2 + decode_val(rd, model.match_len[pos & 3][op_i], model.match_len_extra, 24, 23);
          int r = op == op_e::exp ? 3 : int(op);
          off = op == op_e::exp ? decode_val(rd, model.off[std::min<std::uint32_t>(mlen - 2, num_dist_ctx - 1)], model.off_extra, 32, 32) : rep[r];
          for (int k = r; k > 0; --k) { rep[k] = rep[k - 1]; }
          rep[0] = off;
        }
        for (std::uint32_t j = 0; j < mlen && pos < total_uncomp; ++j) { emit(get_back(off)); }
      }
      writer.finish();
      std::fclose(fp);
      return true;
    }
    catch (...) {
      std::fclose(fp);
      throw;
    }
  }

  fan::bytes_t compress(const std::vector<archive_file_t>& files, compress_params_t params = params_max(), progress_t* prog = nullptr) {
    if (files.size() > std::numeric_limits<std::uint32_t>::max()) { throw std::runtime_error("too many files"); }

    std::size_t total_sz = 4;
    for (const auto& f : files) {
      if (f.path.size() > std::numeric_limits<std::uint16_t>::max()) { throw std::runtime_error("path too long"); }
      total_sz += 2 + f.path.size() + 8 + f.data.size();
    }

    if (prog) { prog->total.store(total_sz, std::memory_order_relaxed); }

    fan::bytes_t src; src.reserve(total_sz);
    std::uint8_t u32_buf[4], u64_buf[8];
    write_le32(u32_buf, std::uint32_t(files.size()));
    src.insert(src.end(), u32_buf, u32_buf + 4);
    for (const auto& f : files) {
      std::uint8_t u16_buf[2];
      write_le16(u16_buf, std::uint16_t(f.path.size()));
      src.insert(src.end(), u16_buf, u16_buf + 2);
      auto* p_path = reinterpret_cast<const std::uint8_t*>(f.path.data());
      src.insert(src.end(), p_path, p_path + f.path.size());
      write_le64(u64_buf, f.data.size());
      src.insert(src.end(), u64_buf, u64_buf + 8);
      src.insert(src.end(), f.data.begin(), f.data.end());
      if (prog) { prog->done.fetch_add(2 + f.path.size() + 8 + f.data.size(), std::memory_order_relaxed); }
    }

    bool bcj = params.bcj && files.size() == 1 && file::is_pe(files[0].data);
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
          std::size_t cs = k * chunk_size, ce = std::min(src.size(), (k + 1) * chunk_size);
          blocks[k] = parse_chunk_optimal(src, cs, ce, finder, params);
          if (prog) { prog->done.fetch_add(ce - cs, std::memory_order_relaxed); }
        }
      });
    }
    workers.clear();

    fan::bytes_t out; out.reserve((src.size() / 5) + 16);
    std::uint8_t header[13];
    write_le32(header, magic_v2);
    write_le64(header + 4, src.size());
    header[12] = bcj;
    out.insert(out.end(), header, header + sizeof(header));

    range_enc_t rc {out};
    auto m_ptr = std::make_unique<stream_model_t>(); stream_model_t& model = *m_ptr;

    std::size_t src_ptr = 0;
    for (std::size_t k = 0; k < nc; ++k) {
      src_ptr = encode_payload(rc, model, src, src_ptr, blocks[k]);
    }
    rc.flush();
    return out;
  }

  std::vector<archive_file_t> decompress(const fan::bytes_t& comp, progress_t* prog = nullptr) {
    std::vector<archive_file_t> files;
    if (comp.size() < 5) { return files; }

    std::size_t idx = 0;
    std::uint32_t magic = read_le32(comp.data());
    bool v2 = magic == magic_v2;
    std::uint64_t total_uncomp64 = 0;

    if (v2) {
      if (comp.size() < 13) { return files; }
      idx = 4;
      total_uncomp64 = read_le64(comp.data() + idx); idx += 8;
    } else {
      total_uncomp64 = magic;
      idx = 4;
    }

    if (total_uncomp64 > std::numeric_limits<std::size_t>::max()) { throw std::runtime_error("archive too large for memory decompress"); }
    std::size_t total_uncomp = std::size_t(total_uncomp64);
    bool bcj = comp[idx++];

    if (prog) { prog->total.store(total_uncomp, std::memory_order_relaxed); }

    fan::bytes_t out; out.reserve(total_uncomp);
    range_dec_t rd {comp, idx};
    auto m_ptr = std::make_unique<stream_model_t>(); stream_model_t& model = *m_ptr;

    std::array<std::uint32_t, 4> rep {1, 1, 1, 1};
    std::size_t next_boundary = chunk_size, last_rep = 0;

    while (out.size() < total_uncomp) {
      if (out.size() == next_boundary) { rep = {1, 1, 1, 1}; next_boundary += chunk_size; }
      std::uint32_t ll = decode_val(rd, model.lit_len[out.size() & 3], model.lit_len_extra, 24, 23);
      for (std::uint32_t j = 0; j < ll && out.size() < total_uncomp; ++j) {
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

      std::size_t old = out.size(); out.resize(std::min<std::size_t>(old + mlen, total_uncomp));
      std::uint8_t* p = out.data() + old;
      const std::uint8_t* s = p - off;
      mlen = std::uint32_t(out.size() - old);

      if (off == 1) { std::memset(p, *s, mlen); }
      else if (off >= mlen) { std::memcpy(p, s, mlen); }
      else {
        std::size_t j = 0;
        for (; j + off <= mlen; j += off) { std::memcpy(p + j, s + j, off); }
        for (; j < mlen; ++j) { p[j] = s[j]; }
      }

      if (prog && (out.size() - last_rep >= 65536 || out.size() >= total_uncomp)) {
        prog->done.store(out.size(), std::memory_order_relaxed);
        last_rep = out.size();
      }
    }

    if (bcj) { fan::fcs::bcj_transform(out, false); }

    if (out.size() < 4) { return files; }
    std::size_t out_idx = 0;
    std::uint32_t num_files = read_le32(out.data() + out_idx); out_idx += 4;
    for (std::uint32_t i = 0; i < num_files; ++i) {
      if (out_idx + 2 > out.size()) { break; }
      std::uint16_t path_len = read_le16(out.data() + out_idx); out_idx += 2;
      if (out_idx + path_len > out.size()) { break; }
      std::string path(reinterpret_cast<const char*>(out.data() + out_idx), path_len); out_idx += path_len;

      std::uint64_t size = 0;
      if (v2) {
        if (out_idx + 8 > out.size()) { break; }
        size = read_le64(out.data() + out_idx); out_idx += 8;
      } else {
        if (out_idx + 4 > out.size()) { break; }
        size = read_le32(out.data() + out_idx); out_idx += 4;
      }
      if (size > out.size() - out_idx) { break; }
      fan::bytes_t data(out.begin() + out_idx, out.begin() + out_idx + std::size_t(size)); out_idx += std::size_t(size);
      files.push_back({std::move(path), std::move(data)});
    }

    return files;
  }

}