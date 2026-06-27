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
      auto* p = d.data() + i; auto b = p[0];
      int s = (b == 0xe8 || b == 0xe9) ? 1 :
        (i + 6 <= n && ((b == 0x0f && (p[1] & 0xf0) == 0x80) || (b == 0xff && (p[1] == 0x15 || p[1] == 0x25)))) ? 2 :
        (i + 7 <= n && b == 0x48 && p[1] == 0x8d && (p[2] & 0xc7) == 0x05) ? 3 : 0;
      if (s) {
        fan::memory::write_le32(p + s, fan::memory::read_le32(p + s) + (enc ? static_cast<std::uint32_t>(i) : -static_cast<std::uint32_t>(i)));
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

  constexpr hybrid_sym_t pack_sym(std::uint32_t v, std::uint32_t t, std::uint32_t base) {
    if (v < t) { return {v, 0, 0}; }
    std::uint32_t r = v - base, k = std::bit_width(r) - 1;
    return {t + k, r - (1u << k), k};
  }

  struct match_t { std::size_t offset = 0, length = 0; int profit = 0; };

  struct compress_params_t {
    std::size_t chain_main = 8192, chain_lazy = 256;
    int lazy_depth = 3;
    bool bcj = true;
    std::size_t block_size = 1 << 23;
    std::size_t dict_size = 1 << 24;
  };

  constexpr compress_params_t params_fast()   { return {32,   16, 0, true, 1 << 21, 1 << 22}; }
  constexpr compress_params_t params_normal() { return {256,  64, 1, true, 1 << 22, 1 << 23}; }
  constexpr compress_params_t params_max()    { return {8192, 256, 6, true, 1 << 23, 1 << 24}; }

  inline constexpr std::size_t min_parallel_block_size = 1 << 16;
  inline constexpr std::size_t blocks_per_thread = 128;
  inline constexpr std::size_t parallel_dict_blocks = 4;
  inline constexpr std::size_t min_hash_size = 1 << 16;

  constexpr compress_params_t threaded_params(compress_params_t params, std::uint64_t total, std::size_t threads) {
    if (threads <= 1 || total == 0) { return params; }
    std::size_t blocks = threads * blocks_per_thread;
    std::size_t target = std::size_t((total + blocks - 1) / blocks);
    params.block_size = std::min(params.block_size, std::max(min_parallel_block_size, target));
    params.dict_size = std::min(params.dict_size, params.block_size * parallel_dict_blocks);
    return params;
  }

  inline void add_progress(fan::progress_t* prog, std::uint64_t v) {
    if (prog && v) { prog->done.fetch_add(v, std::memory_order_relaxed); }
  }

  struct progress_scope_t {
    void set(std::uint64_t v) {
      v = std::min(v, total);
      if (prog && v > value) {
        add_progress(prog, v - value);
        value = v;
      }
    }
    void set_scaled(std::uint64_t begin, std::uint64_t end, std::uint64_t pos, std::uint64_t size) {
      if (size == 0) { set(end); return; }
      pos = std::min(pos, size);
      set(begin + (end - begin) * pos / size);
    }
    void finish() { set(total); }
    fan::progress_t* prog = nullptr;
    std::uint64_t total = 0, value = 0;
  };

  struct match_finder_t {
    match_finder_t(std::size_t dict_size = 1 << 24) {
      std::size_t hash_size = std::bit_ceil(std::max(min_hash_size, dict_size));
      head.assign(hash_size, -1);
      hash_mask = hash_size - 1;
    }
    void reset(std::size_t d_start, std::size_t c_end) {
      std::fill(head.begin(), head.end(), -1);
      base = d_start;
      if (prev.size() < c_end - base) { prev.resize(c_end - base); }
    }
    std::uint32_t hash(const std::uint8_t* p) { return ((fan::memory::read_le32(p) & 0xffffffu) * 0x1e35a7bdu) & std::uint32_t(hash_mask); }
    std::int32_t& link(std::size_t p) { return prev[p - base]; }
    void warm_up(const fan::bytes_t& d, std::size_t ws, std::size_t we, progress_scope_t* progress = nullptr) {
      if (we <= ws) { return; }
      std::size_t progress_pos = ws;
      auto update_progress = [&](std::size_t pos, bool force = false) {
        if (!progress) { return; }
        constexpr std::size_t step = 1 << 12;
        pos = std::min(pos, we);
        std::size_t rounded = force ? pos : pos & ~(step - 1);
        if (rounded > progress_pos) {
          progress->set_scaled(0, progress->total / 50, rounded - ws, we - ws);
          progress_pos = rounded;
        }
      };
      for (std::size_t i = ws; i + 2 < we; ++i) {
        if ((i & 0xfff) == 0) { update_progress(i); }
        insert(d, i);
      }
      update_progress(we, true);
    }
    void insert(const fan::bytes_t& d, std::size_t i) {
      if (i + 4 > d.size() || (i >= base + 2 && i + 1 < d.size() && d[i] == d[i-1] && d[i] == d[i-2] && d[i] == d[i+1])) { return; }
      link(i) = std::exchange(head[hash(d.data() + i)], std::int32_t(i - base));
    }
    match_t find(const fan::bytes_t& src, std::size_t i, std::size_t c_end, const std::array<std::uint32_t, 4>& rep, std::size_t chain, bool push) {
      match_t best;
      if (i + 4 > c_end) { return best; }
      std::uint32_t po32 = fan::memory::read_le32(src.data() + i) & 0xFFFFFFu;
      std::int32_t cur = push ? std::exchange(head[hash(src.data() + i)], std::int32_t(i - base)) : head[hash(src.data() + i)];
      if (push) { link(i) = cur; }
      std::size_t max_avail = c_end - i, best_len = 0, iters = 0;
      const auto* po = src.data() + i; const auto* src_data = src.data();
      while (cur >= 0 && iters++ < chain) {
        std::size_t cur_pos = base + std::size_t(cur);
        if (cur_pos >= i) { break; }
        const auto* pc = src_data + cur_pos;
        if (best_len > 0 && po[best_len] != pc[best_len]) { cur = prev[cur]; continue; }
        if ((fan::memory::read_le32(pc) & 0xFFFFFFu) == po32) {
          std::size_t len = 3 + get_match_len(po + 3, pc + 3, max_avail - 3);
          std::size_t off = i - cur_pos;
          int ob = (off == rep[0]) ? 2 : (off == rep[1] || off == rep[2] || off == rep[3]) ? 4 : 5 + int(pack_sym(off - 1, 32, 31).bits);
          int p = int(len * 8) - ob;
          if (p > best.profit || (p == best.profit && len > best.length)) {
            best = {off, len, p}; best_len = len;
            if (len == max_avail || best_len >= 8192) { break; }
          }
        }
        cur = prev[cur];
      }
      return best;
    }
    std::vector<std::int32_t> head, prev;
    std::size_t base = 0, hash_mask = (1 << 24) - 1;
  };

  inline constexpr int num_lit_ctx = 4096;
  inline constexpr int num_dist_ctx = 16;

  constexpr std::uint32_t lit_ctx(std::uint8_t prev0, std::uint8_t prev1) {
    return std::uint32_t(prev0) | ((std::uint32_t(prev1) & 0xf) << 8);
  }
  inline constexpr std::uint32_t magic_v4 = 0x34334346;

  inline constexpr std::uint64_t compression_progress_scale = 100;
  inline constexpr std::uint64_t compression_encode_progress = 95;
  inline constexpr std::uint64_t compression_write_progress = 5;

  constexpr std::uint64_t compression_progress_total(std::uint64_t total, const compress_params_t&) {
    if (total > std::numeric_limits<std::uint64_t>::max() / compression_progress_scale) { return total; }
    return std::max<std::uint64_t>(total * compression_progress_scale, 1);
  }

  enum class op_e : std::uint8_t { rep0, rep1, rep2, rep3, exp, none };
  struct seq_t { std::uint32_t lit_len; op_e op; std::uint32_t match_len, offset; };
  struct chunk_payload_t { std::vector<seq_t> seqs; };

  chunk_payload_t parse_chunk_optimal(const fan::bytes_t& src, std::size_t d_start, std::size_t c_start, std::size_t c_end, match_finder_t& finder, const compress_params_t& params, progress_scope_t* progress = nullptr) {
    std::size_t block_len = c_end - c_start;
    finder.reset(d_start, c_end); finder.warm_up(src, d_start, c_start, progress);

    struct dp_node_t {
      int cost = std::numeric_limits<int>::max();
      op_e op = op_e::none;
      std::uint32_t len = 0, off = 0;
    };
    std::vector<dp_node_t> dp(block_len + 1);
    dp[0].cost = 0;

    struct rep_state_t { std::array<std::uint32_t, 4> r {1, 1, 1, 1}; };
    std::vector<rep_state_t> rep_at(block_len + 1);

    auto match_cost = [](op_e op, std::size_t len, std::uint32_t off) -> int {
      int lc = (op == op_e::rep0)
        ? 2 + int(pack_sym(std::uint32_t(len - 1), 24, 23).bits)
        : 2 + int(pack_sym(std::uint32_t(len - 2), 24, 23).bits);
      int dc = (op == op_e::exp) ? 5 + int(pack_sym(off - 1, 32, 31).bits) :
               (op == op_e::rep0) ? 1 : 4;
      return lc + dc;
    };

    auto advance_rep = [](std::array<std::uint32_t, 4> rep, op_e op, std::uint32_t off) {
      if (op == op_e::rep0 || op == op_e::none) { return rep; }
      int r = (op == op_e::exp) ? 3 : int(op);
      std::uint32_t v = (op == op_e::exp) ? off : rep[r];
      for (int k = r; k > 0; --k) { rep[k] = rep[k-1]; }
      rep[0] = v;
      return rep;
    };

    std::uint64_t parse_start = progress ? progress->value : 0;
    std::uint64_t parse_end = progress ? progress->total * 60 / 100 : 0;
    auto update_progress = [&](std::size_t pos, bool force = false) {
      if (!progress) { return; }
      constexpr std::size_t step = 1 << 12;
      pos = std::min(pos, block_len);
      std::size_t rounded = force ? pos : pos & ~(step - 1);
      progress->set_scaled(parse_start, parse_end, rounded, block_len);
    };

    constexpr std::size_t nice_len = 256;

    for (std::size_t i = 0; i < block_len; ++i) {
      if ((i & 0xfff) == 0) { update_progress(i); }
      if (dp[i].cost == std::numeric_limits<int>::max()) { continue; }
      std::size_t abs = c_start + i;
      const auto& rep = rep_at[i].r;
      std::size_t avail = c_end - abs;

      // literal
      int lit_cost = dp[i].cost + 8;
      if (lit_cost < dp[i+1].cost) {
        dp[i+1].cost = lit_cost; dp[i+1].op = op_e::none; dp[i+1].len = 1; dp[i+1].off = 0;
        rep_at[i+1].r = rep;
      }

      // rep matches
      for (int r = 0; r < 4; ++r) {
        if (abs < d_start + rep[r]) { continue; }
        std::size_t l = get_match_len(src.data() + abs, src.data() + abs - rep[r], avail);
        std::size_t min_l = (r == 0) ? 1 : 2;
        if (l < min_l) { continue; }
        op_e op = static_cast<op_e>(r);
        
        std::size_t max_ml = std::min(l, nice_len);
        for (std::size_t ml = min_l; ml <= max_ml; ++ml) {
          int c = dp[i].cost + match_cost(op, ml, rep[r]);
          if (c < dp[i+ml].cost) {
            dp[i+ml].cost = c; dp[i+ml].op = op; dp[i+ml].len = std::uint32_t(ml); dp[i+ml].off = rep[r];
            rep_at[i+ml].r = advance_rep(rep, op, rep[r]);
          }
        }
        if (l > nice_len) {
          int c = dp[i].cost + match_cost(op, l, rep[r]);
          if (c < dp[i+l].cost) {
            dp[i+l].cost = c; dp[i+l].op = op; dp[i+l].len = std::uint32_t(l); dp[i+l].off = rep[r];
            rep_at[i+l].r = advance_rep(rep, op, rep[r]);
          }
        }
      }

      // explicit match
      if (avail >= 2) {
        match_t m = finder.find(src, abs, c_end, rep, params.chain_main, true);
        if (m.length >= 2) {
          std::size_t max_ml = std::min(m.length, nice_len);
          for (std::size_t ml = 2; ml <= max_ml; ++ml) {
            int c = dp[i].cost + match_cost(op_e::exp, ml, std::uint32_t(m.offset));
            if (c < dp[i+ml].cost) {
              dp[i+ml].cost = c; dp[i+ml].op = op_e::exp;
              dp[i+ml].len = std::uint32_t(ml); dp[i+ml].off = std::uint32_t(m.offset);
              rep_at[i+ml].r = advance_rep(rep, op_e::exp, std::uint32_t(m.offset));
            }
          }
          if (m.length > nice_len) {
            int c = dp[i].cost + match_cost(op_e::exp, m.length, std::uint32_t(m.offset));
            if (c < dp[i+m.length].cost) {
              dp[i+m.length].cost = c; dp[i+m.length].op = op_e::exp;
              dp[i+m.length].len = std::uint32_t(m.length); dp[i+m.length].off = std::uint32_t(m.offset);
              rep_at[i+m.length].r = advance_rep(rep, op_e::exp, std::uint32_t(m.offset));
            }
          }
        }
      } else { finder.insert(src, abs); }
    }

    // Backward traceback
    chunk_payload_t out; out.seqs.reserve(block_len / 5);
    std::vector<seq_t> rev;
    std::size_t i = block_len;
    while (i > 0) {
      const auto& nd = dp[i];
      if (nd.op == op_e::none) {
        std::uint32_t lc = 0;
        while (i > 0 && dp[i].op == op_e::none && dp[i].len == 1) { ++lc; --i; }
        rev.push_back({lc, op_e::none, 0, 0});
      } else {
        rev.push_back({0, nd.op, nd.len, nd.off});
        i -= nd.len;
      }
    }

    std::reverse(rev.begin(), rev.end());
    std::uint32_t pending_lits = 0;
    for (auto& s : rev) {
      if (s.op == op_e::none) { pending_lits += s.lit_len; }
      else { s.lit_len = pending_lits; pending_lits = 0; out.seqs.push_back(s); }
    }
    if (pending_lits) { out.seqs.push_back({pending_lits, op_e::none, 0, 0}); }

    update_progress(block_len, true);
    return out;
  }

  template <typename Writer>
  struct range_enc_t {
    constexpr void shift_low() {
      if (std::uint32_t(low) < 0xFF000000 || int(low >> 32) != 0) {
        out.write_byte(cache + std::uint8_t(low >> 32));
        out.write_repeat(std::uint8_t(low >> 32) ? 0x00 : 0xFF, cache_size - 1);
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
    Writer& out;
    std::uint64_t low = 0, cache_size = 1;
    std::uint32_t range = -1u;
    std::uint8_t cache = 0;
  };

  template <typename Reader>
  struct range_dec_t {
    constexpr range_dec_t(Reader& r) : in(r) {
      read_byte();
      for (int i = 0; i < 4; ++i) { code = (code << 8) | read_byte(); }
    }
    constexpr std::uint8_t read_byte() { return in.read_byte(); }
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
    Reader& in;
    std::uint32_t range = -1u, code = 0;
  };

  struct stream_model_t {
    constexpr stream_model_t() { reset(); }
    constexpr void reset() {
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

  std::size_t encode_payload(auto& rc, stream_model_t& model, const fan::bytes_t& src, std::size_t src_ptr, std::size_t src_end, const chunk_payload_t& block, progress_scope_t* progress = nullptr) {
    std::size_t src_begin = src_ptr;
    std::uint64_t encode_start = progress ? progress->value : 0;
    auto update_progress = [&] {
      if (progress) { progress->set_scaled(encode_start, progress->total, src_ptr - src_begin, src_end - src_begin); }
    };
    for (const auto& s : block.seqs) {
      encode_val(rc, model.lit_len[src_ptr & 3], model.lit_len_extra, pack_sym(s.lit_len, 24, 23));
      for (std::uint32_t j = 0; j < s.lit_len; ++j) {
        std::uint32_t ctx = lit_ctx(src_ptr >= 1 ? src[src_ptr - 1] : 0, src_ptr >= 2 ? src[src_ptr - 2] : 0);
        rc.encode_tree(model.lit[ctx], src[src_ptr++], 8);
        if ((src_ptr & 0xfff) == 0) { update_progress(); }
      }
      encode_op(rc, model, s.lit_len, s.op, src_ptr);
      if (s.op == op_e::none) { update_progress(); continue; }

      if (s.op == op_e::rep0) {
        encode_val(rc, model.match_len[src_ptr & 3][0], model.match_len_extra, pack_sym(s.match_len - 1, 24, 23));
      } else {
        int op_i = s.op == op_e::exp ? 4 : int(s.op);
        encode_val(rc, model.match_len[src_ptr & 3][op_i], model.match_len_extra, pack_sym(s.match_len - 2, 24, 23));
        if (s.op == op_e::exp) { encode_val(rc, model.off[std::min<std::uint32_t>(s.match_len - 2, num_dist_ctx - 1)], model.off_extra, pack_sym(s.offset - 1, 32, 31)); }
      }
      src_ptr += s.match_len;
      update_progress();
    }
    update_progress();
    return src_ptr;
  }

  template <typename Writer>
  void encode_stream(const fan::io::data_provider_t& provider, Writer& sink, compress_params_t params, std::size_t thread_count, fan::progress_t* prog) {
    std::size_t actual_threads = std::max<std::size_t>(1, thread_count ? thread_count : std::thread::hardware_concurrency());
    
    std::uint64_t total = provider.size();
    if (total == 0) { return; }
    
    std::size_t target_chunk_size = std::max<std::size_t>(1uz << 26, params.block_size * actual_threads * 2);
    std::size_t num_chunks = std::max<std::size_t>(1, (total + target_chunk_size - 1) / target_chunk_size);
    std::size_t chunk_size = total / num_chunks;
    chunk_size = std::max<std::size_t>(params.block_size, ((chunk_size + params.block_size - 1) / params.block_size) * params.block_size);

    fan::bytes_t buf;

    for (std::uint64_t c_start = 0; c_start < total; c_start += chunk_size) {
      std::uint64_t d_start = c_start >= params.dict_size ? c_start - params.dict_size : 0;
      std::uint64_t c_end = std::min<std::uint64_t>(total, c_start + chunk_size);
      provider.read_range(d_start, c_end - d_start, buf);

      std::size_t chunk_len = std::size_t(c_end - c_start);
      std::size_t num_blocks = (chunk_len + params.block_size - 1) / params.block_size;
      std::vector<fan::bytes_t> block_outs(num_blocks);
      std::vector<std::uint8_t> block_ready(num_blocks, 0);
      std::mutex write_mutex;
      std::condition_variable write_cv;

      std::size_t next_job = 0;
      std::size_t writer_idx = 0;
      std::size_t nw = std::min<std::size_t>(num_blocks, actual_threads);
      std::vector<std::jthread> parse_workers;

      std::jthread writer([&] {
        std::uint8_t sz_buf[4];
        for (std::size_t i = 0; i < num_blocks; ++i) {
          fan::bytes_t b_out;
          {
            std::unique_lock lock(write_mutex);
            write_cv.wait(lock, [&] { return block_ready[i] != 0; });
            b_out = std::move(block_outs[i]);
          }
          fan::memory::write_le32(sz_buf, std::uint32_t(b_out.size()));
          sink.write_bytes(std::span<const std::uint8_t>(sz_buf, 4));
          sink.write_bytes(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(b_out.data()), b_out.size()));
          std::size_t b_start = i * params.block_size;
          std::size_t b_end = std::min(chunk_len, b_start + params.block_size);
          add_progress(prog, std::uint64_t(b_end - b_start) * compression_write_progress);

          {
            std::lock_guard lock(write_mutex);
            writer_idx = i + 1;
          }
          write_cv.notify_all();
        }
      });

      for (std::size_t w = 0; w < nw; ++w) {
        parse_workers.emplace_back([&] {
          match_finder_t finder(params.dict_size);
          auto m_ptr = std::make_unique<stream_model_t>();
          for (;;) {
            std::size_t i;
            {
              std::unique_lock lock(write_mutex);
              write_cv.wait(lock, [&] { return next_job >= num_blocks || (next_job - writer_idx < nw * 2); });
              if (next_job >= num_blocks) { break; }
              i = next_job++;
            }
            std::size_t b_start = i * params.block_size;
            std::size_t b_end = std::min(chunk_len, b_start + params.block_size);
            std::size_t local_c_start = std::size_t(c_start - d_start);
            std::size_t local_b_start = local_c_start + b_start;
            std::size_t local_b_end = local_c_start + b_end;
            std::size_t local_d_start = local_b_start >= params.dict_size ? local_b_start - params.dict_size : 0;

            progress_scope_t block_progress {prog, std::uint64_t(b_end - b_start) * compression_encode_progress};
            auto payload = parse_chunk_optimal(buf, local_d_start, local_b_start, local_b_end, finder, params, &block_progress);

            fan::bytes_t b_out; b_out.reserve((b_end - b_start) / 4);
            fan::io::bytes_writer_t bw {b_out}; range_enc_t<fan::io::bytes_writer_t> rc {bw};
            m_ptr->reset();
            encode_payload(rc, *m_ptr, buf, local_b_start, local_b_end, payload, &block_progress);
            rc.flush();

            {
              std::lock_guard lock(write_mutex);
              block_outs[i] = std::move(b_out);
              block_ready[i] = 1;
            }
            block_progress.finish();
            write_cv.notify_all();
          }
        });
      }
      parse_workers.clear();
      writer.join();
    }
  }

  template <typename Reader, typename Emitter>
  void decode_stream(Reader& source, Emitter& emit, std::uint64_t total_uncomp, std::uint32_t block_size, fan::progress_t* prog) {
    fan::bytes_t window(1 << 25, 0);
    std::uint64_t pos = 0, last_rep = 0, window_mask = window.size() - 1;

    auto get_back = [&](std::uint32_t off) -> std::uint8_t {
      if (off == 0 || off > pos || off > window.size()) { throw std::runtime_error("invalid match"); }
      return window[(pos - off) & window_mask];
    };

    std::uint8_t sz_buf[4];
    while (pos < total_uncomp) {
      source.read_exact(std::span<std::uint8_t>(sz_buf, 4));
      fan::bytes_t b_comp(fan::memory::read_le32(sz_buf), 0);
      source.read_exact(b_comp);

      std::uint64_t b_end = std::min<std::uint64_t>(total_uncomp, pos + block_size);
      fan::io::bytes_reader_t br {b_comp, 0}; range_dec_t<fan::io::bytes_reader_t> rd {br};
      
      auto m_ptr = std::make_unique<stream_model_t>();
      auto& model = *m_ptr;
      std::array<std::uint32_t, 4> rep {1, 1, 1, 1};

      while (pos < b_end) {
        std::uint32_t ll = decode_val(rd, model.lit_len[pos & 3], model.lit_len_extra, 24, 23);
        for (std::uint32_t j = 0; j < ll && pos < b_end; ++j) {
          std::uint32_t ctx = lit_ctx(pos >= 1 ? get_back(1) : 0, pos >= 2 ? get_back(2) : 0);
          std::uint8_t b = std::uint8_t(rd.decode_tree(model.lit[ctx], 8));
          emit.put(b); window[pos & window_mask] = b; ++pos;
        }
        if (pos >= b_end) { break; }
        op_e op = decode_op(rd, model, ll, pos);
        if (op == op_e::none) { continue; }

        std::uint32_t mlen = 0, off = 0;
        if (op == op_e::rep0) { mlen = 1 + decode_val(rd, model.match_len[pos & 3][0], model.match_len_extra, 24, 23); off = rep[0]; }
        else {
          int op_i = op == op_e::exp ? 4 : int(op); mlen = 2 + decode_val(rd, model.match_len[pos & 3][op_i], model.match_len_extra, 24, 23);
          int r = op == op_e::exp ? 3 : int(op);
          off = op == op_e::exp ? decode_val(rd, model.off[std::min<std::uint32_t>(mlen - 2, num_dist_ctx - 1)], model.off_extra, 32, 32) : rep[r];
          for (int k = r; k > 0; --k) { rep[k] = rep[k - 1]; }
          rep[0] = off;
        }
        for (std::uint32_t j = 0; j < mlen && pos < b_end; ++j) {
          std::uint8_t b = get_back(off);
          emit.put(b); window[pos & window_mask] = b; ++pos;
        }
      }
      if (prog && (pos - last_rep >= 65536 || pos >= total_uncomp)) {
        prog->done.store(pos, std::memory_order_relaxed); last_rep = pos;
      }
    }
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
    }
    int stride_log2 = delta_stride == 8 ? 3 : delta_stride == 4 ? 2 : delta_stride == 2 ? 1 : 0;
    std::uint8_t flags = std::uint8_t(use_delta ? 2 : 0) | std::uint8_t(stride_log2 << 2);

    params.bcj = false;

    fan::io::file::file_t* fp = nullptr;
    if (fan::io::file::open(&fp, out_path.string(), {"wb"})) { return false; }

    try {
      std::size_t actual_threads = thread_count ? thread_count : std::max<std::size_t>(1, std::thread::hardware_concurrency());
      std::uint64_t total_sz = use_delta ? raw_buf.size() : provider.size();
      params = threaded_params(params, total_sz, actual_threads);
      if (prog) {
        prog->done.store(0, std::memory_order_relaxed);
        prog->total.store(compression_progress_total(total_sz, params), std::memory_order_relaxed);
      }
      fan::progress_t* progress = prog;
      fan::io::file::file_writer_t sink {fp};
      std::uint8_t header[17];
      fan::memory::write_le32(header, magic_v4); fan::memory::write_le64(header + 4, total_sz); fan::memory::write_le32(header + 12, params.block_size); header[16] = flags;
      sink.write_bytes(std::span<const std::uint8_t>(header, sizeof(header)));
      if (use_delta) {
        fan::io::memory_provider_t mp(std::span<std::uint8_t>(raw_buf.data(), raw_buf.size()));
        encode_stream(mp, sink, params, actual_threads, progress);
      } else {
        encode_stream(provider, sink, params, actual_threads, progress);
      }
      fan::io::file::close(fp);
      if (prog) { prog->done.store(prog->total.load(std::memory_order_relaxed), std::memory_order_relaxed); }
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
      std::uint32_t block_size = fan::memory::read_le32(header + 12);
      std::uint8_t flags = header[16];
      bool use_delta = (flags >> 1) & 1;
      int stride_log2 = (flags >> 2) & 7;
      int delta_stride = 1 << stride_log2;
      if (flags & 1) { throw std::runtime_error("streaming bcj decode unsupported"); }
      if (prog) { prog->done.store(0, std::memory_order_relaxed); prog->total.store(total_uncomp, std::memory_order_relaxed); }

      if (use_delta) {
        fan::bytes_t raw; raw.reserve(total_uncomp);
        struct mem_emit_t { fan::bytes_t& out; void put(std::uint8_t b) { out.push_back(b); } } em{raw};
        decode_stream(src, em, total_uncomp, block_size, prog);
        delta_decode(raw, delta_stride);
        fan::io::file::archive_extractor_t writer(out_dir, default_out);
        for (std::uint8_t b : raw) { writer.put(b); }
        writer.finish(); fan::io::file::close(fp);
      } else {
        fan::io::file::archive_extractor_t writer(out_dir, default_out);
        decode_stream(src, writer, total_uncomp, block_size, prog);
        writer.finish(); fan::io::file::close(fp);
      }
      return true;
    } catch (...) {
      fan::io::file::close(fp); throw;
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
    params = threaded_params(params, total_sz, actual_threads);
    if (prog) {
      prog->done.store(0, std::memory_order_relaxed);
      prog->total.store(compression_progress_total(total_sz, params), std::memory_order_relaxed);
    }

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
    if (bcj) { fan::fcs::bcj_transform(raw, true); }

    int delta_stride = 1;
    bool use_delta = false;
    if (!bcj) {
      delta_stride = detect_delta_stride(raw);
      use_delta = delta_stride > 1;
      if (use_delta) { delta_encode(raw, delta_stride); }
    }
    int stride_log2 = delta_stride == 8 ? 3 : delta_stride == 4 ? 2 : delta_stride == 2 ? 1 : 0;
    std::uint8_t flags = std::uint8_t(bcj ? 1 : 0) | std::uint8_t(use_delta ? 2 : 0) | std::uint8_t(stride_log2 << 2);

    fan::bytes_t out; out.reserve((raw.size() / 5) + 17);
    std::uint8_t header[17]; fan::memory::write_le32(header, magic_v4); fan::memory::write_le64(header + 4, raw.size()); fan::memory::write_le32(header + 12, params.block_size); header[16] = flags;
    out.insert(out.end(), header, header + sizeof(header));
    
    fan::io::bytes_writer_t sink {out};
    fan::io::memory_provider_t provider(std::span<std::uint8_t>(reinterpret_cast<std::uint8_t*>(raw.data()), raw.size()));
    encode_stream(provider, sink, params, actual_threads, prog);
    if (prog) { prog->done.store(prog->total.load(std::memory_order_relaxed), std::memory_order_relaxed); }
    
    return out;
  }

  std::vector<fan::io::file_buffer_t> decompress(const fan::bytes_t& comp, fan::progress_t* prog = nullptr) {
    std::vector<fan::io::file_buffer_t> files;
    if (comp.size() < 17 || fan::memory::read_le32(comp.data()) != magic_v4) { throw std::runtime_error("needs FCS4"); }

    std::uint64_t total_uncomp = fan::memory::read_le64(comp.data() + 4);
    std::uint32_t block_size = fan::memory::read_le32(comp.data() + 12);
    std::uint8_t flags = comp[16];
    bool bcj = flags & 1;
    bool use_delta = (flags >> 1) & 1;
    int stride_log2 = (flags >> 2) & 7;
    int delta_stride = 1 << stride_log2;
    
    if (prog) { prog->total.store(total_uncomp, std::memory_order_relaxed); }

    struct mem_source_t {
      const fan::bytes_t& comp; std::size_t idx;
      void read_exact(std::span<std::uint8_t> out) {
        if (idx + out.size() > comp.size()) throw std::runtime_error("eof");
        std::memcpy(out.data(), comp.data() + idx, out.size()); idx += out.size();
      }
      void read_exact(fan::bytes_t& out) {
        if (idx + out.size() > comp.size()) throw std::runtime_error("eof");
        std::memcpy(out.data(), comp.data() + idx, out.size()); idx += out.size();
      }
    } src {comp, 17};
    
    struct mem_emitter_t {
      fan::bytes_t& out;
      void put(std::uint8_t b) { out.push_back(b); }
    };

    fan::bytes_t out; out.reserve(total_uncomp);
    mem_emitter_t writer {out};
    decode_stream(src, writer, total_uncomp, block_size, prog);

    if (use_delta) { fan::fcs::delta_decode(out, delta_stride); }
    if (bcj) { fan::fcs::bcj_transform(out, false); }
    if (out.size() < 4) { return files; }

    std::size_t out_idx = 0;
    std::uint32_t num_files = fan::memory::read_le32(out.data() + out_idx); out_idx += 4;
    struct meta_t { std::string path; std::uint64_t size; };
    std::vector<meta_t> metas(num_files);
    for (auto& m : metas) {
      if (out_idx + 2 > out.size()) { break; }
      std::uint16_t path_len = fan::memory::read_le16(out.data() + out_idx); out_idx += 2;
      if (out_idx + path_len > out.size()) { break; }
      m.path.assign(reinterpret_cast<const char*>(out.data() + out_idx), path_len); out_idx += path_len;
      if (out_idx + 8 > out.size()) { break; }
      m.size = fan::memory::read_le64(out.data() + out_idx); out_idx += 8;
    }
    out_idx += (4 - (out_idx & 3)) & 3;
    for (auto& m : metas) {
      if (out_idx + m.size > out.size()) { break; }
      fan::bytes_t data(out.begin() + out_idx, out.begin() + out_idx + std::size_t(m.size)); out_idx += std::size_t(m.size);
      files.push_back({std::move(m.path), std::move(data)});
    }
    return files;
  }
}