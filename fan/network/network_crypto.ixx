module;

#include <openssl/crypto.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/kdf.h>
#include <openssl/rand.h>
#if OPENSSL_VERSION_MAJOR >= 3
  #include <openssl/provider.h>
#endif

export module fan.network.crypto;

import std;

export namespace fan::network::crypto {
  using x25519_key_t = std::array<std::uint8_t, 32>;
  using confirm_t = std::array<std::uint8_t, 16>;
  constexpr std::size_t tag_size = 16;

  struct session_keys_t {
    x25519_key_t c2s;
    x25519_key_t s2c;
    confirm_t confirm;
    x25519_key_t tcp_c2s;
    x25519_key_t tcp_s2c;
  };

  enum class pin_result_e {
    pinned,
    matched,
    mismatch,
    io_error
  };

  void ensure_providers() {
  #if OPENSSL_VERSION_MAJOR >= 3
    static bool loaded = OSSL_PROVIDER_try_load(nullptr, "default", 1) != nullptr;
    (void)loaded;
  #endif
  }
  const EVP_CIPHER* chacha20_poly1305() {
  #if OPENSSL_VERSION_MAJOR >= 3
    // an explicit fetch skips the provider lookup that EVP_chacha20_poly1305() costs on every init
    static const EVP_CIPHER* cipher = (ensure_providers(), EVP_CIPHER_fetch(nullptr, "ChaCha20-Poly1305", nullptr));
    return cipher;
  #else
    return EVP_chacha20_poly1305();
  #endif
  }
  bool x25519_public(const x25519_key_t& priv, x25519_key_t& pub) {
    ensure_providers();
    EVP_PKEY* k = EVP_PKEY_new_raw_private_key(EVP_PKEY_X25519, nullptr, priv.data(), priv.size());
    std::size_t n = pub.size();
    if (!k) {
      return false;
    }
    bool ok = EVP_PKEY_get_raw_public_key(k, pub.data(), &n) == 1 && n == pub.size();
    EVP_PKEY_free(k);
    return ok;
  }
  bool x25519_keygen(x25519_key_t& priv, x25519_key_t& pub) {
    return RAND_bytes(priv.data(), priv.size()) == 1 && x25519_public(priv, pub);
  }
  bool x25519(const x25519_key_t& priv, const x25519_key_t& peer, std::uint8_t* out) {
    ensure_providers();
    EVP_PKEY* k = EVP_PKEY_new_raw_private_key(EVP_PKEY_X25519, nullptr, priv.data(), priv.size());
    EVP_PKEY* p = EVP_PKEY_new_raw_public_key(EVP_PKEY_X25519, nullptr, peer.data(), peer.size());
    EVP_PKEY_CTX* c = k ? EVP_PKEY_CTX_new(k, nullptr) : nullptr;
    std::size_t n = 32;
    bool ok = c && p
      && EVP_PKEY_derive_init(c) == 1
      && EVP_PKEY_derive_set_peer(c, p) == 1
      && EVP_PKEY_derive(c, out, &n) == 1 && n == 32;
    EVP_PKEY_CTX_free(c);
    EVP_PKEY_free(p);
    EVP_PKEY_free(k);
    return ok;
  }
  bool hkdf(const std::uint8_t* ikm, std::size_t ikm_size, const std::uint8_t* salt, std::size_t salt_size, std::string_view info, std::uint8_t* out, std::size_t out_size) {
    ensure_providers();
    EVP_PKEY_CTX* c = EVP_PKEY_CTX_new_id(EVP_PKEY_HKDF, nullptr);
    bool ok = c
      && EVP_PKEY_derive_init(c) == 1
      && EVP_PKEY_CTX_set_hkdf_md(c, EVP_sha256()) == 1
      && EVP_PKEY_CTX_set1_hkdf_salt(c, salt, salt_size) == 1
      && EVP_PKEY_CTX_set1_hkdf_key(c, ikm, ikm_size) == 1
      && EVP_PKEY_CTX_add1_hkdf_info(c, (const unsigned char*)info.data(), info.size()) == 1
      && EVP_PKEY_derive(c, out, &out_size) == 1;
    EVP_PKEY_CTX_free(c);
    return ok;
  }
  bool derive(const std::uint8_t* dh, const x25519_key_t& client_pub, const x25519_key_t& server_eph_pub, const x25519_key_t& server_static_pub, std::string_view context, session_keys_t& out) {
    std::uint8_t salt[96];
    std::memcpy(salt, client_pub.data(), 32);
    std::memcpy(salt + 32, server_eph_pub.data(), 32);
    std::memcpy(salt + 64, server_static_pub.data(), 32);
    static_assert(sizeof(session_keys_t) == 32 * 4 + 16);
    bool ok = hkdf(dh, 64, salt, sizeof(salt), context, (std::uint8_t*)&out, sizeof(out));
    return ok;
  }
  bool client_handshake(const x25519_key_t& client_priv, const x25519_key_t& client_pub, const x25519_key_t& server_eph_pub, const x25519_key_t& server_static_pub, std::string_view context, session_keys_t& out) {
    std::uint8_t dh[64];
    bool ok = x25519(client_priv, server_eph_pub, dh)
      && x25519(client_priv, server_static_pub, dh + 32)
      && derive(dh, client_pub, server_eph_pub, server_static_pub, context, out);
    OPENSSL_cleanse(dh, sizeof(dh));
    return ok;
  }
  bool server_handshake(const x25519_key_t& static_priv, const x25519_key_t& static_pub, const x25519_key_t& client_pub, std::string_view context, x25519_key_t& eph_pub, session_keys_t& out) {
    x25519_key_t eph_priv;
    std::uint8_t dh[64];
    bool ok = x25519_keygen(eph_priv, eph_pub)
      && x25519(eph_priv, client_pub, dh)
      && x25519(static_priv, client_pub, dh + 32)
      && derive(dh, client_pub, eph_pub, static_pub, context, out);
    OPENSSL_cleanse(dh, sizeof(dh));
    OPENSSL_cleanse(eph_priv.data(), eph_priv.size());
    return ok;
  }
  bool confirm_equal(const confirm_t& a, const std::uint8_t* b) {
    return CRYPTO_memcmp(a.data(), b, a.size()) == 0;
  }
  void cleanse(void* p, std::size_t size) {
    OPENSSL_cleanse(p, size);
  }
  bool aead(bool enc, const x25519_key_t& key, std::uint64_t counter, const std::uint8_t* aad, std::size_t aad_size, std::uint8_t* data, std::size_t size, std::uint8_t* tag) {
    ensure_providers();
    std::uint8_t nonce[12] = {};
    std::memcpy(nonce + 4, &counter, 8);
    // one context per thread, this runs for every packet
    struct cipher_ctx_t {
      ~cipher_ctx_t() {
        EVP_CIPHER_CTX_free(c);
      }
      EVP_CIPHER_CTX* c = EVP_CIPHER_CTX_new();
      bool cipher_set = false;
    };
    thread_local cipher_ctx_t ctx;
    EVP_CIPHER_CTX* c = ctx.c;
    if (c && !ctx.cipher_set) {
      ctx.cipher_set = EVP_CipherInit_ex(c, chacha20_poly1305(), nullptr, nullptr, nullptr, enc) == 1;
    }
    int n = 0;
    bool ok = c && ctx.cipher_set
      && EVP_CipherInit_ex(c, nullptr, nullptr, key.data(), nonce, enc) == 1
      && (enc || EVP_CIPHER_CTX_ctrl(c, EVP_CTRL_AEAD_SET_TAG, tag_size, tag) == 1)
      && EVP_CipherUpdate(c, nullptr, &n, aad, aad_size) == 1
      && (size == 0 || EVP_CipherUpdate(c, data, &n, data, size) == 1)
      && EVP_CipherFinal_ex(c, data + size, &n) == 1
      && (!enc || EVP_CIPHER_CTX_ctrl(c, EVP_CTRL_AEAD_GET_TAG, tag_size, tag) == 1);
    return ok;
  }
  std::string to_hex(const std::uint8_t* data, std::size_t size) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string r(size * 2, 0);
    for (std::size_t i = 0; i < size; ++i) {
      r[i * 2] = digits[data[i] >> 4];
      r[i * 2 + 1] = digits[data[i] & 15];
    }
    return r;
  }
  std::string openssl_error() {
    char buf[256];
    unsigned long e = ERR_get_error();
    if (!e) {
      return "no openssl error queued";
    }
    ERR_error_string_n(e, buf, sizeof(buf));
    ERR_clear_error();
    return buf;
  }
  bool load_or_create_key(const std::string& path, x25519_key_t& priv, x25519_key_t& pub, std::string* error = nullptr) {
    auto fail = [&](std::string msg) {
      if (error) {
        std::error_code ec;
        *error = msg + " (" + std::filesystem::absolute(path, ec).string() + ")";
      }
      return false;
    };
    std::error_code ec;
    if (std::filesystem::exists(path, ec)) {
      std::ifstream in(path, std::ios::binary);
      if (!in.read((char*)priv.data(), priv.size())) {
        return fail("key file is shorter than 32 bytes, delete it to generate a new one");
      }
      return x25519_public(priv, pub) || fail("x25519 public key derivation failed: " + openssl_error());
    }
    ensure_providers();
    if (RAND_bytes(priv.data(), priv.size()) != 1) {
      return fail("RAND_bytes failed: " + openssl_error());
    }
    if (!x25519_public(priv, pub)) {
      return fail("x25519 key creation failed: " + openssl_error() + ", OpenSSL " + OpenSSL_version(OPENSSL_VERSION) + ", OPENSSL_CONF=" + (std::getenv("OPENSSL_CONF") ? std::getenv("OPENSSL_CONF") : "unset"));
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
      return fail("cannot create key file: " + std::generic_category().message(errno));
    }
    out.write((const char*)priv.data(), priv.size());
    out.close();
    if (!out) {
      return fail("cannot write key file");
    }
    std::filesystem::permissions(path, std::filesystem::perms::owner_read | std::filesystem::perms::owner_write, ec);
    return true;
  }
  pin_result_e check_pinned_key(const std::string& path, const std::string& host, const x25519_key_t& key) {
    std::string hex = to_hex(key.data(), key.size());
    std::ifstream in(path);
    std::string h, k;
    while (in >> h >> k) {
      if (h == host) {
        return k == hex ? pin_result_e::matched : pin_result_e::mismatch;
      }
    }
    std::ofstream out(path, std::ios::app);
    out << host << ' ' << hex << '\n';
    return out ? pin_result_e::pinned : pin_result_e::io_error;
  }

  struct replay_window_t {
    bool accept(std::uint64_t n) {
      if (n + window_size <= top) {
        return false;
      }
      if (n >= top) {
        for (std::uint64_t i = top; i <= n && i < top + window_size; ++i) {
          seen.reset(i % window_size);
        }
        top = n + 1;
      }
      else if (seen[n % window_size]) {
        return false;
      }
      seen.set(n % window_size);
      return true;
    }
    static constexpr std::size_t window_size = 1024;
    std::bitset<window_size> seen;
    std::uint64_t top = 0;
  };

  struct secure_channel_t {
    void set(const x25519_key_t& send, const x25519_key_t& recv) {
      *this = {};
      send_key = send;
      recv_key = recv;
      ready = true;
    }
    void set_client(const session_keys_t& keys) {
      set(keys.c2s, keys.s2c);
    }
    void set_server(const session_keys_t& keys) {
      set(keys.s2c, keys.c2s);
    }
    std::size_t seal(std::uint8_t* packet, std::size_t header_size, std::size_t size, std::uint64_t nonce) {
      return ready && aead(true, send_key, nonce, packet, header_size, packet + header_size, size - header_size, packet + size) ? size + tag_size : 0;
    }
    std::size_t open(std::uint8_t* packet, std::size_t header_size, std::size_t size, std::uint64_t nonce) {
      if (!ready || size < header_size + tag_size) {
        return 0;
      }
      size -= tag_size;
      return aead(false, recv_key, nonce, packet, header_size, packet + header_size, size - header_size, packet + size) && window.accept(nonce) ? size : 0;
    }
    x25519_key_t send_key{};
    x25519_key_t recv_key{};
    std::uint64_t send_nonce = 0;
    replay_window_t window;
    bool ready = false;
  };

  struct secure_stream_t {
    void set(const x25519_key_t& send, const x25519_key_t& recv) {
      *this = {};
      send_key = send;
      recv_key = recv;
      ready = true;
    }
    std::vector<char> seal(const char* data, std::size_t size) {
      std::vector<char> out;
      out.reserve(size + (size / max_record + 1) * (2 + tag_size));
      while (size) {
        std::uint16_t n = (std::uint16_t)std::min(size, max_record);
        std::size_t at = out.size();
        out.resize(at + 2 + n + tag_size);
        auto p = (std::uint8_t*)out.data() + at;
        std::memcpy(p, &n, 2);
        std::memcpy(p + 2, data, n);
        aead(true, send_key, send_counter++, p, 2, p + 2, n, p + 2 + n);
        data += n;
        size -= n;
      }
      return out;
    }
    bool open(const char* data, std::size_t size, std::vector<char>& out) {
      pending.insert(pending.end(), data, data + size);
      std::size_t off = 0;
      while (pending.size() - off >= 2) {
        std::uint16_t n;
        std::memcpy(&n, pending.data() + off, 2);
        if (n == 0 || n > max_record) {
          return false;
        }
        if (pending.size() - off < 2 + n + tag_size) {
          break;
        }
        auto p = (std::uint8_t*)pending.data() + off;
        if (!aead(false, recv_key, recv_counter++, p, 2, p + 2, n, p + 2 + n)) {
          return false;
        }
        out.insert(out.end(), p + 2, p + 2 + n);
        off += 2 + n + tag_size;
      }
      pending.erase(pending.begin(), pending.begin() + off);
      return true;
    }
    static constexpr std::size_t max_record = 16384;
    x25519_key_t send_key{};
    x25519_key_t recv_key{};
    std::uint64_t send_counter = 0;
    std::uint64_t recv_counter = 0;
    std::vector<char> pending;
    bool ready = false;
  };
}