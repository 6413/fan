module;

#include <fan/types/json_impl.h>

module fan.types.json;

import std;

namespace nlohmann {
  template <typename t_type> struct adl_serializer<fan::vec2_wrap_t<t_type>> {
    static void to_json(nlohmann::json& j, const fan::vec2_wrap_t<t_type>& v) { j = nlohmann::json{ v.x, v.y }; }
    static void from_json(const nlohmann::json& j, fan::vec2_wrap_t<t_type>& v) { v.x = j[0].get<t_type>(); v.y = j[1].get<t_type>(); }
  };
  template <typename t_type> struct adl_serializer<fan::vec3_wrap_t<t_type>> {
    static void to_json(nlohmann::json& j, const fan::vec3_wrap_t<t_type>& v) { j = nlohmann::json{ v.x, v.y, v.z }; }
    static void from_json(const nlohmann::json& j, fan::vec3_wrap_t<t_type>& v) { v.x = j[0].get<t_type>(); v.y = j[1].get<t_type>(); v.z = j[2].get<t_type>(); }
  };
  template <typename t_type> struct adl_serializer<fan::vec4_wrap_t<t_type>> {
    static void to_json(nlohmann::json& j, const fan::vec4_wrap_t<t_type>& v) { j = nlohmann::json{ v.x, v.y, v.z, v.w }; }
    static void from_json(const nlohmann::json& j, fan::vec4_wrap_t<t_type>& v) { v.x = j[0].get<t_type>(); v.y = j[1].get<t_type>(); v.z = j[2].get<t_type>(); v.w = j[3].get<t_type>(); }
  };
  template <> struct adl_serializer<fan::color> {
    static void to_json(nlohmann::json& j, const fan::color& c) { j = nlohmann::json{ c[0], c[1], c[2], c[3] }; }
    static void from_json(const nlohmann::json& j, fan::color& c) { c.r = j[0]; c.g = j[1]; c.b = j[2]; c.a = j[3]; }
  };
}

namespace fan {
  json_iterator::json_iterator() { m_it = new nlohmann::json::iterator(); }
  json_iterator::~json_iterator() {
    if (m_it) { delete static_cast<nlohmann::json::iterator*>(m_it); }
    if (m_current) { delete m_current; }
  }
  json_iterator::json_iterator(const json_iterator& other) {
    m_it = new nlohmann::json::iterator(*static_cast<nlohmann::json::iterator*>(other.m_it));
    m_current = nullptr;
  }
  json_iterator& json_iterator::operator=(const json_iterator& other) {
    if (this == &other) { return *this; }
    *static_cast<nlohmann::json::iterator*>(m_it) = *static_cast<nlohmann::json::iterator*>(other.m_it);
    return *this;
  }
  json_iterator& json_iterator::operator++() {
    ++(*static_cast<nlohmann::json::iterator*>(m_it));
    return *this;
  }
  json_iterator json_iterator::operator++(int) {
    json_iterator tmp = *this;
    ++(*this);
    return tmp;
  }
  bool json_iterator::operator!=(const json_iterator& other) const {
    return *static_cast<nlohmann::json::iterator*>(m_it) != *static_cast<nlohmann::json::iterator*>(other.m_it);
  }
  bool json_iterator::operator==(const json_iterator& other) const {
    return *static_cast<nlohmann::json::iterator*>(m_it) == *static_cast<nlohmann::json::iterator*>(other.m_it);
  }
  int json_iterator::operator-(const json_iterator& other) const {
    return std::distance(
      *static_cast<nlohmann::json::iterator*>(other.m_it), 
      *static_cast<nlohmann::json::iterator*>(m_it)
    );
  }
  json_iterator json_iterator::operator-(int n) const {
    json_iterator tmp = *this;
    auto& it = *static_cast<nlohmann::json::iterator*>(tmp.m_it);
    it -= n;
    return tmp;
  }
  json_iterator json_iterator::operator+(int n) const {
    json_iterator tmp = *this;
    auto& it = *static_cast<nlohmann::json::iterator*>(tmp.m_it);
    it += n;
    return tmp;
  }
  std::string json_iterator::key() const {
    return static_cast<nlohmann::json::iterator*>(m_it)->key();
  }
  json& json_iterator::operator*() {
    if (!m_current) { m_current = new json(nullptr, true); }
    m_current->m_ptr = &(static_cast<nlohmann::json::iterator*>(m_it)->value());
    return *m_current;
  }
  json& json_iterator::value() { return operator*(); }
  json* json_iterator::operator->() { return &operator*(); }

  json::json() { m_ptr = new nlohmann::json(); m_is_ref = false; }
  json::json(void* ptr, bool is_ref) : m_ptr(ptr), m_is_ref(is_ref) {}
  
  json::json(int v) : json() { *this = v; }
  json::json(std::uint32_t v) : json() { *this = v; }
  json::json(std::int64_t v) : json() { *this = v; }
  json::json(std::uint64_t v) : json() { *this = v; }
  json::json(f32_t v) : json() { *this = v; }
  json::json(f64_t v) : json() { *this = v; }
  json::json(bool v) : json() { *this = v; }
  json::json(char v) : json() { *this = v; }
  json::json(const char* v) : json() { *this = v; }
  json::json(const std::string& v) : json() { *this = v; }
  json::json(std::initializer_list<std::pair<std::string, json>> init) : json() { *this = init; }

  json::~json() {
    if (!m_is_ref && m_ptr) { delete static_cast<nlohmann::json*>(m_ptr); }
  }
  json::json(const json& other) {
    m_is_ref = false;
    if (other.m_ptr) { m_ptr = new nlohmann::json(*static_cast<nlohmann::json*>(other.m_ptr)); }
    else { m_ptr = new nlohmann::json(); }
  }
  json::json(json&& other) noexcept {
    m_ptr = other.m_ptr;
    m_is_ref = other.m_is_ref;
    other.m_ptr = nullptr;
    other.m_is_ref = false;
  }
  json& json::operator=(const json& other) {
    if (this == &other) { return *this; }
    if (!m_ptr) { m_ptr = new nlohmann::json(); }
    *static_cast<nlohmann::json*>(m_ptr) = *static_cast<nlohmann::json*>(other.m_ptr);
    return *this;
  }
  json& json::operator=(json&& other) noexcept {
    if (this == &other) { return *this; }
    if (m_is_ref) {
      if (other.m_ptr) { *static_cast<nlohmann::json*>(m_ptr) = std::move(*static_cast<nlohmann::json*>(other.m_ptr)); }
      return *this;
    }
    if (m_ptr) { delete static_cast<nlohmann::json*>(m_ptr); }
    m_ptr = other.m_ptr;
    m_is_ref = other.m_is_ref;
    other.m_ptr = nullptr;
    other.m_is_ref = false;
    return *this;
  }
  json& json::operator=(std::initializer_list<std::pair<std::string, json>> init) {
    if (!m_ptr) { m_ptr = new nlohmann::json(); }
    auto& j = *static_cast<nlohmann::json*>(m_ptr);
    j = nlohmann::json::object();
    for (const auto& [k, v] : init) {
      j[k] = *static_cast<nlohmann::json*>(v.m_ptr);
    }
    return *this;
  }

  json json::parse(const std::string& raw) {
    json j;
    *static_cast<nlohmann::json*>(j.m_ptr) = nlohmann::json::parse(raw);
    return j;
  }
  json json::load_file(fan::str_view_t path) {
    std::string raw;
    if (fan::io::file::read(path, &raw)) { return json::object(); }
    try { return json::parse(raw); } catch (...) { return json::object(); }
  }
  bool json::save_file(fan::str_view_t path, const json& j, int indent) {
    if (!j.m_ptr) { return false; }
    return fan::io::file::write(path, static_cast<nlohmann::json*>(j.m_ptr)->dump(indent), std::ios_base::binary) == 0;
  }
  bool json::save(fan::str_view_t path, int indent) const { return save_file(path, *this, indent); }
  json json::object() {
    json j;
    *static_cast<nlohmann::json*>(j.m_ptr) = nlohmann::json::object();
    return j;
  }
  json json::array() {
    json j;
    *static_cast<nlohmann::json*>(j.m_ptr) = nlohmann::json::array();
    return j;
  }

  json& json::operator=(int v) { *static_cast<nlohmann::json*>(m_ptr) = v; return *this; }
  json& json::operator=(std::uint32_t v) { *static_cast<nlohmann::json*>(m_ptr) = v; return *this; }
  json& json::operator=(std::int64_t v) { *static_cast<nlohmann::json*>(m_ptr) = v; return *this; }
  json& json::operator=(std::uint64_t v) { *static_cast<nlohmann::json*>(m_ptr) = v; return *this; }
  json& json::operator=(f32_t v) { *static_cast<nlohmann::json*>(m_ptr) = v; return *this; }
  json& json::operator=(f64_t v) { *static_cast<nlohmann::json*>(m_ptr) = v; return *this; }
  json& json::operator=(bool v) { *static_cast<nlohmann::json*>(m_ptr) = v; return *this; }
  json& json::operator=(char v) { *static_cast<nlohmann::json*>(m_ptr) = v; return *this; }
  json& json::operator=(const char* v) { *static_cast<nlohmann::json*>(m_ptr) = v; return *this; }
  json& json::operator=(const std::string& v) { *static_cast<nlohmann::json*>(m_ptr) = v; return *this; }
  json& json::operator=(const fan::color& v) { *static_cast<nlohmann::json*>(m_ptr) = v; return *this; }
  template <typename t_type> json& json::operator=(const fan::vec2_wrap_t<t_type>& v) { *static_cast<nlohmann::json*>(m_ptr) = v; return *this; }
  template <typename t_type> json& json::operator=(const fan::vec3_wrap_t<t_type>& v) { *static_cast<nlohmann::json*>(m_ptr) = v; return *this; }
  template <typename t_type> json& json::operator=(const fan::vec4_wrap_t<t_type>& v) { *static_cast<nlohmann::json*>(m_ptr) = v; return *this; }

  json& json::operator+=(const json& val) {
    if (!m_ptr) *this = json::array();
    if (val.m_ptr) {
      *static_cast<nlohmann::json*>(m_ptr) += *static_cast<nlohmann::json*>(val.m_ptr);
    }
    return *this;
  }

  json json::operator[](const char* key) {
    nlohmann::json& node = (*static_cast<nlohmann::json*>(m_ptr))[key];
    return json(&node, true);
  }
  json json::operator[](const std::string& key) {
    nlohmann::json& node = (*static_cast<nlohmann::json*>(m_ptr))[key];
    return json(&node, true);
  }
  json json::operator[](std::size_t index) {
    nlohmann::json& node = (*static_cast<nlohmann::json*>(m_ptr))[index];
    return json(&node, true);
  }
  json json::operator[](int index) {
    nlohmann::json& node = (*static_cast<nlohmann::json*>(m_ptr))[static_cast<std::size_t>(index)];
    return json(&node, true);
  }
  const json json::operator[](const char* key) const {
    nlohmann::json& node = (*static_cast<nlohmann::json*>(m_ptr))[key];
    return json(&node, true);
  }
  const json json::operator[](const std::string& key) const {
    nlohmann::json& node = (*static_cast<nlohmann::json*>(m_ptr))[key];
    return json(&node, true);
  }
  const json json::operator[](std::size_t index) const {
    nlohmann::json& node = (*static_cast<nlohmann::json*>(m_ptr))[index];
    return json(&node, true);
  }
  const json json::operator[](int index) const {
    nlohmann::json& node = (*static_cast<nlohmann::json*>(m_ptr))[static_cast<std::size_t>(index)];
    return json(&node, true);
  }

  json json::at(const char* key) {
    nlohmann::json& node = static_cast<nlohmann::json*>(m_ptr)->at(key);
    return json(&node, true);
  }
  json json::at(std::size_t index) {
    nlohmann::json& node = static_cast<nlohmann::json*>(m_ptr)->at(index);
    return json(&node, true);
  }
  const json json::at(const char* key) const {
    nlohmann::json& node = static_cast<nlohmann::json*>(m_ptr)->at(key);
    return json(&node, true);
  }
  const json json::at(std::size_t index) const {
    nlohmann::json& node = static_cast<nlohmann::json*>(m_ptr)->at(index);
    return json(&node, true);
  }

  void json::update(const json& other, bool merge_objects) {
    if (!m_ptr) *this = json::object();
    if (other.m_ptr) {
      static_cast<nlohmann::json*>(m_ptr)->update(*static_cast<nlohmann::json*>(other.m_ptr), merge_objects);
    }
  }

  bool json::operator==(int val) const {
    if (!m_ptr) return false;
    return static_cast<nlohmann::json*>(m_ptr)->operator==(val);
  }
  bool json::operator!=(int val) const {
    if (!m_ptr) return true;
    return static_cast<nlohmann::json*>(m_ptr)->operator!=(val);
  }
  bool json::operator==(const json& val) const {
    if (!m_ptr || !val.m_ptr) return m_ptr == val.m_ptr;
    return *static_cast<nlohmann::json*>(m_ptr) == *static_cast<nlohmann::json*>(val.m_ptr);
  }
  bool json::operator!=(const json& val) const {
    return !(*this == val);
  }

  json::value_t json::type() const {
    if (!m_ptr) return value_t::null;
    return static_cast<value_t>(static_cast<nlohmann::json*>(m_ptr)->type());
  }
  bool json::is_string() const { return m_ptr && static_cast<nlohmann::json*>(m_ptr)->is_string(); }
  bool json::is_number() const { return m_ptr && static_cast<nlohmann::json*>(m_ptr)->is_number(); }
  bool json::is_boolean() const { return m_ptr && static_cast<nlohmann::json*>(m_ptr)->is_boolean(); }

  bool json::contains(const char* key) const { return static_cast<nlohmann::json*>(m_ptr)->contains(key); }
  bool json::contains(const std::string& key) const { return static_cast<nlohmann::json*>(m_ptr)->contains(key); }
  bool json::is_object() const { return static_cast<nlohmann::json*>(m_ptr)->is_object(); }
  bool json::is_array() const { return static_cast<nlohmann::json*>(m_ptr)->is_array(); }
  bool json::is_null() const { return static_cast<nlohmann::json*>(m_ptr)->is_null(); }
  std::size_t json::size() const { return static_cast<nlohmann::json*>(m_ptr)->size(); }
  bool json::empty() const {
    if (!m_ptr) { return true; }
    return static_cast<nlohmann::json*>(m_ptr)->empty();
  }

  void json::push_back(const json& val) {
    if (!m_ptr) { *this = json::array(); }
    static_cast<nlohmann::json*>(m_ptr)->push_back(*static_cast<nlohmann::json*>(val.m_ptr));
  }
  void json::push_back(const std::string& val) {
    if (!m_ptr) { *this = json::array(); }
    static_cast<nlohmann::json*>(m_ptr)->push_back(val);
  }
  void json::push_back(const char* val) {
    if (!m_ptr) { *this = json::array(); }
    static_cast<nlohmann::json*>(m_ptr)->push_back(val);
  }

  std::string json::dump(int indent, const char* indent_char, bool ensure_ascii) const {
    if (!m_ptr) { return "null"; }
    return static_cast<nlohmann::json*>(m_ptr)->dump(indent, indent_char[0], ensure_ascii);
  }

  void json::reserve(std::size_t n) {
    if (!m_ptr) *this = json::array();
    auto* internal_json = static_cast<nlohmann::json*>(m_ptr);
    if (internal_json->is_array()) {
      internal_json->get_ptr<nlohmann::json::array_t*>()->reserve(n);
    }
  }

  json::operator std::string_view() const {
    if (!m_ptr) return "";
    return static_cast<nlohmann::json*>(m_ptr)->get_ref<const std::string&>();
  }

  json::operator fan::str_view_t() const {
    if (!m_ptr) return "";
    return static_cast<nlohmann::json*>(m_ptr)->get_ref<const std::string&>();
  }

  json_iterator json::begin() {
    json_iterator it;
    *static_cast<nlohmann::json::iterator*>(it.m_it) = static_cast<nlohmann::json*>(m_ptr)->begin();
    return it;
  }
  json_iterator json::end() {
    json_iterator it;
    *static_cast<nlohmann::json::iterator*>(it.m_it) = static_cast<nlohmann::json*>(m_ptr)->end();
    return it;
  }
  json_iterator json::begin() const {
    json_iterator it;
    *static_cast<nlohmann::json::iterator*>(it.m_it) = static_cast<nlohmann::json*>(m_ptr)->begin();
    return it;
  }
  json_iterator json::end() const {
    json_iterator it;
    *static_cast<nlohmann::json::iterator*>(it.m_it) = static_cast<nlohmann::json*>(m_ptr)->end();
    return it;
  }
  json_iterator json::cbegin() const {
    json_iterator it;
    *static_cast<nlohmann::json::iterator*>(it.m_it) = static_cast<nlohmann::json*>(m_ptr)->begin();
    return it;
  }
  json_iterator json::cend() const {
    json_iterator it;
    *static_cast<nlohmann::json::iterator*>(it.m_it) = static_cast<nlohmann::json*>(m_ptr)->end();
    return it;
  }

  void json::preserve_unknown(const fan::json& source) {
    for (auto it = source.begin(); it != source.end(); ++it) {
      if (!contains(it.key().c_str())) {
        (*this)[it.key()] = it.value();
      }
    }
  }

  std::pair<std::size_t, std::size_t> json_stream_parser_t::find_next_json_bounds(std::string_view s, std::size_t pos) const noexcept {
    pos = s.find('{', pos);
    if (pos == std::string::npos) { return {pos, pos}; }
    int depth = 0;
    bool in_str = false;
    for (std::size_t i = pos; i < s.length(); i++) {
      char c = s[i];
      if (c == '"' && (i == 0 || s[i - 1] != '\\')) { in_str = !in_str; }
      else if (!in_str) {
        if (c == '{') { depth++; }
        else if (c == '}' && --depth == 0) { return {pos, i + 1}; }
      }
    }
    return {pos, std::string::npos};
  }
  std::vector<json_stream_parser_t::parsed_result> json_stream_parser_t::process(std::string_view chunk) {
    std::vector<parsed_result> results;
    buf += chunk;
    std::size_t pos = 0;
    while (pos < buf.length()) {
      auto [start, end] = find_next_json_bounds(buf, pos);
      if (start == std::string::npos) { break; }
      if (end == std::string::npos) { buf = buf.substr(start); break; }
      try { results.push_back({fan::json::parse(std::string(buf.data() + start, end - start)), "", true}); }
      catch (const std::exception& e) { results.push_back({fan::json::object(), e.what(), false}); }
      pos = buf.find('{', end);
      if (pos == std::string::npos) { pos = end; }
    }
    buf = pos < buf.length() ? buf.substr(pos) : "";
    return results;
  }
  void json_stream_parser_t::clear() noexcept { buf.clear(); }

  #define EXPLICIT_GET(T) template <> T json::_get_impl<T>() const { return static_cast<nlohmann::json*>(m_ptr)->get<T>(); }
  
  EXPLICIT_GET(int)
  EXPLICIT_GET(std::uint32_t)
  EXPLICIT_GET(std::int64_t)
  EXPLICIT_GET(std::uint64_t)
  EXPLICIT_GET(std::uint16_t)
  EXPLICIT_GET(std::int16_t)
  EXPLICIT_GET(std::uint8_t)
  EXPLICIT_GET(std::int8_t)
  EXPLICIT_GET(char)
  EXPLICIT_GET(f32_t)
  EXPLICIT_GET(f64_t)
  EXPLICIT_GET(bool)
  EXPLICIT_GET(std::string)
  EXPLICIT_GET(fan::color)
  EXPLICIT_GET(fan::vec2_wrap_t<f32_t>)
  EXPLICIT_GET(fan::vec2_wrap_t<int>)
  EXPLICIT_GET(fan::vec3_wrap_t<f32_t>)
  EXPLICIT_GET(fan::vec3_wrap_t<int>)
  EXPLICIT_GET(fan::vec4_wrap_t<f32_t>)
  EXPLICIT_GET(fan::vec4_wrap_t<int>)
  EXPLICIT_GET(std::vector<std::string>)

  template json& json::operator=<f32_t>(const fan::vec2_wrap_t<f32_t>&);
  template json& json::operator=<int>(const fan::vec2_wrap_t<int>&);
  template json& json::operator=<f32_t>(const fan::vec3_wrap_t<f32_t>&);
  template json& json::operator=<int>(const fan::vec3_wrap_t<int>&);
  template json& json::operator=<f32_t>(const fan::vec4_wrap_t<f32_t>&);
  template json& json::operator=<int>(const fan::vec4_wrap_t<int>&);
}