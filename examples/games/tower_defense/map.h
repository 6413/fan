struct unit_def_t {
  fan::str_view_t name;
  int   cost, health, damage;
  f32_t speed;
  fan::str_view_t json;
};

inline static constexpr auto unit_defs = std::to_array<unit_def_t>({
  { "Rat",    50,  60, 10, 120.f, "rat.json" },
  { "Horse", 150, 150, 25,  80.f, "rat.json" },
  { "Knight",300, 300, 50,  60.f, "rat.json" },
});

struct base_t {
  shape_t   sprite;
  fan::vec2 pos;
  int   hp = 1000, max_hp = 1000;
  f32_t gold_rate = 20.f, gold_accum = 0.f;
};

struct unit_t {
  sprite_t sprite;
  int      id = -1, def_idx = 0, health, damage, target_id = -1;
  f32_t    speed, attack_cd = 0.f;
  bool     dead = false;
  enum struct team_t  { player, enemy } team;
  enum struct state_t { moving, fighting } state = state_t::moving;

  void go_moving()  { if (state == state_t::moving)   return; state = state_t::moving;   sprite.play_sprite_sheet("run"); }
  void go_fighting(){ if (state == state_t::fighting)  return; state = state_t::fighting; sprite.play_sprite_sheet("attack"); }
};

// ---- state ----

std::vector<shape_t> map_shapes;
std::vector<unit_t>  units, pending_units;
base_t               player_base, enemy_base;
int                  player_gold = 200, next_unit_id = 0;

spatial::static_grid_t<int>  static_grid;
spatial::dynamic_grid_t<int> dynamic_grid;
spatial::registry_t<int>     registry;
std::unordered_map<int,int>  id_to_idx;

// ---- helpers ----

base_t& foe_of(const unit_t& u) { return u.team == unit_t::team_t::player ? enemy_base : player_base; }

unit_t* find_unit(int id) {
  auto it = id_to_idx.find(id);
  if (it == id_to_idx.end()) return nullptr;
  unit_t& u = units[it->second];
  return u.dead ? nullptr : &u;
}

// ---- spawn ----

void spawn_unit(unit_t::team_t team, int def_idx) {
  const auto& def = unit_defs[def_idx];
  base_t& own = team == unit_t::team_t::player ? player_base : enemy_base;

  unit_t u;
  u.id = next_unit_id++; u.team = team; u.def_idx = def_idx;
  u.health = def.health; u.damage = def.damage; u.speed = def.speed;
  u.sprite = shape_from_json(def.json);
  u.sprite.set_position(own.pos + fan::vec2{0, fan::random::value(-20.f, 20.f)});
  u.sprite.set_size(u.sprite.get_size() * 0.3f);
  if (team == unit_t::team_t::enemy) { auto s = u.sprite.get_size(); u.sprite.set_size({-s.x, s.y}); }
  u.sprite.play_sprite_sheet("run");
  pending_units.push_back(std::move(u));
}

// ---- update ----

void update_unit(int idx, f32_t dt) {
  unit_t& u = units[idx];
  if (u.dead) return;

  fan::vec2 p = u.sprite.get_position();
  spatial::upsert_object(registry, static_grid, dynamic_grid, u.id,
                         u.sprite.get_aabb(), spatial::movement_dynamic);
  u.attack_cd = std::max(0.f, u.attack_cd - dt);

  if (u.target_id != -1 && !find_unit(u.target_id)) u.target_id = -1;
  if (u.target_id == -1)
    u.target_id = spatial::query_nearest(dynamic_grid, p, 300.f, [&](int id) {
      unit_t* t = find_unit(id); return t && t->team != u.team;
    });

  base_t& foe = foe_of(u);
  fan::vec2 sep = spatial::separation_force(dynamic_grid, u.id, p, 28.f,
    [&](int id) -> fan::vec2 {
      unit_t* o = find_unit(id);
      return o ? fan::vec2(o->sprite.get_position()) : p;
    });

  if (u.state == unit_t::state_t::fighting) {
    if (!u.sprite.is_sprite_sheet_finished()) return;

    if (unit_t* t = find_unit(u.target_id)) {
      if ((t->sprite.get_position() - p).length() < 50.f) {
        if (u.attack_cd == 0.f) {
          u.attack_cd = 0.6f;
          u.go_fighting();
          if ((t->health -= u.damage) <= 0) {
            t->dead = true; t->sprite.set_color({0,0,0,0});
            spatial::remove_and_clean(registry, static_grid, dynamic_grid, t->id);
            id_to_idx.erase(t->id);
            u.target_id = -1;
          }
        }
        return;
      }
      u.go_moving(); return;
    }

    if (std::abs(foe.pos.x - p.x) < 60.f && u.attack_cd == 0.f) {
      u.attack_cd = 0.6f;
      u.go_fighting();
      foe.hp = std::max(0, foe.hp - u.damage);
      return;
    }
    u.go_moving(); return;
  }

  if (unit_t* t = find_unit(u.target_id)) {
    if ((t->sprite.get_position() - p).length() < 40.f) { u.go_fighting(); return; }
    u.sprite.move_towards(t->sprite.get_position() + sep * 0.3f, {u.speed, u.speed}, {-1.f, 0.f});
    return;
  }

  if (std::abs(foe.pos.x - p.x) < 60.f) { u.go_fighting(); return; }
  u.sprite.move_towards(fan::vec2{foe.pos.x, p.y} + sep * 0.3f, {u.speed, u.speed}, {-1.f, 0.f});
}

// ---- ai / gold / gui ----

void enemy_ai_tick(f32_t dt) {
  if (!fan::time::every(8000.f)) return;
  int r = fan::random::value(0, 99);
  spawn_unit(unit_t::team_t::enemy, r < 60 ? 0 : r < 88 ? 1 : 2);
}

void tick_gold(f32_t dt) {
  player_base.gold_accum += player_base.gold_rate * dt;
  int e = (int)player_base.gold_accum;
  player_gold += e; player_base.gold_accum -= e;
}

void draw_gui() {
  fan::vec2 vs = gui::get_display_size();
  { auto w = gui::hud_interactive("##hud");
    gui::gold_text(player_gold);
    gui::healthbar_labeled("Base HP:",  player_base.hp, player_base.max_hp, {150,14}, fan::color(0.2f,0.8f,0.2f,1.f));
    gui::healthbar_labeled("Enemy HP:", enemy_base.hp,  enemy_base.max_hp,  {150,14}, fan::color(0.9f,0.2f,0.2f,1.f), fan::colors::red);
  }
  { auto w = gui::hud_interactive("##buy");
    gui::anchor_bottom_center({-210.f, -100.f});
    gui::button_row(std::span(unit_defs),
      [&](const unit_def_t& d) { return fan::format("{}\n[{}g]", d.name, d.cost); },
      [&](const unit_def_t& d) { return player_gold >= d.cost; },
      {120, 80},
      [&](int i) { player_gold -= unit_defs[i].cost; spawn_unit(unit_t::team_t::player, i); });
  }
  if (enemy_base.hp <= 0 || player_base.hp <= 0) {
    bool win = enemy_base.hp <= 0;
    auto w = gui::hud_interactive("##endscreen");
    gui::set_cursor_pos({0, vs.y / 2.f - 60.f});
    gui::text_centered_outlined_big(win ? "YOU WIN!" : "YOU LOSE!", 48.f,
      win ? fan::color(0.2f,1.f,0.2f,1.f) : fan::color(1.f,0.2f,0.2f,1.f));
    fan::vec2 btn_size = gui::calc_text_size("Restart") + fan::vec2(gui::get_style().FramePadding) * 2.f;
    gui::set_cursor_pos({(vs.x - btn_size.x) / 2.f, vs.y / 2.f + 20.f});
    if (gui::button("Restart")) open(nullptr);
  }
}

// ---- lifecycle ----

void open(void* sod) {
  map_shapes = shapes_from_json("map.json");
  units.clear(); pending_units.clear(); id_to_idx.clear();
  player_gold = 200; next_unit_id = 0;

  fan::vec2 vs = pile.engine.viewport_get_size();
  spatial::static_grid_init(static_grid,   fan::vec2(-100), fan::vec2(64), fan::vec2i(64));
  spatial::dynamic_grid_init(dynamic_grid, fan::vec2(-100), fan::vec2(64), fan::vec2i(64));
  spatial::reset(static_grid, dynamic_grid, registry);

  auto init_base = [&](base_t& base, fan::vec2 pos) {
    base = {};
    base.pos = pos;
    base.sprite = shape_from_json("rat.json");
    base.sprite.set_position(pos);
    base.sprite.set_size(base.sprite.get_size() * 0.5f);
  };
  init_base(player_base, {80.f,        vs.y / 2.f});
  init_base(enemy_base,  {vs.x - 80.f, vs.y / 2.f});
}

void close() {}

void update() {
  f32_t dt = pile.engine.get_delta_time();
  if (enemy_base.hp > 0 && player_base.hp > 0) {
    tick_gold(dt);
    enemy_ai_tick(dt);
    for (auto& u : pending_units) { id_to_idx[u.id] = (int)units.size(); units.push_back(std::move(u)); }
    pending_units.clear();
    int n = (int)units.size();
    for (int i = 0; i < n; ++i) update_unit(i, dt);
    std::erase_if(units, [](auto& u){ return u.dead; });
    id_to_idx.clear();
    for (int i = 0; i < (int)units.size(); ++i) id_to_idx[units[i].id] = i;
  }
  draw_gui(); pile.update();
}