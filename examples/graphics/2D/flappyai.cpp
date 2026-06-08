#include <vector>
#include <algorithm>

import fan;

using namespace fan::graphics;

static constexpr f32_t initial_speed = 1000.f;
static constexpr f32_t jump_velocity = -3000.f;
static constexpr f32_t jump_cooldown_time = 0.1f;
static constexpr f32_t pipe_spacing = 600.f;
static constexpr f32_t gravity_multiplier = 9.f;
static constexpr f32_t gap_half = 200.f;
static constexpr fan::vec2 pipe_half = fan::vec2(32.f, 1000.f);
static constexpr fan::vec2 player_half = fan::vec2(16.f);

static constexpr f32_t pipe_y_min = 0.4f;
static constexpr f32_t pipe_y_max = 1.5f;

static constexpr int mutation_chance = 8;
static constexpr int population_size = 20;
static constexpr int total_hidden_nodes = 6;
static constexpr int total_input_nodes = 4;
static constexpr int total_output_nodes = 1;

static constexpr uint64_t category_bird = 0x1;
static constexpr uint64_t category_pipe = 0x2;

struct pile_t {
  engine_t engine;
};

pile_t* pile = nullptr;

struct pipe_t {
  fan::physics::entity_t body[2];
  fan::graphics::rectangle_t shape[2];

  void push(f32_t x_pos) {
    fan::vec2 vs = pile->engine.viewport_get_size();
    f32_t gap_y = fan::random::value(vs.y * pipe_y_min, vs.y * pipe_y_max);

    for (int i = 0; i < 2; ++i) {
      f32_t sign = i == 0 ? -1.f : 1.f;
      fan::vec2 p = fan::vec2(x_pos, gap_y + sign * (pipe_half.y + gap_half));

      body[i] = pile->engine.physics_context.create_box(
        p, pipe_half, 0.f,
        fan::physics::body_type_e::static_body,
        {.is_sensor = true, .filter = {.categoryBits = category_pipe, .maskBits = category_bird}}
      );

      shape[i] = fan::graphics::rectangle_t{{
        .position = fan::vec3(p, 0),
        .size = pipe_half,
        .color = fan::random::color()
      }};
    }
  }

  void destroy() {
    for (int i = 0; i < 2; ++i)
      body[i].destroy();
  }
};

std::vector<pipe_t> pipes;

struct bird_t {
  fan::physics::entity_t body;
  fan::graphics::rectangle_t shape;

  bool dead = false;
  f32_t fitness = 0.f;
  f32_t jump_cooldown = 0.f;

  std::vector<std::vector<std::vector<f32_t>>> weights;

  void randomize_weights() {
    weights.resize(2);
    weights[0].assign(total_input_nodes, std::vector<f32_t>(total_hidden_nodes));
    weights[1].assign(total_hidden_nodes, std::vector<f32_t>(total_output_nodes));

    for (auto& layer : weights)
      for (auto& prev : layer)
        for (auto& w : prev)
          w = fan::random::value(-1.f, 1.f);
  }

  void init(fan::vec2 start) {
    randomize_weights();

    body = pile->engine.physics_context.create_box(
      start, player_half, 0.f,
      fan::physics::body_type_e::dynamic_body,
      {.fixed_rotation = true, .filter = {.categoryBits = category_bird, .maskBits = category_pipe}}
    );
    body.set_linear_velocity(fan::vec2(initial_speed, 0.f));
    body.set_gravity_scale(gravity_multiplier);

    shape = fan::graphics::rectangle_t{{
      .position = fan::vec3(start, 1),
      .size = player_half,
      .color = fan::random::color().set_alpha(0.5f),
      .blending = true
    }};
  }

  f32_t get_gap_diff() {
    fan::vec2 bpos = body.get_position();
    for (auto& p : pipes) {
      if (bpos.x < p.shape[0].get_position().x + player_half.x * 2.f)
        return (p.shape[0].get_position().y + pipe_half.y + gap_half) - bpos.y;
    }
    return 0.f;
  }

  f32_t get_dist_to_next_pipe() {
    fan::vec2 bpos = body.get_position();
    for (auto& p : pipes) {
      f32_t px = p.shape[0].get_position().x;
      if (bpos.x < px + player_half.x * 2.f)
        return px - bpos.x;
    }
    return 1000.f;
  }

  bool do_ai() {
    std::vector<std::vector<f32_t>> nn(3);
    nn[0].resize(total_input_nodes);
    nn[1].assign(total_hidden_nodes, 0.f);
    nn[2].assign(total_output_nodes, 0.f);

    fan::vec2 vs = pile->engine.viewport_get_size();
    nn[0][0] = body.get_linear_velocity().y / -jump_velocity;
    nn[0][1] = get_gap_diff() / vs.y;
    nn[0][2] = (body.get_position().y - vs.y / 2.f) / vs.y;
    nn[0][3] = get_dist_to_next_pipe() / vs.x;

    for (int a = 0; a < 2; ++a) {
      for (int b = 0; b < (int)nn[a + 1].size(); ++b) {
        for (int c = 0; c < (int)nn[a].size(); ++c)
          nn[a + 1][b] += nn[a][c] * weights[a][c][b];

        f32_t& v = nn[a + 1][b];
        v = v >= 0.f ? 1.f - std::pow(2.f, -v) : std::pow(2.f, v) - 1.f;
      }
    }

    bool output = nn[2][0] >= 0.f;
    jump_cooldown = std::max(0., jump_cooldown - pile->engine.delta_time);
    if (output && jump_cooldown <= 0.f) {
      jump_cooldown = jump_cooldown_time;
      return true;
    }
    return false;
  }

  void reset(fan::vec2 start) {
    dead = false;
    fitness = 0.f;
    jump_cooldown = 0.f;
    body.set_physics_position(start);
    body.set_linear_velocity(fan::vec2(initial_speed, 0.f));
  }

  void mutate() {
    for (auto& layer : weights)
      for (auto& prev : layer)
        for (auto& w : prev)
          if (fan::random::value(0, mutation_chance - 1) == 0)
            w += fan::random::value(-0.3f, 0.3f);
  }

  bool operator>(const bird_t& o) const { return fitness > o.fitness; }
  bool operator<(const bird_t& o) const { return fitness < o.fitness; }

  void crossover(const decltype(weights)& w0, const decltype(weights)& w1) {
    for (int a = 0; a < (int)weights.size(); ++a)
      for (int b = 0; b < (int)weights[a].size(); ++b)
        for (int c = 0; c < (int)weights[a][b].size(); ++c)
          weights[a][b][c] = fan::random::value(0, 1) ? w0[a][b][c] : w1[a][b][c];
    mutate();
  }
};

std::vector<bird_t> birds;

int main() {
  pile = new pile_t;
  pile->engine.update_physics(true);

  fan::vec2 vs = pile->engine.viewport_get_size();
  fan::vec2 start = vs / 2.f;

  birds.resize(population_size);
  for (auto& b : birds)
    b.init(start);

  f32_t next_pipe_x = start.x + vs.x * 0.6f;
  uint32_t generation = 0;
  f32_t best_score = 0.f;

  fan::physics::step_callback_nr_t step_cb = fan::physics::add_physics_step_callback([&]() {
    for (auto& b : birds) {
      if (b.dead) continue;
      fan::vec2 bpos = b.body.get_position();
      if (bpos.y < -(pipe_half.y * 2.f) || bpos.y > vs.y * pipe_y_max + pipe_half.y * 2.f) {
        b.dead = true;
        continue;
      }
      for (auto& p : pipes) {
        for (int i = 0; i < 2; ++i) {
          if (fan::physics::is_on_sensor(b.body, p.body[i])) {
            b.dead = true;
            break;
          }
        }
        if (b.dead) break;
      }
    }
  });

  pile->engine.loop([&] {
    bool all_dead = std::all_of(birds.begin(), birds.end(), [](const bird_t& b){ return b.dead; });

    if (all_dead) {
      std::sort(birds.begin(), birds.end(), std::greater<bird_t>());
      best_score = std::max(best_score, birds[0].fitness);
      ++generation;
      fan::print("gen:", generation, "best:", best_score);

      auto w0 = birds[0].weights;
      auto w1 = birds[1].weights;
      auto w2 = birds[2].weights;

      // keep top 2 but still mutate them slightly
      birds[0].mutate();
      birds[1].mutate();

      std::array<const decltype(bird_t::weights)*, 3> top = {&w0, &w1, &w2};
      for (int i = 2; i < (int)birds.size(); ++i) {
        int a = fan::random::value(0, 2);
        int b = fan::random::value(0, 2);
        birds[i].crossover(*top[a], *top[b]);
      }

      for (auto& b : birds)
        b.reset(start);

      for (auto& p : pipes)
        p.destroy();
      pipes.clear();
      next_pipe_x = start.x + vs.x * 0.6f;
    }

    int leader = 0;
    for (int i = 0; i < (int)birds.size(); ++i) {
      if (!birds[i].dead) { leader = i; break; }
    }

    fan::vec2 cam_pos = birds[leader].body.get_position();
    fan::graphics::camera_set_position(fan::graphics::get_orthographic_render_view().camera, cam_pos - vs / 2.f);

    while (next_pipe_x < cam_pos.x + vs.x) {
      pipes.emplace_back();
      pipes.back().push(next_pipe_x);
      next_pipe_x += pipe_spacing;
    }

    for (int i = (int)pipes.size() - 1; i >= 0; --i) {
      if (pipes[i].shape[0].get_position().x < cam_pos.x - vs.x * 0.5f) {
        pipes[i].destroy();
        pipes.erase(pipes.begin() + i);
      }
    }

    for (auto& b : birds) {
      if (!b.dead) {
        f32_t gap_center_penalty = std::abs(b.get_gap_diff() - gap_half) / vs.y;
        b.fitness = b.body.get_position().x - gap_center_penalty * 50.f;
        if (b.do_ai()) {
          fan::vec2 vel = b.body.get_linear_velocity();
          b.body.set_linear_velocity(fan::vec2(vel.x, jump_velocity));
        }
        fan::vec2 vel = b.body.get_linear_velocity();
        if (vel.x < initial_speed)
          b.body.set_linear_velocity(fan::vec2(initial_speed, vel.y));
      }
      b.shape.set_position(fan::vec3(b.body.get_position(), 1));
    }

    for (auto& p : pipes) {
      for (int i = 0; i < 2; ++i)
        p.shape[i].set_position(fan::vec3(p.body[i].get_position(), 0));
    }
  });

  return 0;
}