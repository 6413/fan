#include <fan/time/time.h>

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>
#include <WITCH/WITCH.h>

import fan;

constexpr static f32_t BCOLStepTime = 0.001;
#define BCOL_set_Dimension 2
#define BCOL_set_IncludePath FAN_INCLUDE_PATH/fan
#define BCOL_set_prefix BCOL
#define BCOL_set_DynamicDeltaFunction \
  ObjectData0->Velocity[1] += delta * 10;
#define BCOL_set_StoreExtraDataInsideObject 1
#define BCOL_set_ExtraDataInsideObject \
  bool IsItPipe;
#include <BCOL/BCOL.h>



f32_t initial_speed = 0.5;

struct pile_t {

  static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
  static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

  pile_t() {
    fan::vec2 window_size = loco.window.get_size();
    camera = loco.open_camera(ortho_x, ortho_y);
    viewport = loco.open_viewport(0, window_size);
  }

  loco_t loco;
  loco_t::camera_t camera;
  loco_t::viewport_t viewport;
};

static constexpr auto gap_size = 0.9;
static constexpr fan::vec2 player_size = 0.01;

// 1 : 64
static constexpr unsigned char MUTATION_PROBABILITY = 64;
static constexpr unsigned char POPULATION_SIZE = 8;
static constexpr unsigned char TOTAL_HIDDEN_NODES = 4;
static constexpr unsigned char TOTAL_INPUT_NODES = 3;
static constexpr unsigned char TOTAL_OUTPUT_NODES = 1;

BCOL_t bcol;

pile_t* pile = new pile_t;
f32_t BCOLDelta = 0;


static constexpr fan::vec2 pipe_size = fan::vec2(0.1, 10000);


//loco_t::shape_t pipes[1000];
struct pipe_t {
  BCOL_t::ObjectID_t oid[2];
  loco_t::shape_t shape[2];
  void push_back(f32_t xpos) {

    f32_t r = fan::random::value_f32(2, 3);

    for (uint32_t i = 0; i < 2; ++i) {

      BCOL_t::ObjectProperties_t p;
      p.Position = xpos;
      p.Position[1] = -1 - pipe_size.y + r;
      if (i == 1) {
        p.Position[1] += pipe_size.y * 2 + gap_size;
      }

      p.ExtraData.IsItPipe = true;
      oid[i] = bcol.NewObject(&p, BCOL_t::ObjectFlag::Constant);

      BCOL_t::ShapeProperties_Rectangle_t sp;
      sp.Position = 0;
      sp.Size = pipe_size;
      bcol.NewShape_Rectangle(oid[i], &sp);

      loco_t::rectangle_t::properties_t pipe_properties;

      pipe_properties.position = *(fan::vec2*)&p.Position;
      pipe_properties.size = sp.Size;
      pipe_properties.camera = pile->camera;
      pipe_properties.viewport = pile->viewport;
      pipe_properties.color = fan::random::color();

      shape[i] = pipe_properties;
    }
  }
};

std::vector<pipe_t> pipes;

struct bird_t{
  BCOL_t::ObjectID_t oid;
  loco_t::shape_t shape;
  
  bool dead = false;
  f32_t fitness;

  std::uniform_int_distribution<unsigned short> mutation_distribution;

  //This is the range in which the weights can be.
  std::uniform_real_distribution<float> node_distribution;

  std::vector<std::vector<std::vector<float>>> weights;

  void init() {
    static std::random_device device;
    static std::mt19937_64 random(device());

    mutation_distribution = decltype(mutation_distribution){ 0, MUTATION_PROBABILITY - 1 };

    node_distribution = decltype(node_distribution)(-1, std::nextafter(1, 2));
    weights.resize(2);

    weights[0].resize(TOTAL_INPUT_NODES, std::vector<float>(TOTAL_HIDDEN_NODES));
    weights[1].resize(TOTAL_HIDDEN_NODES, std::vector<float>(TOTAL_OUTPUT_NODES));

    //This is how I structured the vector of weights.
    for (std::vector<std::vector<float>>& layer : weights)
    {
      for (std::vector<float>& previous_node : layer)
      {
        for (float& next_node : previous_node)
        {
          next_node = node_distribution(random);
        }
      }
    }

    {
      BCOL_t::ObjectProperties_t p;
      p.Position = 0;
      p.ExtraData.IsItPipe = false;
      oid = bcol.NewObject(&p, 0);
      bcol.SetObject_Velocity(oid, fan::vec2(initial_speed, 0));

      BCOL_t::ShapeProperties_Circle_t sp;
      sp.Position = 0;
      sp.Size = player_size.x;
      bcol.NewShape_Circle(oid, &sp);

      loco_t::rectangle_t::properties_t player_properties;
      player_properties.position.x = bcol.GetObject_Position(oid)[0];
      player_properties.position.y = bcol.GetObject_Position(oid)[1];
      player_properties.position.z = 1;
      player_properties.size = sp.Size;
      player_properties.camera = pile->camera;
      player_properties.viewport = pile->viewport;
      player_properties.color = fan::random::color();
      player_properties.color.a = 0.3;
      player_properties.blending = true;

      shape = player_properties;
    }
  }

  float get_gap_difference()
  {
    fan::vec2 player_position = shape.get_position();

    for (auto& a : pipes)
    {
      for (uint32_t i = 0; i < 2; ++i) {
        if (player_position.x < a.shape[i].get_position().x + 2 * player_size.x)
        {
          if (i == 0) {
            return (a.shape[i].get_position().y + a.shape[i].get_size().y + gap_size) - player_position.y;
          }
          else {
            return (a.shape[i].get_position().y - a.shape[i].get_size().y - gap_size) - player_position.y;
          }
        }
      }
    }

    return 0;
  }

  float get_gap_difference2()
  {
    fan::vec2 player_position = shape.get_position();

    for (uint32_t j = 0; j < pipes.size(); ++j)
    {
      for (uint32_t i = 0; i < 2; ++i) {
        if (player_position.x < pipes[j].shape[i].get_position().x + 2 * player_size.x && j + 1 < pipes.size())
        {
          if (i == 0) {
            return (pipes[j + 1].shape[i].get_position().y + pipes[j + 1].shape[i].get_size().y + gap_size) - player_position.y;
          }
          else {
            return (pipes[j + 1].shape[i].get_position().y - pipes[j + 1].shape[i].get_size().y - gap_size) - player_position.y;
          }
        }
      }
    }

    return 0;
  }

  bool do_ai_stuff()
  {
    std::vector<std::vector<float>> neural_network(3);

    neural_network[0].resize(TOTAL_INPUT_NODES);
    neural_network[1].resize(TOTAL_HIDDEN_NODES, 0);
    neural_network[2].resize(TOTAL_OUTPUT_NODES, 0);

    //Input layer.
    neural_network[0][0] = bcol.GetObject_Velocity(oid)[0];
    neural_network[0][1] = get_gap_difference();

    bool b = false;
    //Here we're getting the direction of the first 2 pipes, that are in front of the bird.
    for (auto& pipe : pipes)
    {
      for (uint32_t i = 0; i < 2; ++i) {
        if (shape.get_position().x < pipe.shape[i].get_position().x + player_size.x * 2)
        {
          neural_network[0][2] = i; // direction maybe flipped?
          b = true;
          break;
        }
      }
      if (b) {
        break;
      }
    }

    //I heard that smart people use matrices in a neural network.
    //But I'm not one of them.
    for (unsigned char a = 0; a < neural_network.size() - 1; a++)
    {
      for (unsigned char b = 0; b < neural_network[1 + a].size(); b++)
      {
        for (unsigned char c = 0; c < neural_network[a].size(); c++)
        {
          neural_network[1 + a][b] += neural_network[a][c] * weights[a][c][b];
        }

        if (0 >= neural_network[1 + a][b])
        {
          neural_network[1 + a][b] = pow<float>(2, neural_network[1 + a][b]) - 1;
        }
        else
        {
          neural_network[1 + a][b] = 1 - pow<float>(2, -neural_network[1 + a][b]);
        }
      }
    }

    return 0 <= neural_network[2][0];
  }


  void loop() {
    if (dead) {
      fitness = 0;
      bcol.SetObject_Velocity(oid, fan::vec2(initial_speed, 0));
      bcol.SetObject_Position(oid, 0);
      dead = false;
    }
  }

  bool operator>(bird_t& i_bird)
  {
    return fitness > i_bird.fitness;
  }

  bool operator<(bird_t& i_bird)
  {
    return fitness < i_bird.fitness;
  }

  void crossover(std::mt19937_64& i_random_engine, const std::vector<std::vector<std::vector<float>>>& i_bird_0_weights, const std::vector<std::vector<std::vector<float>>>& i_bird_1_weights)
  {

    for (unsigned char a = 0; a < weights.size(); a++)
    {
      for (unsigned char b = 0; b < weights[a].size(); b++)
      {
        for (unsigned char c = 0; c < weights[a][b].size(); c++)
        {
          if (0 == rand() % 2)
          {
            weights[a][b][c] = i_bird_0_weights[a][b][c];
          }
          else
          {
            weights[a][b][c] = i_bird_1_weights[a][b][c];
          }

          if (0 == mutation_distribution(i_random_engine))
          {
            weights[a][b][c] = node_distribution(i_random_engine);
          }
        }
      }
    }
  }
};

std::vector<bird_t> birds;
int main() {

  {
    BCOL_t::OpenProperties_t OpenProperties;

    bcol.Open(OpenProperties);
    bcol.PreSolve_Shape_cb =
      [](
        BCOL_t* bcol,
        const BCOL_t::ShapeInfoPack_t* sip0,
        const BCOL_t::ShapeInfoPack_t* sip1,
        BCOL_t::Contact_Shape_t* Contact
        ) {
          auto ed = bcol->GetObjectExtraData(sip0->ObjectID);
          bcol->Contact_Shape_DisableContact(Contact);
          if (!ed->IsItPipe && sip1->ShapeEnum == BCOL_t::ShapeEnum_t::Rectangle) {
            for (uint32_t i = 0; i < birds.size(); ++i) {
              if (birds[i].oid == sip0->ObjectID) {
                birds[i].dead = true;
                break;
              }
            }
          }
      };
  }

  // pointer can change for ed->ptr
  birds.resize(6);

  for (uint32_t i = 0; i < birds.size(); ++i) {
    birds[i].init();
  }

  f32_t last_pipe_x = 1;

  //pile->loco.window.add_keys_callback([&](const auto& data) {
  //  if (data.key != fan::key_space) {
  //    return;
  //  }
  //  if (data.state != fan::keyboard_state::press) {
  //    return;
  //  }
  //  bcol.SetObject_Velocity(oid, fan::vec2(0.5, -1));
  //});

  uint32_t generation = 0;

  f32_t score = 0;
  fan::time::clock c;
  c.start(fan::time::nanoseconds(5e+8));

  pile->loco.loop([&] {


    {
      const f32_t BCOLDeltaMax = 2;

      {
        auto d = pile->loco.delta_time;
        BCOLDelta += d;
      }

      if (BCOLDelta > BCOLDeltaMax) {
        BCOLDelta = BCOLDeltaMax;
      }

      while (BCOLDelta >= BCOLStepTime) {
        bcol.Step(BCOLStepTime);

        BCOLDelta -= BCOLStepTime;
      }
    }

    bool restart = true;

    for (uint32_t i = 0; i < birds.size(); ++i) {
      if (!birds[i].dead) {
        restart = false;
        break;
      }
    }

    if (restart == 0) {
      for (uint32_t i = 0; i < birds.size(); ++i) {
        if (!birds[i].dead) {
          birds[i].fitness = bcol.GetObject_Position(birds[i].oid)[0];

          if (birds[i].do_ai_stuff()) {
            f32_t y = bcol.GetObject_Velocity(birds[i].oid)[1];
            y = std::max(y, -10.f);
            bcol.SetObject_Velocity(birds[i].oid, fan::vec2(initial_speed, -1));
          }
        }
      }
    }
    else {

      ++generation;

      static std::random_device device;
      static std::mt19937_64 random(device());

      std::sort(birds.begin(), birds.end(), std::greater());

      for (auto a = 2 + birds.begin(); a != birds.end(); a++)
      {
        a->crossover(random, birds[0].weights, birds[1].weights);
      }

      for (uint32_t i = 0; i < birds.size(); ++i) {
        score = std::max(score, birds[i].fitness);
        birds[i].loop();
      }

      last_pipe_x = 1;
      for (uint32_t i = 0; i < pipes.size(); ++i) {
        bcol.UnlinkObject(pipes[i].oid[0]);
        bcol.RecycleObject(pipes[i].oid[0]);
        bcol.UnlinkObject(pipes[i].oid[1]);
        bcol.RecycleObject(pipes[i].oid[1]);
      }
      pipes.clear();
    }

    if (c.finished()) {
      fan::print(generation, score);
      c.restart();
    }
    
    uint32_t index = 0;
    for (uint32_t i = 0; i < birds.size(); ++i) {
      if (!birds[i].dead) {
        index = i;
        break;
      }
    }

    fan::vec2 player_position = bcol.GetObject_Position(birds[index].oid);

    gloco->camera_set_position(pile->camera, player_position);

    for (uint32_t i = 0; i < birds.size(); ++i) {
      fan::vec2 player_position = bcol.GetObject_Position(birds[i].oid);
      birds[i].shape.set_position(player_position);
    }

    if (last_pipe_x < player_position.x + 3 + pipe_size.x) {
      pipes.resize(pipes.size() + 1);
      pipes[pipes.size() - 1].push_back(last_pipe_x);

      last_pipe_x += .9;
    }
    for (uint32_t i = 0; i < pipes.size(); ++i) {
      if (pipes[i].shape[0].get_position().x < player_position.x - 1) {
        bcol.UnlinkObject(pipes[i].oid[0]);
        bcol.RecycleObject(pipes[i].oid[0]);
        bcol.UnlinkObject(pipes[i].oid[1]);
        bcol.RecycleObject(pipes[i].oid[1]);
        pipes.erase(pipes.begin() + i);
        continue;
      }
    }
    });

  return 0;
}