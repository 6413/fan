#include <fan/pch.h>

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>
#include _INCLUDE_TOKEN(WITCH_INCLUDE_PATH,WITCH.h)

constexpr static f32_t BCOLStepTime = 0.01;
#define ETC_BCOL_set_prefix BCOL
#define ETC_BCOL_set_DynamicDeltaFunction \
  ObjectData0->Velocity.y += delta * 2;
#define ETC_BCOL_set_StoreExtraDataInsideObject 1
#define ETC_BCOL_set_ExtraDataInsideObject \
  bool IsItPipe;
#include _WITCH_PATH(BCOL/BCOL.h)

struct pile_t {

  static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
  static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

  pile_t() {
    fan::vec2 window_size = loco.window.get_size();
    loco.open_camera(&camera, ortho_x, ortho_y);
    loco.open_viewport(&viewport, 0, window_size);
  }

  loco_t loco;
  loco_t::camera_t camera;
  fan::graphics::viewport_t viewport;
};

bool dead = false;

pile_t* pile = new pile_t;
f32_t BCOLDelta = 0;

BCOL_t bcol;

static constexpr fan::vec2 pipe_size = fan::vec2(0.1, 100);

int main() {

  {
    BCOL_t::OpenProperties_t OpenProperties;

    OpenProperties.PreSolve_Shape_cb =
      [](
        BCOL_t* bcol,
        const BCOL_t::ShapeInfoPack_t* sip0,
        const BCOL_t::ShapeInfoPack_t* sip1,
        BCOL_t::Contact_Shape_t* Contact
        ) {
          auto ed = bcol->GetObjectExtraData(sip0->ObjectID);
          if (!ed->IsItPipe) {
            bcol->Contact_Shape_DisableContact(Contact);
            dead = true;
          }
      };

    bcol.Open(&OpenProperties);
  }


  static constexpr auto gap_size = 1;

  //fan::graphics::shape_t pipes[1000];
  struct pipe_t {
    BCOL_t::ObjectID_t oid[2];
    fan::graphics::shape_t shape[2];
    void push_back(f32_t xpos) {

      f32_t r = fan::random::value_f32(0.5, 1);

      for (uint32_t i = 0; i < 2; ++i) {

        BCOL_t::ObjectProperties_t p;
        p.Position = xpos;
        p.Position.y = -1 - pipe_size.y + r;
        if (i == 1) {
          p.Position.y += pipe_size.y * 2 + gap_size;
        }

        p.ExtraData.IsItPipe = true;
        oid[i] = bcol.NewObject(&p, BCOL_t::ObjectFlag::Constant);

        BCOL_t::ShapeProperties_Rectangle_t sp;
        sp.Position = 0;
        sp.Size = pipe_size;
        bcol.NewShape_Rectangle(oid[i], &sp);

        loco_t::shapes_t::rectangle_t::properties_t pipe_properties;

        pipe_properties.position = p.Position;
        pipe_properties.size = sp.Size;
        pipe_properties.camera = &pile->camera;
        pipe_properties.viewport = &pile->viewport;
        pipe_properties.color = fan::colors::green;

        shape[i] = pipe_properties;
      }
    }
  };

  std::vector<pipe_t> pipes;

  struct {
    BCOL_t::ObjectID_t oid;
    fan::graphics::shape_t shape;
  }player;

  {
    BCOL_t::ObjectProperties_t p;
    p.Position = 0;
    p.ExtraData.IsItPipe = false;
    player.oid = bcol.NewObject(&p, 0);
    bcol.SetObject_Velocity(player.oid, fan::vec2(.5, 0));

    BCOL_t::ShapeProperties_Circle_t sp;
    sp.Position = 0;
    sp.Size = 0.05;
    bcol.NewShape_Circle(player.oid, &sp);

    loco_t::shapes_t::rectangle_t::properties_t player_properties;
    player_properties.position.x = bcol.GetObject_Position(player.oid).x;
    player_properties.position.y = bcol.GetObject_Position(player.oid).y;
    player_properties.position.z = 1;
    player_properties.size = sp.Size;
    player_properties.camera = &pile->camera;
    player_properties.viewport = &pile->viewport;
    player_properties.color = fan::colors::yellow;

    player.shape = player_properties;
  }

  f32_t last_pipe_x = bcol.GetObject_Position(player.oid).x + 1;

  pile->loco.window.add_keys_callback([&](const auto& data) {
    if (data.key != fan::key_space) {
      return;
    }
    if (data.state != fan::keyboard_state::press) {
      return;
    }
    bcol.SetObject_Velocity(player.oid, fan::vec2(0.5, -1));
    });

  pile->loco.loop([&] {
    pile->loco.get_fps();


    {
      const f32_t BCOLDeltaMax = 2;

      {
        auto d = pile->loco.get_delta_time();
        BCOLDelta += d;
      }

      if (BCOLDelta > BCOLDeltaMax) {
        BCOLDelta = BCOLDeltaMax;
      }

      while (BCOLDelta >= BCOLStepTime) {
        bcol.Step(BCOLStepTime);

        if (dead) {
          bcol.SetObject_Velocity(player.oid, fan::vec2(.5, 0));
          bcol.SetObject_Position(player.oid, 0);
          last_pipe_x = bcol.GetObject_Position(player.oid).x + 1;
          dead = false;
        }

        BCOLDelta -= BCOLStepTime;
      }
    }

    fan::vec2 player_position = bcol.GetObject_Position(player.oid);

    pile->camera.set_position(player_position);

    player.shape.set_position(player_position);


    if (last_pipe_x < player_position.x + 2 + pipe_size.x) {
      pipes.resize(pipes.size() + 1);
      pipes[pipes.size() - 1].push_back(last_pipe_x);

      last_pipe_x += 1;
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