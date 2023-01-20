#include <WITCH/WITCH.h>
#include _WITCH_PATH(/PR/PR.h)

#include <fan/types/types.h>
#include <fan/math/random.h>

#define loco_opengl
#define loco_window
#define loco_context
#define loco_sprite
#define loco_rectangle
#define loco_circle
#define loco_letter
#define loco_text
#include _FAN_PATH(graphics/loco.h)

constexpr const fan::vec2i DefaultResolution = fan::vec2i(640, 640);

namespace stage {
  enum {
    idle,
    create_vertice_vector,
    create_rectangle,
    create_circle,
    resizing_rectangle,
    resizing_circle,
    moving_shape,
    focused_to_shape,
    MovingShape,
    ResizingShape
  };
}

namespace KeyModifierEnum {
  enum {
    LeftControl = 0x01,
    LeftShift = 0x02,
    LeftAlt = 0x04
  };
}

struct pile_t {
  loco_t loco;

  fan::graphics::viewport_t viewport;
  loco_t::matrices_t matrices;

  loco_t::texturepack TP;
  loco_t::texturepack::ti_t TP_ti;

  fan::graphics::cid_t SpriteCID;

  #define BLL_set_prefix CIDList
  #define BLL_set_Language 1
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_NodeDataType fan::graphics::cid_t
  #define BLL_set_StoreFormat 1
  #include _WITCH_PATH(BLL/BLL.h)
  CIDList_t CIDList[3];

  fan::vec2i MousePosition = DefaultResolution / 2;
  uint32_t KeyModifier;

  uint8_t stage = stage::idle;
  union {
    struct {
      fan::graphics::cid_t* cid;
      fan::vec2i start;
    }resizing_rectangle;
    struct {
      fan::graphics::cid_t* cid;
      fan::vec2i start;
    }resizing_circle;
    struct {
      uint32_t shape;
      fan::graphics::cid_t* cid;
      fan::vec2i grabbed_from;
    }moving_shape;
    struct {
      uint32_t shape;
      fan::graphics::cid_t* cid;
    }focused_to_shape;
    struct {
      uint32_t ShapeType;
      fan::graphics::cid_t* cid;
    }MovingShape;
    struct {
      uint32_t ShapeType;
      fan::graphics::cid_t* cid;
    }ResizingShape;
  }stage_data;
}*pile;

void open_pile() {
  pile = new pile_t;

  fan::vec2i WindowSize = pile->loco.get_window()->get_size();

  pile->viewport.open(pile->loco.get_context());
  pile->viewport.set(pile->loco.get_context(), 0, WindowSize, WindowSize);

  pile->matrices.open(&pile->loco);
  pile->matrices.set_ortho(&pile->loco, fan::vec2(0, WindowSize.x), fan::vec2(0, WindowSize.y));

  pile->loco.get_window()->add_resize_callback([](const fan::window_t::resize_cb_data_t& data) {
    pile->viewport.set(
      pile->loco.get_context(),
      0,
      data.size,
      data.size);
  pile->matrices.set_ortho(
    &pile->loco,
    fan::vec2(0, data.size.x),
    fan::vec2(0, data.size.y));
    });


  pile->KeyModifier = 0;
}
void close_pile() {

}

f32_t GetImagePixelSize() {
  fan::vec2 BlockResolution = pile->TP_ti.size;
  return pile->loco.sprite.get(&pile->SpriteCID, &loco_t::sprite_t::vi_t::size).x * 2 / BlockResolution.x;
}

void PrintCurrentStage() {
  switch (pile->stage) {
  case stage::idle: {
    fan::print("current stage: idle");
    break;
  }
  case stage::focused_to_shape: {
    fan::print("current stage: FocusedToShape");
    break;
  }
  case stage::MovingShape: {
    fan::print("current stage: MovingShape");
    break;
  }
  case stage::ResizingShape: {
    fan::print("current stage: ResizingShape");
    break;
  }
  }
}

void ChangeStage(uint8_t NewStage) {
  pile->stage = NewStage;
  PrintCurrentStage();
}

fan::color get_random_color() {
  fan::vec3 r = fan::random::vec3(0, 1);
  return fan::color(r.x, r.y, r.z, 0.5);
}

fan::vec2 get_position_of_shape(uint32_t shape, fan::graphics::cid_t* cid) {
  switch (shape) {
  case 0: {
    return fan::vec2(0, 0);
  }
  case 1: {
    return pile->loco.rectangle.get(cid, &loco_t::rectangle_t::vi_t::position);
  }
  case 2: {
    return pile->loco.circle.get(cid, &loco_t::circle_t::vi_t::position);
  }
  }
  return fan::vec2(0, 0);
}
void set_position_of_shape(uint32_t shape, fan::graphics::cid_t* cid, fan::vec2 position) {
  switch (shape) {
  case 0: {
    break;
  }
  case 1: {
    pile->loco.rectangle.set(cid, &loco_t::rectangle_t::vi_t::position, position);
    break;
  }
  case 2: {
    pile->loco.circle.set(cid, &loco_t::circle_t::vi_t::position, position);
    break;
  }
  }
}
void remove_shape(uint32_t shape, fan::graphics::cid_t* cid) {
  switch (shape) {
  case 0: {
    break;
  }
  case 1: {
    pile->loco.rectangle.erase(cid);
    break;
  }
  case 2: {
    pile->loco.circle.erase(cid);
    break;
  }
  }
}

bool is_point_inside_shape(fan::vec2 position, uint32_t shape, fan::graphics::cid_t* cid) {
  switch (shape) {
  case 0: {
    return 0;
  }
  case 1: {
    fan::vec2 rp = pile->loco.rectangle.get(cid, &loco_t::rectangle_t::vi_t::position);
    fan::vec2 sp = pile->loco.rectangle.get(cid, &loco_t::rectangle_t::vi_t::size);
    return fan_2d::collision::rectangle::point_inside_no_rotation(position, rp, sp);
  }
  case 2: {
    fan::vec2 cp = pile->loco.circle.get(cid, &loco_t::circle_t::vi_t::position);
    f32_t cr = pile->loco.circle.get(cid, &loco_t::circle_t::vi_t::radius);
    return fan_2d::collision::circle::point_inside(position, cp, cr);
  }
  }
  return 0;
}
uint32_t _shape_end() {
  return 3;
}
void find_touch(uint32_t* p_shape, fan::graphics::cid_t** p_cid) {
  for (uint32_t shape = 0; shape < _shape_end(); shape++) {
    auto nr = pile->CIDList[shape].GetNodeFirst();
    while (nr != pile->CIDList[shape].dst) {
      if (is_point_inside_shape(pile->MousePosition, shape, &pile->CIDList[shape][nr])) {
        *p_shape = shape;
        *p_cid = &pile->CIDList[shape][nr];
        return;
      }
      nr = nr.Next(&pile->CIDList[shape]);
    }
  }
  *p_shape = (uint32_t)-1;
}
void DuplicateShape(uint8_t shape, fan::graphics::cid_t* cid) {
  switch (shape) {
  case 0: {
    break;
  }
  case 1: {
    fan::vec2 position = pile->loco.rectangle.get(cid, &loco_t::rectangle_t::vi_t::position);
    fan::vec2 size = pile->loco.rectangle.get(cid, &loco_t::rectangle_t::vi_t::size);
    loco_t::rectangle_t::properties_t properties;
    properties.position = position;
    properties.position.z = 1;
    properties.size = size;
    properties.color = get_random_color();
    properties.viewport = &pile->viewport;
    properties.matrices = &pile->matrices;
    auto nr = pile->CIDList[1].NewNodeLast();
    pile->loco.rectangle.push_back(&pile->CIDList[1][nr], properties);
    break;
  }
  case 2: {
    fan::vec2 position = pile->loco.circle.get(cid, &loco_t::circle_t::vi_t::position);
    f32_t radius = pile->loco.circle.get(cid, &loco_t::circle_t::vi_t::radius);
    loco_t::circle_t::properties_t properties;
    properties.position = position;
    properties.position.z = 1;
    properties.radius = radius;
    properties.color = get_random_color();
    properties.viewport = &pile->viewport;
    properties.matrices = &pile->matrices;
    auto nr = pile->CIDList[2].NewNodeLast();
    pile->loco.circle.push_back(&pile->CIDList[2][nr], properties);
    break;
  }
  }
}

void resize_rectangle() {
  fan::vec2i start = pile->stage_data.resizing_rectangle.start;
  fan::vec2 size = pile->MousePosition - start;
  pile->loco.rectangle.set(pile->stage_data.resizing_rectangle.cid, &loco_t::rectangle_t::vi_t::size, size);
}
void resize_circle() {
  fan::vec2i start = pile->stage_data.resizing_circle.start;
  fan::vec2 size = pile->MousePosition - start;
  pile->loco.circle.set(pile->stage_data.resizing_circle.cid, &loco_t::circle_t::vi_t::radius, size.length());
}

namespace MoveShapeToDirectionMode {
  enum {
    Pixel = 0,
    ImagePixel = 1
  };
}
void MoveShapeToDirection(uint8_t ShapeType, fan::graphics::cid_t* cid, uint8_t Mode, fan::vec2 Direction) {
  f32_t Unit;
  switch (Mode) {
  case MoveShapeToDirectionMode::Pixel: {
    Unit = 1;
    break;
  }
  case MoveShapeToDirectionMode::ImagePixel: {
    Unit = GetImagePixelSize();
    break;
  }
  }
  fan::vec2 Position = get_position_of_shape(ShapeType, cid);
  Position += Direction * Unit;
  Position -= Position % Unit;
  set_position_of_shape(ShapeType, cid, Position);
}

int main(int argc, char** argv) {
  if (argc != 3) {
    fan::print("usage:");
    fan::print("./exe <TexturePack> <qti name>");
    return 0;
  }

  open_pile();

  {
    pile->TP.open_compiled(&pile->loco, argv[1]);
    auto r = pile->TP.qti(argv[2], &pile->TP_ti);
    if (r != 0) {
      fan::print("failed to find <qti name>");
      return 0;
    }
  }

  {
    fan::vec2i WindowSize = pile->loco.get_window()->get_size();
    loco_t::sprite_t::properties_t properties;
    properties.load_tp(&pile->TP, &pile->TP_ti);
    properties.position = WindowSize / 2;
    properties.position.z = 0;
    properties.size = WindowSize / 2;
    properties.viewport = &pile->viewport;
    properties.matrices = &pile->matrices;
    pile->loco.sprite.push_back(&pile->SpriteCID, properties);
  }

  pile->loco.get_window()->add_keys_callback([](const fan::window_t::keyboard_keys_cb_data_t& data) {
    if (data.key >= fan::input::key_1 && data.key <= fan::input::key_9) {
      if (data.state != fan::keyboard_state::press) {
        return;
      }
      switch (data.key) {
      case fan::input::key_1: {
        ChangeStage(stage::idle);
        break;
      }
      case fan::input::key_2: {
        ChangeStage(stage::create_vertice_vector);
        break;
      }
      case fan::input::key_3: {
        ChangeStage(stage::create_rectangle);
        break;
      }
      case fan::input::key_4: {
        ChangeStage(stage::create_circle);
        break;
      }
      }
      return;
    }
  if (data.key == fan::input::key_a) {
    if (data.state != fan::keyboard_state::press) {
      return;
    }
  }
  if (data.key == fan::input::key_t) {
    if (data.state != fan::keyboard_state::press) {
      return;
    }
    switch (pile->stage) {
    case stage::focused_to_shape: {
      remove_shape(pile->stage_data.focused_to_shape.shape, pile->stage_data.focused_to_shape.cid);
      ChangeStage(stage::idle);
      break;
    }
    }
    return;
  }
  if (data.key == fan::input::key_d) {
    switch (pile->stage) {
    case stage::focused_to_shape: {
      if (data.state != fan::keyboard_state::press) {
        return;
      }
      DuplicateShape(pile->stage_data.focused_to_shape.shape, pile->stage_data.focused_to_shape.cid);
      break;
    }
    case stage::ResizingShape: {
      if (data.state != fan::keyboard_state::press) {
        return;
      }
      uint32_t ShapeType = pile->stage_data.ResizingShape.ShapeType;
      auto cid = pile->stage_data.ResizingShape.cid;
      switch (ShapeType) {
      case 0: {
        fan::throw_error("e");
        break;
      }
      case 1: {
        fan::throw_error("e");
        break;
      }
      case 2: {
        f32_t Size = pile->loco.circle.get(cid, &loco_t::circle_t::vi_t::radius);
        {
          Size /= GetImagePixelSize();
          uint32_t s = Size;
          --s;
          uint8_t leftest = __clz32(s);
          s = 0x80000000 >> leftest;
          Size = s;
          Size *= GetImagePixelSize();
        }
        pile->loco.circle.set(cid, &loco_t::circle_t::vi_t::radius, Size);
        break;
      }
      }
      break;
    }
    }
  }
  if (data.key == fan::input::key_f) {
    switch (pile->stage) {
    case stage::ResizingShape: {
      if (data.state != fan::keyboard_state::press) {
        return;
      }
      uint32_t ShapeType = pile->stage_data.ResizingShape.ShapeType;
      auto cid = pile->stage_data.ResizingShape.cid;
      switch (ShapeType) {
      case 0: {
        fan::throw_error("e");
        break;
      }
      case 1: {
        fan::throw_error("e");
        break;
      }
      case 2: {
        f32_t Size = pile->loco.circle.get(cid, &loco_t::circle_t::vi_t::radius);
        {
          Size /= GetImagePixelSize();
          uint32_t s = Size;
          uint8_t leftest = __clz32(s) - 1;
          s = 0x80000000 >> leftest;
          Size = s;
          Size *= GetImagePixelSize();
        }
        pile->loco.circle.set(cid, &loco_t::circle_t::vi_t::radius, Size);
        break;
      }
      }
      break;
    }
    }
  }
  if (data.key >= fan::input::key_f1 && data.key <= fan::input::key_f4) {
    if (data.state != fan::keyboard_state::press) {
      return;
    }
    if (pile->stage != stage::focused_to_shape) {
      return;
    }
    f32_t scale = data.key - fan::input::key_f1 + 1;
    fan::vec2 block_res = pile->TP_ti.size * scale;
    fan::vec2 block_size = pile->loco.sprite.get(&pile->SpriteCID, &loco_t::sprite_t::vi_t::size) / block_res;
    switch (pile->stage_data.focused_to_shape.shape) {
    case 0: {
      return;
    }
    case 1: {
      fan::vec2 position = pile->loco.rectangle.get(pile->stage_data.focused_to_shape.cid, &loco_t::rectangle_t::vi_t::position);
      position.x = (sint32_t)((position.x + block_size.x / 2) / block_size.x);
      position.x *= block_size.x;
      position.y = (sint32_t)((position.y + block_size.y / 2) / block_size.y);
      position.y *= block_size.y;
      pile->loco.rectangle.set(pile->stage_data.focused_to_shape.cid, &loco_t::rectangle_t::vi_t::position, position);
      break;
    }
    case 2: {
      fan::vec2 position = pile->loco.circle.get(pile->stage_data.focused_to_shape.cid, &loco_t::circle_t::vi_t::position);
      position.x = (sint32_t)((position.x + block_size.x / 2) / block_size.x);
      position.x *= block_size.x;
      position.y = (sint32_t)((position.y + block_size.y / 2) / block_size.y);
      position.y *= block_size.y;
      pile->loco.circle.set(pile->stage_data.focused_to_shape.cid, &loco_t::circle_t::vi_t::position, position);
      break;
    }
    }
    return;
  }
  if (data.key == fan::input::key_o) {
    if (data.state != fan::keyboard_state::press) {
      return;
    }
    std::ofstream f;
    f.open("data");
    {
      uint32_t usage = 0;
      f.write((const char*)&usage, sizeof(uint32_t));
    }
    {
      uint32_t usage = pile->CIDList[1].Usage();
      f.write((const char*)&usage, sizeof(uint32_t));
      auto nr = pile->CIDList[1].GetNodeFirst();
      while (nr != pile->CIDList[1].dst) {
        auto cid = &pile->CIDList[1][nr];
        fan::vec2 position = pile->loco.rectangle.get(cid, &loco_t::rectangle_t::vi_t::position);
        position -= pile->loco.sprite.get(&pile->SpriteCID, &loco_t::sprite_t::vi_t::size);
        position /= GetImagePixelSize();
        f.write((const char*)&position, sizeof(fan::vec2));
        fan::vec2 size = pile->loco.rectangle.get(cid, &loco_t::rectangle_t::vi_t::size);
        size /= GetImagePixelSize();
        f.write((const char*)&size, sizeof(fan::vec2));
        nr.Next(&pile->CIDList[1]);
      }
    }
    {
      uint32_t usage = pile->CIDList[2].Usage();
      f.write((const char*)&usage, sizeof(uint32_t));
      auto nr = pile->CIDList[2].GetNodeFirst();
      while (nr != pile->CIDList[2].dst) {
        auto cid = &pile->CIDList[2][nr];
        fan::vec2 position = pile->loco.circle.get(cid, &loco_t::circle_t::vi_t::position);
        position -= pile->loco.sprite.get(&pile->SpriteCID, &loco_t::sprite_t::vi_t::size);
        position /= GetImagePixelSize();
        f.write((const char*)&position, sizeof(fan::vec2));
        f32_t radius = pile->loco.circle.get(cid, &loco_t::circle_t::vi_t::radius);
        radius /= GetImagePixelSize();
        f.write((const char*)&radius, sizeof(f32_t));
        nr.Next(&pile->CIDList[2]);
      }
    }
    f.close();
  }
  if (data.key == fan::input::key_i) {
    if (data.state != fan::keyboard_state::press) {
      return;
    }
    std::ifstream f;
    f.open("data");
    {
      uint32_t usage;
      f.read((char*)&usage, sizeof(uint32_t));
    }
    {
      uint32_t usage;
      f.read((char*)&usage, sizeof(uint32_t));
      for (uint32_t i = 0; i < usage; i++) {
        fan::vec2 position;
        f.read((char*)&position, sizeof(fan::vec2));
        position *= GetImagePixelSize();
        position += pile->loco.sprite.get(&pile->SpriteCID, &loco_t::sprite_t::vi_t::size);
        fan::vec2 size;
        f.read((char*)&size, sizeof(fan::vec2));
        size *= GetImagePixelSize();

        loco_t::rectangle_t::properties_t properties;
        properties.position = position;
        properties.position.z = 1;
        properties.size = size;
        properties.color = get_random_color();
        properties.viewport = &pile->viewport;
        properties.matrices = &pile->matrices;
        auto nr = pile->CIDList[1].NewNodeLast();
        pile->loco.rectangle.push_back(&pile->CIDList[1][nr], properties);
      }
    }
    {
      uint32_t usage;
      f.read((char*)&usage, sizeof(uint32_t));
      for (uint32_t i = 0; i < usage; i++) {
        fan::vec2 position;
        f.read((char*)&position, sizeof(fan::vec2));
        position *= GetImagePixelSize();
        position += pile->loco.sprite.get(&pile->SpriteCID, &loco_t::sprite_t::vi_t::size);
        f32_t radius;
        f.read((char*)&radius, sizeof(f32_t));
        radius *= GetImagePixelSize();

        loco_t::circle_t::properties_t properties;
        properties.position = position;
        properties.position.z = 1;
        properties.radius = radius;
        properties.color = get_random_color();
        properties.viewport = &pile->viewport;
        properties.matrices = &pile->matrices;
        auto nr = pile->CIDList[2].NewNodeLast();
        pile->loco.circle.push_back(&pile->CIDList[2][nr], properties);
      }
    }
    f.close();
  }
  if (data.key == fan::input::key_q) {
    if (data.state != fan::keyboard_state::press) {
      return;
    }
    if (pile->stage != stage::focused_to_shape) {
      return;
    }
    ChangeStage(stage::MovingShape);
  }
  if (data.key == fan::input::key_w) {
    if (data.state != fan::keyboard_state::press) {
      return;
    }
    if (pile->stage != stage::focused_to_shape) {
      return;
    }
    ChangeStage(stage::ResizingShape);
  }
  if (data.key == fan::input::key_k) {
    if (data.state != fan::keyboard_state::press) {
      return;
    }
    if (pile->stage != stage::focused_to_shape) {
      return;
    }
    uint8_t ShapeType = pile->stage_data.focused_to_shape.shape;
    auto cid = pile->stage_data.focused_to_shape.cid;
    switch (ShapeType) {
    case 0: {
      break;
    }
    case 1: {
      pile->loco.rectangle.set(cid, &loco_t::rectangle_t::vi_t::color, get_random_color());
      break;
    }
    case 2: {
      pile->loco.circle.set(cid, &loco_t::circle_t::vi_t::color, get_random_color());
      break;
    }
    }
  }
  if (data.key == fan::input::key_left_control) {
    if (data.state == fan::keyboard_state::press) {
      pile->KeyModifier |= KeyModifierEnum::LeftControl;
    }
    else if (data.state == fan::keyboard_state::release) {
      pile->KeyModifier ^= KeyModifierEnum::LeftControl;
    }
  }
  if (
    data.key == fan::input::key_up ||
    data.key == fan::input::key_down ||
    data.key == fan::input::key_left ||
    data.key == fan::input::key_right
    ) {
    if (data.state == fan::keyboard_state::release) {
      return;
    }
    switch (pile->stage) {
    case stage::MovingShape: {
      fan::vec2 Direction;
      if (data.key == fan::input::key_up) Direction = fan::vec2(0, -1);
      if (data.key == fan::input::key_down) Direction = fan::vec2(0, 1);
      if (data.key == fan::input::key_left) Direction = fan::vec2(-1, 0);
      if (data.key == fan::input::key_right) Direction = fan::vec2(1, 0);
      uint8_t ShapeType = pile->stage_data.MovingShape.ShapeType;
      auto cid = pile->stage_data.MovingShape.cid;
      uint8_t Mode;
      if (pile->KeyModifier & KeyModifierEnum::LeftControl) {
        Mode = MoveShapeToDirectionMode::ImagePixel;
      }
      else {
        Mode = MoveShapeToDirectionMode::Pixel;
      }
      MoveShapeToDirection(ShapeType, cid, Mode, Direction);
      break;
    }
    case stage::ResizingShape: {
      uint32_t ShapeType = pile->stage_data.ResizingShape.ShapeType;
      auto cid = pile->stage_data.ResizingShape.cid;
      switch (ShapeType) {
      case 0: {
        fan::throw_error("e");
        break;
      }
      case 1: {
        fan::throw_error("e");
        break;
      }
      case 2: {
        f32_t Size = pile->loco.circle.get(cid, &loco_t::circle_t::vi_t::radius);

        f32_t Direction;
        if (data.key == fan::input::key_up) { Direction = -1; }
        if (data.key == fan::input::key_down) { Direction = 1; }

        if (pile->KeyModifier & KeyModifierEnum::LeftControl) {
          Direction *= GetImagePixelSize();
        }

        Size += Direction;
        Size -= fmod(Size, Direction);
        pile->loco.circle.set(cid, &loco_t::circle_t::vi_t::radius, Size);
        break;
      }
      }
      break;
    }
    }
  }
  if (data.key == fan::input::key_tab) {
    if (data.state == fan::keyboard_state::release) {
      return;
    }

    if (pile->KeyModifier & KeyModifierEnum::LeftControl) {
      ChangeStage(stage::idle);
      return;
    }
    switch (pile->stage) {
    case stage::idle: {
      return;
    }
    case stage::focused_to_shape: {
      ChangeStage(stage::idle);
      return;
    }
    case stage::MovingShape: {
      ChangeStage(stage::focused_to_shape);
      return;
    }
    case stage::ResizingShape: {
      ChangeStage(stage::focused_to_shape);
      break;
    }
    }
  }
    });
  pile->loco.get_window()->add_buttons_callback([](const fan::window_t::mouse_buttons_cb_data_t& data) {
    if (data.button == fan::input::mouse_left) {
    mouse_left_switch_label:
      switch (pile->stage) {
      case stage::idle: {
        if (data.state != fan::mouse_state::press) {
          return;
        }
        uint32_t shape;
        fan::graphics::cid_t* cid;
        find_touch(&shape, &cid);
        if (shape == (uint32_t)-1) {
          return;
        }
        ChangeStage(stage::moving_shape);
        pile->stage_data.moving_shape.shape = shape;
        pile->stage_data.moving_shape.cid = cid;
        fan::vec2 shape_position = get_position_of_shape(shape, cid);
        pile->stage_data.moving_shape.grabbed_from.x = pile->MousePosition.x - shape_position.x;
        pile->stage_data.moving_shape.grabbed_from.y = pile->MousePosition.y - shape_position.y;
        break;
      }
      case stage::create_rectangle: {
        if (data.state != fan::mouse_state::press) {
          return;
        }
        loco_t::rectangle_t::properties_t properties;
        properties.position = pile->MousePosition;
        properties.position.z = 1;
        properties.size = fan::vec2(0, 0);
        properties.color = get_random_color();
        properties.viewport = &pile->viewport;
        properties.matrices = &pile->matrices;
        auto nr = pile->CIDList[1].NewNodeLast();
        pile->loco.rectangle.push_back(&pile->CIDList[1][nr], properties);
        ChangeStage(stage::resizing_rectangle);
        pile->stage_data.resizing_rectangle.start = pile->MousePosition;
        pile->stage_data.resizing_rectangle.cid = &pile->CIDList[1][nr];
        break;
      }
      case stage::create_circle: {
        if (data.state != fan::mouse_state::press) {
          return;
        }
        loco_t::circle_t::properties_t properties;
        properties.position = pile->MousePosition;
        properties.position.z = 1;
        properties.radius = 5;
        properties.color = get_random_color();
        properties.viewport = &pile->viewport;
        properties.matrices = &pile->matrices;
        auto nr = pile->CIDList[2].NewNodeLast();
        pile->loco.circle.push_back(&pile->CIDList[2][nr], properties);
        ChangeStage(stage::resizing_circle);
        pile->stage_data.resizing_circle.start = pile->MousePosition;
        pile->stage_data.resizing_circle.cid = &pile->CIDList[2][nr];
        break;
      }
      case stage::resizing_rectangle: {
        ChangeStage(stage::create_rectangle);
        break;
      }
      case stage::resizing_circle: {
        ChangeStage(stage::create_circle);
        break;
      }
      case stage::moving_shape: {
        if (data.state != fan::mouse_state::release) {
          return;
        }
        ChangeStage(stage::focused_to_shape);
        break;
      }
      case stage::focused_to_shape: {
        ChangeStage(stage::idle);
        goto mouse_left_switch_label;
      }
      }
    }
  if (data.button == fan::input::mouse_right) {

  }
    });
  pile->loco.get_window()->add_mouse_move_callback([](const fan::window_t::mouse_move_cb_data_t& data) {
    pile->MousePosition = data.position;
  switch (pile->stage) {
  case stage::resizing_rectangle: {
    resize_rectangle();
    break;
  }
  case stage::resizing_circle: {
    resize_circle();
    break;
  }
  case stage::moving_shape: {
    fan::vec2 grabbed_from = pile->stage_data.moving_shape.grabbed_from;
    fan::vec2 position = pile->MousePosition - grabbed_from;
    set_position_of_shape(pile->stage_data.moving_shape.shape, pile->stage_data.moving_shape.cid, position);
    break;
  }
  }
    });

  pile->loco.loop([] {});

  return 0;
}
