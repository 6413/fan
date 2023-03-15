using Stage_t = stage2_t;

uint8_t KeyTable[0x100];
uint8_t KeyTableIndex = 0;
void key_add(uint8_t key){
  for(uint8_t i = 0; i < KeyTableIndex; i++){
    if(KeyTable[i] == key){
      return;
    }
  }
  KeyTable[KeyTableIndex++] = key;
}
void key_remove(uint8_t key){
  for(uint8_t i = 0; i < KeyTableIndex; i++){
    if(KeyTable[i] == key){
      std::memmove(&KeyTable[i], &KeyTable[i + 1], KeyTableIndex - i);
      KeyTableIndex--;
      return;
    }
  }
}

struct WorldMatrix_t{
  f32_t DownLimitHard = 0.125;
  f32_t UpLimitHard = 1;
  f32_t Scale = this->UpLimitHard;
  f32_t Velocity = 0;
  loco_t::camera_t Camera;
}WorldMatrix;

fan_init_id_t0(
  loco_t::vfi,
  vfiBaseID,
  .mouse_button_cb = [Stage = this](const loco_t::vfi_t::mouse_button_data_t& data){
    if(
      data.button == fan::input::mouse_scroll_up ||
      data.button == fan::input::mouse_scroll_down
    ){
      f32_t a;
      if(data.button == fan::input::mouse_scroll_up) { a = -0.4; }
      if(data.button == fan::input::mouse_scroll_down) { a = +0.4; }
      Stage->WorldMatrix.Velocity += a;
    }
    else{
      if(data.button_state == fan::mouse_state::press){
        data.flag->ignore_move_focus_check = true;
        data.vfi->set_focus_keyboard(data.vfi->get_focus_mouse());
      }
      if(data.button_state == fan::mouse_state::release){
        data.flag->ignore_move_focus_check = false;
      }
    }
    return 0;
  },
  .mouse_move_cb = [](const loco_t::vfi_t::mouse_move_data_t& data){return 0;},
  .keyboard_cb = [Stage = this](const loco_t::vfi_t::keyboard_data_t& data){
    switch(data.key){
      case fan::input::key_escape:{
        /* TODO
        if(data.state == fan::keyboard_state::press && !(game::pile->stage_data.sortie.ReservedFlags & 0x1)){
          game::pile->stage = game::stage_group::sortie_menu;
          // game::stage::sortie_menu::open(); TODO
        }
        else {
          game::pile->stage_data.sortie.ReservedFlags &= ~0x1;
        }
        */
        break;
      }
      case fan::input::key_left:
      case fan::input::key_right:
      case fan::input::key_up:
      case fan::input::key_down:
      case fan::input::key_a:
      case fan::input::key_d:
      case fan::input::key_w:
      case fan::input::key_s:
      {
        if(data.keyboard_state == fan::keyboard_state::press){
          Stage->key_add(data.key);
        }
        else if(data.keyboard_state == fan::keyboard_state::release){
          Stage->key_remove(data.key);
        }
        break;
      }
    }
    return 0;
  },
  .shape_type = loco_t::vfi_t::shape_t::always,
  .shape.always.z = 0
);

BCOL_t bcol;
uint64_t BCOLStepNumber = 0;
f32_t BCOLDelta = 0;

//loco_t::sprite_id_t BackgroundCID;

struct sprite_wrap_t {
  loco_t::id_t cid;
  engine::tp_t::ti_t ti;
};

std::vector<sprite_wrap_t> parallax_background;
//loco_t::sprite_id_t Background_0xfCID;

// uint32_t SunLightIndex; TODO

struct EntityIdentify_t{
  enum{
    FloatingText,
    FloatingDirtPiece,
    Spark,
    Smoke,
    ship,
    block,
    DirtWrap,
    building_FuelStation,
    building_MineralProcess,
    building_JunkShop,
    building_EmendationStation
  };
};
engine::EntityList_t EntityList;

struct DepthList_t{
  uint16_t Base(){
    return OFFSETLESS(this, Stage_t, DepthList)->it * 100 + 1;
  }
  uint16_t Background = Base();
  uint16_t Block = Base() + 0x7f + 0;
  uint16_t BlockDirtWrap = Base() + 0x7f + 1;
  uint16_t Ship = Base() + 0x7f + 2;
  uint16_t OverMining = Base() + 0x7f + 3;
  uint16_t WorldGUI0 = Base() + 0x7f + 4;
}DepthList;

struct{
  uint8_t Rotate : 2, BlockID : 6;
}BlockMap[constants::PlayGround_Size.y][constants::PlayGround_Size.x];
engine::EntityID_t BlockMap_EntityID[constants::PlayGround_Size.y][constants::PlayGround_Size.x];

engine::EntityID_t EntityShipID = (engine::EntityID_t)-1;

struct{
  engine::audio_t::piece_t ambient_0{"audio/ambient/0"};
  engine::audio_t::pieces_t dong{"audio/effect/dong_"};
  engine::audio_t::pieces_t drill{"audio/effect/drill_", ShipUnit::UnitAltCount[(uint32_t)ShipUnit::UnitEnum::Drill]};
  engine::audio_t::pieces_t Engine{"audio/effect/engine_", ShipUnit::UnitAltCount[(uint32_t)ShipUnit::UnitEnum::Engine]};
}audio_pieces;

struct{
  fan::vec2i i_position;
  fan::vec2i i_size;
}rendered_at;

f32_t GetWindowSizeReverseScalar(const fan::vec2 Size){
  fan::vec2 tSize = Size;
  if(tSize.x > 1408){
    tSize *= 1408 / tSize.x;
  }
  if(tSize.y > 792){
    tSize *= 792 / tSize.y;
  }
  return tSize.x / Size.x;
}
fan::vec2 GetWindowSizeToMultipler(fan::vec2 Size){
  Size *= this->WorldMatrix.Scale;
  return Size * GetWindowSizeReverseScalar(Size);
}

fan::vec2 GetWorldMatrixMultipler(fan::vec2 Size){
  return this->GetWindowSizeToMultipler(Size);
}
fan::vec2 GetWorldMatrixMultipler(){
  return GetWorldMatrixMultipler(engine::loco.get_window()->get_size());
}

fan::vec2 GetWorldMatrixPosition() {
  return this->WorldMatrix.Camera.get_camera_position();
}

bool viewable_blocks_do_we_need_redraw(){
  fan::vec2 ms = this->GetWorldMatrixMultipler();
  fan::vec2 f_position = this->GetWorldMatrixPosition();
  fan::vec2i i_position = (f_position - ms) / constants::stage::sortie::BlockSize;
  fan::vec2i i_size = (f_position + ms) / constants::stage::sortie::BlockSize + 1;
  if (i_position != this->rendered_at.i_position) {
    return 1;
  }
  if (i_size != this->rendered_at.i_size) {
    return 1;
  }
  return 0;
}
void clear_viewable_blocks(){
  auto reference = this->EntityList.Begin();
  while(reference != this->EntityList.End()){
    auto next_reference = this->EntityList.Iterate(reference);
    auto IdentifyingAs = this->EntityList.Get(reference)->Behaviour->IdentifyingAs;
    if (
      IdentifyingAs == EntityIdentify_t::block ||
      IdentifyingAs == EntityIdentify_t::DirtWrap
    ){
      this->EntityList.ForceRemove(reference);
    }
    reference = next_reference;
  }
}
void write_viewable_blocks() {
  fan::vec2 ms = this->GetWorldMatrixMultipler();
  fan::vec2 f_position = this->GetWorldMatrixPosition();
  fan::vec2i i_position = (f_position - ms) / constants::stage::sortie::BlockSize;
  fan::vec2i i_size = (f_position + ms) / constants::stage::sortie::BlockSize + 1;
  this->rendered_at.i_position = i_position;
  this->rendered_at.i_size = i_size;
  i_position.x = fan::clamp(i_position.x, 0, constants::PlayGround_Size.x);
  i_position.y = fan::clamp(i_position.y, 0, constants::PlayGround_Size.y);
  i_size.x = fan::clamp(i_size.x, 0, constants::PlayGround_Size.x);
  i_size.y = fan::clamp(i_size.y, 0, constants::PlayGround_Size.y);

  for (uint32_t y = i_position.y; y < i_size.y; y++) {
    for (uint32_t x = i_position.x; x < i_size.x; x++) {
      auto BlockType = this->BlockMap[y][x].BlockID;
      if(BlockType == game::BlockList.AN(&game::BlockList_t::Empty)){
        EntityScope.DirtWrap.add(fan::vec2i(x, y));
      }
      else{
        EntityScope.block.add(
          BlockType,
          fan::vec2i(x, y),
          this->BlockMap[y][x].Rotate);
      }
    }
  }
}

void SetWorldMatrixPosition(fan::vec2 Position){
  this->WorldMatrix.Camera.set_camera_position(Position);
  if (viewable_blocks_do_we_need_redraw()) {
    clear_viewable_blocks();
    write_viewable_blocks();
  }
}
void FocusCamera(fan::vec2 Position, f32_t delta){
  fan::vec2 CameraPosition = this->GetWorldMatrixPosition();
  fan::vec2 ms = this->GetWorldMatrixMultipler() / 2;
  if(delta > (f32_t)1 / 3){
    delta = (f32_t)1 / 3;
  }
  CameraPosition += (Position - CameraPosition) * (delta * 3);
  if(CameraPosition.x - ms.x < 0){
    CameraPosition.x = ms.x;
  }
  else if(CameraPosition.x + ms.x > game::constants::PlayGround_Size.x * game::constants::stage::sortie::BlockSize){
    CameraPosition.x = game::constants::PlayGround_Size.x * game::constants::stage::sortie::BlockSize - ms.x;
  }
  if(CameraPosition.y + ms.y > game::constants::PlayGround_Size.y * game::constants::stage::sortie::BlockSize){
    CameraPosition.y = game::constants::PlayGround_Size.y * game::constants::stage::sortie::BlockSize - ms.y;
  }

  this->SetWorldMatrixPosition(CameraPosition);
}
void FocusCamera_Rough(fan::vec2 Position){
  fan::vec2 CameraPosition = Position;
  fan::vec2 ms = this->GetWorldMatrixMultipler() / 2;
  if(CameraPosition.x - ms.x < 0){
    CameraPosition.x = ms.x;
  }
  else if(CameraPosition.x + ms.x > game::constants::PlayGround_Size.x * game::constants::stage::sortie::BlockSize){
    CameraPosition.x = game::constants::PlayGround_Size.x * game::constants::stage::sortie::BlockSize - ms.x;
  }
  if(CameraPosition.y + ms.y > game::constants::PlayGround_Size.y * game::constants::stage::sortie::BlockSize){
    CameraPosition.y = game::constants::PlayGround_Size.y * game::constants::stage::sortie::BlockSize - ms.y;
  }

  this->SetWorldMatrixPosition(CameraPosition);
}

void UpdateWorldMatrix(fan::vec2 Size){
  {
    f32_t s = this->GetWindowSizeReverseScalar(Size);
    this->WorldMatrix.UpLimitHard = s;
    fan::print(s);
  }
  fan::vec2 m = this->GetWindowSizeToMultipler(Size) / 2;

  this->WorldMatrix.Camera.set_ortho(&engine::loco,
    fan::vec2(-m.x, +m.x),
    fan::vec2(-m.y, +m.y));
}
void UpdateWorldMatrix(){
  UpdateWorldMatrix(engine::loco.get_window()->get_size());
}

fan::color get_level_color_front(uint32_t level){
  if(level < 64){
    f32_t base = 1;
    base -= (f32_t)level / 64 * .25;
    return fan::color(base, base, base);
  }
  else if(level < 96){
    return fan::color(.75, .75, .75);
  }
  else if(level < 192){
    f32_t scale = sin((f32_t)(level - 96) / 96 * fan::math::pi);
    f32_t red = .75 + scale * .25;
    return fan::color(red, .75, .75);
  }
  else if(level < 224){
    return fan::color(.75, .75, .75);
  }
  else if(level < 320){
    f32_t scale = sin((f32_t)(level - 224) / 96 * fan::math::pi);
    f32_t half_scale = sin((f32_t)(level - 224) / 96 * fan::math::pi / 2);
    f32_t base_brightness = .75 - half_scale * .25;
    f32_t green = base_brightness + scale * .25;
    f32_t rest = base_brightness;
    return fan::color(rest, green, rest);
  }
  else if(level < 352){
    return fan::color(.5, .5, .5);
  }
  else if(level < 512){
    f32_t scale = sin((f32_t)(level - 352) / 160 * fan::math::pi / 2);
    f32_t red = .5 + scale * .5;
    return fan::color(red, .5, .5);
  }
  return fan::color(0, 0, 0);
}
fan::color get_level_color_back(uint32_t level){
  if(level < 16){
    f32_t base = 0.75;
    base -= (f32_t)level / 16 * .65;
    return fan::color(base, base, base);
  }
  else if(level < 96){
    return fan::color(.375, .375, .375);
  }
  else if(level < 192){
    f32_t scale = sin((f32_t)(level - 96) / 96 * fan::math::pi);
    f32_t red = .375 + scale * .125;
    return fan::color(red, .375, .375);
  }
  else if(level < 224){
    return fan::color(.375, .375, .375);
  }
  else if(level < 320){
    f32_t scale = sin((f32_t)(level - 224) / 96 * fan::math::pi);
    f32_t half_scale = sin((f32_t)(level - 224) / 96 * fan::math::pi / 2);
    f32_t base_brightness = .375 - half_scale * .125;
    f32_t green = base_brightness + scale * .125;
    f32_t rest = base_brightness;
    return fan::color(rest, green, rest);
  }
  else if(level < 352){
    return fan::color(.25, .25, .25);
  }
  else if(level < 512){
    f32_t scale = sin((f32_t)(level - 352) / 160 * fan::math::pi / 2);
    f32_t red = .25 + scale * .25;
    return fan::color(red, .25, .25);
  }
  return fan::color(0, 0, 0);
}

bool is_block_mineable(fan::vec2i Position){
  uint8_t BlockType = this->BlockMap[Position.y][Position.x].BlockID;
  switch(BlockType){
    case game::BlockList.AN(&game::BlockList_t::Empty):{
      return false;
    }
    case game::BlockList.AN(&game::BlockList_t::Grass):{
      if(
        Position.x >= BuildingInfo::FuelStation::Position.x &&
        Position.x < BuildingInfo::FuelStation::Position.x + 2
      ){
        return false;
      }
      else if(
        Position.x >= BuildingInfo::MineralProcess::Position.x &&
        Position.x < BuildingInfo::MineralProcess::Position.x + 4
      ){
        return false;
      }
      else if(
        Position.x >= BuildingInfo::JunkShop::Position.x &&
        Position.x < BuildingInfo::JunkShop::Position.x + 3
      ){
        return false;
      }
      else if(
        Position.x >= BuildingInfo::EmendationStation::Position.x &&
        Position.x < BuildingInfo::EmendationStation::Position.x + 2
      ){
        return false;
      }
      else{
        return true;
      }
    }
    default:{
      return true;
    }
  }
}
bool is_block_collectable(fan::vec2i Position){
  auto BlockType = this->BlockMap[Position.y][Position.x].BlockID;
  switch(BlockType){
    case game::BlockList.AN(&game::BlockList_t::Empty):
    case game::BlockList.AN(&game::BlockList_t::Grass):
    case game::BlockList.AN(&game::BlockList_t::Dirt):
    {
      return false;
    }
    default:{
      return true;
    }
  }
}
fan::vec2i get_grid_position(fan::vec2 Position){
  fan::vec2i i_position = Position / game::constants::stage::sortie::BlockSize;
  if(Position.y < 0){
    i_position.y--;
  }
  if(Position.x < 0){
    i_position.x--;
  }
  return i_position;
}

bool is_EntityIdentify_Building(uint32_t IdentifyingAs){
  switch(IdentifyingAs){
    case EntityIdentify_t::building_FuelStation:
    case EntityIdentify_t::building_MineralProcess:
    case EntityIdentify_t::building_JunkShop:
    case EntityIdentify_t::building_EmendationStation:
    {
      return true;
    }
    default:{
      return false;
    }
  }
}

struct EntityScope_t{
  #define EntityStructBegin(name) \
    struct CONCAT2(name,_t){ \
      Stage_t *Stage(){ \
        return OFFSETLESS(this, Stage_t, EntityScope.name); \
      }
  #define EntityStructEnd(name) \
    }name;
  /*
  #include "stage2/entity/building_FuelStation/types.h"
  #include "stage2/entity/building_MineralProcess/types.h"
  #include "stage2/entity/building_JunkShop/types.h"
  #include "stage2/entity/building_EmendationStation/types.h"
  */

  #include "stage2/entity/FloatingText/FloatingText.h"
  #include "stage2/entity/FloatingDirtPiece/FloatingDirtPiece.h"
  #include "stage2/entity/Spark/Spark.h"
  #include "stage2/entity/Smoke/Smoke.h"
  #include "stage2/entity/ship/ship.h"
  #include "stage2/entity/block/block.h"
  #include "stage2/entity/DirtWrap/DirtWrap.h"
  #include "stage2/entity/building_FuelStation/building_FuelStation.h"
  //#include "stage2/entity/building_MineralProcess/building_MineralProcess.h"
  #include "stage2/entity/building_JunkShop/building_JunkShop.h"
  //#include "stage2/entity/building_EmendationStation/building_EmendationStation.h"
}EntityScope;

void _ImportHM2CS_err(const char* path, uint32_t FC, sint64_t err) {
  engine::log.write(
    engine::log_t::LogType_t::Error,
    "game::sortie::ImportHM2CS\n  path %s FC %02lx native error %016llx\n",
    path, FC, err);
}
void ImportHM2CS(const char* path, BCOL_t::CompiledShapes_t *CompiledShapes) {
  sint32_t err;
  FS_file_t file;
  err = FS_file_open(path, &file, O_RDONLY);
  if (err != 0) {
    _ImportHM2CS_err(path, 0, err);
    PR_exit(1);
  }
  IO_stat_t s;
  IO_fd_t file_fd;
  FS_file_getfd(&file, &file_fd);
  err = IO_fstat(&file_fd, &s);
  if (err != 0) {
    FS_file_close(&file);
    _ImportHM2CS_err(path, 1, err);
    PR_exit(1);
  }
  IO_off_t Size = IO_stat_GetSizeInBytes(&s);
  uint8_t* Data = A_resize(0, Size);
  FS_ssize_t ReadSize = FS_file_read(&file, Data, Size);
  if (ReadSize != Size) {
    A_resize(Data, 0);
    FS_file_close(&file);
    _ImportHM2CS_err(path, 2, ReadSize);
    PR_exit(1);
  }
  FS_file_close(&file);
  this->bcol.CompiledShapes_open(CompiledShapes);
  this->bcol.ImportHM(Data, Size, CompiledShapes);
  A_resize(Data, 0);
}

void pause(){
  engine::audio.PauseGroup((uint32_t)SoundGroups_t::sortie);
}
void resume(){
  engine::audio.ResumeGroup((uint32_t)SoundGroups_t::sortie);
}

void InitializeBCOL() {
  BCOL_t::OpenProperties_t OpenProperties;

  OpenProperties.GridBlockSize = constants::stage::sortie::BlockSize;
  OpenProperties.PreSolve_Grid_cb =
    [](
      BCOL_t *bcol,
      BCOL_t::ObjectID_t ObjectID,
      BCOL_t::ShapeEnum_t ShapeEnum,
      BCOL_t::ShapeID_t ShapeID,
      fan::_vec2<sint32_t> Grid,
      BCOL_t::Contact_Grid_t *Contact
    ){
      auto Stage = OFFSETLESS(bcol, Stage_t, bcol);

      if ((uint32_t)Grid.x >= constants::PlayGround_Size.x) {
        bcol->Contact_Grid_EnableContact(Contact);
        return;
      }
      if (Grid.y >= constants::PlayGround_Size.y) {
        bcol->Contact_Grid_EnableContact(Contact);
        return;
      }
      if (Grid.y < 0) {
        bcol->Contact_Grid_DisableContact(Contact);
        return;
      }

      if(Stage->BlockMap[Grid.y][Grid.x].BlockID == game::BlockList.AN(&game::BlockList_t::Empty)) {
        bcol->Contact_Grid_DisableContact(Contact);
      }
      else {
        auto ObjectData = bcol->GetObjectExtraData(ObjectID);
        auto Entity = Stage->EntityList.Get(ObjectData->EntityID);
        if(Entity->Behaviour->IdentifyingAs == EntityIdentify_t::ship){
          auto EntityShipData = (EntityScope_t::ship_t::EntityData_t *)Entity->UserPTR;

          fan::vec2i ShipGrid = Stage->EntityScope.ship.get_grid_position();
          fan::vec2i GridDiff = Grid - ShipGrid;
          if(fan::math::abs(GridDiff.x) > 0 && GridDiff.y == 0){
            if(SIGN(EntityShipData->UserWantedDirection.x) == SIGN(GridDiff.x)){
              EntityShipData->StageData.Idle.WantedToDrillGrid = GridDiff;
            }
          }
          else if(GridDiff.x == 0 && fan::math::abs(GridDiff.y) > 0){
            if(SIGN(EntityShipData->UserWantedDirection.y) == SIGN(GridDiff.y)){
              EntityShipData->StageData.Idle.WantedToDrillGrid = GridDiff;
            }
          }
          bcol->Contact_Grid_EnableContact(Contact);
        }
        else{
          bcol->Contact_Grid_EnableContact(Contact);
        }
      }
    };
  OpenProperties.PostSolve_Grid_cb =
    [](
      BCOL_t* bcol,
      BCOL_t::ObjectID_t ObjectID,
      BCOL_t::ShapeEnum_t ShapeEnum,
      BCOL_t::ShapeID_t ShapeID,
      fan::_vec2<sint32_t> Grid,
      BCOL_t::ContactResult_Grid_t* ContactResult
    ){
      auto Stage = OFFSETLESS(bcol, Stage_t, bcol);

      auto ObjectData = bcol->GetObjectExtraData(ObjectID);
      auto Entity = Stage->EntityList.Get(ObjectData->EntityID);
      switch(Entity->Behaviour->IdentifyingAs) {
        case EntityIdentify_t::ship:{
          auto EntityShipData = (EntityScope_t::ship_t::EntityData_t *)Entity->UserPTR;
          auto Normal = bcol->ContactResult_Grid_GetNormal(ContactResult);

          /* enough? */
          if (Normal.y < -0.25) {
            EntityShipData->StageData.Idle.TouchedGroundAt = Stage->BCOLStepNumber;
          }

          break;
        }
        default: {
          break;
        }
      }
    };

  OpenProperties.PreSolve_Shape_cb =
    [](
      BCOL_t *bcol,
      BCOL_t::ObjectID_t ObjectID0,
      BCOL_t::ShapeEnum_t ShapeEnum0,
      BCOL_t::ShapeID_t ShapeID0,
      BCOL_t::ObjectID_t ObjectID1,
      BCOL_t::ShapeEnum_t ShapeEnum1,
      BCOL_t::ShapeID_t ShapeID1,
      BCOL_t::Contact_Shape_t *Contact
    ){
      auto Stage = OFFSETLESS(bcol, Stage_t, bcol);

      auto EntityID0 = bcol->GetObjectExtraData(ObjectID0)->EntityID;
      auto EntityID1 = bcol->GetObjectExtraData(ObjectID1)->EntityID;
      auto Entity0 = Stage->EntityList.Get(EntityID0);
      auto Entity1 = Stage->EntityList.Get(EntityID1);
      switch(Entity0->Behaviour->IdentifyingAs){
        case EntityIdentify_t::ship:{
          if (Stage->is_EntityIdentify_Building(Entity1->Behaviour->IdentifyingAs)) {
            bcol->Contact_Shape_DisableContact(Contact);

            auto EntityData = (EntityScope_t::ship_t::EntityData_t *)Entity0->UserPTR;
            uint64_t BCOLStepNumber = Stage->BCOLStepNumber;

            /* neccassary if ship hitbox model has more than one shape */
            if (BCOLStepNumber == EntityData->StageData.Idle.TouchedBuildingAt) {
              return;
            }

            if (BCOLStepNumber - 1 == EntityData->StageData.Idle.TouchedBuildingAt) {
              EntityData->StageData.Idle.TouchedBuildingAt = BCOLStepNumber;
              return;
            }

            EntityData->StageData.Idle.TouchedBuildingAt = BCOLStepNumber;
            switch (Entity1->Behaviour->IdentifyingAs) {
              case EntityIdentify_t::building_FuelStation: {
                PR_abort();
                // game::pile->stage = game::stage_group::sortie_in_building_FuelStation;
                // sortie_in_building_FuelStation::open();
                break;
              }
              case EntityIdentify_t::building_MineralProcess: {
                PR_abort();
                // game::pile->stage = game::stage_group::sortie_in_building_MineralProcess;
                // sortie_in_building_MineralProcess::open();
                break;
              }
              case EntityIdentify_t::building_JunkShop: {
                PR_abort();
                // game::pile->stage = game::stage_group::sortie_in_building_JunkShop;
                // sortie_in_building_JunkShop::open();
                break;
              }
              case EntityIdentify_t::building_EmendationStation: {
                PR_abort();
                // game::pile->stage = game::stage_group::sortie_in_building_EmendationStation;
                // sortie_in_building_EmendationStation::open();
                break;
              }
            }
          }
          else {
            bcol->Contact_Shape_EnableContact(Contact);
          }
          break;
        }
        default: {
          bcol->Contact_Shape_EnableContact(Contact);
          break;
        }
      }
    };

  this->bcol.Open(&OpenProperties);
}

fan::graphics::cid_t LightCids[10];

void open(auto& loco) {
  engine::loco.vfi.set_focus_keyboard(*vfiBaseID);

  // game::pile->stage_data.sortie.image_light.create(&game::pile->engine.context, 0, 0); TODO

  {
    this->WorldMatrix.Camera.open(&engine::loco);
    UpdateWorldMatrix();
  }

  engine::loco.lighting.ambient = 0;
  {
    loco_t::light_sun_t::properties_t p;
    p.camera = &this->WorldMatrix.Camera;
    p.viewport = &game::pile->viewport;
    p.size = fan::vec2(10000, 10000);
    p.color = fan::color(0.7, 0.9, 0.9, 1) * 1.5;
    //p.type = 1;
    p.position = fan::vec3(400, -400, 0);
    engine::loco.light_sun.push_back(&LightCids[0], p);
  }

  {
    //loco_t::sprite_t::properties_t sp;
    //sp.camera = &this->WorldMatrix.Camera;
    //sp.viewport = &game::pile->viewport;
    //sp.position = fan::vec3(-0.5, 0.5, 0);
    //sp.size = fan::vec2(10000, 10000);
    //static loco_t::image_t image;
    //image.create(&engine::loco, fan::colors::white, 1);
    //sp.image = &image;

    //engine::loco.sprite.push_back(&LightCids[1], sp);

    //loco_t::light_t::properties_t p;
    //p.camera = &this->WorldMatrix.Camera;
    //p.viewport = &game::pile->viewport;
    //p.position = fan::vec3(1000, -1000, 0);
    //p.size = fan::vec2(200, 100);
    ////p.color = fan::color(1, 1, 0.5, 1);
    //p.type = 0;
    //engine::loco.light.push_back(&LightCids[2], p);
  }

  {
    loco_t::light_t::properties_t p;
    p.camera = &this->WorldMatrix.Camera;
    p.viewport = &game::pile->viewport;
    p.position = fan::vec3(400, 0, 0);
    p.size = fan::vec2(200, 200);
    p.color = fan::color(1, 0.7, 0, 1);
    p.type = 0;
    engine::loco.light.push_back(&LightCids[3], p);
  }

  {
    loco_t::light_t::properties_t p;
    p.camera = &this->WorldMatrix.Camera;
    p.viewport = &game::pile->viewport;
    p.position = fan::vec3(400, 0, 0);
    p.size = fan::vec2(10, 10);
    p.color = fan::color(0.09, 0.28, 0.33, 1);
    p.type = 0;
    engine::loco.light.push_back(&LightCids[4], p);
  }

  InitializeBCOL();

  #include "stage2/GeneratePlayground.h"

  this->EntityScope.ship.add(fan::vec2(400, -100));

  EntityScope.building_FuelStation.add(BuildingInfo::FuelStation::Position);
  EntityScope.building_JunkShop.add(BuildingInfo::JunkShop::Position);

  this->write_viewable_blocks();

  // this->EntityScope.ship.FocusCamera_Rough(); TODO need initial camera set

  {
    parallax_background.resize(10);
    stage_loader_t::stage_open_properties_t op;
    op.camera = &this->WorldMatrix.Camera;
    op.viewport = &game::pile->viewport;
    op.theme = &game::pile->gui_theme;
    op.itToDepthMultiplier = 0;
    game::pile->StageList._StageLoader.push_and_open_stage<game::pile_t::StageList_t::stage_loader_t::stage::stage_parallax_t>(op);
    engine::tp.qti("background", &parallax_background[0].ti);
    engine::tp.qti("background1", &parallax_background[1].ti);
    engine::tp.qti("background2", &parallax_background[2].ti);
    engine::tp.qti("background3", &parallax_background[3].ti);

    f32_t t = (game::constants::stage::sortie::BlockSize * game::constants::PlayGround_Size.x) / parallax_background[0].ti.size.x;
    fan::vec2 s = fan::vec2(parallax_background[0].ti.size * t) / 2;
    //fan::vec2 s(1088.00000, 136);
    //s *= 10;
    fan::vec2 p;
    p.x = game::constants::PlayGround_Size.x * game::constants::stage::sortie::BlockSize / 2;
    //p.x = 0;
    p.y = -s.y;
    // put down
   // p.y += s.y / 2;

  /*  parallax_background[0].cid[{
      .position = {p, this->DepthList.Background },
      .parallax_factor = .9,
      .size = s,
      .matrices = &this->WorldMatrix.Matrices,
      .viewport = &game::pile->viewport,
      .ti = &parallax_background[0].ti
    }];*/

    //p.x = game::constants::PlayGround_Size.x * game::constants::stage::sortie::BlockSize / 2;
    //p.y = -s.y;
    // put down
   // p.y += s.y / 2;
    //p.x = 400;
   /* parallax_background[1].cid[{
      .position = { p, this->DepthList.Background + 1},
        .parallax_factor = .7,
        .size = s,
        .matrices = &this->WorldMatrix.Matrices,
        .viewport = &game::pile->viewport,
        .ti = &parallax_background[1].ti
    }];*/

  /*  parallax_background[2].cid[{
      .position = { p, this->DepthList.Background + 2 },
        .parallax_factor = .5,
        .size = s,
        .matrices = &this->WorldMatrix.Matrices,
        .viewport = &game::pile->viewport,
        .ti = &parallax_background[3].ti
    }];
    parallax_background[3].cid[{
      .position = { p, this->DepthList.Background + 3 },
        .parallax_factor = .2,
        .size = s,
        .matrices = &this->WorldMatrix.Matrices,
        .viewport = &game::pile->viewport,
        .ti = &parallax_background[2].ti
    }];*/
  }

  // InitializeLighting(); TODO
}

void close(auto& loco){
  // game::pile->stage_data.sortie.light.close(); TODO

  engine::audio.StopGroup((uint32_t)game::SoundGroups_t::sortie);

  this->bcol.Close();
}

void window_resize_callback(auto& loco){
  this->UpdateWorldMatrix();
}

void update(auto& loco){
  const f32_t StepTime = 0.01;
  const f32_t BCOLDeltaMax = 2;

  this->BCOLDelta += engine::loco.get_delta_time();

  if(this->BCOLDelta > BCOLDeltaMax){
    engine::log.write(engine::log_t::LogType_t::Warning, "game stage sortie BCOLDelta is above BCOLDeltaMax\n");
    this->BCOLDelta = BCOLDeltaMax;
  }

  while(this->BCOLDelta >= StepTime){
    if(MATH_abs_f32(this->WorldMatrix.Velocity) > 0.01){
      this->WorldMatrix.Scale += this->WorldMatrix.Velocity * StepTime;
      if(this->WorldMatrix.Scale < this->WorldMatrix.DownLimitHard){
        this->WorldMatrix.Scale = this->WorldMatrix.DownLimitHard;
      }
      else if(this->WorldMatrix.Scale > this->WorldMatrix.UpLimitHard){
        this->WorldMatrix.Scale = this->WorldMatrix.UpLimitHard;
      }
      this->WorldMatrix.Velocity /= StepTime * 4 + 1;
      this->UpdateWorldMatrix();
    }

    auto StepEnd = [&](){
      this->BCOLDelta -= StepTime;
      this->BCOLStepNumber++;
    };

    this->bcol.Step(StepTime);
    this->EntityList.Step(StepTime);

    StepEnd();
  }

  // light probably needs entity

  fan::vec2 ship_pos = EntityScope.ship.GetWorldPosition();

  engine::loco.light.set(&LightCids[3], &loco_t::light_t::vi_t::position, ship_pos);

  // dont look this bloat xd
  auto ShipEntity = EntityList.Get(EntityShipID);
  using ShipEntityData_t = Stage_t::EntityScope_t::ship_t::EntityData_t;
  auto ShipEntityData = (ShipEntityData_t*)ShipEntity->UserPTR;

  // smoke is the only cid
  static const fan::string smoke_id = "entity_smoke";

  auto& model_list = engine::_model_list_filler.model_list;

  const auto& instance = model_list.get_instance(ShipEntityData->ModelID.internal_id);

  ship_pos.x += instance.size.x / 6;
  ship_pos.y -= instance.size.y / 20;
  engine::loco.light.set(&LightCids[4], &loco_t::light_t::vi_t::position, ship_pos);
  static f32_t var = 0;
  static f32_t direction = 1;
  fan::color c = fan::color(0.09, 0.28, 0.33, 1) * 4;
  c.r *= var;
  c.g *= var;
  c.b *= var;
  engine::loco.light.set(&LightCids[4], &loco_t::light_t::vi_t::color, c);
  var += engine::loco.get_delta_time() * 1.25 * direction;
  if (var >= 1) {
    var = 1;
    direction = -1;
  }
  else if (var <= 0) {
    var = 0;
    direction = 1;
  }
}
