using Stage_t = stage2_t;

loco_t::vfi_t::shape_id_t vfiBaseID;

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

f32_t WorldMatrixMultipler;
f32_t WorldMatrixMultiplerVelocity;
loco_t::matrices_t world_matrices;

BCOL_t bcol;
uint64_t BCOLStepNumber = 0;
f32_t BCOLDelta = 0;

engine::TP::ti_t ti_background;
// hp and fuel images - bar and material
engine::TP::ti_t ti_ship_info[3];

uint32_t ShipLightIndex[2];
uint32_t ThrusterLightIndex;
uint32_t SunLightIndex;

// bit table
// 0 - will ignore escape key when exiting building
uint8_t ReservedFlags = 0;

struct EntityIdentify_t{
  enum{
    FloatingText,
    FloatingDirtPiece,
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

struct{
  uint8_t Rotate : 2, BlockID : 6;
}BlockMap[constants::PlayGround_Size.y][constants::PlayGround_Size.x];
uint32_t BlockMap_EntityID[constants::PlayGround_Size.y][constants::PlayGround_Size.x];

engine::EntityID_t EntityShipID = (engine::EntityID_t)-1;

struct{
  fan::audio::piece_t ambient_0;
  fan::audio::piece_t dong_0;
  fan::audio::piece_t dong_1;
  fan::audio::piece_t dong_2;
  fan::audio::piece_t drill[ShipUnit::UnitAltCount[(uint32_t)ShipUnit::UnitEnum::Drill]];
  fan::audio::piece_t Engine[ShipUnit::UnitAltCount[(uint32_t)ShipUnit::UnitEnum::Engine]];
}audio_pieces;

/* entityless cids */
struct{
  fan::opengl::cid_t Background;
}elcids;

struct{
  /*
    TODO
    should we assign i_position to 0 here
    we check at viewable_blocks_do_we_need_redraw:
    if (i_position != rendered_at.i_position) {
    before initializing
  */
  fan::vec2i i_position = 0;
  fan::vec2i i_size;
}rendered_at;

fan::vec2 GetWorldMatrixMultipler(fan::vec2 Size){
  return Size / Size.max() *
    constants::stage::sortie::BlockSize *
    this->WorldMatrixMultipler;
}
fan::vec2 GetWorldMatrixMultipler(){
  return GetWorldMatrixMultipler(game::pile->loco.get_window()->get_size());
}

fan::vec2 GetWorldMatrixPosition() {
  return this->world_matrices.get_camera_position();
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
      IdentifyingAs == game::entity_group::block ||
      IdentifyingAs == game::entity_group::DirtWrap
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
  if (i_position.x < 0) {
    i_position.x = 0;
  }
  else if (i_position.x >= constants::PlayGround_Size.x) {
    i_position.x = constants::PlayGround_Size.x;
  }
  if (i_position.y < 0) {
    i_position.y = 0;
  }
  else if (i_position.y >= constants::PlayGround_Size.y) {
    i_position.y = constants::PlayGround_Size.y;
  }
  if (i_size.x < 0) {
    i_size.x = 0;
  }
  else if (i_size.x >= constants::PlayGround_Size.x) {
    i_size.x = constants::PlayGround_Size.x;
  }
  if (i_size.y < 0) {
    i_size.y = 0;
  }
  else if (i_size.y >= constants::PlayGround_Size.y) {
    i_size.y = constants::PlayGround_Size.y;
  }
  for (uint32_t y = i_position.y; y < i_size.y; y++) {
    for (uint32_t x = i_position.x; x < i_size.x; x++) {
      uint8_t Rotate = this->BlockMap[y][x].Rotate;
      game::Blocks::MapEnum_t BlockType = (game::Blocks::MapEnum_t)this->BlockMap[y][x].BlockID;
      switch (BlockType) {
        case game::Blocks::MapEnum_t::empty: {
          // EntityScope.DirtWrap.add(fan::vec2i(x, y)); TODO
          continue;
        }
        case game::Blocks::MapEnum_t::grass:
        case game::Blocks::MapEnum_t::dirt:
        case game::Blocks::MapEnum_t::iron:
        case game::Blocks::MapEnum_t::bronze:
        case game::Blocks::MapEnum_t::silver:
        case game::Blocks::MapEnum_t::gold:
        case game::Blocks::MapEnum_t::platinum:
        case game::Blocks::MapEnum_t::einsteinium:
        case game::Blocks::MapEnum_t::emerald:
        case game::Blocks::MapEnum_t::ruby:
        case game::Blocks::MapEnum_t::diamond:
        {
          EntityScope.block.add(game::Blocks::MapEnumToEnum(BlockType), fan::vec2i(x, y), Rotate);
          break;
        }
      }
    }
  }
}

void SetWorldMatrixPosition(fan::vec2 Position){
  this->world_matrices.set_camera_position(Position);
  if (viewable_blocks_do_we_need_redraw()) {
    clear_viewable_blocks();
    write_viewable_blocks();
  }
}

void UpdateWorldMatrix(fan::vec2 Size){
  fan::vec2 m = Size / Size.max() * constants::stage::sortie::BlockSize * this->WorldMatrixMultipler;

  this->world_matrices.set_ortho(&game::pile->loco,
    fan::vec2(-m.x, +m.x),
    fan::vec2(-m.y, +m.y));
}
void UpdateWorldMatrix(){
  UpdateWorldMatrix(game::pile->loco.get_window()->get_size());
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
  game::Blocks::MapEnum_t BlockType = (game::Blocks::MapEnum_t)this->BlockMap[Position.y][Position.x].BlockID;
  switch(BlockType){
    case Blocks::MapEnum_t::empty:
    {
      return 0;
    }
    case Blocks::MapEnum_t::grass:
    {
      if(
        Position.x >= BuildingInfo::FuelStation::Position.x &&
        Position.x < BuildingInfo::FuelStation::Position.x + 2
      ){
        return 0;
      }
      else if(
        Position.x >= BuildingInfo::MineralProcess::Position.x &&
        Position.x < BuildingInfo::MineralProcess::Position.x + 4
      ){
        return 0;
      }
      else if(
        Position.x >= BuildingInfo::JunkShop::Position.x &&
        Position.x < BuildingInfo::JunkShop::Position.x + 3
      ){
        return 0;
      }
      else if(
        Position.x >= BuildingInfo::EmendationStation::Position.x &&
        Position.x < BuildingInfo::EmendationStation::Position.x + 2
      ){
        return 0;
      }
      else{
        return 1;
      }
    }
    case Blocks::MapEnum_t::dirt:
    case Blocks::MapEnum_t::iron:
    case Blocks::MapEnum_t::bronze:
    case Blocks::MapEnum_t::silver:
    case Blocks::MapEnum_t::gold:
    case Blocks::MapEnum_t::platinum:
    case Blocks::MapEnum_t::einsteinium:
    case Blocks::MapEnum_t::emerald:
    case Blocks::MapEnum_t::ruby:
    case Blocks::MapEnum_t::diamond:
    {
      return 1;
    }
  }
}
bool is_block_collectable(fan::vec2i Position){
  game::Blocks::MapEnum_t BlockType = (game::Blocks::MapEnum_t)this->BlockMap[Position.y][Position.x].BlockID;
  switch(BlockType){
    case Blocks::MapEnum_t::empty:
    case Blocks::MapEnum_t::grass:
    case Blocks::MapEnum_t::dirt:
    {
      return 0;
    }
    case Blocks::MapEnum_t::iron:
    case Blocks::MapEnum_t::bronze:
    case Blocks::MapEnum_t::silver:
    case Blocks::MapEnum_t::gold:
    case Blocks::MapEnum_t::platinum:
    case Blocks::MapEnum_t::einsteinium:
    case Blocks::MapEnum_t::emerald:
    case Blocks::MapEnum_t::ruby:
    case Blocks::MapEnum_t::diamond:
    {
      return 1;
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
    case EntityIdentify_t::FloatingText:
    case EntityIdentify_t::FloatingDirtPiece:
    case EntityIdentify_t::ship:
    case EntityIdentify_t::DirtWrap:
    case EntityIdentify_t::block:
    {
      return 0;
    }
    case EntityIdentify_t::building_FuelStation:
    case EntityIdentify_t::building_MineralProcess:
    case EntityIdentify_t::building_JunkShop:
    case EntityIdentify_t::building_EmendationStation:
    {
      return 1;
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
  #include "stage2/entity/FloatingText/types.h"
  #include "stage2/entity/FloatingDirtPiece/types.h"
  #include "stage2/entity/DirtWrap/types.h"
  #include "stage2/entity/building_FuelStation/types.h"
  #include "stage2/entity/building_MineralProcess/types.h"
  #include "stage2/entity/building_JunkShop/types.h"
  #include "stage2/entity/building_EmendationStation/types.h"
  */

  //#include "stage2/entity/FloatingText/FloatingText.h"
  //#include "stage2/entity/FloatingDirtPiece/FloatingDirtPiece.h"
  #include "stage2/entity/ship/ship.h"
  #include "stage2/entity/block/block.h"
  //#include "stage2/entity/DirtWrap/DirtWrap.h"
  //#include "stage2/entity/building_FuelStation/building_FuelStation.h"
  //#include "stage2/entity/building_MineralProcess/building_MineralProcess.h"
  //#include "stage2/entity/building_JunkShop/building_JunkShop.h"
  //#include "stage2/entity/building_EmendationStation/building_EmendationStation.h"
}EntityScope;

void _ImportHM2CS_err(const char* path, uint32_t FC, sint64_t err) {
  engine::log::write(
    engine::log::LogType_t::Error,
    "game::sortie::ImportHM2CS\n  path %s FC %02lx native error %016llx\n",
    path, FC, err);
}
void ImportHM2CS(const char* path, BCOL_CompiledShapes_t* CompiledShapes) {
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
  BCOL_CompiledShapes_open(&this->bcol, CompiledShapes);
  BCOL_ImportHM(&this->bcol, Data, Size, CompiledShapes);
  A_resize(Data, 0);
}

void pause(){
  fan::audio::audio_pause_group(&game::pile->audio, (uint32_t)SoundGroups_t::sortie);
}
void resume(){
  fan::audio::audio_resume_group(&game::pile->audio, (uint32_t)SoundGroups_t::sortie);
}

void InitializeBCOL() {
  BCOL_OpenProperties_t OpenProperties;

  OpenProperties.GridBlockSize = constants::stage::sortie::BlockSize;
  OpenProperties.PreSolve_Grid_cb =
    [](
      BCOL_t *bcol,
      BCOL_ObjectID_t ObjectID,
      uint8_t ShapeEnum,
      BCOL_ShapeID_t ShapeID,
      sint32_t GridY,
      sint32_t GridX,
      BCOL_Contact_Grid_t* Contact
    ){
      auto Stage = OFFSETLESS(bcol, Stage_t, bcol);

      if ((uint32_t)GridX >= constants::PlayGround_Size.x) {
        BCOL_Contact_Grid_EnableContact(Contact);
        return;
      }
      if (GridY >= constants::PlayGround_Size.y) {
        BCOL_Contact_Grid_EnableContact(Contact);
        return;
      }
      if (GridY < 0) {
        BCOL_Contact_Grid_DisableContact(Contact);
        return;
      }

      if(Stage->BlockMap[GridY][GridX].BlockID == (uint8_t)game::Blocks::MapEnum_t::empty) {
        BCOL_Contact_Grid_DisableContact(Contact);
      }
      else {
        BCOL_Contact_Grid_EnableContact(Contact);
      }
    };
  OpenProperties.PostSolve_Grid_cb =
    [](
      BCOL_t* bcol,
      BCOL_ObjectID_t ObjectID,
      uint8_t ShapeEnum,
      BCOL_ShapeID_t ShapeID,
      sint32_t GridY,
      sint32_t GridX,
      BCOL_ContactResult_Grid_t* ContactResult
    ){
      auto Stage = OFFSETLESS(bcol, Stage_t, bcol);

      BCOL_ObjectExtraData_t *ObjectData = BCOL_GetObjectExtraData(bcol, ObjectID);
      auto Entity = Stage->EntityList.Get(ObjectData->EntityID);
      switch(Entity->Behaviour->IdentifyingAs) {
        case EntityIdentify_t::ship:{
          auto EntityShipData = (EntityScope_t::ship_t::EntityData_t *)Entity->UserPTR;
          f32_t NormalY = BCOL_ContactResult_Grid_GetNormalY(ContactResult);
          f32_t NormalX = BCOL_ContactResult_Grid_GetNormalX(ContactResult);

          /* enough? */
          if (NormalY < -0.25) {
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
      BCOL_ObjectID_t ObjectID0,
      uint8_t ShapeEnum0,
      BCOL_ShapeID_t ShapeID0,
      BCOL_ObjectID_t ObjectID1,
      uint8_t ShapeEnum1,
      BCOL_ShapeID_t ShapeID1,
      BCOL_Contact_Shape_t *Contact
    ){
      auto Stage = OFFSETLESS(bcol, Stage_t, bcol);

      uint32_t EntityID0 = BCOL_GetObjectExtraData(bcol, ObjectID0)->EntityID;
      uint32_t EntityID1 = BCOL_GetObjectExtraData(bcol, ObjectID1)->EntityID;
      auto Entity0 = Stage->EntityList.Get(EntityID0);
      auto Entity1 = Stage->EntityList.Get(EntityID1);
      switch(Entity0->Behaviour->IdentifyingAs){
        case EntityIdentify_t::ship:{
          if (Stage->is_EntityIdentify_Building(Entity1->Behaviour->IdentifyingAs)) {
            BCOL_Contact_Shape_DisableContact(Contact);

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
            BCOL_Contact_Shape_EnableContact(Contact);
          }
          break;
        }
        default: {
          BCOL_Contact_Shape_EnableContact(Contact);
          break;
        }
      }
    };

  BCOL_open(&this->bcol, &OpenProperties);
}

void open(auto* loco) {
  {
    loco_t::vfi_t::properties_t vfip;
    vfip.mouse_button_cb = [](const loco_t::vfi_t::mouse_button_data_t& data){
      fan::print("salsa mouse_button_cb\n");
      if (data.button_state == fan::mouse_state::press) {
        data.flag->ignore_move_focus_check = true;
        data.vfi->set_focus_keyboard(data.vfi->get_focus_mouse());
      }
      if (data.button_state == fan::mouse_state::release) {
        data.flag->ignore_move_focus_check = false;
      }
      return 0;
    };
    vfip.mouse_move_cb = [](const loco_t::vfi_t::mouse_move_data_t& data){return 0;};
    vfip.keyboard_cb = [Stage = this](const loco_t::vfi_t::keyboard_data_t& data){
      fan::print("salsa keyboard_cb\n");
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
        /* TODO
        case fan::input::mouse_scroll_up:
        case fan::input::mouse_scroll_down:
        {
          f32_t a;
          if(data.key == fan::input::mouse_scroll_up) { a = -0.4; }
          if(data.key == fan::input::mouse_scroll_down) { a = +0.4; }
          Stage->WorldMatrixMultiplerVelocity += a;
          break;
        }
        */
      }
      return 0;
    };
    vfip.shape_type = loco_t::vfi_t::shape_t::always;
    vfip.shape.always.z = 0;
    vfiBaseID = game::pile->loco.push_back_input_hitbox(vfip);
  }

  // game::pile->stage_data.sortie.image_light.create(&game::pile->engine.context, 0, 0); TODO

  this->WorldMatrixMultipler = constants::stage::sortie::WorldMatrixMultipler::UpLimitHard;
  this->WorldMatrixMultiplerVelocity = 0;
  {
    this->world_matrices.open(&game::pile->loco);
    UpdateWorldMatrix();
  }

  InitializeBCOL();

  game::pile->audio.piece_open(&this->audio_pieces.ambient_0, "audio/ambient/0");
  game::pile->audio.piece_open(&this->audio_pieces.dong_0, "audio/effect/dong_0");
  game::pile->audio.piece_open(&this->audio_pieces.dong_1, "audio/effect/dong_1");
  game::pile->audio.piece_open(&this->audio_pieces.dong_2, "audio/effect/dong_2");
  for (uint32_t i = 0; i < ShipUnit::UnitAltCount[(uint32_t)ShipUnit::UnitEnum::Drill]; i++) {
    game::pile->audio.piece_open(&this->audio_pieces.drill[i], "audio/effect/drill_" + std::to_string(i));
  }
  for (uint32_t i = 0; i < ShipUnit::UnitAltCount[(uint32_t)ShipUnit::UnitEnum::Engine]; i++) {
    game::pile->audio.piece_open(&this->audio_pieces.Engine[i], "audio/effect/engine_" + std::to_string(i));
  }

  #include "stage2/GeneratePlayground.h"

  this->EntityScope.ship.add(fan::vec2(400, -100));

  this->EntityScope.ship.FocusCamera_Rough();

  this->write_viewable_blocks();

  // InitializeBackground(); TODO

  // InitializeShipFlame(); TODO

  // InitializeGui(); TODO
  // game::pile->stage_data.sortie.flame.enable_draw(&game::pile->engine.context); TODO

  // InitializeLighting(); TODO
}

void close(auto* loco){
  // game::pile->stage_data.sortie.flame.close(); TODO
  // game::pile->stage_data.sortie.light.close(); TODO

  fan::audio::audio_stop_group(&game::pile->audio, (uint32_t)game::SoundGroups_t::sortie);

  BCOL_close(&this->bcol);
}

void window_resize_callback(auto* loco){
		
}

void update(auto* loco){
  const f32_t StepTime = 0.01;
  const f32_t BCOLDeltaMax = 2;

  this->BCOLDelta += game::pile->loco.get_delta_time();

  if(this->BCOLDelta > BCOLDeltaMax){
    engine::log::write(engine::log::LogType_t::Warning, "game stage sortie BCOLDelta is above BCOLDeltaMax\n");
    this->BCOLDelta = BCOLDeltaMax;
  }

  while(this->BCOLDelta >= StepTime){
    if(MATH_abs_f32(this->WorldMatrixMultiplerVelocity) > 0.01){
      this->WorldMatrixMultipler += this->WorldMatrixMultiplerVelocity * StepTime;
      if(this->WorldMatrixMultipler < game::constants::stage::sortie::WorldMatrixMultipler::DownLimitHard){
        this->WorldMatrixMultipler = game::constants::stage::sortie::WorldMatrixMultipler::DownLimitHard;
      }
      else if(this->WorldMatrixMultipler > game::constants::stage::sortie::WorldMatrixMultipler::UpLimitHard){
        this->WorldMatrixMultipler = game::constants::stage::sortie::WorldMatrixMultipler::UpLimitHard;
      }
      this->WorldMatrixMultiplerVelocity /= StepTime * 4 + 1;
      this->UpdateWorldMatrix();
    }

    auto StepEnd = [&](){
      this->BCOLDelta -= StepTime;
      this->BCOLStepNumber++;
    };

    BCOL_Step(&this->bcol, StepTime);
    this->EntityList.Step(StepTime);

    StepEnd();
  }
}
