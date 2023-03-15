engine::audio_t::piece_t main_menu_piece{"audio/main_menu"};
engine::audio_t::SoundPlayID_t SoundPlayID;

void open(auto& loco) {
  game::pile->StageList.Add<game::pile_t::StageList_t::stage_loader_t::stage::stage1_t>(this->stage_id);

  engine::audio_t::PropertiesSoundPlay_t Properties;
  Properties.GroupID = (uint32_t)game::SoundGroups_t::menu;
  Properties.Flags.Loop = true;
  //SoundPlayID = engine::audio.SoundPlay(&main_menu_piece, &Properties);
}

void close(auto& loco){
  engine::audio_t::PropertiesSoundStop_t p{.FadeOutTo = 2};
  //engine::audio.SoundStop(SoundPlayID, &p);
}

void window_resize_callback(auto& loco){
		
}

void update(auto& loco){
	
}
