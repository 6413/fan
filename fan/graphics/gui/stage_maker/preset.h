stage_loader_t::stage_common_t stage_common = {
  .open = _stage_open,
  .close = _stage_close,
  .update = _stage_update
};

static void _stage_open(void* ptr, void *sod) {
  ((lstd_current_type*)ptr)->open(sod);
}

static void _stage_close(void* ptr) {
  ((lstd_current_type*)ptr)->close();
  delete (lstd_current_type*)ptr;
}  

static void _stage_update(void* ptr){
  ((lstd_current_type*)ptr)->update();
}

struct structor_t{
  structor_t(const stage_loader_t::stage_open_properties_t& op) {
    auto outside = OFFSETLESS(this, lstd_current_type, structor);
    auto nr = gstage->stage_list.NewNodeLast();
    outside->stage_common.stage_id = nr;
    outside->stage_common.parent_id = op.parent_id;
    gstage->stage_list[nr].stage = outside;
    if (outside->stage_common.stage_id.Prev(&gstage->stage_list) != gstage->stage_list.src) {
      outside->stage_common.it = ((stage_loader_t::stage_common_t *)gstage->stage_list[outside->stage_common.stage_id.Prev(&gstage->stage_list)].stage)->it + 1;
    }
    else {
      outside->stage_common.it = 0;
    }

    if (!fan::string(lstd_current_type::stage_name).empty()) {
      gstage->load_fgm(outside, op, lstd_current_type::stage_name);
    }
    gstage->stage_list[outside->stage_common.stage_id].update_nr = gloco->m_update_callback.NewNodeLast();
    gloco->m_update_callback[gstage->stage_list[outside->stage_common.stage_id].update_nr] = [&, outside](loco_t* loco) {
      outside->update();
    };
  }
}structor;
