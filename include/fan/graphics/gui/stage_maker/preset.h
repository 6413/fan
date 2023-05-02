stage_common_t stage_common = {
  .open = _stage_open,
  .close = _stage_close,
  .window_resize = _stage_window_resize,
  .update = _stage_update
};
static void _stage_open(void* ptr) {
  ((lstd_current_type*)ptr)->open();
}

static void _stage_close(void* ptr) {
  ((lstd_current_type*)ptr)->close();
}  

static void _stage_window_resize(void* ptr){
	((lstd_current_type*)ptr)->window_resize();
}

static void _stage_update(void* ptr){
  ((lstd_current_type*)ptr)->update();
}

//lstd_current_type() {
//
//}
//lstd_current_type() {
//
//}