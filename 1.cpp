#include "../WITCH/WITCH.h"

constexpr static f32_t bcol_step_time = 0.01;
#define ETC_BCOL_set_prefix bcol
#define ETC_BCOL_set_DynamicDeltaFunction \
  //ObjectData0->Velocity.y += delta * 2;
#define ETC_BCOL_set_StoreExtraDataInsideObject 1
#define ETC_BCOL_set_ExtraDataInsideObject \
  bcol_t::ShapeID_t shape_id;
#include "../WITCH/ETC/BCOL/BCOL.h"

int main() {
	
}