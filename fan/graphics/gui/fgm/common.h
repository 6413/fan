// kinda legacy
struct version_001_t {

};

static constexpr uint32_t version_001 = 1;

static constexpr uint32_t current_version = version_001;
using current_version_t = version_001_t;

#undef only_struct_data
#undef model_maker_loader