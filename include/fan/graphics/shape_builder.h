#if defined(loco_opengl)
	#include _FAN_PATH(graphics/opengl/2D/objects/shape_builder.h)
#elif defined(loco_vulkan)
	#include _FAN_PATH(graphics/vulkan/2D/objects/shape_builder.h)
#endif