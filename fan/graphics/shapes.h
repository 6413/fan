#if defined(fan_3D)
  #define IF_FAN_3D(X) X(rectangle3d) X(line3d)
#else
  #define IF_FAN_3D(X)
#endif

#define GEN_SHAPES(X, SKIP) \
		X(sprite) X(text) SKIP(X(hitbox)) SKIP(X(mark)) X(line) X(rectangle) \
		X(light) X(unlit_sprite) X(circle) X(capsule) X(polygon) X(grid) \
		X(vfi) X(particles) X(universal_image_renderer) X(gradient) \
		SKIP(X(light_end)) X(shader_shape) IF_FAN_3D(X) X(shadow)