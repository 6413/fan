R"(
#version 450

layout(location = 0) out vec4 o_color;
layout(location = 1) out vec4 rcolor;

layout(location = 0) in vec4 instance_color;
layout(location = 1) in vec2 texture_coordinate;
layout(location = 2) in float render_size;

layout(binding = 2) uniform sampler2D _t[16];

layout( push_constant ) uniform constants_t {
	uint texture_id;
	uint camera_id;
}constants;

void main() {
  rcolor = vec4(0);
  float distance = texture(_t[constants.texture_id], texture_coordinate).r;

  float smoothing = 1.0 / (log(render_size * 100) * 2);
  float width = 0.2;
  float alpha = smoothstep(width, width + smoothing, distance);

  float border_width = 0.5;
  float border_edge =  0.1;

  float outline_alpha = smoothstep(border_width, border_width + border_edge, distance);

  o_color = vec4(instance_color.rgb, alpha);
  if (o_color.a < 0.1) {
		discard;
	}
}
)"