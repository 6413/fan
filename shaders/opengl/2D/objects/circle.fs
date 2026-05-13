#version 330

layout (location = 0) out vec4 o_attachment0;

in vec4 instance_color;
in vec3 instance_position;
in float instance_radius;
in vec3 frag_position;
in vec2 texture_coordinate;
flat in uint flags;
in vec4 instance_outline_color;
in float instance_outline_width;

uniform float camera_zoom;

out vec4 color;

void main() {
  float distance = length(frag_position.xy - instance_position.xy);
  float smooth_edge = 2.0 / camera_zoom;

  float outer = instance_radius;
  float inner = instance_radius - instance_outline_width;

  float outer_alpha = 1.0 - smoothstep(
    outer - smooth_edge, 
    outer, 
    distance
  );

  float inner_alpha = 1.0 - smoothstep(
    inner - smooth_edge, 
    inner, 
    distance
  );

  vec4 result = mix(
    instance_outline_color,
    instance_color,
    inner_alpha
  );
  result.a *= outer_alpha;
  o_attachment0 = result;
}
