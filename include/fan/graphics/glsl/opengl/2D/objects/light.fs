R"(
#version 330

layout (location = 2) out vec4 o_attachment2;

uniform sampler2D _t00;
uniform sampler2D _t01;
uniform sampler2D _t02;

in vec4 instance_color;
in vec3 instance_position;
in vec2 instance_size;
in vec3 frag_position;
in vec2 texture_coordinate;
flat in uint instance_light_type;

const float gradient_depth = 600.0;

const vec3 u_sky_top_color = vec3(0.5, 0.8, 0.9) / 2;
const vec3 u_sky_bottom_color = vec3(0.5, 0.8, 0.9) / 30;

void main() {
  vec4 t = vec4(texture(_t02, texture_coordinate).rgb, 1);
  vec4 t2 = vec4(texture(_t00, texture_coordinate).rgb, 1);
  
  o_attachment2 = instance_color;
  switch(instance_light_type) {
    case 0u: {
      float distance = length(frag_position - instance_position);
      float radius = instance_size.x;
      float smooth_edge = radius;
      float intensity = 1.0 - smoothstep(radius / 3 -smooth_edge, radius, distance);
      o_attachment2 *= intensity;
      break;
    }
    case 1u: {
      float distance = length(frag_position - instance_position);
      float radius = instance_size.x;
      float smooth_edge = radius;
      float intensity = 1.0 - smoothstep(radius / 3 -smooth_edge, radius, distance);
      if (frag_position.y > 0) {
        float y_intensity = clamp((frag_position.y - 0) / gradient_depth, 0, 1);
        intensity = mix(intensity, 0, y_intensity);
        intensity += 1.0 - smoothstep(radius / 3 -smooth_edge, radius, distance) * 2;
      }
      o_attachment2 *= intensity;
      break;
    }
    case 2u: {
      float y = abs(frag_position.y) / 1000;
      float step = smoothstep(0, 1, y);
      o_attachment2 = vec4(mix(u_sky_bottom_color, u_sky_top_color, step), 1.0);
      
      break;
    }
  }
}
)"