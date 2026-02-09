inline static const char* grass_shader_fragment = R"(#version 330

layout (location = 0) out vec4 o_attachment0;

in vec2 texture_coordinate;
in vec4 instance_color;
flat in float object_seed;

uniform sampler2D _t00;
uniform float grass_time;
uniform float grass_wind;

void main() {
  vec2 uv = texture_coordinate;

float sway_factor = pow(1.0 - uv.y, 1.5);
float phase = object_seed * 10.0 + uv.y * 5.0;
float amp = mix(0.8, 1.2, fract(sin(object_seed * 123.456) * 999.0));

float sway = sin(grass_time * 3.0 + uv.y * 3.0 + phase)
           * sway_factor
           * 0.02
           * grass_wind
           * amp;

float flutter = sin(grass_time * 4.0 + uv.y * 40.0 + phase)
              * 0.005
              * grass_wind;

float vertical_drop = sin(grass_time * 2.0 + phase)
                    * 0.01
                    * grass_wind;

uv.x += sway + flutter;
uv.y += vertical_drop;

vec4 c = texture(_t00, uv) * instance_color;

if (c.a < 0.1) {
  discard;
}

o_attachment0 = c;

}
)";