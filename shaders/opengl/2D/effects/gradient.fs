#version 330

layout(location = 0) out vec4 o_attachment0;

uniform sampler2D _t01; // light map
uniform vec3 lighting_ambient;
uniform vec2 window_size;

in vec4 instance_color;

void main() {
  vec3 base = instance_color.rgb;
  vec3 light_color = texture(_t01, gl_FragCoord.xy / window_size).rgb;
  float lowest = min(min(light_color.r, light_color.g), light_color.b);
  float brightness = dot(base, vec3(0.2126, 0.7152, 0.0722));
  vec3 mixed = base * lowest + brightness * light_color * (1.0 - lowest);
  vec3 ambient = base * lighting_ambient;
  vec3 final_color = ambient + mixed;

  o_attachment0 = vec4(final_color, instance_color.a);
}