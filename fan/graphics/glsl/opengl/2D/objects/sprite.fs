
#version 330

layout (location = 0) out vec4 o_attachment0;
layout (location = 2) out vec4 o_attachment2;

in vec2 texture_coordinate;
in vec2 size;
in vec4 instance_color;
flat in uint fs_flags;
flat in int element_id;

uniform sampler2D _t00;
uniform sampler2D _t01;
uniform vec3 lighting_ambient;
uniform vec2 window_size;
uniform float m_time;

float rand(vec2 co) {
    float seed = dot(co, vec2(12.9898, 78.233));
    float rand = fract(sin(seed) * 43758.5453);
    return rand;
}

// Function to generate random value between min and max
float randomValue(float minValue, float maxValue, vec2 seed) {
    return mix(minValue, maxValue, rand(seed));
}

void main() {

  vec2 tc = texture_coordinate;

  if ((fs_flags & 0x1u) == 0x1u) {
    float speed = 0.3;
  vec2 Wave = vec2(randomValue(0.8, 0.9, vec2(float(element_id), float(element_id))), 2);

    tc += vec2(cos((tc.y/Wave.x + (m_time + (m_time * (float(randomValue(1.0, 100.0, vec2(float(element_id), float(element_id)))))) / 100.0) * speed) * Wave.y), 0.0) / size * (1.0 - tc.y);
  }

  vec4 tex_color = texture(_t00, tc) * instance_color;

  if (tex_color.a <= 0.25) {
    discard;
  }

  vec4 t = vec4(texture(_t01, gl_FragCoord.xy / window_size).rgb, 1);

  tex_color.rgb *= lighting_ambient + t.rgb;

  o_attachment0 = tex_color;

  // t.rgb
  float brightness = dot(t.rgb, vec3(0.2126, 0.7152, 0.0722));
  //if(brightness > 1.0) {
    if ((fs_flags & 0x2u) == 0x2u) {
      o_attachment2 = vec4(t.rgb, 1);
    }
    else {
      //o_attachment2 += vec4(0, 0, 0, 0);
    }
    //o_attachment1 = vec4(t.rgb, 1);
    //o_attachment1 = vec4(0, 0, 0, 1);
//  }
  //else {
   // o_attachment2 = vec4(0, 0, 0, 0);
    //o_attachment1 = vec4(0, 0, 0, 1);
//  }
}