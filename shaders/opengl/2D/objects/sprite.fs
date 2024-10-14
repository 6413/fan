
#version 330

layout (location = 0) out vec4 o_attachment0;
//layout (location = 2) out vec4 o_attachment2;

in vec2 texture_coordinate;
in vec2 size;
in vec4 instance_color;
flat in uint fs_flags;
flat in float object_seed;

in vec3 instance_position;
in vec3 frag_position;


uniform sampler2D _t00;
uniform sampler2D _t01;
uniform sampler2D _t02;
uniform vec3 lighting_ambient;
uniform vec2 window_size;
uniform float m_time;
uniform vec2 offset;

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

    vec4 tex_color = vec4(1, 1, 1, 1);
  //if (fs_flags == floatBitsToUint(1.0)) {
  //  float speed = 0.3;
  //vec2 Wave = vec2(randomValue(0.8, 0.9, vec2(float(element_id), float(element_id))), 2);
  //
  //  tc += vec2(cos((tc.y/Wave.x + (m_time + (m_time * (float(randomValue(1.0, 100.0, vec2(float(element_id), float(element_id)))))) / 100.0) * speed) * Wave.y), 0.0) / size * (1.0 - tc.y);
  //}
  //
  if (fs_flags == floatBitsToUint(2.0)) {
    float speed = 0.03; // Adjust this value to change the speed of the lava
    vec2 tc_noise = tc * 0.5 + vec2(0, -m_time * speed); // Scale the texture coordinates

    // Sample the noise texture
    vec4 noise_col = texture(_t02, tc_noise);

    // Add another layer of noise at a different scale
    vec2 tc_noise2 = tc * 0.1 + vec2(0, -m_time * speed * 0.5);
    vec4 noise_col2 = texture(_t02, tc_noise2);

    // Combine the two noise values
    vec4 noise_combined = mix(noise_col, noise_col2, 0.5);

    // Add a time-dependent offset to the texture coordinates
    vec2 tc_offset = tc + vec2(0, -m_time * speed) + fract(vec2(object_seed, object_seed) * 5.324);

    // Use the combined noise value to offset the texture coordinates for the lava texture
    tex_color = texture(_t00, tc_offset + 0.6 * noise_combined.rg) * instance_color;
  }
  else  if (fs_flags == floatBitsToUint(4.0)) {
    float speed = 2.03; // Adjust this value to change the speed of the waves

    // Calculate the distance from the center
    vec2 center = vec2(0.5, 0.5);
    vec2 from_center = tc - center;
    float distance_from_center = length(from_center);

    // Use the distance from the center to create a radial wave effect
    vec2 tc_noise = tc + (from_center / distance_from_center) * cos(distance_from_center * 10.0 - m_time * speed) * 0.01;

    // Sample the noise texture
    vec4 noise_col = texture(_t02, tc_noise);

    // Add another layer of noise at a different scale
    vec2 tc_noise2 = tc + (from_center / distance_from_center) * cos(distance_from_center * 20.0 - m_time * speed * 0.5) * 0.005;
    vec4 noise_col2 = texture(_t02, tc_noise2);

    // Combine the two noise values
    vec4 noise_combined = mix(noise_col, noise_col2, 0.5);

    // Add a time-dependent offset to the texture coordinates
    vec2 tc_offset = tc + (from_center / distance_from_center) * sin(distance_from_center * 15.0 - m_time * speed) * 0.01 + fract(vec2(object_seed, object_seed) * 5.324);

    // Use the combined noise value to offset the texture coordinates for the wave texture
    tex_color = texture(_t00, tc_offset + 0.6 * noise_combined.rg) * instance_color;
  }
  else {
    tex_color = texture(_t00, tc) * instance_color;
  }
    

  if (tex_color.a <= 0.25) {
    discard;
  }

  vec4 lighting_texture = vec4(texture(_t01, gl_FragCoord.xy / window_size).rgb, 1);

  tex_color.rgb *= lighting_ambient + lighting_texture.rgb;

  o_attachment0 = tex_color;

  // t.rgb
  //float brightness = dot(lighting_texture.rgb, vec3(0.2126, 0.7152, 0.0722));
  //if(brightness > 1.0) {
    if ((fs_flags & 0x2u) == 0x2u) {
      //o_attachment2 = vec4(t.rgb, 1);
    }
}