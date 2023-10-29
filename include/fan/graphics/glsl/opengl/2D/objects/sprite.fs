R"(
#version 330

layout (location = 0) out vec4 o_attachment0;

in vec2 texture_coordinate;
in vec4 instance_color;

uniform sampler2D _t00;
uniform sampler2D _t01;
uniform vec3 lighting_ambient;
uniform vec2 window_size;

const float offset = 1.0 / 300.0;  

void main()
{
    vec2 offsets[9] = vec2[](
        vec2(-offset,  offset), // top-left
        vec2( 0.0f,    offset), // top-center
        vec2( offset,  offset), // top-right
        vec2(-offset,  0.0f),   // center-left
        vec2( 0.0f,    0.0f),   // center-center
        vec2( offset,  0.0f),   // center-right
        vec2(-offset, -offset), // bottom-left
        vec2( 0.0f,   -offset), // bottom-center
        vec2( offset, -offset)  // bottom-right    
    );

float kernel[9] = float[](
  1, 1, 1,
  1, -10, 1,
  1, 1, 1
);
    
  o_attachment0 = texture(_t00, vec2(texture_coordinate.x, texture_coordinate.y)) * instance_color;

  if (o_attachment0.a <= 0.5) {
    discard;
  }


//vec3 sampleTex[9];
  //for(int i = 0; i < 9; i++)
  //{
  //    sampleTex[i] = vec3(texture(_t00, texture_coordinate.st + offsets[i]));
  //}
  //for(int i = 0; i < 9; i++)
  //    o_attachment0.rgb += sampleTex[i] * kernel[i];

  //  o_attachment0 *= 5;

  vec4 t = vec4(texture(_t01, gl_FragCoord.xy / window_size).rgb, 1);

  //o_attachment0.rgb *= lighting_ambient + t.rgb;

  //if (o_attachment0.r + o_attachment0.g + o_attachment0.b < 0.1) {
  //if (o_attachment0.r + o_attachment0.g + o_attachment0.b < 0.1) {
  //  discard;
  //  discard;
  //}
  //}
}
)"