R"(
  #version 330

  in vec2 texture_coordinate;

  in vec4 instance_color;
  in float texture_id;

  out vec4 o_color;

  uniform int which;

  uniform sampler2D _t00;
  uniform sampler2D _t01;
  uniform sampler2D _t02;

  const float offset = 1.0 / 300.0;

  const vec2 offsets[9] = vec2[](
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

  const float kernel[9] = float[](
      1, 1, 1,
      1, -8, 1,
      1, 1, 1
  );

  void main() {

    //vec3 sampleTex[9];
    //for(int i = 0; i < 9; i++)
    //{
    //    sampleTex[i] = vec3(texture(_t00, texture_coordinate.st + offsets[i]));
    //}
    //vec3 col = vec3(0.0);
    //for(int i = 0; i < 9; i++)
    //    col += sampleTex[i] * kernel[i];

    o_color = texture(_t00, texture_coordinate);
    const vec3 W = vec3(0.2125, 0.7154, 0.0721);
    //vec3 intensity = vec3(dot(o_color.rgb, W));
    //o_color.a = 1;
    //o_color. = vec4(1, 0, 0, 1);
    //o_color.rgb = mix(intensity, o_color.rgb, vec3(0.5));

    vec4 black_image = o_color;

    float brightness = (black_image.r * W.x) + (black_image.g * W.y) + (black_image.b * W.z);
    if (brightness <= 0.0) {
      black_image = vec4(0);
    }

    if (which == 1) {
        o_color = black_image;
        clamp(o_color.r, 0, 1);
        clamp(o_color.g, 0, 1);
        clamp(o_color.b, 0, 1);
        clamp(o_color.a, 0, 1);
    }
    if (which == 2) {
      vec4 t = texture(_t02, texture_coordinate) * 5;
      clamp(t.r, 0, 1);
      clamp(t.g, 0, 1);
      clamp(t.b, 0, 1);
      clamp(t.a, 0, 1);

      o_color += t;
      clamp(o_color.r, 0, 1);
      clamp(o_color.g, 0, 1);
      clamp(o_color.b, 0, 1);
      clamp(o_color.a, 0, 1);
    }
    //o_color = black_image;
  }
)"