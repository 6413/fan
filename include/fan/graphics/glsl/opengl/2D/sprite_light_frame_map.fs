R"(
#version 130

in vec2 texture_coordinate;

in vec4 light_data;

out vec4 color;

uniform sampler2D texture_sampler0;

flat in uint RenderOPCode0;
flat in uint RenderOPCode1;
in float AspectRatio;

void main() {

  vec2 flipped_y = vec2(texture_coordinate.x, 1.0 - texture_coordinate.y);


//  color = texture(texture_sampler0, flipped_y);
color = light_data;
  uint RenderType = RenderOPCode0 & 0xffu;
  switch(RenderType){
    case 0u:{
      
      break;
    }
  }
}
)"
