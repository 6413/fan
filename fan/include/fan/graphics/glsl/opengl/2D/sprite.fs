R"(
#version 130

in vec2 texture_coordinate;

in vec4 i_color;
in vec2 f_position;

out vec4 color;

uniform sampler2D texture_sampler0;

in float AspectRatio;

vec3 c8tof(uint color){
  vec3 c;
  c.x = float((color & 0xe0u) << 0u) / float(0xff);
  c.y = float((color & 0x1cu) << 3u) / float(0xff);
  c.z = float((color & 0x03u) << 6u) / float(0xff);
  return c;
}

float get_triangle_up(vec2 Position, float Area){
  float r;
  Position.x *= 1.55;
  Position.y += 0.375 * Area;
  float sum_relative = abs(Position.x) + abs(Position.y);
  float min_area = Area - .09375;
  float max_area = Area + .09375;
  if(Position.y > .046875 && sum_relative > min_area && sum_relative < max_area){
    r += 1.0 - 1.0 / .09375 * distance(Area, sum_relative);
  }
  if(Position.y < .09375){
    if(abs(Position.x) < Area){
      if(Position.y > 0){
        r += 1.0 - 1.0 / .046875 * distance(.046875, Position.y);
      }
    }
  }
  return r;
}
float get_triangle_down(vec2 Position, float Area){
  float r = 0;
  Position.x *= 1.55;
  Position.y -= 0.375 * Area;
  float sum_relative = abs(Position.x) + abs(Position.y);
  float min_area = Area - .09375;
  float max_area = Area + .09375;
  if(Position.y < -.046875 && sum_relative > min_area && sum_relative < max_area){
    r += 1.0 - 1.0 / .09375 * distance(Area, sum_relative);
  }
  if(Position.y > -.09375){
    if(abs(Position.x) < Area){
      if(Position.y < 0){
        r += 1.0 - 1.0 / .046875 * distance(-.046875, Position.y);
      }
    }
  }
  return r;
}

float get_david_star(vec2 Position, float Area){
  float r = 0.0;
  r += get_triangle_up(Position, Area);
  r += get_triangle_down(Position, Area);
  return min(1.0, r);
}

float get_cross(vec2 Position, float Area){
  float r = 0.0;
  float AreaMultiply0125 = Area * .125;
  if(abs(Position.x) < AreaMultiply0125){
    if(abs(Position.y) < Area * .75){
      float rt0 = 1.0 - abs(Position.x) / AreaMultiply0125;
      if(abs(Position.y) > Area * 0.625){
        float rt1 = 1.0 - distance(Area * 0.625, abs(Position.y)) / AreaMultiply0125;
        rt0 *= rt1;
      }
      r += rt0;
    }
  }
  if(Position.y > AreaMultiply0125 && Position.y < Area * 0.375){
    if(abs(Position.x) < Area * 0.75){
      float rt0 = 1.0 - distance(Area * 0.25, Position.y) / AreaMultiply0125;
      if(abs(Position.x) > Area * 0.625){
        float rt1 = 1.0 - distance(Area * 0.625, abs(Position.x)) / AreaMultiply0125;
        rt0 *= rt1;
      }
      r += rt0;
    }
  }
  return min(1.0, r);
}

float get_case3(vec2 Position, float DotPerArea, float DotSizeHard, float DotSizeSoft, vec2 Cut, float Delta){
  if(Position.x < -Cut.x || Position.x > Cut.x){
    return 0.0;
  }
  if(Position.y < -Cut.y || Position.y > Cut.y){
    return 0.0;
  }

  Position *= vec2(1.0) / Cut;

  float RealAspectRatio = AspectRatio * Cut.x / Cut.y;

  vec2 SizeDot = vec2(1.0 / RealAspectRatio, 1.0) / DotPerArea;
  vec2 SizeDotD2 = SizeDot / 2.0;

  float Multipler = 1.0;
  float PosProg = (Position.x + 1.0) / 2.0;
  float DotPerAreaX = DotPerArea * RealAspectRatio;
  float DeltaDotPerAreaX = DotPerAreaX * Delta;
  if(floor(PosProg * DotPerAreaX) > floor(DeltaDotPerAreaX)){
    return 0.0;
  }
  if(PosProg * DotPerAreaX > floor(DeltaDotPerAreaX)){
    Multipler = DeltaDotPerAreaX - floor(DeltaDotPerAreaX);
  }

  vec2 PosMod = mod((Position + 1.0) / 2.0, SizeDot);
  PosMod /= SizeDot;
  float dist = distance(PosMod, vec2(0.5));
  float r;
  if(dist > DotSizeHard * 0.5){
    if(dist < DotSizeSoft * 0.5){
      float diff = DotSizeSoft * 0.5 - DotSizeHard * 0.5;
      float v = dist - DotSizeHard * 0.5;
      v *= 1.0 / diff;
      r = 1.0 - v;
    }
    else{
      r = 0.0;
    }
  }
  else{
    r = 1.0;
  }
  return r * Multipler;
}

uniform vec2 mouse_position;

void main() {

  vec2 flipped_y = vec2(texture_coordinate.x, 1.0 - texture_coordinate.y);

  color = texture(texture_sampler0, flipped_y);

  color *= i_color;
}
)"
