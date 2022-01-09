R"(
#version 130

in vec2 texture_coordinate;

in vec4 i_color;

out vec4 color;

uniform sampler2D texture_sampler;

flat in uint RenderOPCode;

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

void main() {

  vec2 flipped_y = vec2(texture_coordinate.x, 1.0 - texture_coordinate.y);

  color = texture(texture_sampler, flipped_y);

	uint RenderType = RenderOPCode & 0xffu;
	switch(RenderType){
		case 0u:{
  		color *= i_color;
			break;
		}
		case 1u:{
			color *= i_color;
			vec3 RenderColor;
			{
				uint color8 = RenderOPCode >> 8u & 0xffu;
				RenderColor = c8tof(color8);
			}
			float RenderAlpha;
			{
				uint alpha6 = RenderOPCode >> 16u & 0x3fu;
				alpha6 <<= 2u;
				RenderAlpha = float(alpha6) / float(0xff);
			}
			float RenderDelta;
			{
				uint delta10 = RenderOPCode >> 22u & 0x3ffu;
				RenderDelta = float(delta10) / float(0x3ff);
			}
			vec2 relative_coordinate = texture_coordinate * 2.0 - 1.0;
			if(RenderDelta < .2){
				RenderAlpha *= RenderDelta * 5.0;
			}
			else if(RenderDelta > .8){
				RenderAlpha *= (1.0 - RenderDelta) * 5.0;
			}
			if(length(relative_coordinate) < RenderDelta){
				RenderColor *= RenderAlpha;
				color.xyz *= 1.0 - RenderAlpha;
				float david_star = get_david_star(relative_coordinate, RenderDelta);
				color.xyz += RenderColor * david_star;
			}
			else if(length(relative_coordinate) > RenderDelta && length(relative_coordinate) < RenderDelta + .1){
				RenderColor *= RenderAlpha;
				color.xyz *= 1.0 - RenderAlpha;
				color.xyz += RenderColor;
			}
			break;
		}
		case 2u:{
			color *= i_color;
			vec3 RenderColor;
			{
				uint color8 = RenderOPCode >> 8u & 0xffu;
				RenderColor = c8tof(color8);
			}
			float RenderAlpha;
			{
				uint alpha6 = RenderOPCode >> 16u & 0x3fu;
				alpha6 <<= 2u;
				RenderAlpha = float(alpha6) / float(0xff);
			}
			float RenderDelta;
			{
				uint delta10 = RenderOPCode >> 22u & 0x3ffu;
				RenderDelta = float(delta10) / float(0x3ff);
			}
			vec2 relative_coordinate = texture_coordinate * 2.0 - 1.0;
			if(RenderDelta < .2){
				RenderAlpha *= RenderDelta * 5.0;
			}
			else if(RenderDelta > .8){
				RenderAlpha *= (1.0 - RenderDelta) * 5.0;
			}
			if(length(relative_coordinate) < RenderDelta){
				RenderColor *= RenderAlpha;
				color.xyz *= 1.0 - RenderAlpha;
				float cross = get_cross(relative_coordinate, RenderDelta);
				color.xyz += RenderColor * cross;
			}
			else if(length(relative_coordinate) > RenderDelta && length(relative_coordinate) < RenderDelta + .1){
				RenderColor *= RenderAlpha;
				color.xyz *= 1.0 - RenderAlpha;
				color.xyz += RenderColor;
			}
			break;
		}
	}
}
)"

		