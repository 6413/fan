#version 130

in vec2 texture_coordinate;
in float transparency;
in vec2 blur_texture_coordinates[22];

out vec4 color;

uniform sampler2D texture_sampler;

const vec2 DOWNSAMPLE_OFFSETS[] = vec2[]
(
      vec2(0.5, 0.5),
      vec2(0.5, 2.5),
      vec2(2.5, 2.5),
      vec2(2.5, 0.5)
);

void main() {

    vec4 blur = vec4(0);

	vec2 flipped_y = vec2(texture_coordinate.x, 1.0 - texture_coordinate.y);
  
//  	blur += texture(texture_sampler, blur_texture_coordinates[0]) * 0.0093;
//    blur += texture(texture_sampler, blur_texture_coordinates[1]) * 0.028002;
//    blur += texture(texture_sampler, blur_texture_coordinates[2]) * 0.065984;
//    blur += texture(texture_sampler, blur_texture_coordinates[3]) * 0.121703;
//    blur += texture(texture_sampler, blur_texture_coordinates[4]) * 0.175713;
//    blur += texture(texture_sampler, blur_texture_coordinates[5]) * 0.198596;
//    blur += texture(texture_sampler, blur_texture_coordinates[6]) * 0.175713;
//    blur += texture(texture_sampler, blur_texture_coordinates[7]) * 0.121703;
//    blur += texture(texture_sampler, blur_texture_coordinates[8]) * 0.065984;
//    blur += texture(texture_sampler, blur_texture_coordinates[9]) * 0.028002;
//    blur += texture(texture_sampler, blur_texture_coordinates[10]) * 0.0093;

//    blur += texture(texture_sampler, blur_texture_coordinates[11]) * 0.0093;
//    blur += texture(texture_sampler, blur_texture_coordinates[12]) * 0.028002;
//    blur += texture(texture_sampler, blur_texture_coordinates[13]) * 0.065984;
//    blur += texture(texture_sampler, blur_texture_coordinates[14]) * 0.121703;
//    blur += texture(texture_sampler, blur_texture_coordinates[15]) * 0.175713;
//    blur += texture(texture_sampler, blur_texture_coordinates[16]) * 0.198596;
//    blur += texture(texture_sampler, blur_texture_coordinates[17]) * 0.175713;
//    blur += texture(texture_sampler, blur_texture_coordinates[18]) * 0.121703;
//    blur += texture(texture_sampler, blur_texture_coordinates[19]) * 0.065984;
//    blur += texture(texture_sampler, blur_texture_coordinates[20]) * 0.028002;
//    blur += texture(texture_sampler, blur_texture_coordinates[21]) * 0.0093;

//    //blur /= 2;

//    vec4 original = texture(texture_sampler, flipped_y);

//    color = original * blur;

//	float r = 19;

//    float xs = 300, ys = 300;

//    vec2 pos = blur_texture_coordinates[0];

// float x,y,xx,yy,rr=r*r,dx,dy,w,w0;
//w0=0.3780/pow(r,1.975);
//vec2 p;
//vec4 col=vec4(0.0,0.0,0.0,0.0);
//for (dx=1.0/xs,x=-r,p.x=0.5+(pos.x*0.5)+(x*dx);x<=r;x++,p.x+=dx){ xx=x*x;
// for (dy=1.0/ys,y=-r,p.y=0.5+(pos.y*0.5)+(y*dy);y<=r;y++,p.y+=dy){ yy=y*y;
//  if (xx+yy<=rr)
//    {
//    w=w0*exp((-xx-yy)/(2.0*rr));
//    col+=texture2D(texture_sampler,p)*w;
//    }}}
//    color=col;

	color = texture(texture_sampler, flipped_y);

	//color.a -= clamp(1.0 - transparency, 0, 1);
}