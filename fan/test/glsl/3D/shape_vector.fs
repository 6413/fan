#version 330 core

flat in int mode;
flat in int v;

in vec2 texture_coordinate;
in vec4 color;

out vec4 shape_texture;

in vec3 vertex_position;
uniform vec3 player_position;

uniform sampler2D texture_sampler;
uniform samplerCube skybox;

void main() {
    switch(mode) {
        case 1: {
            shape_texture = color;
            break;
		}
        case 2: {

            vec3 I = normalize(vertex_position - player_position);
            
            // 0 = up, 1 = down, 2 = front, 3 = right, 4 = back, 5 = left

            vec3 cases[] = vec3[](
                vec3(0, 1, 0), 
                vec3(0, -1, 0), 
                vec3(0, 0, -1), 
                vec3(1, 0, 0), 
                vec3(0, 0, 1),
                vec3(-1, 0, 0)
            );

           // vec3 r = refract(I, cases[v / 6], 1.0 / 32);

            vec3 r = reflect(I, cases[v / 6]);

            vec4 test = vec4(texture(skybox, r).rgb, 1);

            vec4 real = texture2D(texture_sampler, texture_coordinate);

            shape_texture = test;


            break;
        }
	}
} 