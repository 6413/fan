#version 330

in vec2 texture_coordinate;

layout (location = 0) out vec4 o_attachment0;

uniform sampler2D _t00;
uniform vec3 view_angles;
uniform float tilt;
uniform float zoom;
uniform float depth;
uniform float m_time;

#define PI 3.14159265359

mat4 rotate(mat4 m, vec3 angles) {
    float cx = cos(angles.x);
    float sx = sin(angles.x);
    float cy = cos(angles.y);
    float sy = sin(angles.y);
    float cz = cos(angles.z);
    float sz = sin(angles.z);

    mat4 rotationX = mat4(1.0, 0.0, 0.0, 0.0,
                          0.0, cx, -sx, 0.0,
                          0.0, sx, cx, 0.0,
                          0.0, 0.0, 0.0, 1.0);

    mat4 rotationY = mat4(cy, 0.0, sy, 0.0,
                          0.0, 1.0, 0.0, 0.0,
                          -sy, 0.0, cy, 0.0,
                          0.0, 0.0, 0.0, 1.0);

    mat4 rotationZ = mat4(cz, -sz, 0.0, 0.0,
                          sz, cz, 0.0, 0.0,
                          0.0, 0.0, 1.0, 0.0,
                          0.0, 0.0, 0.0, 1.0);

    mat4 matrix = rotationX * rotationY * rotationZ * m;
    return matrix;
}

vec3 quantizeToPalette(vec3 color, vec3[27] palette) {
    // Initialize with the first color in the palette
    vec3 closestColor = palette[0];
    float minDistance = distance(color, closestColor);

    // Iterate over the remaining colors in the palette
    for (int i = 1; i < 27; i++) {
        float dist = distance(color, palette[i]);
        if (dist < minDistance) {
            minDistance = dist;
            closestColor = palette[i];
        }
    }

    return closestColor;
}

vec3 normalizeColor(vec3 color) {
    return color / 255.0;
}

void main() {
    vec3 actual = texture(_t00, texture_coordinate).rgb;

        vec2 position = -1.0 + 2.0 * texture_coordinate;
    position *= zoom;

    // Apply tilt (rotate around the x-axis)
    float tilt_angle = tilt;
    float tilt_s = sin(tilt_angle);
    float tilt_c = cos(tilt_angle);
    position = vec2(
        position.x,
        position.y * tilt_c + position.y * tilt_s
    );

    // Apply pitch (rotate around the y-axis)
    float pitch_angle = 0;
    float pitch_s = sin(pitch_angle);
    float pitch_c = cos(pitch_angle);
    position = vec2(
        position.x * pitch_c - position.y * pitch_s,
        position.x * pitch_s + position.y * pitch_c
    );

    // Apply pan
    position += vec2(0, 0.0);

    // Apply spin
    float spin_angle = 0;
    float spin_s = sin(spin_angle);
    float spin_c = cos(spin_angle);
    position = vec2(
        position.x * spin_c - position.y * spin_s,
        position.x * spin_s + position.y * spin_c
    );

    // Normalize position to range [-1, 1]

    // Convert normalized position to spherical coordinates
    float x2y2 = position.x * position.x + position.y * position.y;
    vec3 sphere_pnt = vec3(2.0 * position, x2y2 - 1.0) / (x2y2 + 1.0);

    mat4 m = mat4(1);
    m = rotate(m, view_angles);

    sphere_pnt = vec3(m * vec4(sphere_pnt, 1.0));

    vec2 sampleUV = vec2(
        (atan(sphere_pnt.y, sphere_pnt.x) / PI + 1.0) * 0.5,
        (asin(sphere_pnt.z) / PI + 0.5)
    );

    vec2 b = vec2(1.0 - sampleUV.x, sampleUV.y);
    vec2 a = texture_coordinate;

    // Sample the texture
    vec2 final = vec2(0, 0);
    if (depth > 100) {
      final = b;
    }
    else {
      final = a;
    }

vec3 colors[27] = vec3[](
    vec3(0.988, 0.914, 0.310), // #fce94f
    vec3(0.929, 0.831, 0.000), // #edd400
    vec3(0.769, 0.627, 0.000), // #c4a000
    vec3(0.541, 0.886, 0.204), // #8ae234
    vec3(0.451, 0.820, 0.086), // #73d216
    vec3(0.306, 0.604, 0.024), // #4e9a06
    vec3(0.988, 0.686, 0.243), // #fcaf3e
    vec3(0.961, 0.474, 0.000), // #f57900
    vec3(0.808, 0.361, 0.000), // #ce5c00
    vec3(0.447, 0.624, 0.811), // #729fcf
    vec3(0.204, 0.396, 0.643), // #3465a4
    vec3(0.125, 0.290, 0.529), // #204a87
    vec3(0.678, 0.498, 0.659), // #ad7fa8
    vec3(0.459, 0.313, 0.482), // #75507b
    vec3(0.361, 0.208, 0.400), // #5c3566
    vec3(0.914, 0.725, 0.431), // #e9b96e
    vec3(0.757, 0.490, 0.067), // #c17d11
    vec3(0.561, 0.349, 0.008), // #8f5902
    vec3(0.937, 0.161, 0.161), // #ef2929
    vec3(0.800, 0.000, 0.000), // #cc0000
    vec3(0.643, 0.000, 0.000), // #a40000
    vec3(0.933, 0.933, 0.925), // #eeeeec
    vec3(0.827, 0.843, 0.812), // #d3d7cf
    vec3(0.729, 0.737, 0.714), // #babdb6
    vec3(0.533, 0.541, 0.522), // #888a85
    vec3(0.333, 0.333, 0.325), // #555753
    vec3(0.180, 0.204, 0.212)  // #2e3436
);


    vec3 quantized = quantizeToPalette(actual, colors);

     // Calculate error
    vec3 error = actual - quantized;

    // Distribute error to neighboring pixels (Floyd-Steinberg)
    // Note: This is a simplified example and assumes you have access to neighboring pixels.
    // In a real shader, you would need to use a texture to store and retrieve the error values.
    vec3 error_distribution = error / 16.0;
    vec3 dither = vec3(0);
    
    //dither = quantizeToPalette(texture(_t00, texture_coordinate + vec2(1.0, 0.0)).rgb, colors) + error_distribution * 7.0;
    //dither += quantizeToPalette(texture(_t00, texture_coordinate +  vec2(-1.0, 1.0)).rgb, colors) + error_distribution * 3.0;
    //dither += quantizeToPalette(texture(_t00, texture_coordinate +  vec2(0.0, 1.0)).rgb, colors) + error_distribution * 5.0;
    //dither += quantizeToPalette(texture(_t00, texture_coordinate +  vec2(1.0, 1.0)).rgb, colors) + error_distribution * 1.0;
    //dither *= error_distribution;
    // texture(_t00, texture_coordinate + vec2(-1.0, 1.0)) += error_distribution * 3.0;
    // texture(_t00, texture_coordinate + vec2(0.0, 1.0)) += error_distribution * 5.0;
    // texture(_t00, texture_coordinate + vec2(1.0, 1.0)) += error_distribution * 1.0;

    o_attachment0 = vec4(texture(_t00, a).rgb, 1.0);
}