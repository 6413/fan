#version 130

precision highp float;

out vec4 ShapeColor;

in vec4 color;

void main()
{
    ShapeColor = color;
} 