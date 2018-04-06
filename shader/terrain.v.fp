#version 400

layout (location = 0) in vec2 position;

out vec2 posV;

void main()
{
	posV = position;
}
