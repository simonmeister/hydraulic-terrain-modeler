#version 400

layout(quads, fractional_even_spacing, cw) in;

uniform sampler2D heightTexture;
uniform float gridSpacing;
uniform mat4 pv;
uniform float patchSize;

in vec2 posTC[];
out vec2 uvTE;
out float discardTE;

#define sizeFactor 1.0/patchSize
#define uv gl_TessCoord

float height(float u, float v) 
{
	return texture(heightTexture, vec2(u,v)).r;
}

void main() 
{
	ivec2 tSize = textureSize(heightTexture,0);
	vec2 patchNum = tSize * sizeFactor;

	// Compute texture coordinates
	uvTE = posTC[0].xy + uv.st/patchNum;

	
	// Compute vertex position [0 ... 1] to [0 ... tSize * gridSpacing]
	vec4 pos;
	pos.x = uvTE.s * tSize.x * gridSpacing;
	pos.z = uvTE.t * tSize.y * gridSpacing;
	pos.y = height(uvTE.s, uvTE.t);
	pos.w = 1.0;
	gl_Position = pv * pos;

	//number of patches was rounded up to make rendering of non-evenly-divisible-by-64-patches possible
	//this method will generate some vertices that cant be assigned to a heightmap position. 
	//These positions are going to be discarded.
	if(pos.x > tSize.x || pos.z > tSize.y)
		discardTE = 1.0;
	else
		discardTE = 0.0;
}