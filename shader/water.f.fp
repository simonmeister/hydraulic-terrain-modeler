#version 400

uniform	sampler2D heightTexture;
uniform float gridSpacing;
uniform samplerCube reflectionCubemap;
uniform vec3 cameraPosition;
uniform vec2 mousePos;
uniform float brushRadius;
uniform vec3 markerColor;
uniform float transparencyBase;
uniform float transparencyMult;
uniform vec3 waterBase;
uniform int showMarker;
uniform int useReflections;
uniform int useTransparency;

in vec2 uvTE;
in float discardTE;
out vec4 outColor;

float height(float u, float v) 
{
	return texture(heightTexture, vec2(u,v)).g;
}

float combHeight(float u, float v) 
{
	vec4 t = texture(heightTexture, vec2(u,v));
	return t.g+t.r;
}

//------------------------------------------------------------------------
// Constants for Spherical harmonics, taken from 
// The OpenGl Shading Language 3 */
//------------------------------------------------------------------------

const float C1 = 0.429043;
const float C2 = 0.511664;
const float C3 = 0.743125;
const float C4 = 0.886227;
const float C5 = 0.247708;

			// brighter/blue-ish
const vec3 L00 = vec3( 0.38, 0.43, 0.45);
const vec3 L1m1 = vec3( 0.29, 0.36, 0.41);
const vec3 L10 = vec3( 0.04, 0.03, 0.01);
const vec3 L11 = vec3(-0.10, -0.10, -0.09);
const vec3 L2m2 = vec3(-0.06, -0.06, -0.04);
const vec3 L2m1 = vec3( 0.01, -0.01, -0.05);
const vec3 L20 = vec3(-0.09, -0.13, -0.15);
const vec3 L21 = vec3(-0.06, -0.05, -0.04);
const vec3 L22 = vec3(0.02, 0.00, -0.05); 

void main() 
{

	//discard zero-height fragments for water
	if( height(uvTE.x,uvTE.y) < 0.001)
		discard;

	//discard unused fragments
	if( discardTE > 0.7)
		discard;
	//------------------------------------------------------------------------
	// compute marker contribution
	//------------------------------------------------------------------------

	vec2 tSize = textureSize(heightTexture,0);
	float aspectRatio = tSize.x/tSize.y;

    float dist = distance(vec2(uvTE.x*aspectRatio,uvTE.y), vec2(mousePos.x*aspectRatio, mousePos.y));

	vec3 markerGlow;
    if(dist >  brushRadius && dist < brushRadius* 1.15 && showMarker == 1)
	{
        markerGlow = markerColor;
    }
    else
	{
		markerGlow = vec3(1.0, 1.0, 1.0);
    }

	//------------------------------------------------------------------------
	// Approximate normal
	//------------------------------------------------------------------------
	vec2 dstep = 1.0 /tSize;

	vec3 deltaX = vec3(2.0 * gridSpacing,
				height(uvTE.s + dstep.x, uvTE.t) - height(uvTE.s - dstep.x, uvTE.t) , 
				0.0);
								
	vec3 deltaZ = vec3( 0.0, 
				height(uvTE.s, uvTE.t + dstep.y) - height(uvTE.s, uvTE.t - dstep.y) , 
				2.0 * gridSpacing) ;
	
	vec3 normalF = normalize(cross(deltaZ, deltaX));

	//---------------------
	vec3 color;
	float heightMag = (height(uvTE.s,uvTE.t)
	                           / 100.0 * transparencyMult ) + transparencyBase;
	//------------------------------------------------------------------------
	// Sample cubemap for fake-reflection
	//------------------------------------------------------------------------
	
	if(useReflections == 1)
	{
		vec3 position = vec3(uvTE.s, combHeight(uvTE.s,uvTE.t), uvTE.t) * textureSize(heightTexture,0).r * 10 ;
		vec3 toCamera = normalize(cameraPosition - position) ;
		vec3 r = reflect(toCamera, normalF);
		color= texture(reflectionCubemap, r).rgb;

		vec3 mult = waterBase * markerGlow;
		if(useTransparency == 1)
			outColor = vec4(color * mult, heightMag);
		else
			outColor = vec4(color * mult, 1.0);
	}
	else
	{
		//------------------------------------------------------------------------
		// Compute lighting using SH
		//------------------------------------------------------------------------
		vec3 lightColor = 
			(C1 * L22 * (normalF.x * normalF.x - normalF.y * normalF.y) +
			C3 * L20 * normalF.z * normalF.z + C4 * L00 - C5 * L20 +
			2.0 * C1 * L2m2 * normalF.x * normalF.y +
			2.0 * C1 * L21 * normalF.x * normalF.z +
			2.0 * C1 * L2m1 * normalF.y * normalF.z +
			2.0 * C2 * L11 * normalF.x +
			2.0 * C2 * L1m1 * normalF.y +
			2.0 * C2 * L10 * normalF.z);

		color = vec3(heightMag,pow(heightMag,2),1.0);

		vec3 mult = waterBase * markerGlow;
		if(useTransparency == 1)
			outColor = vec4(color * mult, heightMag);
		else
			outColor = vec4(color * mult *lightColor, 1.0);
	}

}
