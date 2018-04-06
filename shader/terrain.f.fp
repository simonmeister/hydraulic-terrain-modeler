#version 400

uniform	sampler2D heightTexture;
uniform sampler2D matTexture;
uniform sampler1D materialColorTexture;

uniform float gridSpacing;
uniform vec2 mousePos;
uniform vec2 constantMarkerPosition;
uniform float brushRadius;
uniform float constantMarkerRadius;
uniform vec3 markerColor;
uniform vec3 constantMarkerColor;
uniform float terrainMult;
uniform int occlusionSamples;
uniform int terrainLighting;
uniform int showMarker;
uniform int showConstantMarker;


in vec2 uvTE;
in float discardTE;
out vec4 outColor;

float height(float u, float v) 
{
	return texture(heightTexture, vec2(u,v)).r;
}

vec3 materialColor(float u, float v) 
{
	// scale matID from [0...matCount/255] to [0...matCount] 
	float matID = texture(matTexture, vec2(u,v)).r * 255;
	// scale matID from [0...matCount] to [0...1] 
	//matID /= textureSize(materialColorTexture, 0 );
	return texture(materialColorTexture, matID).rgb;
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

//make occlusion samples and some sample settings + diffuse color base uniform 
#define occlusionSampleRaylen 5.0
#define PI 3.14159265358979323846


void main() 
{
	//discard unused fragments
	if( discardTE > 0.9)
		discard;

	//------------------------------------------------------------------------
	// compute marker contributions to cell color
	//------------------------------------------------------------------------
	vec3 markerGlow;

	vec2 tsize = textureSize(heightTexture,0);
	float dist;

	vec3 brushMarkerColor = vec3(1.0, 1.0, 1.0);
	vec3 constMarkerColor = vec3(1.0, 1.0, 1.0);
	if(showMarker == 1)
	{
		if(tsize.x < tsize.y)
		{
			float aspectRatio = tsize.y/tsize.x;
			dist = distance(vec2(uvTE.x,uvTE.y*aspectRatio), vec2(mousePos.x, mousePos.y*aspectRatio));
		}
		else 
		{
			float aspectRatio = tsize.x/tsize.y;
			dist = distance(vec2(uvTE.x*aspectRatio,uvTE.y), vec2(mousePos.x*aspectRatio, mousePos.y));
		}
	
		if(dist >  brushRadius && dist < brushRadius* 1.15 )
		{
			brushMarkerColor = markerColor;
		}
    }
	if(showConstantMarker == 1)
	{
		if(tsize.x < tsize.y)
		{
			float aspectRatio = tsize.y/tsize.x;
			dist = distance(vec2(uvTE.x,uvTE.y*aspectRatio), vec2(constantMarkerPosition.x, constantMarkerPosition.y*aspectRatio));
		}
		else 
		{
			float aspectRatio = tsize.x/tsize.y;
			dist = distance(vec2(uvTE.x*aspectRatio,uvTE.y), vec2(constantMarkerPosition.x*aspectRatio, constantMarkerPosition.y));
		}
	
		if(dist >  constantMarkerRadius && dist < constantMarkerRadius* 1.15)
		{
			constMarkerColor = constantMarkerColor;
		}
    }

	markerGlow = constMarkerColor * brushMarkerColor;
	//------------------------------------------------------------------------
	// Approximate normal
	//------------------------------------------------------------------------

	vec2 dstep = 1.0 /tsize;
	vec3 deltaX = vec3(2.0 * gridSpacing,
				height(uvTE.s + dstep.x, uvTE.t) - height(uvTE.s - dstep.x, uvTE.t) , 
				0.0) ;
				
	vec3 deltaZ = vec3( 0.0, 
				height(uvTE.s, uvTE.t + dstep.y) - height(uvTE.s, uvTE.t - dstep.y) , 
				2.0 * gridSpacing) ;
	
	vec3 normalF = normalize(cross(deltaZ, deltaX));

	//------------------------------------------------------------------------
	// Compute lighting using SH
	//------------------------------------------------------------------------
	vec3 L00,L1m1,L10,L11,L2m2,L2m1,L20,L21,L22;

	if(terrainLighting == 2)
	{
		// darker
		L00 = vec3( 0.871297, 0.875222, 0.864470);
		L1m1 = vec3( 0.175058, 0.245335, 0.312891);
		L10 = vec3( 0.034675, 0.036107, 0.037362);
		L11 = vec3(-0.004629, -0.029448, -0.048028);
		L2m2 = vec3(-0.120535, -0.121160, -0.117507);
		L2m1 = vec3( 0.003242, 0.003624, 0.007511);
		L20 = vec3(-0.028667, -0.024926, -0.020998);
		L21 = vec3(-0.077539, -0.086325, -0.091591);
		L22 = vec3(-0.161784, -0.191783, -0.219152);
	}
	else if(terrainLighting == 1) 
	{
		//light
		L00 =	vec3( 0.6841148, 0.6929004, 0.7069543);
		L1m1 =	vec3( 0.3173355, 0.3694407, 0.4406839);
		L10 =	vec3(-0.1747193, -0.1737154, -0.1657420);
		L11 =	vec3(-0.4496467, -0.4155184, -0.3416573);
		L2m2 =	vec3(-0.1690202, -0.1703022, -0.1525870);
		L2m1 =	vec3(-0.0837808, -0.0940454, -0.1027518);
		L20 =	vec3(-0.0319670, -0.0214051, -0.0147691);
		L21 =	vec3( 0.1641816, 0.1377558, 0.1010403);
		L22 =	vec3( 0.3697189, 0.3097930, 0.2029923); 
	}
	else if(terrainLighting == 0) 
	{
		// brighter/blue-ish
		L00 = vec3( 0.38, 0.43, 0.45);
		L1m1 = vec3( 0.29, 0.36, 0.41);
		L10 = vec3( 0.04, 0.03, 0.01);
		L11 = vec3(-0.10, -0.10, -0.09);
		L2m2 = vec3(-0.06, -0.06, -0.04);
		L2m1 = vec3( 0.01, -0.01, -0.05);
		L20 = vec3(-0.09, -0.13, -0.15);
		L21 = vec3(-0.06, -0.05, -0.04);
		L22 = vec3(0.02, 0.00, -0.05); 
	}

	vec3 lightColor = 
		(C1 * L22 * (normalF.x * normalF.x - normalF.y * normalF.y) +
		C3 * L20 * normalF.z * normalF.z + C4 * L00 - C5 * L20 +
		2.0 * C1 * L2m2 * normalF.x * normalF.y +
		2.0 * C1 * L21 * normalF.x * normalF.z +
		2.0 * C1 * L2m1 * normalF.y * normalF.z +
		2.0 * C2 * L11 * normalF.x +
		2.0 * C2 * L1m1 * normalF.y +
		2.0 * C2 * L10 * normalF.z);

	//------------------------------------------------------------------------
	// Compute occlusion
	//------------------------------------------------------------------------
	vec3 pos = vec3(uvTE.s, height(uvTE.s,uvTE.t), uvTE.t);
	float occlusion = 0.0;
    for(int i=1; i<occlusionSamples; i++)
	{
        float s = float(i)/32.0;
        float a = sqrt(s * 1024.0);
        float b = sqrt(s);
        float x = sin(a) * b * occlusionSampleRaylen;
        float y = cos(a) * b * occlusionSampleRaylen;

        vec3 samplePos = vec3(x,height(x, y),y);
        vec3 sampleDir = normalize(samplePos - pos);

        float lambert = clamp(dot(normalF, sampleDir), 0.0, 1.0);
        float distanceScale = 0.23/sqrt(length(samplePos - pos));
        occlusion += distanceScale * lambert;
    }

	//------------------------------------------------------------------------
	// Combine
	//------------------------------------------------------------------------
	vec3 diffuse = lightColor * materialColor(uvTE.x, uvTE.y) * markerGlow;

	outColor = (terrainMult * vec4(diffuse,1.0)) - occlusion * 0.07;
	//float tmp = 1.0 - occlusion * 0.1;
	//outColor = vec4(tmp,tmp,tmp,1.0);
	//outColor = vec4(1.0,0.0,0.0,1.0);
}
