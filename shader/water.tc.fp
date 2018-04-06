#version 400

layout(vertices = 1) out;


uniform float gridSpacing;

uniform mat4 pv;
uniform ivec2 viewportDim;

uniform sampler2D heightTexture;

uniform int pixelsPerEdge;
uniform float patchSize;

in vec2 posV[];
out vec2 posTC[];

#define ID gl_InvocationID

float height(float u, float v) 
{
	return texture(heightTexture, vec2(u,v)).g;
}

// Checks if a segment is at least partially inside the frustum
bool segmentInFrustum(vec4 p1, vec4 p2) 
{
	if ((p1.x < -p1.w && p2.x < -p2.w) || (p1.x > p1.w && p2.x > p2.w) ||
//		(p1.y < -p1.w && p2.y < -p2.w) || (p1.y > p1.w && p2.y > p2.w) ||
		(p1.z < -p1.w && p2.z < -p2.w) || (p1.z > p1.w && p2.z > p2.w))
		return false;
	else
		return true;
}

// Measures the screen size of segment p1-p2
float screenSphereSize(vec4 p1, vec4 p2) 
{
	vec4 viewCenter = (p1+p2) * 0.5;
	vec4 viewUp = viewCenter;
	viewUp.y += distance(p1,p2);
	vec4 p1Proj = viewCenter;
	vec4 p2Proj = viewUp;

	vec4 p1NDC, p2NDC;
	p1NDC = p1Proj/p1Proj.w;
	p2NDC = p2Proj/p2Proj.w;
	
	return( clamp(length((p2NDC.xy - p1NDC.xy) * viewportDim * 0.5) / (pixelsPerEdge), 1.0, patchSize));
}

//Transforms a heightmap position into screenspace
vec4 screenSpace( vec2 mapSize, vec2 uv )
{
	vec2 sc = uv * mapSize * gridSpacing;
	return pv * vec4(sc.x, height(uv.x,uv.y), sc.y, 1.0);
}

void main() 
{
	vec2 iLevel;
	vec4 oLevel;

	vec4 posTransV[4];
	vec2 pAux;
	vec2 posTCAux[4];

	ivec2 tSize = textureSize(heightTexture,0);
	vec2 div = patchSize / tSize;
	
	//compute the 4 vertex positions from lower left base vertex
	posTCAux[0] = posV[0];
	posTCAux[1] = posV[0] + vec2(0.0, div.y);
	posTCAux[2] = posV[0] + vec2(div.x,0.0);
	posTCAux[3] = posV[0] + vec2(div.x,div.y);
	
	//transform vertices into screen space
	posTransV[0] = screenSpace(tSize, posTCAux[0]);
	posTransV[1] = screenSpace(tSize, posTCAux[1]);
	posTransV[2] = screenSpace(tSize, posTCAux[2]);
	posTransV[3] = screenSpace(tSize, posTCAux[3]);

	if (segmentInFrustum(posTransV[ID], posTransV[ID+1]) ||
		segmentInFrustum(posTransV[ID], posTransV[ID+2]) ||
		segmentInFrustum(posTransV[ID+2], posTransV[ID+3]) ||
		segmentInFrustum(posTransV[ID+3], posTransV[ID+1])) 			
	{				
		// screen size dependent lod
		oLevel = vec4(screenSphereSize(posTransV[ID], posTransV[ID+1]),
					screenSphereSize(posTransV[ID+0], posTransV[ID+2]),
					screenSphereSize(posTransV[ID+2], posTransV[ID+3]),
					screenSphereSize(posTransV[ID+3], posTransV[ID+1]));
		iLevel = vec2(max(oLevel[1] , oLevel[3]) , max(oLevel[0] , oLevel[2]) );		
	}
	else 
	{
		oLevel = vec4(0);
		iLevel = vec2(0);		
	} 
	posTC[ID] = posV[ID];
	gl_TessLevelOuter[0] = oLevel[0];
	gl_TessLevelOuter[1] = oLevel[1];
	gl_TessLevelOuter[2] = oLevel[2];
	gl_TessLevelOuter[3] = oLevel[3];
	gl_TessLevelInner[0] = iLevel[0];
	gl_TessLevelInner[1] = iLevel[1];
}