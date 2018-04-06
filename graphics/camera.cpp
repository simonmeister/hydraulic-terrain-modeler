#include <cmath>
#include "camera.h"

#ifndef PI
#define PI 3.14159265f
#endif

Camera::Camera(glm::vec3 pos, glm::vec3 dir, glm::vec3 pup)
	:homepos(pos)
	,homedir(glm::normalize(dir))
	,up(pup)
	,tmat(glm::mat4(1.0))
{
	resetToHome();
}

Camera::~Camera()
{
}

void Camera::moveCamera(moveType t, float amount)
{
	if(t == FORWARD)
	{
		position += direction * amount;
	}
	else if(t == BACK)
	{
		position -= direction * amount;
	}
	else if(t == RIGHT)
	{
		position += glm::normalize(glm::cross(direction,up)) * amount;
	}
	else if(t == LEFT)
	{
		position -= glm::normalize(glm::cross(direction,up)) * amount;
	}

	updateTransform();
}

void Camera::rotateCamera(float rx, float ry)
{
	//check for max. pitch
	float pitch = acos(glm::dot(up,direction));
	pitch = pitch * 180.0f / PI;
	if((pitch > 179.0 && ry < 0.0) || ( pitch < 0.1 && ry > 0.0))
		    ry = 0.0;

	//yaw
	glm::mat4 dirRot = glm::rotate(glm::mat4(1.0),rx,up);
	//pitch
	dirRot = glm::rotate(dirRot,ry,glm::cross(direction,up));
	//apply
	glm::vec4 camDir4 = glm::vec4(direction.x,direction.y,direction.z,0.0);
	camDir4 = dirRot * camDir4;
	direction = glm::normalize(glm::vec3(camDir4.x,camDir4.y,camDir4.z));

	updateTransform();
}

void Camera::updateTransform()
{
	tmat = glm::lookAt(position,position+direction,up);
}
void Camera::resetToHome()
{
	position = homepos;
	direction = homedir;
	updateTransform();
}

void Camera::setHome(glm::vec3 pos, glm::vec3 dir)
{
	homepos = pos;
	homedir = glm::normalize(dir);
}

