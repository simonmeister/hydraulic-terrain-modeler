#ifndef CAMERA_H
#define CAMERA_H

#include "util/glm.h"

class Camera 
{
public:
	Camera(glm::vec3 pos, glm::vec3 dir, glm::vec3 up = glm::vec3(0.0,1.0,0.0));
	virtual ~Camera();

	inline glm::mat4 getCameraTransform() const 
	{ return tmat; }

	inline glm::vec3 getPosition() const 
	{ return position; }

	inline glm::vec3 getDirection() const
	{ return direction; }

	inline glm::vec3 getUpDirection() const
	{ return up; }

	enum moveType { FORWARD = 0, BACK, LEFT, RIGHT };

	void moveCamera(moveType t, float amount);

	void rotateCamera(float rx, float ry);

	void resetToHome();

	void setHome(glm::vec3 pos, glm::vec3 dir); 

	void updateTransform();


private:
	glm::vec3 position;
	glm::vec3 direction;
	glm::vec3 up;
	glm::mat4 tmat;

	glm::vec3 homepos;
	glm::vec3 homedir;
};

#endif //CAMERA_H
