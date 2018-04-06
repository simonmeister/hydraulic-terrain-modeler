#ifndef GLShader_H
#define GLShader_H

#include <vector>
#include <iostream>
#include <string>

#include "util/global.h"
#include "3rdParty/glew/glew.h"



/*Very simple shader-handling class */
class GLShader 
{
public:
	GLShader();
	~GLShader();

	void addShaderFromFile(const std::string& filename, GLenum shaderType);	
	void link();

	//A program returned from here may be invalid. To validate, call isUsable first.
	inline GLuint getProgram() const 
	{
		return m_progID;
	}

	inline bool isUsable() const 
	{ 
		glValidateProgram(m_progID);
		GLint valid;
		glGetProgramiv(m_progID,GL_VALIDATE_STATUS,&valid);
		return m_linked && (valid == 1);
	}

	inline GLint getUniformLocation(const std::string& name) 
	{
		GLint unif = glGetUniformLocation(m_progID,name.c_str());
		if( unif == -1 ) 
		{
			std::cerr << "Uniform location \"" << name << "\" does not exist!" << std::endl;
			return -1;
		}
		return unif;
	}
  
private:
	GLShader(const GLShader& other){}

	//Accelerate 
	bool m_linked;
	GLuint m_progID;
	std::vector<GLint> m_shaders;
};
#endif // GLShader_H
