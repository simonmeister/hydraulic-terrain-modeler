#include "GLShader.h"

#include <sstream>
#include <fstream>

using std::cerr;
using std::endl;

static void stringFromFile(const std::string& fn, std::string& destination)
{
	std::ifstream ffile(fn);
    if( !ffile.is_open() || !ffile.good())
    {
        cerr << "Can't open file \"" << fn << "\"" << endl;
    }
	std::stringstream fss;
	fss << ffile.rdbuf();
    ffile.close();
    destination.append(fss.str());
}

GLShader::GLShader()
	: m_linked(false) 
{}

GLShader::~GLShader()
{
	if(m_linked)
		glDeleteProgram(m_progID);
	for(auto iter = m_shaders.begin(); iter != m_shaders.end(); ++iter) {
		glDeleteShader(*iter);
	}
}

void GLShader::link() 
{
	if(m_linked)
		return;

	m_progID = glCreateProgram();

	for(auto iter = m_shaders.begin(); iter != m_shaders.end(); ++iter) 
		glAttachShader(m_progID,*iter);

	glLinkProgram(m_progID);

	GLint ret;
	glGetProgramiv(m_progID,GL_LINK_STATUS,&ret);
	if( ret == GL_TRUE ) 
	{
		m_linked = true;
		return;
	}
	//Linking failed!
	m_linked = false;
	GLint len;
	glGetProgramiv(m_progID,GL_INFO_LOG_LENGTH,&len);
	if(len > 0) 
	{
		char* log = new char[len];
		glGetProgramInfoLog(m_progID, len, 0 , log);
		cerr << "Program failed to link: " << log << endl; 
		delete[] log;
	}
	glDeleteProgram(m_progID);
}

void GLShader::addShaderFromFile(const std::string& fn, GLenum tp) 
{
	std::string f;
	stringFromFile(fn,f);
	//Build shader
	GLint ns = glCreateShader(tp);
	const char* cstr = f.c_str();
	glShaderSource(ns,1,(const GLchar**)(&cstr),NULL);
	glCompileShader(ns);

	GLint ret;
	glGetShaderiv(ns,GL_COMPILE_STATUS,&ret);
	if( ret == GL_TRUE )
	{
		m_shaders.push_back(ns);
		return;
	}

	//Compilation failed!
	GLint len;
	glGetShaderiv(ns,GL_INFO_LOG_LENGTH,&len);
	if(len>0)
	{
		char* log = new char[len];
		glGetShaderInfoLog(ns, len, 0 , log);
		cerr << "Shader \"" << fn << "\" failed to compile: " << log << endl; 
		delete[] log;
	}
	glDeleteShader(ns);
}
