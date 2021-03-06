#Specify the version being used as well as the language
cmake_minimum_required(VERSION 2.8.11)

#Name your project here
project(JACK)

#set the module directory
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")

#find the STIM library
find_package(STIM)
#if the STIM library isn't found, download it
#if(NOT STIM_INCLUDE_DIRS)
#	file(REMOVE_RECURSE ${CMAKE_BINARY_DIR}/stimlib)	#remove the stimlib directory if it exists
#	set(STIM_GIT "https://git.stim.ee.uh.edu/codebase/stimlib.git")
#	execute_process(COMMAND git clone --depth 1 ${STIM_GIT} WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
#	set(STIM_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/stimlib" CACHE TYPE PATH)
#endif(NOT STIM_INCLUDE_DIRS)


#find the GLUT library for visualization
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
#GLEW is required for MS Visual C
if(MSVC)
	find_package(GLEW REQUIRED)
	include_directories(${GLEW_INCLUDE_DIR})
endif(MSVC)
#GLFW is required for MS Visual C
if(MSVC)
	find_package(GLFW REQUIRED)
	include_directories(${GLFW_INCLUDE_DIR})
endif(MSVC)

#find OpenCV
find_package(OpenCV REQUIRED)
add_definitions(-DUSING_OPENCV)

#find the pthreads package
find_package(Threads)

#find the X11 package
find_package(X11)

if(CMAKE_COMPILER_IS_GNUCC)
set(CMAKE_CXX_FLAGS "-std=c++11 -O2")
endif(CMAKE_COMPILER_IS_GNUCC)

#build the executable in the binary directory on MS Visual Studio
if ( MSVC )
	SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
	SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")
endif ( MSVC )

include_directories(
					${OpenGL_INCLUDE_DIRS}
					${GLUT_INCLUDE_DIR}
					${STIM_INCLUDE_DIRS}
					${OpenCV_INCLUDE_DIRS}
					)

#create an executable file
add_executable(JACK "main.cpp")

#set the link libraries
target_link_libraries(JACK
				${OPENGL_LIBRARIES}
				${GLUT_LIBRARIES}
            	${ANN_LIBRARY}
				${X11_LIBRARIES}
				${OpenCV_LIBS}
		  )
if(MSVC)
	target_link_libraries(JACK ${GLEW_LIBRARY})
endif(MSVC)
if(MSVC)
	target_link_libraries(JACK ${GLFW_LIBRARY})
endif(MSVC)
