#
# Windows users: define the GLEW_PATH environment variable to point
# to the root glew directory, which contains:
#		lib/Release/Win32/glew32.lib AND/OR lib/Release/x64/glew32.lib
#		include/GL/glew.h

#Try to find GLEW library and include path.
# Once done this will define
#
# GLEW_FOUND
# GLEW_INCLUDE_DIR
# GLEW_LIBRARY
# 

IF (WIN32)
	FIND_PATH( GLEW_INCLUDE_DIR GL/glew.h
		$ENV{GLEW_PATH}/include
		$ENV{PROGRAMFILES}/GLEW/include
		${PROJECT_SOURCE_DIR}/src/nvgl/glew/include
		DOC "The directory where GL/glew.h resides")
	if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
		FIND_LIBRARY( GLEW_LIBRARY
			NAMES glew GLEW glew32 glew32s
			PATHS
			$ENV{GLEW_PATH}/lib/Release/x64
			$ENV{PROGRAMFILES}/GLEW/lib
			${PROJECT_SOURCE_DIR}/src/nvgl/glew/bin
			${PROJECT_SOURCE_DIR}/src/nvgl/glew/lib
			DOC "The GLEW library")
	else( CMAKE_SIZEOF_VOID_P EQUAL 8 )
		FIND_LIBRARY( GLEW_LIBRARY
			NAMES glew GLEW glew32 glew32s
			PATHS
			$ENV{GLEW_PATH}/lib/Release/Win32
			$ENV{PROGRAMFILES}/GLEW/lib
			${PROJECT_SOURCE_DIR}/src/nvgl/glew/bin
			${PROJECT_SOURCE_DIR}/src/nvgl/glew/lib
			DOC "The GLEW library")
	endif( CMAKE_SIZEOF_VOID_P EQUAL 8 )
ELSE (WIN32)
	FIND_PATH( GLEW_INCLUDE_DIR GL/glew.h
		/usr/include
		/usr/local/include
		/sw/include
		/opt/local/include
		DOC "The directory where GL/glew.h resides")
	FIND_LIBRARY( GLEW_LIBRARY
		NAMES GLEW glew
		PATHS
		/usr/lib64
		/usr/lib
		/usr/local/lib64
		/usr/local/lib
		/sw/lib
		/opt/local/lib
		DOC "The GLEW library")
ENDIF (WIN32)

IF (GLEW_INCLUDE_DIR)
	SET( GLEW_FOUND 1 CACHE STRING "Set to 1 if GLEW is found, 0 otherwise")
ELSE (GLEW_INCLUDE_DIR)
	SET( GLEW_FOUND 0 CACHE STRING "Set to 1 if GLEW is found, 0 otherwise")
ENDIF (GLEW_INCLUDE_DIR)

MARK_AS_ADVANCED( 
	GLEW_FOUND 
	GLEW_INCLUDE_DIR
	GLEW_LIBRARY
)