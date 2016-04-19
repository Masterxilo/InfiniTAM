#include "itmlibdefines.h"
#include "itmintrinsics.h"
#include "coordinatesystem.h"
#include "ITMPose.h"
struct BeginGLRender {
    /** Fills the depth buffer and framebuffer of 
    the current OpenGL context with the specified images
    and sets up the modelview-projection/viewtransform/rasterization pipeline state
    such that points, subsequently specified within glBegin-glEnd given in world coordinates to glVertex
    will appear at the intended position.
    
    The depth image gives depth of the colored points in 
    eyeCoordinate positive z direction, in the range (viewFrustum_min, viewFrustum_max).
    z with z < viewFrustum_min are not rendered.

    The images are assumed to have been rendered from the camera described by:
    - their resolution (must be the same)
    - the intrinsics and eye coordinate system specified.
    
    TODO this could take an ITMView.
    
    It is assumed that the opengl window (backbuffer) 
    is at least as big as the supplied image - the viewport is set
    to write to that part exclusively.*/
    BeginGLRender(
        ITMUChar4Image* colorImage, //!< assumed rgba
        ITMFloatImage* depthImage,
        ITMIntrinsics depthCameraIntrinsics,
        ITMPose* cameraEyeCoordinates
        );

    ~BeginGLRender();
};