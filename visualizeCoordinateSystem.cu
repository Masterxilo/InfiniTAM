#include <iostream>
using namespace std;
#include <gl/freeglut.h>
#include "visualizeCoordinateSystem.h"

#include "itmpixelutils.h"


BeginGLRender::BeginGLRender(
    ITMUChar4Image* colorImage,
    ITMFloatImage* depthImage,
    ITMIntrinsics depthCameraIntrinsics,
    ITMPose* cameraEyeCoordinates
    ) {
    const uint W = colorImage->noDims.width;
    const uint H = colorImage->noDims.height;
    assert(W == depthImage->noDims.width &&
        H == depthImage->noDims.height);

    // invert standard OpenGl viewport along y axis
    // to make y increase downwards
    //glViewport(0, H, W, -H); // does not work!

    glViewport(0, 0, W, H);

    glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LESS);
    glutReportErrors();
    glClearColor(0, 0, 0, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


    // fill depth buffer with depthImage
    // (which are freeview.raycastResult's z values (multiplied by voxelSize)
    // and perspectively project such that
    //      [viewFrustum_min, viewFrustum_max] |-> [0,1]
    // Note: This is not a linear transformation, see projectionMatrix.nb

    auto gl_depthbuffer_float_image = new Image<float>(depthImage->noDims);
    
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            int i = pixelLocId(x, y, depthImage->noDims);
        auto depthImageEyeZ = depthImage->GetData()[i];


         i = pixelLocId(x, H - 1-y, depthImage->noDims);// must invert y axis for destination
        if (depthImageEyeZ < viewFrustum_min) {
            gl_depthbuffer_float_image->GetData()[i] = 1; // raycast hit nothing -> z-buffer value = max = 1
            continue;
        }
        assert(depthImageEyeZ <= viewFrustum_max);

        const float z = // projectionMatrix.nb
            viewFrustum_max*(depthImageEyeZ - viewFrustum_min) /
            (depthImageEyeZ*(viewFrustum_max - viewFrustum_min));

        gl_depthbuffer_float_image->GetData()[i] = z;
        assert(z >= 0 && z <= 1);
    }
    assert(gl_depthbuffer_float_image->noDims == Vector2i(W, H));

    typedef void(*glWindowPos3fT)(float, float, float);
    auto glWindowPos3f = (glWindowPos3fT)wglGetProcAddress("glWindowPos3f");
    assert(glWindowPos3f);

    // Copy to gl depth buffer
    glColorMask(0, 0, 0, 0);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glWindowPos3f(0, 0, 0); // required 
    glDrawPixels(W, H,
        GL_DEPTH_COMPONENT, GL_FLOAT,
        gl_depthbuffer_float_image->GetData());
    glutReportErrors();
    glColorMask(1, 1, 1, 1);


    // fill color buffer


    assert(colorImage->noDims == Vector2i(W, H));
    glDepthMask(0);

    // draw pixels starts drawing at current raster pos
    // somehow cannot set it in global coodinates but this works

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glWindowPos3f(0, 0, 0); // required 

    const int x = 0;
    for (int y = H - 1; y >= 0; y--) {
        //for (int x = 0; x < W; x++) 
            glWindowPos3f(x, H-1-y, 0); // required 


            const  int i = pixelLocId(x, y, colorImage->noDims);
            glDrawPixels(W,1,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                colorImage->GetData()[i]);
        }
    /*
    glWindowPos3f(0, 0, 0); // required 
    glDrawPixels(W, H,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        colorImage->GetData());*/
    glutReportErrors();
    glDepthMask(1);

    // projection matrix setup
    // c.f. projectionMatrix.nb
    float fx = depthCameraIntrinsics.projectionParamsSimple.fx;
    float fy = depthCameraIntrinsics.projectionParamsSimple.fy;
    float cx = depthCameraIntrinsics.projectionParamsSimple.px;
    float cy = depthCameraIntrinsics.projectionParamsSimple.py;
    float width = W;
    float height = H;
    float zmin = viewFrustum_min;
    float zmax = viewFrustum_max;

    float columnMajorProjectionMatrix[16] = {
        // col1
        (2 * fx) / width,
        0,
        0,
        0,
        // 2
        0,
        (2 * fy) / height,
        0,
        0,

        // 3
        -1 + (2 * cx) / width,
        -1 + (2 * cy) / height,
        (zmax + zmin) / (zmax - zmin),
        1,

        // 4
        0,
        0,
        (2 * zmax*zmin) / (-zmax + zmin),
        0
    };
    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf(columnMajorProjectionMatrix);
    cout << "gl projection matrix: " << endl << Matrix4f(columnMajorProjectionMatrix) << endl;

    // finally, invert y-axis
    glScalef(1,-1,1);

    // world-to-view (eye) transform
    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixf(cameraEyeCoordinates->GetM().m);
    cout << "gl mv matrix: " << endl << cameraEyeCoordinates->GetM() << endl;

    // ready!

    // draw something
    glBegin(GL_LINES);
    float zoff = 0;

    // draw major axis of world coordinate system with unit length
#define ax(x,y,z)\
    glColor4f((x),(y),(z),1);\
    glVertex3f(0, 0, zoff);\
    glVertex3f((x),(y),(z)+zoff);

    ax(1, 0, 0);
    ax(0, 1, 0);
    ax(0, 0, 1);

    glEnd();
}

BeginGLRender::~BeginGLRender() {

}