#include <gl/freeglut.h>
#include "fileutils.h"

#include "image.h"
#include "itmlibdefines.h"
#include "positive.h"
#include "CoordinateSystem.h"
#include "CameraImage.h"

using namespace ORUtils;
const int  W = 640;
const int H = 480;

void saveRGBA(std::string filename) {
    auto image = new ITMUChar4Image(Vector2i(W, H));
    glutReportErrors();
    glReadPixels(0, 0, W, H,
        GL_RGBA, GL_UNSIGNED_BYTE,
        image->GetData());
    glutReportErrors();
    png::SaveImageToFile(image, filename);
    puts("saved rgb");
}

void saveDepth(std::string filename) {
    auto depth = new ITMShortImage(Vector2i(W, H));
    glutReportErrors();
    glReadPixels(0, 0, W, H, 
        GL_DEPTH_COMPONENT, GL_SHORT,
        depth->GetData());
    glutReportErrors();
    png::SaveImageToFile(depth, filename);
    puts("saved depth");
}

static void render(void)
{
    glViewport(0, H, W, -H);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glutReportErrors();
    glClearColor(0,0,0, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT| GL_DEPTH_BUFFER_BIT);

    
    // fill depth buffer with freeview.raycastResult.dump's z values (multiply vectors by voxelSize
    // and stretch to viewFrustum_min - viewFrustum_max == [0,1])
    
        auto depth = new Image<Vector4f>(Vector2i(W, H));
        dump::ReadImageFromFile(depth, "freeview.raycastResult.dump");
        glutReportErrors(); 
        //glRasterPos2f(0,0);
        auto depth_float_image = new Image<float>(depth->noDims);
        for (int i = 0; i < depth->dataSize; i++) {
            auto rr = depth->GetData()[i];
            if (rr.w != 1.0) continue; // raycast hit nothing
            auto z = rr.z*voxelSize; // convert raycastResult to world coordinates

            z = viewFrustum_max*(z - viewFrustum_min) / 
                (z*(viewFrustum_max - viewFrustum_min))
                
                + 0.01 // a little bias to move it away
                ;// (z - viewFrustum_min) / (viewFrustum_max - viewFrustum_min); // scale z to 0-1 according to 

            depth_float_image->GetData()[i] = z;
            assert(z >= 0 && z <= 1);
        }
        assert(depth_float_image->noDims == Vector2i(W, H));
/*
        glColorMask(0, 0, 0, 0);
        glDrawPixels(W, H,
            GL_DEPTH_COMPONENT, GL_FLOAT,
            depth_float_image->GetData());
        glutReportErrors();
        glColorMask(1,1,1,1);
    
    
    // fill color buffer with render.png
    {
        auto render = new ITMUChar4Image(Vector2i(W, H));
        png::ReadImageFromFile(render, "render.png");
        glutReportErrors();
        //glRasterPos2f(0,0);
        assert(render->noDims == Vector2i(W, H));
        glDepthMask(0);
        glDrawPixels(W, H,
            GL_RGBA, 
            GL_UNSIGNED_BYTE,
            render->GetData());
        glutReportErrors();
        glDepthMask(1);
    }*/

    glBegin(GL_TRIANGLES);
    /*glColor4f(1, 1,1, 1);


    glVertex3f(0, 0, 0);
    glVertex3f(1, 0, 0);
    glVertex3f(0, 1, 0);
    
    glColor4f(1, 0, 0, 1);
    glVertex3f(0, 0, -1);
    glVertex3f(1, 0, -1);
    glVertex3f(0, 1, 1);
*/
    glColor4f(1, 1, 1, 1);
    
    glEnd();

    //
    glMatrixMode(GL_PROJECTION);
    ITMIntrinsics intrin;
    float fx = intrin.projectionParamsSimple.fx;
    float fy = intrin.projectionParamsSimple.fy;
    float cx = intrin.projectionParamsSimple.px;
    float cy = intrin.projectionParamsSimple.py;
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
    glLoadMatrixf(columnMajorProjectionMatrix);
    puts("load");
    glutReportErrors();
    glBegin(GL_LINES);
    
    glColor4f(0, 1,0, 1);
    glVertex3f(0, 0, (zmin + zmax) / 2);
    glVertex3f(0.1, 0, (zmin + zmax) / 2);
    glVertex3f(0, 1, (zmin + zmax) / 2);
/*
    for (int i = 0; i < depth->dataSize; i++) {
        auto rr = depth->GetData()[i];
        if (rr.w != 1.0) continue; // raycast hit nothing
        rr = rr*voxelSize; // convert raycastResult to world coordinates


        glVertex3f(rr.x, rr.y, rr.z);
    }
    */

    glEnd();

    glBegin(GL_LINES);
    float zoff = (zmin + zmax) / 2;
#define ax(x,y,z)\
    glColor4f((x),(y),(z),1);\
    glVertex3f(0, 0, zoff);\
    glVertex3f((x),(y),(z)+zoff);

    ax(1, 0, 0);
    ax(0, 1, 0);
    ax(0, 0, 1);

    glEnd();

    glLoadIdentity();
    puts("load identity");
    glutReportErrors();
    //


    glFinish();
    //for (int i = 0; i < 2; i++)  // only after doing this twice will the framebuffer be correct - but the screen will not
    glutSwapBuffers();

    saveRGBA("rgba.png");
    saveDepth("depth_out.png");

    glutReportErrors();
    puts("ok");
}

void doit(positive x) {

}



CPU_AND_GPU void testCS(CoordinateSystem* o) {
    auto g = CoordinateSystem::global();
    assert(g);

    Point a = Point(o, Vector3f(0.5, 0.5, 0.5));
    assert(a.coordinateSystem == o);
    Point b = g->convert(a);
    assert(b.coordinateSystem == g);
    assert(!(b == Point(g, Vector3f(1, 1, 1))));
    assert((b == Point(g, Vector3f(0.25, 0.25, 0.25))));

    // Thus:
    Point c = o->convert(Point(g, Vector3f(1, 1, 1)));
    assert(c.coordinateSystem == o);
    assert(!(c == Point(o, Vector3f(1, 1, 1))));
    assert((c == Point(o, Vector3f(2, 2, 2))));

    Point d = o->convert(c);
    assert(c == d);

    Point e = o->convert(g->convert(c));
    assert(c == e);
    assert(g->convert(c) == Point(g, Vector3f(1, 1, 1)));

    Point f = o->convert(g->convert(o->convert(c)));
    assert(c == f);

    // +
    Point q = Point(g, Vector3f(1, 1, 2)) + Vector(g, Vector3f(1, 1, 0));
    assert(q.location == Vector3f(2, 2, 2));

    // -
    {
        Vector t = Point(g, Vector3f(0, 0, 0)) - Point(g, Vector3f(1, 1, 2));
        assert(t.direction == Vector3f(-1, -1, -2));
    }

    {
        Vector t = Vector(g, Vector3f(0, 0, 0)) - Vector(g, Vector3f(1, 1, 2));
        assert(t.direction == Vector3f(-1, -1, -2));
    }

    // dot (angle if the coordinate system is orthogonal and the vectors unit)
    assert(Vector(o, Vector3f(1, 2, 3)).dot(Vector(o, Vector3f(3, 2, 1))) == 1 * 3 + 2 * 2 + 1 * 3);
}

KERNEL ktestCS(CoordinateSystem* o) {
    testCS(o);
}
__managed__ CoordinateSystem* cs;
CPU_AND_GPU void testCi(
    const DepthImage* const di,
    const PointImage* const pi) {
    Vector2i imgSize(640, 480);
    assert(di->location() == Point(cs, Vector3f(0, 0, 0)));
    {
        auto r = di->getRayThroughPixel(Vector2i(0, 0), 1);
        assert(r.origin == Point(cs, Vector3f(0, 0, 0)));
        assert(!(r.direction == Vector(cs, Vector3f(0, 0, 1))));
    }
    {
        auto r = di->getRayThroughPixel(imgSize / 2, 1);
        assert(r.origin == Point(cs, Vector3f(0, 0, 0)));
        assert(r.direction == Vector(cs, Vector3f(0, 0, 1)));
    }
    {
        auto r = di->getRayThroughPixel(imgSize / 2, 2);
        assert(r.origin == Point(cs, Vector3f(0, 0, 0)));
        assert(r.direction == Vector(cs, Vector3f(0, 0, 2)));
    }
    {
        auto r = di->getPointForPixel(Vector2i(0, 0));
        assert(r == Point(cs, Vector3f(0, 0, 0)));
    }
    {
        auto r = di->getPointForPixel(Vector2i(1, 0));
        assert(!(r == Point(cs, Vector3f(0, 0, 0))));
        assert(r.location.z == 1);
        auto ray = di->getRayThroughPixel(Vector2i(1, 0), 1);
        assert(ray.endpoint() == r);
    }


    assert(pi->location() == Point(cs, Vector3f(0, 0, 0)));
    assert(CoordinateSystem::global()->convert(pi->location()) == Point(CoordinateSystem::global(), Vector3f(0, 0, 1)));
    assert(
        cs->convert(Point(CoordinateSystem::global(), Vector3f(0, 0, 0)))
        ==
        Point(cs, Vector3f(0, 0, -1))
        );

    {
        auto r = pi->getPointForPixel(Vector2i(0, 0));
        assert(r == Point(cs, Vector3f(0, 0, 0)));
    }
    {
        auto r = pi->getPointForPixel(Vector2i(1, 0));
        assert(r == Point(cs, Vector3f(1, 1, 1)));
    }

    Vector2f pt_image;
    assert(pi->project(Point(CoordinateSystem::global(), Vector3f(0, 0, 2)), pt_image));
    assert(pt_image == (1 / 2.f) * imgSize.toFloat());// *(1 / 2.f));

    assert(pi->project(Point(di->eyeCoordinates, Vector3f(0, 0, 1)), pt_image));
    assert(pt_image == (1 / 2.f) * imgSize.toFloat());// *(1 / 2.f));
    assert(!pi->project(Point(CoordinateSystem::global(), Vector3f(0, 0, 0)), pt_image));

    assert(Point(di->eyeCoordinates, Vector3f(0, 0, 1))
        ==
        di->eyeCoordinates->convert(Point(CoordinateSystem::global(), Vector3f(0, 0, 2)))
        );
}

KERNEL ktestCi(
    const DepthImage* const di,
    const PointImage* const pi) {

    testCi(di, pi);
}
void testCameraImage() {
    ITMIntrinsics intrin;
    Vector2i imgSize(640, 480);
    auto depthImage = new ITMFloatImage(imgSize);
    auto pointImage = new ITMFloat4Image(imgSize);

    depthImage->GetData()[1] = 1;
    pointImage->GetData()[1] = Vector4f(1,1,1,1);
    // must submit manually
    depthImage->UpdateDeviceFromHost();
    pointImage->UpdateDeviceFromHost();

    Matrix4f cameraToWorld;
    cameraToWorld.setIdentity();
    cameraToWorld.setTranslate(Vector3f(0, 0, 1));
    cs = new CoordinateSystem(cameraToWorld);
    auto di = new DepthImage(depthImage, cs, intrin.projectionParamsSimple.all);
    auto pi = new PointImage(pointImage, cs, intrin.projectionParamsSimple.all);

    testCi(di, pi);
    ktestCi << <1, 1 >> >(di, pi);
    
}

int main(int argc, char** argv)
{
    testCameraImage();
    cudaDeviceSynchronize();
    // o gives points with twice as large coordinates as the global coordinate system
    Matrix4f m;
    m.setIdentity();
    m.setScale(0.5); // scale down by half to get the global coordinates of the point
    auto o = new CoordinateSystem(m);

    testCS(o);
    ktestCS<<<1,1>>>(o);
    cudaDeviceSynchronize();


    assert(_fpclass(-0.f) == _FPCLASS_NZ);
    assert(_fpclass(-0.f * 5.f) == _FPCLASS_NZ);
    assert(_fpclass(-0.f * -5.f) == _FPCLASS_PZ);
    assert(-0.f >= 0.f);
    doit(positive(1.3f));
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(W, H);
    glutCreateWindow("Hello World");
    glutDisplayFunc(&render);

    glutMainLoop();
}