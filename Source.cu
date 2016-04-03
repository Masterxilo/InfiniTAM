#include <gl/freeglut.h>
#include "fileutils.h"
#include "image.h"
#include "itmlibdefines.h"
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
    {
        auto depth = new Image<Vector4f>(Vector2i(W, H));
        dump::ReadImageFromFile(depth, "freeview.raycastResult.dump");
        glutReportErrors(); 
        //glRasterPos2f(0,0);
        /*
        DrawPixels can draw color, depth, and stencil data. But only one at a time; the other two are picked up the current raster state, unconditionally.

        The operation is orthogonal to all raster tests. So if you want to draw depth data without modifiying color, ColorMask(0,0,0,0). Or color data without depth writes, DepthMask(0), or Disable(DEPTH_TEST).
        */
        auto depth_float_image = new Image<float>(depth->noDims);
        for (int i = 0; i < depth->dataSize; i++) {
            auto rr = depth->GetData()[i];
            if (rr.w != 1.0) continue; // raycast hit nothing
            auto z = rr.z*voxelSize;
            z = (z - viewFrustum_min) / (viewFrustum_max - viewFrustum_min);

            depth_float_image->GetData()[i] = z;
            assert(z >= 0 && z <= 1);
        }
        assert(depth_float_image->noDims == Vector2i(W, H));

        glColorMask(0, 0, 0, 0);
        glDrawPixels(W, H,
            GL_DEPTH_COMPONENT, GL_FLOAT,
            depth_float_image->GetData());
        glutReportErrors();
        glColorMask(1,1,1,1);
    }
    // fill color buffer with render.png
    {
        auto render = new ITMUChar4Image(Vector2i(W, H));
        png::ReadImageFromFile(render, "render.png");
        glutReportErrors();
        //glRasterPos2f(0,0);
        /*
        DrawPixels can draw color, depth, and stencil data. But only one at a time; the other two are picked up the current raster state, unconditionally.

        The operation is orthogonal to all raster tests. So if you want to draw depth data without modifiying color, ColorMask(0,0,0,0). Or color data without depth writes, DepthMask(0), or Disable(DEPTH_TEST).
        */
        assert(render->noDims == Vector2i(W, H));
        glDepthMask(0);
        glDrawPixels(W, H,
            GL_RGBA, 
            GL_UNSIGNED_BYTE,
            render->GetData());
        glutReportErrors();
        glDepthMask(1);
    }

    glBegin(GL_TRIANGLES);
    /*glColor4f(1, 1,1, 1);


    glVertex3f(0, 0, 0);
    glVertex3f(1, 0, 0);
    glVertex3f(0, 1, 0);*/
    
    glColor4f(1, 0, 0, 1);
    glVertex3f(0, 0, -1);
    glVertex3f(1, 0, -1);
    glVertex3f(0, 1, 1);

    glColor4f(1, 1, 1, 1);
    
    glEnd();

    glFinish();
    //for (int i = 0; i < 2; i++)  // only after doing this twice will the framebuffer be correct - but the screen will not
    glutSwapBuffers();

    saveRGBA("rgba.png");
    saveDepth("depth_out.png");

    glutReportErrors();
    puts("ok");
}

#include <float.h>
/**
Any non-infinite, non-NAN, non-negative float. 
IEEE -0 is also prohibited (even though 
    assert(_fpclass(-0.f) == _FPCLASS_NZ);
    assert(-0.f >= 0.f);
 holds)
*/
class positive {
private:
    const float value;
    
public:
    /// Utility: -0 is converted to +0
    static float nz2pz(float x) {
        if (_fpclass(x) == _FPCLASS_NZ)
            return +0.f;
        return x;
    }
    /// Value verified at runtime
    explicit positive(float x) : value(nz2pz(x)) {
        assert(_fpclass(value) == _FPCLASS_PD || _fpclass(value) == _FPCLASS_PN || _fpclass(value) == _FPCLASS_PZ);
    }                         

    // Trick to avoid initialization with anything but float: declare but dont define for all basic types:
    explicit positive(unsigned char);
    explicit positive(char);
    explicit positive(unsigned short);
    explicit positive(short);
    explicit positive(unsigned int);
    explicit positive(int);
    explicit positive(unsigned long);
    explicit positive(long);
   /* explicit positive(unsigned long long);
    explicit positive(long long);*/
    explicit positive(double);
    explicit positive(long double);
    explicit positive(wchar_t);
};

void doit(positive x) {

}

class Point;
class Vector;
class CoordinateSystem;
static __managed__ CoordinateSystem* globalcs = 0;


/// Coordinate systems are identical when their pointers are.
class CoordinateSystem : public Managed {
private:
    CoordinateSystem(const CoordinateSystem&);
    void operator=(const CoordinateSystem&);

    const Matrix4f toGlobal;
    const Matrix4f fromGlobal;
    CPU_AND_GPU Point toGlobalPoint(Point p)const;
    CPU_AND_GPU Point fromGlobalPoint(Point p)const;
    CPU_AND_GPU Vector toGlobalVector(Vector p)const;
    CPU_AND_GPU Vector fromGlobalVector(Vector p)const;
public:
    explicit CoordinateSystem(const Matrix4f& toGlobal) : toGlobal(toGlobal), fromGlobal(toGlobal.getInv()) {
        assert(toGlobal.GetR().det() != 0);
    }

    CPU_AND_GPU static CoordinateSystem* global() {
#ifndef __CUDA_ARCH__
        if (!globalcs) {
            Matrix4f m;
            m.setIdentity();
            ::globalcs = new CoordinateSystem(m);
        }
#endif
        assert(globalcs);
        return globalcs;
    }

    CPU_AND_GPU Point convert(Point p)const;
    CPU_AND_GPU Vector convert(Vector p)const;
};

// Entries are considered equal only when they have the same coordinates.
// They are comparable only if in the same coordinate system.
class CoordinateEntry {
public:
    const CoordinateSystem* coordinateSystem;
    friend CoordinateSystem;
    CPU_AND_GPU CoordinateEntry(const CoordinateSystem* coordinateSystem) : coordinateSystem(coordinateSystem) {}
};

class Vector : public CoordinateEntry {
private:
    friend Point;
    friend CoordinateSystem;
public:
    const Vector3f direction;
    // copy constructor ok
    // assignment will not be possible

    CPU_AND_GPU explicit Vector(const CoordinateSystem* coordinateSystem, Vector3f direction) : CoordinateEntry(coordinateSystem), direction(direction) {
    }
    CPU_AND_GPU bool operator==(const Vector& rhs) const {
        assert(coordinateSystem == rhs.coordinateSystem);
        return direction == rhs.direction;
    }
    CPU_AND_GPU float dot(const Vector& rhs) const {
        assert(coordinateSystem == rhs.coordinateSystem);
        return ORUtils::dot(direction, rhs.direction);
    }
};

class Point : public CoordinateEntry {
private:
    friend CoordinateSystem;
public:
    const Vector3f location;
    // copy constructor ok
    // assignment will not be possible

    CPU_AND_GPU explicit Point(const CoordinateSystem* coordinateSystem, Vector3f location) : CoordinateEntry(coordinateSystem), location(location) {
    }
    CPU_AND_GPU bool operator==(const Point& rhs) const {
        assert(coordinateSystem == rhs.coordinateSystem);
        return location == rhs.location;
    }
    CPU_AND_GPU Point operator+(const Vector& rhs) const {
        assert(coordinateSystem == rhs.coordinateSystem);
        return Point(coordinateSystem, location + rhs.direction);
    }
    // points from rhs to this
    CPU_AND_GPU Vector operator-(const Point& rhs) const {
        assert(coordinateSystem == rhs.coordinateSystem);
        return Vector(coordinateSystem, location - rhs.location);
    }
};

class Ray {
public:
    const Point origin;
    const Vector direction;

    CPU_AND_GPU Ray(Point& origin, Vector& direction) : origin(origin), direction(direction) {
        assert(origin.coordinateSystem == direction.coordinateSystem);
    }
    CPU_AND_GPU Point endpoint() {
        Point p = origin + direction;
        assert(p.coordinateSystem == origin.coordinateSystem);
        return p;
    }
};

#include "itmpixelutils.h"
template<typename T>
class CameraImage : public Managed {
private:
    void operator=(const CameraImage& ci);
    CameraImage(const CameraImage&);
public:
    const Image<T>* image;
    const CoordinateSystem* const eyeCoordinates; // const ITMPose* const pose // pose->GetM is fromGlobal matrix of coord system; <- inverse is toGlobal
    const Vector4f cameraIntrinsics;// const ITMIntrinsics* const cameraIntrinsics;

    CameraImage(
        const Image<T>* image,
        const CoordinateSystem* const eyeCoordinates,
        const Vector4f cameraIntrinsics) :
        image(image), eyeCoordinates(eyeCoordinates), cameraIntrinsics(cameraIntrinsics) {}

    CPU_AND_GPU Vector2i imgSize()const {
        return image->noDims;
    }

    CPU_AND_GPU Vector4f projParams() const {
        return cameraIntrinsics; // ->projectionParamsSimple.all;
    }
    /// 0,0,0 in eyeCoordinates
    CPU_AND_GPU Point location() const {
        return Point(eyeCoordinates, Vector3f(0, 0, 0));
    }
    CPU_AND_GPU Ray getRayForPixel(Vector2i pixel, float depth) const {
        assert(pixel.x >= 0 && pixel.x < image->noDims.width);
        assert(pixel.y >= 0 && pixel.y < image->noDims.height);
        Vector4f f = depthTo3D(projParams(), pixel.x, pixel.y, depth);
        assert(f.z == depth);
        return Ray(location(), Vector(eyeCoordinates, f.toVector3()));
    }
    /// \see project
    /// \returns false when point projects outside of image
    CPU_AND_GPU bool project(Point p, Vector2f& pt_image) const {
        Point p_ec = eyeCoordinates->convert(p);
        assert(p_ec.coordinateSystem == eyeCoordinates);
        return ::project(projParams(), imgSize(), Vector4f(p_ec.location, 1.f), pt_image);
    }
};

class DepthImage : public CameraImage<float> {
public:
    DepthImage(
        const Image<float>* image,
        const CoordinateSystem* const eyeCoordinates,
        const Vector4f cameraIntrinsics) :
        CameraImage(image, eyeCoordinates, cameraIntrinsics) {}

    CPU_AND_GPU Point getPointForPixel(Vector2i pixel) const {
        float depth = sampleNearest(image->GetData(), pixel.x, pixel.y, imgSize());
        Ray r = getRayForPixel(pixel, depth);
        Point p = r.endpoint();
        assert(p.coordinateSystem == eyeCoordinates);
        return p;
    }
};

class PointImage : public CameraImage<Vector4f> {
public:
    PointImage(
        const Image<Vector4f>* image,
        const CoordinateSystem* const eyeCoordinates,
        const Vector4f cameraIntrinsics) :
        CameraImage(image, eyeCoordinates, cameraIntrinsics) {}

    CPU_AND_GPU Point getPointForPixel(Vector2i pixel) const {
        return Point(eyeCoordinates, sampleNearest(image->GetData(), pixel.x, pixel.y, imgSize()).toVector3());
    }
};


CPU_AND_GPU Point CoordinateSystem::toGlobalPoint(Point p) const {
    return Point(global(), Vector3f(this->toGlobal * Vector4f(p.location, 1)));
}
CPU_AND_GPU Point CoordinateSystem::fromGlobalPoint(Point p) const {
    assert(p.coordinateSystem == global());
    return Point(this, Vector3f(this->fromGlobal * Vector4f(p.location, 1)));
}
CPU_AND_GPU Vector CoordinateSystem::toGlobalVector(Vector v) const {
    return Vector(global(), this->toGlobal.GetR() *v.direction);
}
CPU_AND_GPU Vector CoordinateSystem::fromGlobalVector(Vector v) const {
    assert(v.coordinateSystem == global());
    return Vector(this, this->fromGlobal.GetR() *v.direction);
}
CPU_AND_GPU Point CoordinateSystem::convert(Point p) const {
    Point o = this->fromGlobalPoint(p.coordinateSystem->toGlobalPoint(p));
    assert(o.coordinateSystem == this);
    return o;
}
CPU_AND_GPU Vector CoordinateSystem::convert(Vector p) const {
    Vector o = this->fromGlobalVector(p.coordinateSystem->toGlobalVector(p));
    assert(o.coordinateSystem == this);
    return o;
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
    Vector t = Point(g, Vector3f(0,0,0)) - Point(g, Vector3f(1, 1, 2));
    assert(t.direction == Vector3f(-1, -1, -2));

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
        auto r = di->getRayForPixel(Vector2i(0, 0), 1);
        assert(r.origin == Point(cs, Vector3f(0, 0, 0)));
        assert(!(r.direction == Vector(cs, Vector3f(0, 0, 1))));
    }
    {
        auto r = di->getRayForPixel(imgSize / 2, 1);
        assert(r.origin == Point(cs, Vector3f(0, 0, 0)));
        assert(r.direction == Vector(cs, Vector3f(0, 0, 1)));
    }
    {
        auto r = di->getRayForPixel(imgSize / 2, 2);
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
        auto ray = di->getRayForPixel(Vector2i(1, 0), 1);
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

}
void testCameraImage() {
    ITMIntrinsics intrin;
    Vector2i imgSize(640, 480);
    auto depthImage = new ITMFloatImage(imgSize);
    auto pointImage = new ITMFloat4Image(imgSize);

    depthImage->GetData()[1] = 1;
    pointImage->GetData()[1] = Vector4f(1,1,1,1);

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