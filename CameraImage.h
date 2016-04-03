#include "image.h"
#include "itmlibdefines.h"
#include "itmpixelutils.h"


/// Base class storing a camera calibration, eyecoordinate system and an image taken with a camera thusly calibrated.
template<typename T>
class CameraImage : public Managed {
private:
    void operator=(const CameraImage& ci);
    CameraImage(const CameraImage&);
public:
    const Image<T>*const image;
    const CoordinateSystem* const eyeCoordinates; // const ITMPose* const pose // pose->GetM is fromGlobal matrix of coord system; <- inverse is toGlobal
    const Vector4f cameraIntrinsics;// const ITMIntrinsics* const cameraIntrinsics;

    CameraImage(
        const Image<T>*const image,
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
    CPU_AND_GPU Ray getRayThroughPixel(Vector2i pixel, float depth) const {
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

/// Constructs getRayThroughPixel endpoints for depths specified in an image.
class DepthImage : public CameraImage<float> {
public:
    DepthImage(
        const Image<float>*const image,
        const CoordinateSystem* const eyeCoordinates,
        const Vector4f cameraIntrinsics) :
        CameraImage(image, eyeCoordinates, cameraIntrinsics) {}

    CPU_AND_GPU Point getPointForPixel(Vector2i pixel) const {
        float depth = sampleNearest(image->GetData(), pixel.x, pixel.y, imgSize());
        Ray r = getRayThroughPixel(pixel, depth);
        Point p = r.endpoint();
        assert(p.coordinateSystem == eyeCoordinates);
        return p;
    }
};

/// Treats a raster of locations as points in eye-coordinates
class PointImage : public CameraImage<Vector4f> {
public:
    PointImage(
        const Image<Vector4f>*const image,
        const CoordinateSystem* const eyeCoordinates,
        const Vector4f cameraIntrinsics) :
        CameraImage(image, eyeCoordinates, cameraIntrinsics) {}

    CPU_AND_GPU Point getPointForPixel(Vector2i pixel) const {
        return Point(eyeCoordinates, sampleNearest(image->GetData(), pixel.x, pixel.y, imgSize()).toVector3());
    }
};

/// Treats a raster of locations and normals as rays, specified in eye-coordinates.
/// Pixel (x,y) is associated to the ray startin at pointImage[x,y] into the direction normalImage[x,y],
/// where both coordinates are taken to be eyeCoordinates.
class RayImage : public PointImage {
public:
    RayImage(
        const Image<Vector4f>*const pointImage,
        const Image<Vector4f>*const normalImage,
        const CoordinateSystem* const eyeCoordinates,
        const Vector4f cameraIntrinsics) :
        PointImage(pointImage, eyeCoordinates, cameraIntrinsics), normalImage(normalImage) {}
    const Image<Vector4f>* const normalImage;

    CPU_AND_GPU Ray getRayForPixel(Vector2i pixel) const {
        Point origin = getPointForPixel(pixel);
        auto direction = sampleNearest(normalImage->GetData(), pixel.x, pixel.y, imgSize()).toVector3();
        return Ray(origin, Vector(eyeCoordinates, direction));
    }
};
