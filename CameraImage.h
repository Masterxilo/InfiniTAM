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
    Image<T>*const image; // TODO should the image have to be const?
    const CoordinateSystem* const eyeCoordinates; // const ITMPose* const pose // pose->GetM is fromGlobal matrix of coord system; <- inverse is toGlobal // TODO should this encapsulate a copy?
    const Vector4f cameraIntrinsics;// const ITMIntrinsics* const cameraIntrinsics;

    CameraImage(
        const Image<T>*const image,
        const CoordinateSystem* const eyeCoordinates,
        const Vector4f cameraIntrinsics) :
        image(image), eyeCoordinates(eyeCoordinates), cameraIntrinsics(cameraIntrinsics) {
        assert(image->noDims.area() > 1);
    }

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
#define EXTRA_BOUNDS true
    /// \see project
    /// \returns false when point projects outside of image
    CPU_AND_GPU bool project(Point p, Vector2f& pt_image, bool extraBounds = false) const {
        Point p_ec = eyeCoordinates->convert(p);
        assert(p_ec.coordinateSystem == eyeCoordinates);
        if (extraBounds)
            return ::projectExtraBounds(projParams(), imgSize(), Vector4f(p_ec.location, 1.f), pt_image);
        else
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

/// Treats a raster of locations as points in pointCoordinates.
///
/// The assumption is that the location in image[x,y] was recorded by
/// intersecting a ray through pixel (x,y) of this camera with something.
/// The coordinate system 'pointCoordinates' does not have to be the same as eyeCoordinates, it might be 
/// global() coordinates.
/// getPointForPixel will return a point in pointCoordinates with .location as specified in the image.
///
/// Note: This is a lot different from the depth image, where the assumption is always that the depths are 
/// the z component in eyeCoordinates. Here, the coordinate system of the data in the image can be anything.
///
/// We use this to store intersection points (in world coordinates) obtained by raytracing from a certain camera location.
class PointImage : public CameraImage<Vector4f> {
public:
    PointImage(
        const Image<Vector4f>*const image,
        const CoordinateSystem* const pointCoordinates,

        const CoordinateSystem* const eyeCoordinates,
        const Vector4f cameraIntrinsics) :
        CameraImage(image, eyeCoordinates, cameraIntrinsics), pointCoordinates(pointCoordinates) {}
    const CoordinateSystem* const pointCoordinates;

    CPU_AND_GPU Point getPointForPixel(Vector2i pixel) const {
        return Point(pointCoordinates, sampleNearest(image->GetData(), pixel.x, pixel.y, imgSize()).toVector3());
    }

    CPU_AND_GPU Point getPointForPixelInterpolated(Vector2f pixel, bool& out_isIllegal) const {
        out_isIllegal = false;
        // TODO should this always consider holes?
        auto point = interpolateBilinear<Vector4f, WITH_HOLES>(
            image->GetData(),
            pixel,
            imgSize());
        if (!isLegalColor(point)) out_isIllegal = true;
        // TODO handle holes
        return Point(pointCoordinates, point.toVector3());
    }
};

/// Treats a raster of locations and normals as rays, specified in pointCoordinates.
/// Pixel (x,y) is associated to the ray startin at pointImage[x,y] into the direction normalImage[x,y],
/// where both coordinates are taken to be pointCoordinates.
class RayImage : public PointImage {
public:
    RayImage(
        const Image<Vector4f>*const pointImage,
        const Image<Vector4f>*const normalImage,
        const CoordinateSystem* const pointCoordinates,

        const CoordinateSystem* const eyeCoordinates,
        const Vector4f cameraIntrinsics) :
        PointImage(pointImage, pointCoordinates, eyeCoordinates, cameraIntrinsics), normalImage(normalImage) {
        assert(normalImage->noDims == pointImage->noDims);
    }
    const Image<Vector4f>* const normalImage;

    CPU_AND_GPU Ray getRayForPixel(Vector2i pixel) const {
        Point origin = getPointForPixel(pixel);
        auto direction = sampleNearest(normalImage->GetData(), pixel.x, pixel.y, imgSize()).toVector3();
        return Ray(origin, Vector(pointCoordinates, direction));
    }

    CPU_AND_GPU Ray getRayForPixelInterpolated(Vector2f pixel, bool& out_isIllegal) const {
        out_isIllegal = false;
        Point origin = getPointForPixelInterpolated(pixel, out_isIllegal);
        // TODO should this always consider holes?
        auto direction = interpolateBilinear<Vector4f, WITH_HOLES>(
            normalImage->GetData(),
            pixel,
            imgSize());
        if (!isLegalColor(direction)) out_isIllegal = true;
        // TODO handle holes
        return Ray(origin, Vector(pointCoordinates, direction.toVector3()));
    }
};
