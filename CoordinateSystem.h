#pragma once
#include "itmlibdefines.h"

class Point;
class Vector;
class Ray;
class CoordinateSystem;
extern __managed__ CoordinateSystem* globalcs;


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

    /// The world or global space coodinate system.
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
    CPU_AND_GPU Ray convert(Ray p)const;
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
    CPU_AND_GPU Vector operator*(const float rhs) const {
        return Vector(coordinateSystem, direction * rhs);
    }
    CPU_AND_GPU float dot(const Vector& rhs) const {
        assert(coordinateSystem == rhs.coordinateSystem);
        return ORUtils::dot(direction, rhs.direction);
    }
    CPU_AND_GPU Vector operator-(const Vector& rhs) const {
        assert(coordinateSystem == rhs.coordinateSystem);
        return Vector(coordinateSystem, direction - rhs.direction);
    }
};

class Point : public CoordinateEntry {
private:
    friend CoordinateSystem;
public:
    const Vector3f location;
    // copy constructor ok

    // Assignment // TODO use a reference instead
    CPU_AND_GPU void operator=(const Point& rhs) {
        coordinateSystem = rhs.coordinateSystem;
        const_cast<Vector3f&>(location) = rhs.location;
    }

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


inline CPU_AND_GPU Point CoordinateSystem::toGlobalPoint(Point p) const {
    return Point(global(), Vector3f(this->toGlobal * Vector4f(p.location, 1)));
}
inline CPU_AND_GPU Point CoordinateSystem::fromGlobalPoint(Point p) const {
    assert(p.coordinateSystem == global());
    return Point(this, Vector3f(this->fromGlobal * Vector4f(p.location, 1)));
}
inline CPU_AND_GPU Vector CoordinateSystem::toGlobalVector(Vector v) const {
    return Vector(global(), this->toGlobal.GetR() *v.direction);
}
inline CPU_AND_GPU Vector CoordinateSystem::fromGlobalVector(Vector v) const {
    assert(v.coordinateSystem == global());
    return Vector(this, this->fromGlobal.GetR() *v.direction);
}
inline CPU_AND_GPU Point CoordinateSystem::convert(Point p) const {
    Point o = this->fromGlobalPoint(p.coordinateSystem->toGlobalPoint(p));
    assert(o.coordinateSystem == this);
    return o;
}
inline CPU_AND_GPU Vector CoordinateSystem::convert(Vector p) const {
    Vector o = this->fromGlobalVector(p.coordinateSystem->toGlobalVector(p));
    assert(o.coordinateSystem == this);
    return o;
}
inline CPU_AND_GPU Ray CoordinateSystem::convert(Ray p) const {
    return Ray(convert(p.origin), convert(p.direction));
}
