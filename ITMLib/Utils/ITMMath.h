#pragma once

#include "MathUtils.h"

#ifndef NULL
#define NULL 0
#endif


typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;

#include "Vector.h"
#include "Matrix.h"
using namespace ORUtils;

typedef class Matrix3<float> Matrix3f;
typedef class Matrix4<float> Matrix4f;

typedef class Vector2<short> Vector2s;
typedef class Vector2<int> Vector2i;
typedef class Vector2<float> Vector2f;
typedef class Vector2<double> Vector2d;

typedef class Vector3<short> Vector3s;
typedef class Vector3<double> Vector3d;
typedef class Vector3<int> Vector3i;
typedef class Vector3<uint> Vector3ui;
typedef class Vector3<uchar> Vector3u;
typedef class Vector3<float> Vector3f;

typedef class Vector4<float> Vector4f;
typedef class Vector4<int> Vector4i;
typedef class Vector4<short> Vector4s;
typedef class Vector4<uchar> Vector4u;

typedef class Vector6<float> Vector6f;

#ifndef TO_INT_ROUND3
#define TO_INT_ROUND3(x) (x).toIntRound()
#endif

#ifndef TO_INT_ROUND4
#define TO_INT_ROUND4(x) (x).toIntRound()
#endif

#ifndef TO_INT_FLOOR3
#define TO_INT_FLOOR3(inted, coeffs, in) inted = (in).toIntFloor(coeffs)
#endif

#ifndef TO_SHORT_FLOOR3
#define TO_SHORT_FLOOR3(x) (x).toShortFloor()
#endif

#ifndef TO_UCHAR3
#define TO_UCHAR3(x) (x).toUChar()
#endif

#ifndef TO_FLOAT3
#define TO_FLOAT3(x) (x).toFloat()
#endif

#ifndef TO_SHORT3
#define TO_SHORT3(p) Vector3s(p.x, p.y, p.z)
#endif

#ifndef TO_VECTOR3
#define TO_VECTOR3(a) (a).toVector3()
#endif

#ifndef IS_EQUAL3
#define IS_EQUAL3(a,b) (((a).x == (b).x) && ((a).y == (b).y) && ((a).z == (b).z))
#endif


