#include "FAN/Math.hpp"

//Mat4x4 operator*(const Mat4x4& lhs, const Mat4x4& rhs) {
//	return Mat4x4(
//		Vec4(lhs.vec[0].x * rhs.vec[0].x + lhs.vec[0].y * rhs.vec[1].x + lhs.vec[0].z * rhs.vec[2].x + lhs.vec[0].a * rhs.vec[3].x,
//			lhs.vec[0].x * rhs.vec[0].y + lhs.vec[0].y * rhs.vec[1].y + lhs.vec[0].z * rhs.vec[2].y + lhs.vec[0].a * rhs.vec[3].y,
//			lhs.vec[0].x * rhs.vec[0].z + lhs.vec[0].y * rhs.vec[1].z + lhs.vec[0].z * rhs.vec[2].z + lhs.vec[0].a * rhs.vec[3].z,
//			lhs.vec[0].x * rhs.vec[0].a + lhs.vec[0].y * rhs.vec[1].a + lhs.vec[0].z * rhs.vec[2].a + lhs.vec[0].a * rhs.vec[3].a),
//		Vec4(lhs.vec[1].x * rhs.vec[0].x + lhs.vec[1].y * rhs.vec[1].x + lhs.vec[1].z * rhs.vec[2].x + lhs.vec[1].a * rhs.vec[3].x,
//			lhs.vec[1].x * rhs.vec[0].y + lhs.vec[1].y * rhs.vec[1].y + lhs.vec[1].z * rhs.vec[2].y + lhs.vec[1].a * rhs.vec[3].y,
//			lhs.vec[1].x * rhs.vec[0].z + lhs.vec[1].y * rhs.vec[1].z + lhs.vec[1].z * rhs.vec[2].z + lhs.vec[1].a * rhs.vec[3].z,
//			lhs.vec[1].x * rhs.vec[0].a + lhs.vec[1].y * rhs.vec[1].a + lhs.vec[1].z * rhs.vec[2].a + lhs.vec[1].a * rhs.vec[3].a),
//		Vec4(lhs.vec[2].x * rhs.vec[0].x + lhs.vec[2].y * rhs.vec[1].x + lhs.vec[2].z * rhs.vec[2].x + lhs.vec[2].a * rhs.vec[3].x,
//			lhs.vec[2].x * rhs.vec[0].y + lhs.vec[2].y * rhs.vec[1].y + lhs.vec[2].z * rhs.vec[2].y + lhs.vec[2].a * rhs.vec[3].y,
//			lhs.vec[2].x * rhs.vec[0].z + lhs.vec[2].y * rhs.vec[1].z + lhs.vec[2].z * rhs.vec[2].z + lhs.vec[2].a * rhs.vec[3].z,
//			lhs.vec[2].x * rhs.vec[0].a + lhs.vec[2].y * rhs.vec[1].a + lhs.vec[2].z * rhs.vec[2].a + lhs.vec[2].a * rhs.vec[3].a),
//		Vec4(lhs.vec[3].x * rhs.vec[0].x + lhs.vec[3].y * rhs.vec[1].x + lhs.vec[3].z * rhs.vec[2].x + lhs.vec[3].a * rhs.vec[3].x,
//			lhs.vec[3].x * rhs.vec[0].y + lhs.vec[3].y * rhs.vec[1].y + lhs.vec[3].z * rhs.vec[2].y + lhs.vec[3].a * rhs.vec[3].y,
//			lhs.vec[3].x * rhs.vec[0].z + lhs.vec[3].y * rhs.vec[1].z + lhs.vec[3].z * rhs.vec[2].z + lhs.vec[3].a * rhs.vec[3].z,
//			lhs.vec[3].x * rhs.vec[0].a + lhs.vec[3].y * rhs.vec[1].a + lhs.vec[3].z * rhs.vec[2].a + lhs.vec[3].a * rhs.vec[3].a)
//	);
//}