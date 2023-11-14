#define __BCOL_P(p0) CONCAT3(BCOL_set_prefix, _, p0)
#define __BCOL_PP(p0) CONCAT4(_, BCOL_set_prefix, _, p0)

#include "Types/Types.h"

#undef __BCOL_PP
#undef __BCOL_P
