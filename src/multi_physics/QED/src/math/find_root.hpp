#ifndef PICSAR_MULTIPHYSICS_FIND_ROOT
#define PICSAR_MULTIPHYSICS_FIND_ROOT

//Should be included by all the src files of the library
#include "../qed_commons.h"

#include "math_constants.h"

#include <boost/math/tools/roots.hpp>

#include <limits>
#include <utility>

namespace picsar{
namespace multi_physics{
namespace math{
    /**
    * This module is a wrapper around the default root finding algorithm
    * provided by the boost library.
    */

    /**
    * This function performs the integration of the function f(x)
    * in the interval (a,b) using the method specified in the template parameter
    * (not usable on GPUs).
    *
    * @tparam RealType the floating point type to be used
    * @tparam QuadAlgo the quadrature method to be used
    * @param[in] f the function which should be integrated
    * @param[in] a the left boundary of the integration region
    * @param[in] b the right boundary of the integration region
    * @return the integral of f in (a,b)
    */
    template<
        typename RealType, typename Functor>
    std::pair<bool, RealType>
    find_root(const Functor& func, RealType guess, bool is_rising,
        const RealType factor = static_cast<RealType>(2),
        const unsigned int max_it = 30, const unsigned int digit_loss = 3)
    {
        boost::uintmax_t it = max_it;
        const auto get_digits = static_cast<unsigned int>(
            std::numeric_limits<RealType>::digits - digit_loss);
        const auto tol =
            boost::math::tools::eps_tolerance<RealType>{get_digits};
        const auto r =
            boost::math::tools::bracket_and_solve_root(
                func, guess, factor, is_rising, tol, it);

        return std::make_pair(it < max_it, (r.second + r.first)*half<RealType>);

        //return std::make_pair(it < max_it, (r.second + r.first)*half<RealType>);
    }
}
}
}

#endif //PICSAR_MULTIPHYSICS_FIND_ROOT
