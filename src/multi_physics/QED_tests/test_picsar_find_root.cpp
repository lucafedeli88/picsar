//####### Test module for find_root function ####################################

//Define Module name
 #define BOOST_TEST_MODULE "math/find_root"

//Include Boost unit tests library & library for floating point comparison
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>

#include <array>

#include "math_constants.h"
#include "find_root.hpp"

using namespace picsar::multi_physics::math;

//Tolerance for double precision calculations
const double double_tolerance = 1.0e-6;

//Tolerance for single precision calculations
const float float_tolerance = 1.0e-3;

//Templated tolerance
template <typename T>
T constexpr tolerance()
{
    if(std::is_same<T,float>::value)
        return float_tolerance;
    else
        return double_tolerance;
}

// ------------- Tests --------------

// ***Test find_root algorithm

template<typename RealType>
constexpr void test_find_root_1()
{
    const auto a = static_cast<RealType>(10.0);
    const auto func = [=](RealType x){return x*x*x - a;};

    const auto guess = static_cast<RealType>(1.0);
    const bool is_rising = true;
    const RealType factor = static_cast<RealType>(2);
    const unsigned int max_it = 20;
    const unsigned int digit_loss = 3;
    const auto res = find_root(func, guess, is_rising,
        factor, max_it, digit_loss);

    const RealType exp_res = std::cbrt(a);

    BOOST_CHECK_EQUAL(res.first, true);
    BOOST_CHECK_SMALL((res.second-exp_res)/exp_res, tolerance<RealType>());
}

BOOST_AUTO_TEST_CASE( picsar_find_root_1 )
{
    test_find_root_1 <double>();
    test_find_root_1 <float>();
}

template<typename RealType>
constexpr void test_find_root_2()
{
    const auto a = static_cast<RealType>(10.0);
    const auto func = [=](RealType x){return x*x*x - a;};

    const auto guess = static_cast<RealType>(1.0);
    const bool is_rising = true;
    const auto res = find_root(func, guess, is_rising);

    const RealType exp_res = std::cbrt(a);

    BOOST_CHECK_EQUAL(res.first, true);
    BOOST_CHECK_SMALL((res.second-exp_res)/exp_res, tolerance<RealType>());
}

BOOST_AUTO_TEST_CASE( picsar_find_root_2 )
{
    test_find_root_2 <double>();
    test_find_root_2 <float>();
}

template<typename RealType>
constexpr void test_find_root_3()
{
    const auto func = [=](RealType x){return std::tanh(x) - half<RealType>;};

    const auto guess = static_cast<RealType>(1.0);
    const bool is_rising = true;
    const auto res = find_root(func, guess, is_rising);

    const RealType exp_res = std::atanh(half<RealType>);

    BOOST_CHECK_EQUAL(res.first, true);
    BOOST_CHECK_SMALL((res.second-exp_res)/exp_res, tolerance<RealType>());
}

BOOST_AUTO_TEST_CASE( picsar_find_root_3 )
{
    test_find_root_3 <double>();
    test_find_root_3 <float>();
}
