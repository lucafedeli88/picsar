//####### Test module for physical and mathematical constants ##################

//Define Module name
 #define BOOST_TEST_MODULE "phys/unit_conversion"

//Will automatically define a main for this test
 #define BOOST_TEST_DYN_LINK

#include <cmath>

//Include Boost unit tests library & library for floating point comparison
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include "math_constants.h"
#include "phys_constants.h"

using namespace picsar::multi_physics::phys;
using namespace picsar::multi_physics::math;

//Tolerance for double precision calculations
const double double_tolerance = 1.0e-14;

//Tolerance for single precision calculations
const float float_tolerance = 1.0e-7;

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

// ***Test math constants

template<typename RealType>
void test_case_const_math()
{
    const auto exp_pi =
        static_cast<RealType>(3.14159265358979323846264338327950288);

    const auto exp_half = static_cast<RealType>(0.5);
    const auto exp_one = static_cast<RealType>(1.0);
    const auto exp_two = static_cast<RealType>(2.0);
    const auto exp_three = static_cast<RealType>(3.0);
    const auto exp_one_third = static_cast<RealType>(1.0/3.0);
    const auto exp_two_thirds = static_cast<RealType>(2.0/3.0);

    BOOST_CHECK_SMALL((pi<RealType>-exp_pi)/exp_pi, tolerance<RealType>());
    BOOST_CHECK_SMALL(zero<RealType>, tolerance<RealType>());
    BOOST_CHECK_SMALL((half<RealType>-exp_half)/exp_half, tolerance<RealType>());
    BOOST_CHECK_SMALL((one<RealType>-exp_one)/exp_one, tolerance<RealType>());
    BOOST_CHECK_SMALL((two<RealType>-exp_two)/exp_two, tolerance<RealType>());
    BOOST_CHECK_SMALL((three<RealType>-exp_three)/exp_three, tolerance<RealType>());
    BOOST_CHECK_SMALL((one_third<RealType>-exp_one_third)/exp_one_third, tolerance<RealType>());
    BOOST_CHECK_SMALL((two_thirds<RealType>-exp_two_thirds)/exp_two_thirds, tolerance<RealType>());
}

BOOST_AUTO_TEST_CASE( picsar_const_math )
{
    test_case_const_math<double>();
    test_case_const_math<float>();
}

// *******************************

// ***Test physical constants

template<typename RealType>
void test_case_const_phys()
{
    const auto exp_electron_mass =
        static_cast<RealType>(9.1093837015e-31);
    const auto exp_elementary_charge =
        static_cast<RealType>(1.602176634e-19);
    const auto exp_light_speed =
        static_cast<RealType>(299792458.);
    const auto exp_reduced_plank =
        static_cast<RealType>(1.054571817e-34);
    const auto exp_vacuum_permittivity =
        static_cast<RealType>(8.8541878128e-12);
    const auto exp_vacuum_permeability =
        static_cast<RealType>(1.25663706212e-6);
    const auto exp_fine_structure =
        static_cast<RealType>(0.0072973525693);
    const auto exp_eV =
        static_cast<RealType>(1.602176634e-19);
    const auto exp_KeV =
        static_cast<RealType>(1.602176634e-16);
    const auto exp_MeV =
        static_cast<RealType>(1.602176634e-13);
    const auto exp_GeV =
        static_cast<RealType>(1.602176634e-10);
    const auto exp_sqrt_4_pi_fine_structure =
            static_cast<RealType>(sqrt(
                4.0*pi<double>*fine_structure<double>));
    const auto exp_classical_electron_radius =
        static_cast<RealType>(2.81794032620493e-15);
    const auto exp_schwinger_field =
        static_cast<RealType>(1.32328547494817e18);
    const auto exp_tau_e =
        static_cast<RealType>(9.39963715232933e-24);

    BOOST_CHECK_SMALL(
        (electron_mass<RealType>-exp_electron_mass)/exp_electron_mass,
        tolerance<RealType>());

    BOOST_CHECK_SMALL(
        (elementary_charge<RealType>-exp_elementary_charge)/exp_elementary_charge,
        tolerance<RealType>());

    BOOST_CHECK_SMALL(
        (light_speed<RealType>-exp_light_speed)/exp_light_speed,
        tolerance<RealType>());

    BOOST_CHECK_SMALL(
        (reduced_plank<RealType>-exp_reduced_plank)/exp_reduced_plank,
        tolerance<RealType>());

    BOOST_CHECK_SMALL(
        (vacuum_permittivity<RealType>-exp_vacuum_permittivity)/exp_vacuum_permittivity,
        tolerance<RealType>());

    BOOST_CHECK_SMALL(
        (vacuum_permeability<RealType>-exp_vacuum_permeability)/exp_vacuum_permeability,
        tolerance<RealType>());

    BOOST_CHECK_SMALL(
        (fine_structure<RealType>-exp_fine_structure)/exp_fine_structure,
        tolerance<RealType>());

    BOOST_CHECK_SMALL(
        (eV<RealType>-exp_eV)/exp_eV,
        tolerance<RealType>());

    BOOST_CHECK_SMALL(
        (KeV<RealType>-exp_KeV)/exp_KeV,
        tolerance<RealType>());

    BOOST_CHECK_SMALL(
        (MeV<RealType>-exp_MeV)/exp_MeV,
        tolerance<RealType>());

    BOOST_CHECK_SMALL(
        (GeV<RealType>-exp_GeV)/exp_GeV,
        tolerance<RealType>());

    BOOST_CHECK_SMALL(
            (sqrt_4_pi_fine_structure<RealType>-exp_sqrt_4_pi_fine_structure)/
            exp_sqrt_4_pi_fine_structure, tolerance<RealType>());

    BOOST_CHECK_SMALL(
        (classical_electron_radius<RealType>-exp_classical_electron_radius)/exp_classical_electron_radius,
        tolerance<RealType>());

    BOOST_CHECK_SMALL(
        (schwinger_field<RealType>-exp_schwinger_field)/exp_schwinger_field,
        tolerance<RealType>());

    BOOST_CHECK_SMALL(
        (tau_e<RealType>-exp_tau_e)/exp_tau_e,
        tolerance<RealType>());
}

BOOST_AUTO_TEST_CASE( picsar_const_phys )
{
    test_case_const_phys<double>();
    test_case_const_phys<float>();
}

// *******************************
