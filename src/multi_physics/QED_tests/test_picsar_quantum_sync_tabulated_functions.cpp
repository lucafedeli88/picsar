//####### Test module for picsar_tables ####################################

//Define Module name
 #define BOOST_TEST_MODULE "phys/quantum_sync/tabulated_functions"

//Will automatically define a main for this test
 #define BOOST_TEST_DYN_LINK

 #include<array>
 #include<utility>

//Include Boost unit tests library & library for floating point comparison
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>

#include "quantum_sync_engine_tabulated_functions.hpp"

//Tolerance for double precision calculations
const double double_tolerance = 1.5e-1; //Warning! High error for chi >= 100

//Tolerance for single precision calculations
const float float_tolerance = 1.5e-1; //Warning! High error for chi >= 100

using namespace picsar::multi_physics::phys::quantum_sync;

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

template <typename RealType>
void check_int_K_5_3_replacement()
{
    const auto cases = std::array<std::pair<RealType,RealType>,5>{
        std::make_pair<RealType,RealType>( 1e-5, 4629.2044114881355),
        std::make_pair<RealType,RealType>( 1e-4, 995.9088308508012),
        std::make_pair<RealType,RealType>( 1e-2, 44.497250411420913),
        std::make_pair<RealType,RealType>( 1, 0.651422815355309),
        std::make_pair<RealType,RealType>( 10, 1.9223826430338323e-05)};

    for (const auto cc : cases)
    {
        const auto res = inner_integral(cc.first);
        const auto sol = cc.second;
        BOOST_CHECK_SMALL((res-sol)/sol,  tolerance<RealType>());
    }

}

// ***Test replacement of the integral of K_5_3
BOOST_AUTO_TEST_CASE( picsar_quantum_sync_int_K_5_3_replacement)
{
    check_int_K_5_3_replacement<double>();
    check_int_K_5_3_replacement<float>();
}


template <typename RealType>
void check_dndt_table()
{
    const auto cases = std::array<std::pair<double,double>,8>{
        std::make_pair( 0.0001, 2.16486358e-04),
        std::make_pair( 0.001, 2.16307103e-03),
        std::make_pair( 0.01, 2.14576898e-02),
        std::make_pair( 0.1,  2.01157780e-01),
        std::make_pair( 1.0 , 1.50538892e+00),
        std::make_pair( 10.0, 8.36547559e+00),
        std::make_pair( 100.0, 4.05810801e+01),
        std::make_pair( 1000.0, 1.90176330e+02)};

    for (const auto cc : cases)
    {
        const auto res = compute_G_function(static_cast<RealType>(cc.first));
            BOOST_CHECK_SMALL((res - static_cast<RealType>(cc.second))/
                static_cast<RealType>(cc.second), tolerance<RealType>());
    }
}

// ***Test Quantum Synchrotron dndt table
BOOST_AUTO_TEST_CASE( picsar_quantum_sync_dndt_G_function)
{
    check_dndt_table<double>();
    check_dndt_table<float>();
}
