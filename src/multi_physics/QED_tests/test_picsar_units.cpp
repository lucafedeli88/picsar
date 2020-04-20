//####### Test module for unit conversion ####################################

//Define Module name
 #define BOOST_TEST_MODULE "phys/unit_conversion"

//Will automatically define a main for this test
 #define BOOST_TEST_DYN_LINK

#include <array>
#include <algorithm>
#include <functional>

//Include Boost unit tests library & library for floating point comparison
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include "math_constants.h"
#include "phys_constants.h"
#include "unit_conversion.hpp"

using namespace picsar::multi_physics::phys;
using namespace picsar::multi_physics::math;

//Tolerance for double precision calculations
const double double_tolerance = 1.0e-12;

//Tolerance for single precision calculations
const float float_tolerance = 1.0e-4;

//Templated tolerance
template <typename T>
T constexpr tolerance()
{
    if(std::is_same<T,float>::value)
        return float_tolerance;
    else
        return double_tolerance;
}

//Auxiliary functions for tests
template<typename RealType>
struct val_pack{
    RealType SI;
    RealType omega;
    RealType lambda;
    RealType hl;
};

enum quantity{
    mass,
    charge,
    velocity,
    momentum,
    energy,
    length,
    area,
    volume,
    ttime, //to avoid clash with "time" function
    rate,
    E,
    B
};

template <quantity Quantity>
struct fact{
    template<unit_system From, typename RealType>
    static constexpr RealType to_SI_from();

    template<unit_system From, typename RealType>
    static constexpr RealType to_SI_from(RealType reference_quantity);

    template<unit_system From, unit_system To, typename RealType>
    static constexpr RealType from_to();

    template<unit_system From,  unit_system To, typename RealType>
    static constexpr RealType from_to(
        RealType from_reference_quantity,
        RealType to_reference_quantity);
};

template<>
struct fact<quantity::mass>
{
    template<unit_system From, typename RealType>
    static constexpr RealType to_SI_from(){
        return fact_mass_to_SI_from<From, RealType>();
    }

    template<unit_system From, unit_system To, typename RealType>
    static constexpr RealType from_to(){
        return fact_mass_from_to<From, To, RealType>();
    }
};

template<>
struct fact<quantity::charge>
{
    template<unit_system From, typename RealType>
    static constexpr RealType to_SI_from(){
        return fact_charge_to_SI_from<From, RealType>();
    }

    template<unit_system From, unit_system To, typename RealType>
    static constexpr RealType from_to(){
        return fact_charge_from_to<From, To, RealType>();
    }
};

template<>
struct fact<quantity::velocity>
{
    template<unit_system From, typename RealType>
    static constexpr RealType to_SI_from(){
        return fact_velocity_to_SI_from<From, RealType>();
    }

    template<unit_system From, unit_system To, typename RealType>
    static constexpr RealType from_to(){
        return fact_velocity_from_to<From, To, RealType>();
    }
};

template<>
struct fact<quantity::momentum>
{
    template<unit_system From, typename RealType>
    static constexpr RealType to_SI_from(){
        return fact_momentum_to_SI_from<From, RealType>();
    }

    template<unit_system From, unit_system To, typename RealType>
    static constexpr RealType from_to(){
        return fact_momentum_from_to<From, To, RealType>();
    }
};

template<>
struct fact<quantity::energy>
{
    template<unit_system From, typename RealType>
    static constexpr RealType to_SI_from(){
        return fact_energy_to_SI_from<From, RealType>();
    }

    template<unit_system From, unit_system To, typename RealType>
    static constexpr RealType from_to(){
        return fact_energy_from_to<From, To, RealType>();
    }
};

template<>
struct fact<quantity::length>
{
    template<unit_system From, typename RealType>
    static constexpr RealType to_SI_from(RealType reference_quantity = 1.0)
    {
        return fact_length_to_SI_from<From, RealType>(reference_quantity);
    }

    template<unit_system From, unit_system To, typename RealType>
    static constexpr RealType from_to(
        RealType reference_quantity_from = 1.0,
        RealType reference_quantity_to = 1.0){
        return fact_length_from_to<From, To, RealType>(
            reference_quantity_from,
            reference_quantity_to);
    }
};

template<>
struct fact<quantity::area>
{
    template<unit_system From, typename RealType>
    static constexpr RealType to_SI_from(RealType reference_quantity = 1.0)
    {
        return fact_area_to_SI_from<From, RealType>(reference_quantity);
    }

    template<unit_system From, unit_system To, typename RealType>
    static constexpr RealType from_to(
        RealType reference_quantity_from = 1.0,
        RealType reference_quantity_to = 1.0){
        return fact_area_from_to<From, To, RealType>(
            reference_quantity_from,
            reference_quantity_to);
    }
};

template<>
struct fact<quantity::volume>
{
    template<unit_system From, typename RealType>
    static constexpr RealType to_SI_from(RealType reference_quantity = 1.0)
    {
        return fact_volume_to_SI_from<From, RealType>(reference_quantity);
    }

    template<unit_system From, unit_system To, typename RealType>
    static constexpr RealType from_to(
        RealType reference_quantity_from = 1.0,
        RealType reference_quantity_to = 1.0){
        return fact_volume_from_to<From, To, RealType>(
            reference_quantity_from,
            reference_quantity_to);
    }
};

template<>
struct fact<quantity::ttime>
{
    template<unit_system From, typename RealType>
    static constexpr RealType to_SI_from(RealType reference_quantity = 1.0)
    {
        return fact_time_to_SI_from<From, RealType>(reference_quantity);
    }
};

template<>
struct fact<quantity::rate>
{
    template<unit_system From, typename RealType>
    static constexpr RealType to_SI_from(RealType reference_quantity = 1.0)
    {
        return fact_rate_to_SI_from<From, RealType>(reference_quantity);
    }
};

template<typename RealType, quantity Quantity>
constexpr void test_to_SI(val_pack<RealType> vals)
{
    const auto fact_SI = fact<Quantity>::
        template to_SI_from<unit_system::SI, RealType>();
    const auto fact_omega = fact<Quantity>::
        template to_SI_from<unit_system::norm_omega, RealType>();
    const auto fact_lambda = fact<Quantity>::
        template to_SI_from<unit_system::norm_lambda, RealType>();
    const auto fact_hl = fact<Quantity>::
        template to_SI_from<unit_system::heaviside_lorentz, RealType>();

    const auto res_SI2SI = vals.SI*fact_SI;
    const auto res_omega2SI = vals.omega*fact_omega;
    const auto res_lambda2SI = vals.lambda*fact_lambda;
    const auto res_hl2SI = vals.hl*fact_hl;
    const auto all_res = std::array<RealType,4>{
        res_SI2SI, res_omega2SI, res_lambda2SI, res_hl2SI};
    for (const auto& res : all_res)
        BOOST_CHECK_SMALL((res-vals.SI)/vals.SI, tolerance<RealType>());
}

template<typename RealType, quantity Quantity>
constexpr void test_to_SI(val_pack<RealType> vals,
    RealType reference_omega,
    RealType reference_length)
{
    const auto fact_SI = fact<Quantity>::
        template to_SI_from<unit_system::SI, RealType>();
    const auto fact_omega = fact<Quantity>::
        template to_SI_from<unit_system::norm_omega, RealType>(reference_omega);
    const auto fact_lambda = fact<Quantity>::
        template to_SI_from<unit_system::norm_lambda, RealType>(reference_length);
    const auto fact_hl = fact<Quantity>::
        template to_SI_from<unit_system::heaviside_lorentz, RealType>();

    const auto res_SI2SI = vals.SI*fact_SI;
    const auto res_omega2SI = vals.omega*fact_omega;
    const auto res_lambda2SI = vals.lambda*fact_lambda;
    const auto res_hl2SI = vals.hl*fact_hl;
    const auto all_res = std::array<RealType,4>{
        res_SI2SI, res_omega2SI, res_lambda2SI, res_hl2SI};
    for (const auto& res : all_res)
        BOOST_CHECK_SMALL((res-vals.SI)/vals.SI, tolerance<RealType>());
}

template<typename RealType, quantity Quantity>
constexpr void test_from_to(val_pack<RealType> vals)
{
    const auto from_SI_to_all = std::array<RealType,4>
    {
        vals.SI*fact<Quantity>::
            template from_to<unit_system::SI, unit_system::SI, RealType>(),
        vals.SI*fact<Quantity>::
            template from_to<unit_system::SI, unit_system::norm_omega, RealType>(),
        vals.SI*fact<Quantity>::
            template from_to<unit_system::SI, unit_system::norm_lambda, RealType>(),
        vals.SI*fact<Quantity>::
            template from_to<unit_system::SI, unit_system::heaviside_lorentz, RealType>()
    };

    const auto from_omega_to_all = std::array<RealType,4>
    {
        vals.omega*fact<Quantity>::
            template from_to<unit_system::norm_omega, unit_system::SI, RealType>(),
        vals.omega*fact<Quantity>::
            template from_to<unit_system::norm_omega, unit_system::norm_omega, RealType>(),
        vals.omega*fact<Quantity>::
            template from_to<unit_system::norm_omega, unit_system::norm_lambda, RealType>(),
        vals.omega*fact<Quantity>::
            template from_to<unit_system::norm_omega, unit_system::heaviside_lorentz, RealType>()
    };

    const auto from_lambda_to_all = std::array<RealType,4>
    {
        vals.lambda*fact<Quantity>::
            template from_to<unit_system::norm_lambda, unit_system::SI, RealType>(),
        vals.lambda*fact<Quantity>::
            template from_to<unit_system::norm_lambda, unit_system::norm_omega, RealType>(),
        vals.lambda*fact<Quantity>::
            template from_to<unit_system::norm_lambda, unit_system::norm_lambda, RealType>(),
        vals.lambda*fact<Quantity>::
            template from_to<unit_system::norm_lambda, unit_system::heaviside_lorentz, RealType>()
    };

    const auto from_hl_to_all = std::array<RealType,4>
    {
        vals.hl*fact<Quantity>::
            template from_to<unit_system::heaviside_lorentz, unit_system::SI, RealType>(),
        vals.hl*fact<Quantity>::
            template from_to<unit_system::heaviside_lorentz, unit_system::norm_omega, RealType>(),
        vals.hl*fact<Quantity>::
            template from_to<unit_system::heaviside_lorentz, unit_system::norm_lambda, RealType>(),
        vals.hl*fact<Quantity>::
            template from_to<unit_system::heaviside_lorentz, unit_system::heaviside_lorentz, RealType>()
    };

    const auto fact_SI = fact<Quantity>::
        template to_SI_from<unit_system::SI, RealType>();
    const auto fact_omega = fact<Quantity>::
        template to_SI_from<unit_system::norm_omega, RealType>();
    const auto fact_lambda = fact<Quantity>::
        template to_SI_from<unit_system::norm_lambda, RealType>();
    const auto fact_hl = fact<Quantity>::
        template to_SI_from<unit_system::heaviside_lorentz, RealType>();

    const auto all_facts = std::array<RealType, 4>{
        fact_SI, fact_omega, fact_lambda, fact_hl};

    const auto all_data = std::array<std::array<RealType, 4>, 4>{
        from_SI_to_all,
        from_omega_to_all,
        from_lambda_to_all,
        from_hl_to_all
    };

    for (auto data: all_data)
    {
        std::transform( data.begin(), data.end(),
            all_facts.begin(), data.begin(),
            std::multiplies<RealType>());

        for (const auto& res : data){
            BOOST_CHECK_SMALL((res-vals.SI)/vals.SI, tolerance<RealType>());
        }

    }
}

template<typename RealType, quantity Quantity>
constexpr void test_from_to(
    val_pack<RealType> vals,
    RealType reference_omega,
    RealType reference_length)

{
    const auto from_SI_to_all = std::array<RealType,4>
    {
        vals.SI*fact<Quantity>::
            template from_to<unit_system::SI, unit_system::SI, RealType>(1.0, 1.0),
        vals.SI*fact<Quantity>::
            template from_to<unit_system::SI, unit_system::norm_omega, RealType>(1.0, reference_omega),
        vals.SI*fact<Quantity>::
            template from_to<unit_system::SI, unit_system::norm_lambda, RealType>(1.0, reference_length),
        vals.SI*fact<Quantity>::
            template from_to<unit_system::SI, unit_system::heaviside_lorentz, RealType>(1.0, 1.0)
    };

    const auto from_omega_to_all = std::array<RealType,4>
    {
        vals.omega*fact<Quantity>::
            template from_to<unit_system::norm_omega, unit_system::SI, RealType>(reference_omega, 1.0),
        vals.omega*fact<Quantity>::
            template from_to<unit_system::norm_omega, unit_system::norm_omega, RealType>(reference_omega, reference_omega),
        vals.omega*fact<Quantity>::
            template from_to<unit_system::norm_omega, unit_system::norm_lambda, RealType>(reference_omega, reference_length),
        vals.omega*fact<Quantity>::
            template from_to<unit_system::norm_omega, unit_system::heaviside_lorentz, RealType>(reference_omega, 1.0)
    };

    const auto from_lambda_to_all = std::array<RealType,4>
    {
        vals.lambda*fact<Quantity>::
            template from_to<unit_system::norm_lambda, unit_system::SI, RealType>(reference_length, 1.0),
        vals.lambda*fact<Quantity>::
            template from_to<unit_system::norm_lambda, unit_system::norm_omega, RealType>(reference_length, reference_omega),
        vals.lambda*fact<Quantity>::
            template from_to<unit_system::norm_lambda, unit_system::norm_lambda, RealType>(reference_length, reference_length),
        vals.lambda*fact<Quantity>::
            template from_to<unit_system::norm_lambda, unit_system::heaviside_lorentz, RealType>(reference_length, 1.0)
    };

    const auto from_hl_to_all = std::array<RealType,4>
    {
        vals.hl*fact<Quantity>::
            template from_to<unit_system::heaviside_lorentz, unit_system::SI, RealType>(1.0, 1.0),
        vals.hl*fact<Quantity>::
            template from_to<unit_system::heaviside_lorentz, unit_system::norm_omega, RealType>(1.0, reference_omega),
        vals.hl*fact<Quantity>::
            template from_to<unit_system::heaviside_lorentz, unit_system::norm_lambda, RealType>(1.0, reference_length),
        vals.hl*fact<Quantity>::
            template from_to<unit_system::heaviside_lorentz, unit_system::heaviside_lorentz, RealType>(1.0, 1.0)
    };

    const auto fact_SI = fact<Quantity>::
        template to_SI_from<unit_system::SI, RealType>();
    const auto fact_omega = fact<Quantity>::
        template to_SI_from<unit_system::norm_omega, RealType>();
    const auto fact_lambda = fact<Quantity>::
        template to_SI_from<unit_system::norm_lambda, RealType>();
    const auto fact_hl = fact<Quantity>::
        template to_SI_from<unit_system::heaviside_lorentz, RealType>();

    const auto all_facts = std::array<RealType, 4>{
        fact_SI, fact_omega, fact_lambda, fact_hl};

    const auto all_data = std::array<std::array<RealType, 4>, 4>{
        from_SI_to_all,
        from_omega_to_all,
        from_lambda_to_all,
        from_hl_to_all
    };

    for (auto data: all_data)
    {
        std::transform( data.begin(), data.end(),
            all_facts.begin(), data.begin(),
            std::multiplies<RealType>());

        for (const auto& res : data){
            BOOST_CHECK_SMALL((res-vals.SI)/vals.SI, tolerance<RealType>());
        }

    }
}

// ------------- Tests --------------

// ***Test energy reference for Heaviside Lorentz units
template<typename RealType>
void test_case_hl_reference_energy()
{
    BOOST_CHECK_SMALL(
        (heaviside_lorentz_reference_energy<RealType>-MeV<RealType>)/MeV<RealType>,
        tolerance<RealType>());
}

BOOST_AUTO_TEST_CASE( picsar_unit_conv_heaviside_lorentz_ref_energy )
{
    test_case_hl_reference_energy<double>();
    test_case_hl_reference_energy<float>();
}

// ***Test mass conversion to SI and all to all
template<typename RealType>
void test_case_mass()
{
    constexpr auto mass_SI = electron_mass<RealType>;
    constexpr auto mass_omega = static_cast<RealType>(1.0);
    constexpr auto mass_lambda = static_cast<RealType>(1.0);
    constexpr auto mass_hl = electron_mass<RealType>*light_speed<RealType>*
        light_speed<RealType>/heaviside_lorentz_reference_energy<RealType>;
    constexpr auto all_masses = val_pack<RealType>{mass_SI, mass_omega, mass_lambda, mass_hl};

    test_to_SI<RealType, quantity::mass>(all_masses);
    test_from_to<RealType, quantity::mass>(all_masses);
}

BOOST_AUTO_TEST_CASE( picsar_unit_conv_mass )
{
    test_case_mass<double>();
    test_case_mass<float>();
}

// ***Test charge conversion to SI and all to all
template<typename RealType>
void test_case_charge()
{
    constexpr auto charge_SI = elementary_charge<RealType>;
    constexpr auto charge_omega = static_cast<RealType>(1.0);
    constexpr auto charge_lambda = static_cast<RealType>(1.0);
    constexpr auto charge_hl = sqrt_4_pi_fine_structure<RealType>;
    constexpr auto all_charges = val_pack<RealType>{charge_SI, charge_omega, charge_lambda, charge_hl};

    test_to_SI<RealType, quantity::charge>(all_charges);
    test_from_to<RealType, quantity::charge>(all_charges);
}

BOOST_AUTO_TEST_CASE( picsar_unit_conv_charge )
{
    test_case_charge<double>();
    test_case_charge<float>();
}

// ***Test velocity conversion to SI and all to all
template<typename RealType>
void test_case_velocity()
{
    constexpr auto velocity_SI = light_speed<RealType>;
    constexpr auto velocity_omega = static_cast<RealType>(1.0);
    constexpr auto velocity_lambda = static_cast<RealType>(1.0);
    constexpr auto velocity_hl = static_cast<RealType>(1.0);
    constexpr auto all_velocities = val_pack<RealType>{velocity_SI, velocity_omega, velocity_lambda, velocity_hl};

    test_to_SI<RealType, quantity::velocity>(all_velocities);
    test_from_to<RealType, quantity::velocity>(all_velocities);
}

BOOST_AUTO_TEST_CASE( picsar_unit_conv_velocity )
{
    test_case_velocity<double>();
    test_case_velocity<float>();
}

// ***Test momentum conversion to SI and all to all
template<typename RealType>
void test_case_momentum()
{
    constexpr auto momentum_SI = electron_mass<RealType>*light_speed<RealType>;
    constexpr auto momentum_omega = static_cast<RealType>(1.0);
    constexpr auto momentum_lambda = static_cast<RealType>(1.0);
    constexpr auto momentum_hl = static_cast<RealType>(
        electron_mass<double>*light_speed<double>*light_speed<double>/
        heaviside_lorentz_reference_energy<double>);
    constexpr auto all_momenta = val_pack<RealType>{momentum_SI, momentum_omega, momentum_lambda, momentum_hl};

    test_to_SI<RealType, quantity::momentum>(all_momenta);
    test_from_to<RealType, quantity::momentum>(all_momenta);
}

BOOST_AUTO_TEST_CASE( picsar_unit_conv_momentum )
{
    test_case_momentum<double>();
    test_case_momentum<float>();
}

// ***Test energy conversion to SI and all to all
template<typename RealType>
void test_case_energy()
{
    constexpr auto energy_SI = GeV<RealType>;
    constexpr auto energy_omega = static_cast<RealType>(
        GeV<double>/electron_mass<double>/light_speed<double>/light_speed<double>);
    constexpr auto energy_lambda = static_cast<RealType>(
        GeV<double>/electron_mass<double>/light_speed<double>/light_speed<double>);
    constexpr auto energy_hl = static_cast<RealType>(
        GeV<double>/heaviside_lorentz_reference_energy<double>);
    constexpr auto all_energies = val_pack<RealType>{energy_SI, energy_omega, energy_lambda, energy_hl};

    test_to_SI<RealType, quantity::energy>(all_energies);
    test_from_to<RealType, quantity::energy>(all_energies);
}

BOOST_AUTO_TEST_CASE( picsar_unit_conv_energy )
{
    test_case_energy<double>();
    test_case_energy<float>();
}

// ***Test length conversion to SI and all to all
template<typename RealType>
void test_case_length()
{
    constexpr auto reference_length = static_cast<RealType>(800.0e-9);
    constexpr auto reference_omega = static_cast<RealType>(
        2.0*pi<double>*light_speed<double>/reference_length);

    constexpr auto length_SI = reference_length;
    constexpr auto length_omega = static_cast<RealType>(2.0* pi<double>);
    constexpr auto length_lambda = static_cast<RealType>(1.0);
    constexpr auto length_hl = static_cast<RealType>(
        heaviside_lorentz_reference_energy<double>*reference_length/
        reduced_plank<double>/light_speed<double>);

    constexpr auto all_lenghts = val_pack<RealType>{length_SI, length_omega, length_lambda, length_hl};

    test_to_SI<RealType, quantity::length>(
        all_lenghts, reference_omega, reference_length);
    test_from_to<RealType, quantity::length>(
        all_lenghts, reference_omega, reference_length);
}

BOOST_AUTO_TEST_CASE( picsar_unit_conv_length)
{
    test_case_length<double>();
    test_case_length<float>();
}

// ***Test area conversion to SI
template<typename RealType>
void test_case_area_to_SI()
{
    constexpr auto reference_length = static_cast<RealType>(800.0e-9);
    constexpr auto reference_omega = static_cast<RealType>(
        2.0*pi<double>*light_speed<double>/reference_length);

    constexpr auto area_SI = reference_length*reference_length;
    constexpr auto area_omega = static_cast<RealType>(4.0* pi<double>*pi<double>);
    constexpr auto area_lambda = static_cast<RealType>(1.0);
    constexpr auto area_hl = static_cast<RealType>(
        heaviside_lorentz_reference_energy<double>*
        heaviside_lorentz_reference_energy<double>*
        reference_length*reference_length/
        reduced_plank<double>/reduced_plank<double>/
        light_speed<double>/light_speed<double>);

    constexpr auto all_areas = val_pack<RealType>{area_SI, area_omega, area_lambda, area_hl};

    test_to_SI<RealType, quantity::area>(
        all_areas, reference_omega, reference_length);
}

BOOST_AUTO_TEST_CASE( picsar_unit_conv_area_to_SI )
{
    test_case_area_to_SI<double>();
    test_case_area_to_SI<float>();
}

// ***Test volume conversion to SI
template<typename RealType>
void test_case_volume_to_SI()
{
    constexpr auto reference_length = static_cast<RealType>(800.0e-9);
    constexpr auto reference_omega = static_cast<RealType>(
        2.0*pi<double>*light_speed<double>/reference_length);

    constexpr auto volume_SI = reference_length*reference_length*reference_length;
    constexpr auto volume_omega = static_cast<RealType>(8.0*
        pi<double>*pi<double>*pi<double>);
    constexpr auto volume_lambda = static_cast<RealType>(1.0);
    constexpr auto volume_hl = static_cast<RealType>(
        heaviside_lorentz_reference_energy<double>*
        heaviside_lorentz_reference_energy<double>*
        heaviside_lorentz_reference_energy<double>*
        reference_length*reference_length*reference_length/
        reduced_plank<double>/reduced_plank<double>/reduced_plank<double>/
        light_speed<double>/light_speed<double>/light_speed<double>);

    constexpr auto all_volumes = val_pack<RealType>{volume_SI, volume_omega, volume_lambda, volume_hl};

    test_to_SI<RealType, quantity::volume>(
        all_volumes, reference_omega, reference_length);
}

BOOST_AUTO_TEST_CASE( picsar_unit_conv_volume_to_SI )
{
    test_case_volume_to_SI<double>();
    test_case_volume_to_SI<float>();
}

// ***Test time conversion to SI
template<typename RealType>
void test_case_time_to_SI()
{
    constexpr auto reference_length = static_cast<RealType>(800.0e-9);
    constexpr auto reference_omega = static_cast<RealType>(
        2.0*pi<double>*light_speed<double>/reference_length);

    constexpr auto time_SI = static_cast<RealType>(reference_length/light_speed<double>);
    constexpr auto time_omega = static_cast<RealType>(2.0*pi<double>);
    constexpr auto time_lambda = static_cast<RealType>(1.0);
    constexpr auto time_hl = static_cast<RealType>(
        (reference_length/light_speed<double>)*
        heaviside_lorentz_reference_energy<double>/
        reduced_plank<double>);

    constexpr auto all_times = val_pack<RealType>{time_SI, time_omega, time_lambda, time_hl};

    test_to_SI<RealType, quantity::ttime>(
        all_times, reference_omega, reference_length);
}

BOOST_AUTO_TEST_CASE( picsar_unit_conv_time_to_SI )
{
    test_case_time_to_SI<double>();
    test_case_time_to_SI<float>();
}

// ***Test rate conversion to SI
template<typename RealType>
void test_case_rate_to_SI()
{
    constexpr auto reference_length = static_cast<RealType>(800.0e-9);
    constexpr auto reference_omega = static_cast<RealType>(
        2.0*pi<double>*light_speed<double>/reference_length);

    constexpr auto rate_SI = static_cast<RealType>(light_speed<double>/reference_length);
    constexpr auto rate_omega = static_cast<RealType>(1/(2.0*pi<double>));
    constexpr auto rate_lambda = static_cast<RealType>(1.0);
    constexpr auto rate_hl = static_cast<RealType>(
        (light_speed<double>/reference_length)*
        reduced_plank<double>/
        heaviside_lorentz_reference_energy<double>);

    constexpr auto all_rates = val_pack<RealType>{rate_SI, rate_omega, rate_lambda, rate_hl};

    test_to_SI<RealType, quantity::rate>(
        all_rates, reference_omega, reference_length);
}

BOOST_AUTO_TEST_CASE( picsar_unit_conv_rate_to_SI )
{
    test_case_rate_to_SI<double>();
    test_case_rate_to_SI<float>();
}
