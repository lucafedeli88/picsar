#ifndef __OMP_EXAMPLE_COMMONS__
#define __OMP_EXAMPLE_COMMONS__

//This file contains common functions and constants used by the two OMP examples

//Include OMP
#include <omp.h>
//__________________________________________________

//------------------------
#define PXRMP_HAS_OPENMP
//__________________________________________________

//Include PICSAR QED
#include <picsar_qed/physics/chi_functions.hpp>
#include <picsar_qed/physics/gamma_functions.hpp>
#include <picsar_qed/physics/breit_wheeler/breit_wheeler_engine_core.hpp>
#include <picsar_qed/physics/breit_wheeler/breit_wheeler_engine_tables.hpp>
#include <picsar_qed/physics/breit_wheeler/breit_wheeler_engine_tables_generator.hpp>
#include <picsar_qed/physics/quantum_sync/quantum_sync_engine_core.hpp>
#include <picsar_qed/physics/quantum_sync/quantum_sync_engine_tables.hpp>
#include <picsar_qed/physics/quantum_sync/quantum_sync_engine_tables_generator.hpp>
//__________________________________________________

#include <iostream>
#include <string>
#include <random>
#include <cstdlib>
#include <chrono>

//Some namespace aliases
namespace pxr =  picsar::multi_physics::phys;
namespace pxr_bw = picsar::multi_physics::phys::breit_wheeler;
namespace pxr_qs = picsar::multi_physics::phys::quantum_sync;
namespace pxr_m =  picsar::multi_physics::math;
//__________________________________________________


//Some useful physical constants and functions
template<typename T = double>
constexpr T mec = static_cast<T>(pxr::electron_mass<> * pxr::light_speed<>);

template<typename T = double>
constexpr T mec2= static_cast<T>(pxr::electron_mass<> * pxr::light_speed<> * pxr::light_speed<>);

const double Es = pxr::schwinger_field<>;
//__________________________________________________

template<typename Real>
using vec3 = std::array<Real,3>;

/**
* Data structure to emulate particle data in a Particle-In-Cell code.
* Momenta are in a num_particlesx3 vector, while field components and optical depths
* are each one in a separate vector of size num_particles.
*
* @tparam Real the floating point type to be used
*/
template<typename Real>
struct ParticleData{
    static constexpr int num_components = 3;
    int num_particles = 0;

    std::vector<vec3<Real>> m_momentum;
    struct {
        std::vector<Real> Ex;
        std::vector<Real> Ey;
        std::vector<Real> Ez;
        std::vector<Real> Bx;
        std::vector<Real> By;
        std::vector<Real> Bz;
        std::vector<Real> opt;
    } m_fields;
};

/**
* An auxiliary function to call typeid(T).name()
*
* @tparam T a typename
* @return the name of T as a string
*/
template<typename T>
std::string get_type_name()
{
    return typeid(T).name();
}

/**
* Initializes a 2D vector with N elements.
* The values are initialized randomly between min_val
* and max_val.
*
*/
template <typename Real, typename RandGenPool>
std::vector<vec3<Real>> init_vec3_with_random_content(
    const Real min_val, const Real max_val,
    const int N,
    RandGenPool& gen_pool)
{
    auto vec = std::vector<vec3<Real>>(N);
    auto unf = std::uniform_real_distribution<Real>{min_val, max_val};

    const auto len = static_cast<int>(vec.size());

    #pragma omp parallel for simd
    for(int i = 0; i < len; ++i){
        int tid = omp_get_thread_num();
        auto& gen = gen_pool[tid];
        vec[i][0] = unf(gen);
        vec[i][1] = unf(gen);
        vec[i][2] = unf(gen);
    }

    return vec;
}


/**
* Initializes a 1D vector with N elements.
* The values are initialized randomly between min_val
* and max_val.
*
*/
template <typename Real, typename RandGenPool>
std::vector<Real> init_vec_with_random_content(
    const Real min_val, const Real max_val,
    const int N,
    RandGenPool& gen_pool)
{
    auto vec = std::vector<Real>(N);
    const auto len = static_cast<int>(vec.size());

    #pragma omp parallel for simd
    for(int i = 0; i < len; ++i){
        auto unf = std::uniform_real_distribution<Real>{min_val, max_val};
        const int tid = omp_get_thread_num();
        auto& gen = gen_pool[tid];
        vec[i] = unf(gen);
    }

    return vec;
}

/**
* Initializes particle data
*
* @tparam Real the floating point type to be used
* @param N the number of particles
* @param[in] Pmin the minimum value of the momentum
* @param[in] Pmax the maximum value of the momentum
* @param[in] Emin the minimum value of the momentum
* @param[in] Emax the maximum value of the momentum
* @param[in] Bmin the minimum value of the momentum
* @param[in] Bmax the maximum value of the momentum
* @param[in,out] rand_pool a random pool
* @return the particle data
*/
template<typename Real, typename RandGenPool>
ParticleData<Real> create_particles(const int N,
Real Pmin, Real Pmax, Real Emin, Real Emax, Real Bmin, Real Bmax,
RandGenPool& gen_pool)
{
    ParticleData<Real> pdata;
    pdata.num_particles = N;

    pdata.m_momentum = init_vec3_with_random_content(
        Pmin, Pmax, N, gen_pool);
    pdata.m_fields.Ex = init_vec_with_random_content(
        Emin, Emax, N, gen_pool);
    pdata.m_fields.Ey = init_vec_with_random_content(
        Emin, Emax, N, gen_pool);
    pdata.m_fields.Ez = init_vec_with_random_content(
        Emin, Emax, N, gen_pool);
    pdata.m_fields.Bx = init_vec_with_random_content(
        Bmin, Bmax, N, gen_pool);
    pdata.m_fields.By = init_vec_with_random_content(
        Bmin, Bmax, N, gen_pool);
    pdata.m_fields.Bz = init_vec_with_random_content(
        Bmin, Bmax, N, gen_pool);
    pdata.m_fields.opt = init_vec_with_random_content(
        Real(0.0), Real(0.0), N, gen_pool);
    return pdata;
}

/**
* Checks a 1D vector for NaN and/or infs
*
*/
template <typename Real>
bool check(const std::vector<Real>& field,
    const bool check_nan = true, const bool check_inf = false)
{
    bool flag = true;
    const int N = field.size();

    if(check_nan && !check_inf){
        #pragma omp parallel for simd
        for(int i = 0; i < N; ++i){
            if(std::isnan(field[i])){
                flag = false;
            }
        }
    }
    else if(!check_nan && check_inf){
        #pragma omp parallel for simd
        for(int i = 0; i < N; ++i){
            if(std::isinf(field[i])){
                flag = false;
            }
        }
    }
    else if(check_nan && check_inf){
        #pragma omp parallel for simd
        for(int i = 0; i < N; ++i){
            if(std::isinf(field[i]) || std::isnan(field[i]) ){
                flag = false;
            }
        }
    }

    return flag;
}

/**
* Checks a 2D vector for NaN and/or infs
*
*/
template <typename Real>
bool check3(const std::vector<vec3<Real>>& field,
    const bool check_nan = true, const bool check_inf = false)
{
    bool flag = true;
    const int N = field.size();

    if(check_nan && !check_inf){
        #pragma omp parallel for simd
        for(int i = 0; i < N; ++i){
            const bool cond =
                std::isnan(field[i][0]) ||
                std::isnan(field[i][1]) ||
                std::isnan(field[i][2]);
            if(cond){
                flag = false;
            }
        }
    }
    else if(!check_nan && check_inf){
        #pragma omp parallel for simd
        for(int i = 0; i < N; ++i){
            const bool cond =
                std::isinf(field[i][0]) ||
                std::isinf(field[i][1]) ||
                std::isinf(field[i][2]);
            if(cond){
                flag = false;
            }
        }
    }
    else if(check_nan && check_inf){
        #pragma omp parallel for simd
        for(int i = 0; i < N; ++i){
            const bool cond =
                std::isinf(field[i][0]) || std::isnan(field[i][0]) ||
                std::isinf(field[i][1]) || std::isnan(field[i][1]) ||
                std::isinf(field[i][2]) || std::isnan(field[i][2]);
            if(cond){
                flag = false;
            }
        }
    }

    return flag;
}


std::vector<std::mt19937> get_gen_pool(const unsigned int seed);

#endif //__OMP_EXAMPLE_COMMONS__
