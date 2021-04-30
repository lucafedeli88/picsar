#ifndef __CUDA_EXAMPLE_COMMONS__
#define __CUDA_EXAMPLE_COMMONS__

//This file contains common functions and constants used by the two CUDA examples

//Include CUDA & thrust
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>

//__________________________________________________

//Sets PICSAR QED macros
#define PXRMP_WITH_GPU
#define PXRMP_GPU_QUALIFIER __host__ __device__
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
#include <cstdlib>

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

    __host__ __device__
    static int idx(int i, int cc){
        return num_components*i + cc;
    }

    Real* m_momentum;
    struct {
        Real* Ex;
        Real* Ey;
        Real* Ez;
        Real* Bx;
        Real* By;
        Real* Bz;
        Real* opt;
    } m_fields;
};

//A thin wrapper around thrust::device_vector
template<typename RealType>
class ThrustDeviceWrapper : public thrust::device_vector<RealType>
{
    public:
    template<typename... Args>
    ThrustDeviceWrapper(Args&&... args) : thrust::device_vector<RealType>(std::forward<Args>(args)...)
    {}

    const RealType* data() const
    {
        return thrust::raw_pointer_cast(thrust::device_vector<RealType>::data());
    }
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

template <typename Real>
__global__ void rescale(
    int N, Real* d_ptr, const Real min, const Real max)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < N){
        d_ptr[i] =  d_ptr[i] * (max-min) + min;
    }
}

void w_curandGenerateUniform(
    curandGenerator_t& gen,
    double* d_ptr,
    const unsigned int N)
{
    curandGenerateUniformDouble(gen, d_ptr, N);
}

void w_curandGenerateUniform(
    curandGenerator_t& gen,
    float* d_ptr,
    const unsigned int N
    )
{
    curandGenerateUniform(gen, d_ptr, N);
}

/**
* Initializes a 1D vector with N elements.
* The values are initialized randomly between min_val
* and max_val.
*
*/
template <typename Real>
void init_vec_with_random_content(
    Real* d_ptr,
    const Real min_val, const Real max_val,
    const unsigned int N,
    curandGenerator_t& gen)
{
    w_curandGenerateUniform(gen, d_ptr, N);
    rescale<<<(N+255)/256,256>>>(N, d_ptr, min_val, max_val);
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
template<typename Real>
ParticleData<Real> create_particles(const int N,
Real Pmin, Real Pmax, Real Emin, Real Emax, Real Bmin, Real Bmax,
curandGenerator_t& gen)
{
    ParticleData<Real> pdata;
    pdata.num_particles = N;

    cudaMalloc(&pdata.m_momentum, pdata.num_components*N*sizeof(Real));
    cudaMalloc(&pdata.m_fields.Ex,  N*sizeof(Real));
    cudaMalloc(&pdata.m_fields.Ey,  N*sizeof(Real));
    cudaMalloc(&pdata.m_fields.Ez,  N*sizeof(Real));
    cudaMalloc(&pdata.m_fields.Bx,  N*sizeof(Real));
    cudaMalloc(&pdata.m_fields.By,  N*sizeof(Real));
    cudaMalloc(&pdata.m_fields.Bz,  N*sizeof(Real));
    cudaMalloc(&pdata.m_fields.opt,  N*sizeof(Real));

    init_vec_with_random_content(pdata.m_momentum,
        Pmin, Pmax, N*pdata.num_components, gen);
    init_vec_with_random_content(pdata.m_fields.Ex,
        Emin, Emax, N, gen);
    init_vec_with_random_content(pdata.m_fields.Ey,
        Emin, Emax, N, gen);
    init_vec_with_random_content(pdata.m_fields.Ez,
        Emin, Emax, N, gen);
    init_vec_with_random_content(pdata.m_fields.Bx,
        Bmin, Bmax, N, gen);
    init_vec_with_random_content(pdata.m_fields.By,
        Bmin, Bmax, N, gen);
    init_vec_with_random_content(pdata.m_fields.Bz,
        Bmin, Bmax, N, gen);
    init_vec_with_random_content(pdata.m_fields.opt,
        Real(0.0), Real(0.0), N, gen);
    return pdata;
}

template<typename Real>
void free_particles(ParticleData<Real>& pdata )
{
    cudaFree(pdata.m_momentum);
    cudaFree(pdata.m_fields.Ex);
    cudaFree(pdata.m_fields.Ey);
    cudaFree(pdata.m_fields.Ez);
    cudaFree(pdata.m_fields.Bx);
    cudaFree(pdata.m_fields.By);
    cudaFree(pdata.m_fields.Bz);
    cudaFree(pdata.m_fields.opt);
}


/**
* Checks a 1D vector for NaN and/or infs
*
*/
template <typename Real>
bool check(const std::vector<Real>& field,
    const bool check_nan = true, const bool check_inf = false)
{
    if(check_nan && !check_inf){
        for(const auto el : field){
            if( std::isnan(el))
                return false;
        }
    }
    else if(!check_nan && check_inf){
        for(const auto el : field){
            if( std::isinf(el))
                return false;
        }
    }
    else if(check_nan && check_inf){
        for(const auto el : field){
            if( std::isinf(el) || std::isnan(el))
                return false;
        }
    }

    return true;
}

#endif //__CUDA_EXAMPLE_COMMONS__
