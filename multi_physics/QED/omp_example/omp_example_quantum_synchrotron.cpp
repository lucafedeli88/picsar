#include "omp_example_commons.hpp"

#include <cstdio>

// QUANTUM SYNCHROTRON EMISSION

//Parameters of the test case
const unsigned int how_many_particles = 20'000'000;
const double dt_test= 1e-18;
const double table_chi_min = 0.01;
const double table_chi_max = 1000.0;
const double table_frac_min = 1.0e-12;
const int table_chi_size = 128;
const int table_frac_size = 128;
const unsigned int random_seed = 22051988;
const double E_min = -0.01*Es;
const double E_max = 0.01*Es;
const double B_min = E_min/pxr::light_speed<>;
const double B_max = E_max/pxr::light_speed<>;
const double P_min = -100*mec<>;
const double P_max = 100*mec<>;
//__________________________________________________

/**
* Generates the dN/dt lookup table
*
* @tparam Real the floating point type to be used
* @tparam Vector the vector type to be used
* @param[in] chi_min the minimum chi parameter
* @param[in] chi_max the maximum chi parameter
* @param[in] chi_size the size of the lookup table along the chi axis
* @return the lookup table
*/
template <typename Real, typename Vector>
auto generate_dndt_table(Real chi_min, Real chi_max, int chi_size)
{
    std::cout << "Preparing dndt table [" << get_type_name<Real>()
        << ", " << chi_size <<"]...\n";
    std::cout.flush();

    pxr_qs::dndt_lookup_table_params<Real> qs_params{chi_min, chi_max, chi_size};

	auto table = pxr_qs::dndt_lookup_table<
        Real, Vector>{qs_params};

    table.generate();

    return table;
}

/**
* Generates the photon emission lookup table
*
* @tparam Real the floating point type to be used
* @tparam Vector the vector type to be used
* @param[in] chi_min the minimum chi parameter
* @param[in] chi_max the maximum chi parameter
* @param[in] chi_size the size of the lookup table along the chi axis
* @param[in] frac_size the size of the lookup table along the frac axis
* @return the lookup table
*/
template <typename Real, typename Vector>
auto generate_photon_emission_table(
    Real chi_min, Real chi_max, Real frac_min, int chi_size, int frac_size)
{
    std::cout << "Preparing photon emission table [" << get_type_name<Real>()
        << ", " << chi_size << " x " << frac_size <<"]...\n";
    std::cout.flush();

    pxr_qs::photon_emission_lookup_table_params<Real> qs_params{
        chi_min, chi_max, frac_min, chi_size, frac_size};

	auto table = pxr_qs::photon_emission_lookup_table<
        Real, Vector>{qs_params};

    table.template generate();

    return table;
}

template <typename Real>
__global__
void fill_opt(int N, Real* opt_data, Real* unf_zero_one)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < N){
        opt_data[i] = pxr_qs::get_optical_depth<Real>(
            Real(1.0)-unf_zero_one[i]);
    }
}


/**
* Tests the initialization of the optical depth
*
* @tparam Real the floating point type to be used
* @tparam TableType the lookup table type
* @param[in,out] pdata the particle data
* @param[in,out] rand_pool a random pool
* @return a bool success flag and the elapsed time in ms, packed in a pair
*/
template <typename Real, typename RandGenPool>
std::pair<bool, double>
fill_opt_test(
    ParticleData<Real>& pdata,
    RandGenPool& gen_pool)
{
    const auto N = pdata.num_particles;
    Real* rand;
    std::vector<Real> t_opt(N);

    cudaMalloc(&rand,N*sizeof(Real));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    w_curandGenerateUniform(gen, rand, N);
    fill_opt<<< (N+255)/256, 256 >>>(N, pdata.m_fields.opt, rand);
    cudaEventRecord(stop);

    cudaMemcpy(t_opt.data(), pdata.m_fields.opt, N*sizeof(Real), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);

    float time = 0.0;
    cudaEventElapsedTime(&time, start, stop);

    cudaFree(rand);

    return std::make_pair(check(t_opt, true, false), time);
}

template <typename Real, typename TableType>
__global__
void evolve_opt(const int N,
   const Real*__restrict__ mom,
   const Real*__restrict__ ex,
   const Real*__restrict__ ey,
   const Real*__restrict__ ez,
   const Real*__restrict__ bx,
   const Real*__restrict__ by,
   const Real*__restrict__ bz,
   Real*__restrict__ opt,
   const Real dt,
   const TableType ref_table)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < N){
        const auto& px = mom[ParticleData<Real>::idx(i,0)];
        const auto& py = mom[ParticleData<Real>::idx(i,1)];
        const auto& pz = mom[ParticleData<Real>::idx(i,2)];

        const auto ppx = px/mec<Real>;
	    const auto ppy = py/mec<Real>;
	    const auto ppz = pz/mec<Real>;
        const auto ee =
            pxr::compute_gamma_ele_pos<Real>(px, py, pz)*mec2<Real>;

        const auto chi =
            pxr::chi_ele_pos<Real, pxr::unit_system::SI>(
                px, py ,pz, ex[i], ey[i], ez[i], bx[i], by[i], bz[i]);
        pxr_qs::evolve_optical_depth<Real, TableType>(
                ee, chi, dt, opt[i], ref_table);
    }
}

template <typename Real, typename TableType>
std::pair<bool, double>
evolve_optical_depth(
    ParticleData<Real>& pdata,
    const TableType& ref_table,
    Real dt)
{
    const auto N = pdata.num_particles;
    std::vector<Real> t_opt(N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    evolve_opt<<< (N+255)/256, 256 >>>(
        N,
        pdata.m_momentum,
        pdata.m_fields.Ex, pdata.m_fields.Ey, pdata.m_fields.Ez,
        pdata.m_fields.Bx, pdata.m_fields.By, pdata.m_fields.Bz,
        pdata.m_fields.opt, dt, ref_table);
    cudaEventRecord(stop);

    cudaMemcpy(t_opt.data(), pdata.m_fields.opt, N*sizeof(Real), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float time = 0.0;
    cudaEventElapsedTime(&time, start, stop);

    return std::make_pair(check(t_opt, true, false), time);
}

template <typename Real, typename TableType>
__global__
void phot_gen(const int N,
   Real*__restrict__ mom,
   const Real*__restrict__ ex,
   const Real*__restrict__ ey,
   const Real*__restrict__ ez,
   const Real*__restrict__ bx,
   const Real*__restrict__ by,
   const Real*__restrict__ bz,
   const Real*__restrict__ rand,
   const TableType ref_table,
   Real*__restrict__ mom_phot)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < N){
        const auto& px = mom[ParticleData<Real>::idx(i,0)];
        const auto& py = mom[ParticleData<Real>::idx(i,1)];
        const auto& pz = mom[ParticleData<Real>::idx(i,2)];

        const auto unf_zero_one = Real(1.0)-rand[i];

        const auto chi = pxr::chi_ele_pos<Real, pxr::unit_system::SI>(
            px, py, pz, ex[i], ey[i], ez[i], bx[i], by[i], bz[i]);

        auto pp = pxr_m::vec3<Real>{px, py, pz};

        auto e_phot = pxr_m::vec3<Real>{0,0,0};

        pxr_qs::generate_photon_update_momentum<Real, TableType, pxr::unit_system::SI>(
            chi, pp,
            unf_zero_one,
            ref_table,
            e_phot);

        mom_phot[ParticleData<Real>::idx(i,0)] = e_phot[0];
        mom_phot[ParticleData<Real>::idx(i,1)] = e_phot[1];
        mom_phot[ParticleData<Real>::idx(i,2)] = e_phot[2];

        mom[ParticleData<Real>::idx(i,0)] = pp[0];
        mom[ParticleData<Real>::idx(i,1)] = pp[1];
        mom[ParticleData<Real>::idx(i,2)] = pp[2];
    }
}

/**
* Tests
*
* @tparam Real the floating point type to be used
* @tparam TableType the lookup table type
* @param[in,out] pdata the particle data
* @param[in] ref_table the pair production lookup table
* @param[in,out] gen a random pool
* @return a bool success flag and the elapsed time in ms, packed in a pair
*/
template <typename Real, typename TableType>
std::pair<bool, double>
generate_photons(
    ParticleData<Real>& pdata,
    const TableType ref_table,
    curandGenerator_t& gen)
{
    const auto N = pdata.num_particles;
    std::vector<Real> t_phot_mom(
        ParticleData<Real>::num_components*N);
    std::vector<Real> t_part_mom(
        ParticleData<Real>::num_components*N);

    Real *rand, *phot_momentum, *part_momentum;
    cudaMalloc(&rand,N*sizeof(Real));
    cudaMalloc(&phot_momentum,
        N*ParticleData<Real>::num_components*sizeof(Real));
    cudaMalloc(&part_momentum,
            N*ParticleData<Real>::num_components*sizeof(Real));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    w_curandGenerateUniform(gen, rand, N);
    phot_gen<<< (N+255)/256, 256 >>>(
        N,
        pdata.m_momentum,
        pdata.m_fields.Ex, pdata.m_fields.Ey, pdata.m_fields.Ez,
        pdata.m_fields.Bx, pdata.m_fields.By, pdata.m_fields.Bz,
        rand, ref_table, phot_momentum);
    cudaEventRecord(stop);

    cudaMemcpy(t_phot_mom.data(), phot_momentum,
        N*ParticleData<Real>::num_components*sizeof(Real), cudaMemcpyDeviceToHost);

    cudaMemcpy(t_part_mom.data(), pdata.m_momentum,
        N*ParticleData<Real>::num_components*sizeof(Real), cudaMemcpyDeviceToHost);


    cudaEventSynchronize(stop);

    float time = 0.0;
    cudaEventElapsedTime(&time, start, stop);

    cudaFree(rand);
    cudaFree(phot_momentum);
    cudaFree(part_momentum);

    return std::make_pair(
        check(t_phot_mom, true, true) && check(t_part_mom, true, true),
        time);
}


/**
* Performs tests with a given precision
*
* @tparam Real the floating point type to be used
* @param[in,out] rand_pool a random pool
*/
template <typename Real>
void do_test(curandGenerator_t& rand_pool)
{
    auto particle_data = create_particles<Real>(
        how_many_particles,
        P_min, P_max, E_min, E_max, B_min, B_max, rand_pool);

    const auto dndt_table =
        generate_dndt_table<Real, ThrustDeviceWrapper<Real>>(
            table_chi_min,
            table_chi_max,
            table_chi_size);

    const auto phot_em_table =
    generate_photon_emission_table<Real,ThrustDeviceWrapper<Real>>(
            table_chi_min,
            table_chi_max,
            table_frac_min,
            table_chi_size,
            table_frac_size);

    const auto dndt_table_view = dndt_table.get_view();
    const auto phot_em_table_view = phot_em_table.get_view();

    bool fill_opt_success = false; double fill_opt_time = 0.0;
    std::tie(fill_opt_success, fill_opt_time) =
        fill_opt_test<Real>(particle_data, rand_pool);
    std::cout << ( fill_opt_success? "[ OK ]":"[ FAIL ]" )
        << "  Fill Optical Depth : " << fill_opt_time << " ms" << std::endl;

    bool evolve_opt_success = false; double evolve_opt_time = 0.0;
    std::tie(evolve_opt_success, evolve_opt_time) =
        evolve_optical_depth<Real>(
            particle_data, dndt_table_view, dt_test);
    std::cout << ( evolve_opt_success? "[ OK ]":"[ FAIL ]" )
        << "  Evolve Optical Depth : " << evolve_opt_time << " ms" << std::endl;

    bool phot_em_success = false; double phot_em_time = 0.0;
    std::tie(phot_em_success, phot_em_time) =
        generate_photons<Real>(
            particle_data, phot_em_table_view, rand_pool);
    std::cout << ( phot_em_success? "[ OK ]":"[ FAIL ]" )
        << "  Photon Emission : " << phot_em_time << " ms" << std::endl;

    free_particles(particle_data);
}

int main(int argc, char** argv)
{

    std::cout << "*** CUDA example: begin ***" << std::endl;

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, random_seed);

    std::cout << "   --- Double precision test ---" << std::endl;
    do_test<double>(gen);
    std::cout << "   --- END ---" << std::endl;

    std::cout << "   --- Single precision test ---" << std::endl;
    do_test<float>(gen);
    std::cout << "   --- END ---" << std::endl;

    std::cout << "___ END ___" << std::endl;
    exit(EXIT_SUCCESS);
}
