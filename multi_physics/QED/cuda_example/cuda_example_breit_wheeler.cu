#include "cuda_example_commons.hpp"

#include <cstdio>

// BREIT-WHEELER PAIR PRODUCTION

//Parameters of the test case
const unsigned int how_many_particles = 20'000'000;
const double dt_test= 1e-18;
const double table_chi_min = 0.01;
const double table_chi_max = 1000.0;
const int table_chi_size = 128;
const int table_frac_size = 128;
const int random_seed = 22051988;
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

    pxr_bw::dndt_lookup_table_params<Real> bw_params{chi_min, chi_max, chi_size};

	auto table = pxr_bw::dndt_lookup_table<
        Real, Vector>{bw_params};

    table.generate();

    return table;
}

/**
* Generates the pair production lookup table
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
auto generate_pair_table(Real chi_min, Real chi_max, int chi_size, int frac_size)
{
    std::cout << "Preparing pair production table [" << get_type_name<Real>()
        << ", " << chi_size << " x " << frac_size <<"]...\n";
    std::cout.flush();

    pxr_bw::pair_prod_lookup_table_params<Real> bw_params{
        chi_min, chi_max, chi_size, frac_size};

	auto table = pxr_bw::pair_prod_lookup_table<
        Real, Vector>{bw_params};

    table.template generate();

    return table;
}

template <typename Real>
__global__
void fill_opt(int N, Real* opt_data, Real* unf_zero_one)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < N){
        opt_data[i] = pxr_bw::get_optical_depth<Real>(
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
template <typename Real>
std::pair<bool, double>
fill_opt_test(
    ParticleData<Real>& pdata,
    curandGenerator_t& gen)
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
        const auto chi = pxr::chi_photon<Real, pxr::unit_system::SI>(
            px, py, pz, ex[i], ey[i], ez[i], bx[i], by[i], bz[i]);

        const auto ppx = px/mec<Real>;
	    const auto ppy = py/mec<Real>;
	    const auto ppz = pz/mec<Real>;
        const auto en = (sqrt(ppx*ppx + ppy*ppy + ppz*ppz))*mec2<Real>;

        pxr_bw::evolve_optical_depth<Real,TableType>(
            en, chi, dt, opt[i],
            ref_table);
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
void pair_gen(const int N,
   const Real*__restrict__ mom,
   const Real*__restrict__ ex,
   const Real*__restrict__ ey,
   const Real*__restrict__ ez,
   const Real*__restrict__ bx,
   const Real*__restrict__ by,
   const Real*__restrict__ bz,
   const Real*__restrict__ rand,
   const TableType ref_table,
   Real*__restrict__ mom_ele,
   Real*__restrict__ mom_pos)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < N){
        const auto& px = mom[ParticleData<Real>::idx(i,0)];
        const auto& py = mom[ParticleData<Real>::idx(i,1)];
        const auto& pz = mom[ParticleData<Real>::idx(i,2)];

        const auto unf_zero_one = Real(1.0)-rand[i];

        const auto chi = pxr::chi_photon<Real, pxr::unit_system::SI>(
            px, py, pz, ex[i], ey[i], ez[i], bx[i], by[i], bz[i]);

        const auto pp = pxr_m::vec3<Real>{px, py, pz};

        auto e_mom = pxr_m::vec3<Real>{0,0,0};
        auto p_mom = pxr_m::vec3<Real>{0,0,0};

        pxr_bw::generate_breit_wheeler_pairs<Real, TableType, pxr::unit_system::SI>(
            chi, pp,
            unf_zero_one,
            ref_table,
            e_mom, p_mom);

        mom_ele[ParticleData<Real>::idx(i,0)] = e_mom[0];
        mom_ele[ParticleData<Real>::idx(i,1)] = e_mom[1];
        mom_ele[ParticleData<Real>::idx(i,2)] = e_mom[2];
        mom_pos[ParticleData<Real>::idx(i,0)] = p_mom[0];
        mom_pos[ParticleData<Real>::idx(i,1)] = p_mom[1];
        mom_pos[ParticleData<Real>::idx(i,2)] = p_mom[2];
    }
}

/**
* Tests pair production
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
generate_pairs(
    ParticleData<Real>& pdata,
    const TableType ref_table,
    curandGenerator_t& gen)
{
    const auto N = pdata.num_particles;
    std::vector<Real> t_ele_mom(
        ParticleData<Real>::num_components*N);
    std::vector<Real> t_pos_mom(
        ParticleData<Real>::num_components*N);

    Real *rand, *ele_momentum, *pos_momentum;
    cudaMalloc(&rand,N*sizeof(Real));
    cudaMalloc(&ele_momentum,
        N*ParticleData<Real>::num_components*sizeof(Real));
    cudaMalloc(&pos_momentum,
        N*ParticleData<Real>::num_components*sizeof(Real));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    w_curandGenerateUniform(gen, rand, N);
    pair_gen<<< (N+255)/256, 256 >>>(
        N,
        pdata.m_momentum,
        pdata.m_fields.Ex, pdata.m_fields.Ey, pdata.m_fields.Ez,
        pdata.m_fields.Bx, pdata.m_fields.By, pdata.m_fields.Bz,
        rand, ref_table, ele_momentum, pos_momentum);
    cudaEventRecord(stop);

    cudaMemcpy(t_ele_mom.data(), pdata.m_fields.opt,
        N*ParticleData<Real>::num_components*sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(t_pos_mom.data(), pdata.m_fields.opt,
        N*ParticleData<Real>::num_components*sizeof(Real), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);

    float time = 0.0;
    cudaEventElapsedTime(&time, start, stop);

    cudaFree(rand);
    cudaFree(ele_momentum);
    cudaFree(pos_momentum);

    return std::make_pair(
        check(t_ele_mom, true, true) && check(t_pos_mom, true, true),
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
    //correct_low_momenta(particle_data);

    const auto dndt_table =
        generate_dndt_table<Real, ThrustDeviceWrapper<Real>>(
            table_chi_min,
            table_chi_max,
            table_chi_size);

    const auto pair_table =
        generate_pair_table<Real,ThrustDeviceWrapper<Real>>(
            table_chi_min,
            table_chi_max,
            table_chi_size,
            table_frac_size);

    const auto dndt_table_view = dndt_table.get_view();
    const auto pair_table_view = pair_table.get_view();

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

    bool pair_prod_success = false; double pair_prod_time = 0.0;
    std::tie(pair_prod_success, pair_prod_time) =
        generate_pairs<Real>(
            particle_data, pair_table_view, rand_pool);
    std::cout << ( pair_prod_success? "[ OK ]":"[ FAIL ]" )
        << "  Pair Production : " << pair_prod_time << " ms" << std::endl;

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
