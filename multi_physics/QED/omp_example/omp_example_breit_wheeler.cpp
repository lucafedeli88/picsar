#include "omp_example_commons.hpp"

#include <cstdio>

// BREIT-WHEELER PAIR PRODUCTION

//Parameters of the test case
const unsigned int how_many_particles = 20'000'000;
const double dt_test= 1e-18;
const double table_chi_min = 0.01;
const double table_chi_max = 1000.0;
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

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for simd
    for (int i = 0; i < N; ++i){
        auto unf = std::uniform_real_distribution<Real>{Real(0.0), Real(1.0)};
        const int tid = omp_get_thread_num();
        auto& gen = gen_pool[tid];
        pdata.m_fields.opt[i] = pxr_bw::get_optical_depth<Real>(unf(gen));
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end-start;
    return std::make_pair(check(pdata.m_fields.opt, true, false), elapsed.count()*1000.0);
}

template <typename Real, typename TableType>
std::pair<bool, double>
evolve_optical_depth(
    ParticleData<Real>& pdata,
    const TableType& ref_table,
    Real dt)
{
    const auto N = pdata.num_particles;

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for simd
    for (int i = 0; i < N; ++i){

        const auto& px = pdata.m_momentum[i][0];
        const auto& py = pdata.m_momentum[i][1];
        const auto& pz = pdata.m_momentum[i][2];

        const auto chi = pxr::chi_photon<Real, pxr::unit_system::SI>(
            px, py, pz,
            pdata.m_fields.Ex[i],
            pdata.m_fields.Ey[i],
            pdata.m_fields.Ez[i],
            pdata.m_fields.Bx[i],
            pdata.m_fields.By[i],
            pdata.m_fields.Bz[i]);

        const auto ppx = px/mec<Real>;
	    const auto ppy = py/mec<Real>;
	    const auto ppz = pz/mec<Real>;
        const auto en = (sqrt(ppx*ppx + ppy*ppy + ppz*ppz))*mec2<Real>;

        pxr_bw::evolve_optical_depth<Real,TableType>(
            en, chi, dt,
            pdata.m_fields.opt[i],
            ref_table);

    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end-start;

    return std::make_pair(check(pdata.m_fields.opt, true, false), elapsed.count()*1000.0);
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
template <typename Real, typename TableType, typename RandGenPool>
std::pair<bool, double>
generate_pairs(
    ParticleData<Real>& pdata,
    const TableType ref_table,
    RandGenPool& gen_pool)
{
    const auto N = pdata.num_particles;
    std::vector<vec3<Real>> ele_mom(N);
    std::vector<vec3<Real>> pos_mom(N);

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for simd
    for (int i = 0; i < N; ++i){
        auto unf = std::uniform_real_distribution<Real>{Real(0.0), Real(1.0)};
        const int tid = omp_get_thread_num();
        auto& gen = gen_pool[tid];

        const auto& px = pdata.m_momentum[i][0];
        const auto& py = pdata.m_momentum[i][1];
        const auto& pz = pdata.m_momentum[i][2];

        const auto chi = pxr::chi_photon<Real, pxr::unit_system::SI>(
            px, py, pz,
            pdata.m_fields.Ex[i],
            pdata.m_fields.Ey[i],
            pdata.m_fields.Ez[i],
            pdata.m_fields.Bx[i],
            pdata.m_fields.By[i],
            pdata.m_fields.Bz[i]);

        const auto pp = pxr_m::vec3<Real>{px, py, pz};

        auto e_mom = pxr_m::vec3<Real>{0,0,0};
        auto p_mom = pxr_m::vec3<Real>{0,0,0};

        pxr_bw::generate_breit_wheeler_pairs<Real, TableType, pxr::unit_system::SI>(
            chi, pp,
            unf(gen),
            ref_table,
            e_mom, p_mom);

        std::copy(e_mom.begin(), e_mom.end(), ele_mom[i].begin());
        std::copy(p_mom.begin(), p_mom.end(), pos_mom[i].begin());
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end-start;

    return std::make_pair(
        check3(ele_mom, true, true) && check3(pos_mom, true, true),
        elapsed.count()*1000.0);
}

/**
* Corrects momenta which are too low to allow the decay of
* a photon into a pair.
*
* @tparam Real the floating point type to be used
* @param[in,out] pdata the particle data
*/
template<typename Real>
void correct_low_momenta(ParticleData<Real>& pdata)
{
    const auto N = pdata.num_particles;

    auto& mom = pdata.m_momentum;

    #pragma omp parallel for simd
    for (int i = 0; i < N; ++i){

        auto& px = mom[i][0];
        auto& py = mom[i][1];
        auto& pz = mom[i][2];

        const auto gamma_gamma =
            pxr::compute_gamma_photon<Real>(px, py, pz);

        const auto bb = Real(2.1);

        if(gamma_gamma == Real(0.0) ){
            const auto cc = bb/std::sqrt(Real(3.0));
            px = cc*mec<Real>;
            py = cc*mec<Real>;
            pz = cc*mec<Real>;
        }
        else if (gamma_gamma < Real(2.0)){
            const auto cc = bb/gamma_gamma;
            px *= cc;
            py *= cc;
            pz *= cc;
        }
    }
}


/**
* Performs tests with a given precision
*
* @tparam Real the floating point type to be used
* @param[in,out] rand_pool a random pool
*/
template <typename Real, typename RandGenPool>
void do_test(RandGenPool& gen_pool)
{
    auto particle_data = create_particles<Real>(
        how_many_particles,
        P_min, P_max, E_min, E_max, B_min, B_max, gen_pool);
    correct_low_momenta(particle_data);

    const auto dndt_table =
        generate_dndt_table<Real, std::vector<Real>>(
            table_chi_min,
            table_chi_max,
            table_chi_size);

    const auto pair_table =
        generate_pair_table<Real, std::vector<Real>>(
            table_chi_min,
            table_chi_max,
            table_chi_size,
            table_frac_size);

    const auto dndt_table_view = dndt_table.get_view();
    const auto pair_table_view = pair_table.get_view();

    bool fill_opt_success = false; double fill_opt_time = 0.0;
    std::tie(fill_opt_success, fill_opt_time) =
        fill_opt_test<Real>(particle_data, gen_pool);
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
            particle_data, pair_table_view, gen_pool);
    std::cout << ( pair_prod_success? "[ OK ]":"[ FAIL ]" )
        << "  Pair Production : " << pair_prod_time << " ms" << std::endl;
}

int main(int argc, char** argv)
{

    std::cout << "*** OMP example: begin ***" << std::endl;

    auto gen_pool = get_gen_pool(random_seed);

    std::cout << "   --- Double precision test ---" << std::endl;
    do_test<double>(gen_pool);
    std::cout << "   --- END ---" << std::endl;

    std::cout << "   --- Single precision test ---" << std::endl;
    do_test<float>(gen_pool);
    std::cout << "   --- END ---" << std::endl;

    std::cout << "___ END ___" << std::endl;
    exit(EXIT_SUCCESS);
}
