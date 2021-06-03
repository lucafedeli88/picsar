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

    #pragma ivdep
    #pragma omp parallel for
    for (int i = 0; i < N; ++i){
        auto unf = std::uniform_real_distribution<Real>{Real(0.0), Real(1.0)};
        const int tid = omp_get_thread_num();
        auto& gen = gen_pool[tid];
        pdata.m_fields.opt[i] = pxr_qs::get_optical_depth<Real>(unf(gen));
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

    #pragma ivdep
    #pragma omp parallel for
    for (int i = 0; i < N; ++i){

        const auto& px = pdata.m_momentum[i][0];
        const auto& py = pdata.m_momentum[i][1];
        const auto& pz = pdata.m_momentum[i][2];

        const auto chi = pxr::chi_ele_pos<Real, pxr::unit_system::SI>(
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
        const auto ee =
            pxr::compute_gamma_ele_pos<Real>(px, py, pz)*mec2<Real>;

        pxr_qs::evolve_optical_depth<Real, TableType>(
                ee, chi, dt, pdata.m_fields.opt[i], ref_table);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end-start;

    return std::make_pair(check(pdata.m_fields.opt, true, false), elapsed.count()*1000.0);
}

template <typename Real, typename TableType, typename RandGenPool>
std::pair<bool, double>
generate_photons(
    ParticleData<Real>& pdata,
    const TableType ref_table,
    RandGenPool& gen_pool)
{
    const auto N = pdata.num_particles;
    std::vector<vec3<Real>> momentum_phot(N);

    auto start = std::chrono::high_resolution_clock::now();

    #pragma ivdep
    #pragma omp parallel for
    for (int i = 0; i < N; ++i){
        auto unf = std::uniform_real_distribution<Real>{Real(0.0), Real(1.0)};
        const int tid = omp_get_thread_num();
        auto& gen = gen_pool[tid];

        const auto& px = pdata.m_momentum[i][0];
        const auto& py = pdata.m_momentum[i][1];
        const auto& pz = pdata.m_momentum[i][2];

        const auto chi = pxr::chi_ele_pos<Real, pxr::unit_system::SI>(
            px, py, pz,
            pdata.m_fields.Ex[i],
            pdata.m_fields.Ey[i],
            pdata.m_fields.Ez[i],
            pdata.m_fields.Bx[i],
            pdata.m_fields.By[i],
            pdata.m_fields.Bz[i]);

        auto pp = pxr_m::vec3<Real>{px, py, pz};

        auto phot_mom = pxr_m::vec3<Real>{0,0,0};

        pxr_qs::generate_photon_update_momentum<Real, TableType, pxr::unit_system::SI>(
            chi, pp,
            unf(gen),
            ref_table,
            phot_mom);

        std::copy(phot_mom.begin(), phot_mom.end(), pdata.m_momentum[i].begin());
        std::copy(pp.begin(), pp.end(), pdata.m_momentum[i].begin());
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end-start;

    return std::make_pair(
        check3(momentum_phot, true, true) && check3(pdata.m_momentum, true, true),
        elapsed.count()*1000.0);
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

    const auto dndt_table =
        generate_dndt_table<Real, std::vector<Real>>(
            table_chi_min,
            table_chi_max,
            table_chi_size);

    const auto phot_em_table =
    generate_photon_emission_table<Real, std::vector<Real>>(
            table_chi_min,
            table_chi_max,
            table_frac_min,
            table_chi_size,
            table_frac_size);

    const auto dndt_table_view = dndt_table.get_view();
    const auto phot_em_table_view = phot_em_table.get_view();

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

    bool phot_em_success = false; double phot_em_time = 0.0;
    std::tie(phot_em_success, phot_em_time) =
        generate_photons<Real>(
            particle_data, phot_em_table_view, gen_pool);
    std::cout << ( phot_em_success? "[ OK ]":"[ FAIL ]" )
        << "  Photon Emission : " << phot_em_time << " ms" << std::endl;
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
