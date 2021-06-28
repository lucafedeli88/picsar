#ifndef __OMP_EXAMPLE_QS_TABLEGEN__
#define __OMP_EXAMPLE_QS_TABLEGEN__

#include "omp_example_commons.hpp"

pxr_qs::dndt_lookup_table<float, std::vector<float>>
xgenerate_dndt_table(
    float chi_min,
    float chi_max,
    int chi_size);

pxr_qs::dndt_lookup_table<double, std::vector<double>>
xgenerate_dndt_table(
    double chi_min,
    double chi_max,
    int chi_size);

/**
* Generates the dN/dt lookup table
*
* @tparam Real the floating point type to be used
* @param[in] chi_min the minimum chi parameter
* @param[in] chi_max the maximum chi parameter
* @param[in] chi_size the size of the lookup table along the chi axis
* @return the lookup table
*/
template <typename Real>
auto generate_dndt_table(Real chi_min, Real chi_max, int chi_size)
{
    return xgenerate_dndt_table(
        chi_min,
        chi_max,
        chi_size);
}

pxr_qs::photon_emission_lookup_table<float, std::vector<float>>
xgenerate_photon_emission_table(
    float chi_min,
    float chi_max,
    float frac_min,
    int chi_size,
    int frac_size);

pxr_qs::photon_emission_lookup_table<double, std::vector<double>>
xgenerate_photon_emission_table(
    double chi_min,
    double chi_max,
    double frac_min,
    int chi_size,
    int frac_size);

/**
* Generates the photon emission lookup table
*
* @tparam Real the floating point type to be used
* @param[in] chi_min the minimum chi parameter
* @param[in] chi_max the maximum chi parameter
* @param[in] chi_size the size of the lookup table along the chi axis
* @param[in] frac_size the size of the lookup table along the frac axis
* @return the lookup table
*/
template <typename Real>
auto generate_photon_emission_table(
    Real chi_min, Real chi_max, Real frac_min, int chi_size, int frac_size)
{
    return xgenerate_photon_emission_table(
        chi_min,
        chi_max,
        frac_min,
        chi_size,
        frac_size
    );
}

#endif //__OMP_EXAMPLE_QS_TABLEGEN__
