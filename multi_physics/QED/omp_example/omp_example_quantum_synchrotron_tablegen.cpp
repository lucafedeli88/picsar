#include "omp_example_commons.hpp"

#include "omp_example_quantum_synchrotron_tablegen.h"

pxr_qs::dndt_lookup_table<float, std::vector<float>>
xgenerate_dndt_table(
    float chi_min, float chi_max, int chi_size)
{
    std::cout << "Preparing dndt table [" << get_type_name<float>()
        << ", " << chi_size <<"]...\n";
    std::cout.flush();

    pxr_qs::dndt_lookup_table_params<float> qs_params{chi_min, chi_max, chi_size};

	auto table = pxr_qs::dndt_lookup_table<
        float, std::vector<float>>{qs_params};

    table.generate();

    return table;
}

pxr_qs::dndt_lookup_table<double, std::vector<double>>
xgenerate_dndt_table(
    double chi_min, double chi_max, int chi_size)
{
    std::cout << "Preparing dndt table [" << get_type_name<double>()
        << ", " << chi_size <<"]...\n";
    std::cout.flush();

    pxr_qs::dndt_lookup_table_params<double> qs_params{chi_min, chi_max, chi_size};

	auto table = pxr_qs::dndt_lookup_table<
        double, std::vector<double>>{qs_params};

    table.generate();

    return table;
}

pxr_qs::photon_emission_lookup_table<float, std::vector<float>>
xgenerate_photon_emission_table(
    float chi_min, float chi_max, float frac_min, int chi_size, int frac_size)
{
    std::cout << "Preparing photon emission table [" << get_type_name<float>()
        << ", " << chi_size << " x " << frac_size <<"]...\n";
    std::cout.flush();

    pxr_qs::photon_emission_lookup_table_params<float> qs_params{
        chi_min, chi_max, frac_min, chi_size, frac_size};

	auto table = pxr_qs::photon_emission_lookup_table<
        float, std::vector<float>>{qs_params};

    table.template generate();

    return table;
}

pxr_qs::photon_emission_lookup_table<double, std::vector<double>>
xgenerate_photon_emission_table(
    double chi_min, double chi_max, double frac_min, int chi_size, int frac_size)
{
    std::cout << "Preparing photon emission table [" << get_type_name<double>()
        << ", " << chi_size << " x " << frac_size <<"]...\n";
    std::cout.flush();

    pxr_qs::photon_emission_lookup_table_params<double> qs_params{
        chi_min, chi_max, frac_min, chi_size, frac_size};

	auto table = pxr_qs::photon_emission_lookup_table<
        double, std::vector<double>>{qs_params};

    table.template generate();

    return table;
}
