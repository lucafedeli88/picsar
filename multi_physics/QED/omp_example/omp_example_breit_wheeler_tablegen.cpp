#include "omp_example_commons.hpp"

#include "omp_example_breit_wheeler_tablegen.h"

#include <cstdio>

pxr_bw::dndt_lookup_table<float, std::vector<float>>
xgenerate_dndt_table(
    float chi_min,
    float chi_max,
    int chi_size)
{
    std::cout << "Preparing dndt table [" << get_type_name<float>()
        << ", " << chi_size <<"]...\n";
    std::cout.flush();

    pxr_bw::dndt_lookup_table_params<float> bw_params{chi_min, chi_max, chi_size};

	auto table = pxr_bw::dndt_lookup_table<
        float, std::vector<float>>{bw_params};

    table.generate();

    return table;
}

pxr_bw::dndt_lookup_table<double, std::vector<double>>
xgenerate_dndt_table(
    double chi_min,
    double chi_max,
    int chi_size)
{
    std::cout << "Preparing dndt table [" << get_type_name<double>()
        << ", " << chi_size <<"]...\n";
    std::cout.flush();

    pxr_bw::dndt_lookup_table_params<double> bw_params{chi_min, chi_max, chi_size};

	auto table = pxr_bw::dndt_lookup_table<
        double, std::vector<double>>{bw_params};

    table.generate();

    return table;
}

pxr_bw::pair_prod_lookup_table<float, std::vector<float>>
xgenerate_pair_table(
    float chi_min,
    float chi_max,
    int chi_size,
    int frac_size)
{
    std::cout << "Preparing pair production table [" << get_type_name<float>()
        << ", " << chi_size << " x " << frac_size <<"]...\n";
    std::cout.flush();

    pxr_bw::pair_prod_lookup_table_params<float> bw_params{
        chi_min, chi_max, chi_size, frac_size};

	auto table = pxr_bw::pair_prod_lookup_table<
        float, std::vector<float>>{bw_params};

    table.template generate();

    return table;
}

pxr_bw::pair_prod_lookup_table<double, std::vector<double>>
xgenerate_pair_table(
    double chi_min,
    double chi_max,
    int chi_size,
    int frac_size)
{
    std::cout << "Preparing pair production table [" << get_type_name<double>()
        << ", " << chi_size << " x " << frac_size <<"]...\n";
    std::cout.flush();

    pxr_bw::pair_prod_lookup_table_params<double> bw_params{
        chi_min, chi_max, chi_size, frac_size};

	auto table = pxr_bw::pair_prod_lookup_table<
        double, std::vector<double>>{bw_params};

    table.template generate();

    return table;
}
