/**
* This program tests the generation of the lookup tables of the QED library and
* shows how to convert them into a binary file.
* For each table it produces also a csv file which can be inspected with pyplot
* or gnuplot.
*/

#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <omp.h>

#include "../QED/src/physics/breit_wheeler/breit_wheeler_engine_tables_generator.hpp"

namespace px_bw = picsar::multi_physics::phys::breit_wheeler;

template<typename RealType, typename TableType>
void write_csv_1d_table(const TableType& table,
    const RealType left, const RealType right,
    const int how_many,
    bool log_scale, const std::string& file_name)
{
    auto coords = std::vector<RealType>(how_many);
    if(log_scale){
            std::generate(coords.begin(), coords.end(), [=,i = 0]() mutable{
            return std::exp(
                std::log(left) + (i++)*(std::log(right)-std::log(left))/(how_many-1));
        });
    }
    else
    {
        std::generate(coords.begin(), coords.end(), [=,i = 0]() mutable{
            return left + (i++)*(right-left)/(how_many-1);
        });
    }

    auto res = std::vector<RealType>(how_many);
    #pragma omp parallel
    for(int i = 0 ; i < how_many; ++i){
        res[i] = table.interp(coords[i]);
    }

    std::ofstream of{file_name};
    for (int i = 0; i < how_many; ++i){
        of << coords[i] << ", " <<  res[i] << "\n";
    }
    of.close();
}

template<typename RealType, typename TableType>
void write_csv_2d_table(const TableType& table,
    const RealType x1, const RealType x2,
    const RealType y1, const RealType y2,
    const int how_many_x, const int how_many_y,
    bool log_scale_x, bool log_scale_y, const std::string& file_name)
{
    auto coords_x = std::vector<RealType>(how_many_x);
    auto coords_y = std::vector<RealType>(how_many_y);
    if(log_scale_x){
            std::generate(coords_x.begin(), coords_x.end(), [=,i = 0]() mutable{
            return std::exp(
                std::log(x1) + (i++)*(std::log(x2)-std::log(x1))/(how_many_x-1));
        });
    }
    else
    {
        std::generate(coords_x.begin(), coords_x.end(), [=,i = 0]() mutable{
            return x1 + (i++)*(x2-x1)/(how_many_x-1);
        });
    }

    if(log_scale_y){
            std::generate(coords_y.begin(), coords_y.end(), [=,i = 0]() mutable{
            return std::exp(
                std::log(y1) + (i++)*(std::log(y2)-std::log(y1))/(how_many_y-1));
        });
    }
    else
    {
        std::generate(coords_y.begin(), coords_y.end(), [=,i = 0]() mutable{
            return y1 + (i++)*(y2-y1)/(how_many_y-1);
        });
    }

    auto res = std::vector<RealType>(how_many_x * how_many_y);
    #pragma omp parallel
    for(int i = 0 ; i < how_many_x; ++i){
        for(int j = 0 ; j < how_many_y; ++j){
            res[i*how_many_y + j] = table.interp(coords_x[i], coords_y[j]);
        }
    }

    std::ofstream of{file_name};
    for(int i = 0 ; i < how_many_x; ++i){
        for(int j = 0 ; j < how_many_y; ++j){
            of << coords_x[i] << ", " <<  coords_y[j]  << ", " << res[i*how_many_y+j]/coords_x[i] << "\n";
        }
    }
    of.close();
}

template<
    typename RealType,
    px_bw::generation_policy Policy = px_bw::generation_policy::regular>
void generate_breit_wheeler_alt_pair_prod_table(
    px_bw::pair_prod_lookup_table_params<RealType> bw_params,
    const std::string& file_name)
{
    auto table = px_bw::alt_pair_prod_lookup_table<
        RealType, std::vector<RealType>>{
            bw_params};

    table.template generate<Policy>();

    const auto raw_data = table.serialize();

    std::ofstream of{file_name};
    of.write (raw_data.data(),raw_data.size());
    of.close();


    write_csv_2d_table(table, bw_params.chi_phot_min*0.5f, bw_params.chi_phot_max*2.f,
        RealType(0.0), RealType(1.0)-std::numeric_limits<RealType>::epsilon(), bw_params.chi_phot_how_many*3,
        bw_params.frac_how_many*3, true, false, file_name + ".csv");

}

int main(int argc, char** argv)
{
   auto params = px_bw::default_pair_prod_lookup_table_params<double>;
   params.chi_phot_how_many = 13;
   params.frac_how_many = 15;
   std::cout << "** Double precision table ** \n" << std::endl;
    generate_breit_wheeler_alt_pair_prod_table<double>(
        params,
        "bw_alt_pairprod_d");
    std::cout << "____________________________ \n" << std::endl;

    /*std::cout << "** Single precision tables calculated in double precision ** \n" << std::endl;
    generate_breit_wheeler_alt_pair_prod_table<float,
        px_bw::generation_policy::force_internal_double>(
        px_bw::default_pair_prod_lookup_table_params<float>,
        "bw_alt_pairprod_fd");
    std::cout << "____________________________ \n" << std::endl;

    std::cout << "** Single precision tables ** \n" << std::endl;
    generate_breit_wheeler_alt_pair_prod_table<float>(
        px_bw::default_pair_prod_lookup_table_params<float>,
        "bw_alt_pairprod_f");
     std::cout << "____________________________ \n" << std::endl;*/

    return 0;
}
