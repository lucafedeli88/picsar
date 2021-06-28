#include "omp_example_commons.hpp"

std::vector<std::mt19937> get_gen_pool(const unsigned int seed)
{
    const auto max_num_threads = omp_get_max_threads();
    auto rand_pool = std::vector<std::mt19937>(max_num_threads);

    auto aux_gen = std::ranlux48{seed};

    std::generate(
        rand_pool.begin(), rand_pool.end(),
        [&](){
        return std::mt19937{aux_gen()};});

    return rand_pool;
}
