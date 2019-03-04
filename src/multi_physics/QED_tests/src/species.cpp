#include "species.h"

using namespace testbed;

species::species(std::string name):name(name){}

species::~species(){}

std::string species::get_name() const{
    return name;
}

double species::get_mass() const{
    return mass;
}

double species::get_charge() const{
    return charge;
}

int species::get_num_particles() const{
    return pos[0].size();
}

void species::add_simple_process(simple_process proc, int ID){
    if (simple_processes.find(ID) != simple_processes.end())
        throw std::logic_error("ID already assigned");
    else
        simple_processes.insert(std::pair<int, simple_process>(ID, proc));
}

void species::do_simple_processes(ttime dt){
    for(auto& proc : simple_processes){
        proc.second(pos, mom,  fields, optical_depth, mass, charge, dt);
    }
}

void species::add_particle(position part_pos, momentum part_mom){
    for(int i = 0; i < 3; i++){
        pos[i].push_back(part_pos[i]);
        mom[i].push_back(part_mom[i]);
    }
    for(int i = 0; i < 6; i++){
        fields[i].push_back(0);
    }
    optical_depth.push_back(0.0);

}

void species::remove_particle(int particle_index){
    for (auto& p : pos)
        p.erase(p.begin() + particle_index);

    for (auto& m : mom)
        m.erase(m.begin() + particle_index);

    for (auto& f: fields)
        f.erase(f.begin() + particle_index);

    optical_depth.erase(optical_depth.begin() + particle_index);
}


void species::calc_fields(em_field_function em_function, ttime tt){
    int num_particles = pos[0].size();

    for (int i = 0; i < num_particles; i++){
        em_field EB = em_function({pos[0][i], pos[1][i], pos[2][i]}, tt);
        for (size_t j = 0; j < EB.size(); j++){
            fields[j][i] = EB[j];
        }
    }
}

void species::print_on_disk(std::string prefix, int step_num) const{
    std::ofstream out_file(prefix + "_" + name + "_" + std::to_string(step_num) +".dat");

    out_file << "# " << header << std::endl;

    int num_particles = pos[0].size();
    out_file << "#N = " << num_particles << std::endl;

    out_file << "#x\ty\tz\tpx\tpy\tpz\tEx\tEy\tEz\tBx\tBy\tBz" << std::endl;

    for (int i = 0; i < num_particles; i++){
        out_file << pos[0][i] << " " << pos[1][i] << " " << pos[2][i] << " ";
        out_file << mom[0][i] << " " << mom[1][i] << " " << mom[2][i] << " ";
        out_file << fields[0][i] << " " << fields[1][i] << " " << fields[2][i] << " ";
        out_file << fields[3][i] << " " << fields[4][i] << " " << fields[5][i] << std::endl;
    }

    out_file.close();
}

std::tuple<positions_list, momenta_list,
em_field_list,  std::vector<double>> species::get_copy_of_all_data(){
    return std::tuple<positions_list, momenta_list,
    em_field_list,  std::vector<double>>{pos, mom, fields, optical_depth};
}

positions_list species::get_copy_of_positions(){
    return pos;
}

momenta_list species::get_copy_of_momenta(){
    return mom;
}

std::tuple<positions_list&, momenta_list&,
em_field_list&,  std::vector<double>&> species::get_ref_of_all_data(){
    return std::tuple<positions_list&, momenta_list&,
    em_field_list&,  std::vector<double>&>{pos, mom, fields, optical_depth};
}

positions_list& species::get_ref_of_positions(){
    return pos;
}
momenta_list& species::get_ref_of_momenta(){
    return mom;
}

std::vector<double>& species::get_ref_of_optical_depth(){
    return optical_depth;
}
