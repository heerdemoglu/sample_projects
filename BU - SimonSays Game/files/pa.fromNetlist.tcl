
# PlanAhead Launch Script for Post-Synthesis floorplanning, created by Project Navigator

create_project -name SimonSaysV8 -dir "C:/Users/Emre Erdemoglu/Desktop/EEE102Project/SimonSaysV8/planAhead_run_2" -part xc3s100ecp132-5
set_property design_mode GateLvl [get_property srcset [current_run -impl]]
set_property edif_top_file "C:/Users/Emre Erdemoglu/Desktop/EEE102Project/SimonSaysV8/SimonTop.ngc" [ get_property srcset [ current_run ] ]
add_files -norecurse { {C:/Users/Emre Erdemoglu/Desktop/EEE102Project/SimonSaysV8} }
set_param project.paUcfFile  "SimonTop.ucf"
add_files "SimonTop.ucf" -fileset [get_property constrset [current_run]]
open_netlist_design
