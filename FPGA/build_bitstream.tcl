# Batch script to open the generated project and build bitstream
set proj "[file normalize "./vivado_ecg_proj/ecg_proj.xpr"]"
if {![file isfile $proj]} {
  puts "ERROR: Project not found: $proj"
  exit 1
}
open_project $proj
update_ip_catalog
upgrade_ip -quiet [get_ips *]
update_compile_order -fileset sources_1
reset_run synth_1
reset_run impl_1
launch_runs synth_1 -jobs 4
wait_on_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
set run_dir [get_property DIRECTORY [get_runs impl_1]]
set bitfile [file join $run_dir "design_1_wrapper.bit"]
if {[file exists $bitfile]} {
  puts "INFO: Bitstream generated: $bitfile"
} else {
  puts "ERROR: Bitstream not found in $run_dir"
  exit 2
}
exit 0
