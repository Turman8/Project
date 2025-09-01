# Export hardware XSA (with bitstream if available)
set proj "[file normalize "./vivado_ecg_proj/ecg_proj.xpr"]"
if {![file isfile $proj]} {
  puts "ERROR: Project not found: $proj"
  exit 1
}
open_project $proj
set outdir [file normalize "./vivado_ecg_proj/export"]
file mkdir $outdir
set xsa [file join $outdir "design_1_wrapper.xsa"]
# Try to include bitstream if present
set run_dir [get_property DIRECTORY [get_runs impl_1]]
set bitfile [file join $run_dir "design_1_wrapper.bit"]
if {[file exists $bitfile]} {
  write_hw_platform -fixed -include_bit -file $xsa
} else {
  puts "WARN: Bitstream not found; exporting XSA without bit..."
  write_hw_platform -fixed -file $xsa
}
puts "INFO: Exported $xsa"
exit 0
