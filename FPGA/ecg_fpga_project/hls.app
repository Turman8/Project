<project xmlns="com.autoesl.autopilot.project" name="ecg_fpga_project" top="ecg_classify_trained" ideType="classic" projectType="C/C++">
    <files>
        <file name="hls_source/weights.h" sc="0" tb="false" cflags="" blackbox="false" csimflags=""/>
        <file name="hls_source/ecg_params.vh" sc="0" tb="false" cflags="" blackbox="false" csimflags=""/>
        <file name="hls_source/ecg_trained_classifier.cpp" sc="0" tb="false" cflags="" blackbox="false" csimflags=""/>
        <file name="../../testbench/tb_ecg_classifier.cpp" sc="0" tb="1" cflags="-Wno-unknown-pragmas" blackbox="false" csimflags=""/>
    </files>
    <includePaths/>
    <libraryPaths/>
    <Simulation argv="">
        <SimFlow name="csim" ldflags="" mflags="" clean="true" setup="false" optimizeCompile="false"/>
    </Simulation>
    <solutions xmlns="">
        <solution name="solution1" status="active"/>
    </solutions>
</project>

