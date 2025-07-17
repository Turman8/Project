<AutoPilot:project xmlns:AutoPilot="com.autoesl.autopilot.project" projectType="C/C++" name="hls_project" ideType="classic" top="ecg_classify_manual_fixed">
    <files>
        <file name="hls_source/params.vh" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="hls_source/weights.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="hls_source/classifier.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="hls_source/classifier.cpp" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="../../testbench/testbench.cpp" sc="0" tb="1" cflags="-Wno-unknown-pragmas" csimflags="" blackbox="false"/>
    </files>
    <solutions>
        <solution name="solution1" status=""/>
    </solutions>
    <Simulation argv="">
        <SimFlow name="csim" setup="false" optimizeCompile="false" clean="true" ldflags="" mflags=""/>
    </Simulation>
</AutoPilot:project>

