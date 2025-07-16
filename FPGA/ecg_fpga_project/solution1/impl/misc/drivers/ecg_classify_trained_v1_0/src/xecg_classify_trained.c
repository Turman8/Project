// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.1.2 (64-bit)
// Tool Version Limit: 2024.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
/***************************** Include Files *********************************/
#include "xecg_classify_trained.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XEcg_classify_trained_CfgInitialize(XEcg_classify_trained *InstancePtr, XEcg_classify_trained_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Control_BaseAddress = ConfigPtr->Control_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XEcg_classify_trained_Start(XEcg_classify_trained *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XEcg_classify_trained_ReadReg(InstancePtr->Control_BaseAddress, XECG_CLASSIFY_TRAINED_CONTROL_ADDR_AP_CTRL) & 0x80;
    XEcg_classify_trained_WriteReg(InstancePtr->Control_BaseAddress, XECG_CLASSIFY_TRAINED_CONTROL_ADDR_AP_CTRL, Data | 0x01);
}

u32 XEcg_classify_trained_IsDone(XEcg_classify_trained *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XEcg_classify_trained_ReadReg(InstancePtr->Control_BaseAddress, XECG_CLASSIFY_TRAINED_CONTROL_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XEcg_classify_trained_IsIdle(XEcg_classify_trained *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XEcg_classify_trained_ReadReg(InstancePtr->Control_BaseAddress, XECG_CLASSIFY_TRAINED_CONTROL_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XEcg_classify_trained_IsReady(XEcg_classify_trained *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XEcg_classify_trained_ReadReg(InstancePtr->Control_BaseAddress, XECG_CLASSIFY_TRAINED_CONTROL_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XEcg_classify_trained_EnableAutoRestart(XEcg_classify_trained *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XEcg_classify_trained_WriteReg(InstancePtr->Control_BaseAddress, XECG_CLASSIFY_TRAINED_CONTROL_ADDR_AP_CTRL, 0x80);
}

void XEcg_classify_trained_DisableAutoRestart(XEcg_classify_trained *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XEcg_classify_trained_WriteReg(InstancePtr->Control_BaseAddress, XECG_CLASSIFY_TRAINED_CONTROL_ADDR_AP_CTRL, 0);
}

void XEcg_classify_trained_Set_features(XEcg_classify_trained *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XEcg_classify_trained_WriteReg(InstancePtr->Control_BaseAddress, XECG_CLASSIFY_TRAINED_CONTROL_ADDR_FEATURES_DATA, (u32)(Data));
    XEcg_classify_trained_WriteReg(InstancePtr->Control_BaseAddress, XECG_CLASSIFY_TRAINED_CONTROL_ADDR_FEATURES_DATA + 4, (u32)(Data >> 32));
}

u64 XEcg_classify_trained_Get_features(XEcg_classify_trained *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XEcg_classify_trained_ReadReg(InstancePtr->Control_BaseAddress, XECG_CLASSIFY_TRAINED_CONTROL_ADDR_FEATURES_DATA);
    Data += (u64)XEcg_classify_trained_ReadReg(InstancePtr->Control_BaseAddress, XECG_CLASSIFY_TRAINED_CONTROL_ADDR_FEATURES_DATA + 4) << 32;
    return Data;
}

void XEcg_classify_trained_Set_probabilities(XEcg_classify_trained *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XEcg_classify_trained_WriteReg(InstancePtr->Control_BaseAddress, XECG_CLASSIFY_TRAINED_CONTROL_ADDR_PROBABILITIES_DATA, (u32)(Data));
    XEcg_classify_trained_WriteReg(InstancePtr->Control_BaseAddress, XECG_CLASSIFY_TRAINED_CONTROL_ADDR_PROBABILITIES_DATA + 4, (u32)(Data >> 32));
}

u64 XEcg_classify_trained_Get_probabilities(XEcg_classify_trained *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XEcg_classify_trained_ReadReg(InstancePtr->Control_BaseAddress, XECG_CLASSIFY_TRAINED_CONTROL_ADDR_PROBABILITIES_DATA);
    Data += (u64)XEcg_classify_trained_ReadReg(InstancePtr->Control_BaseAddress, XECG_CLASSIFY_TRAINED_CONTROL_ADDR_PROBABILITIES_DATA + 4) << 32;
    return Data;
}

void XEcg_classify_trained_Set_predicted_class(XEcg_classify_trained *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XEcg_classify_trained_WriteReg(InstancePtr->Control_BaseAddress, XECG_CLASSIFY_TRAINED_CONTROL_ADDR_PREDICTED_CLASS_DATA, (u32)(Data));
    XEcg_classify_trained_WriteReg(InstancePtr->Control_BaseAddress, XECG_CLASSIFY_TRAINED_CONTROL_ADDR_PREDICTED_CLASS_DATA + 4, (u32)(Data >> 32));
}

u64 XEcg_classify_trained_Get_predicted_class(XEcg_classify_trained *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XEcg_classify_trained_ReadReg(InstancePtr->Control_BaseAddress, XECG_CLASSIFY_TRAINED_CONTROL_ADDR_PREDICTED_CLASS_DATA);
    Data += (u64)XEcg_classify_trained_ReadReg(InstancePtr->Control_BaseAddress, XECG_CLASSIFY_TRAINED_CONTROL_ADDR_PREDICTED_CLASS_DATA + 4) << 32;
    return Data;
}

void XEcg_classify_trained_InterruptGlobalEnable(XEcg_classify_trained *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XEcg_classify_trained_WriteReg(InstancePtr->Control_BaseAddress, XECG_CLASSIFY_TRAINED_CONTROL_ADDR_GIE, 1);
}

void XEcg_classify_trained_InterruptGlobalDisable(XEcg_classify_trained *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XEcg_classify_trained_WriteReg(InstancePtr->Control_BaseAddress, XECG_CLASSIFY_TRAINED_CONTROL_ADDR_GIE, 0);
}

void XEcg_classify_trained_InterruptEnable(XEcg_classify_trained *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XEcg_classify_trained_ReadReg(InstancePtr->Control_BaseAddress, XECG_CLASSIFY_TRAINED_CONTROL_ADDR_IER);
    XEcg_classify_trained_WriteReg(InstancePtr->Control_BaseAddress, XECG_CLASSIFY_TRAINED_CONTROL_ADDR_IER, Register | Mask);
}

void XEcg_classify_trained_InterruptDisable(XEcg_classify_trained *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XEcg_classify_trained_ReadReg(InstancePtr->Control_BaseAddress, XECG_CLASSIFY_TRAINED_CONTROL_ADDR_IER);
    XEcg_classify_trained_WriteReg(InstancePtr->Control_BaseAddress, XECG_CLASSIFY_TRAINED_CONTROL_ADDR_IER, Register & (~Mask));
}

void XEcg_classify_trained_InterruptClear(XEcg_classify_trained *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XEcg_classify_trained_WriteReg(InstancePtr->Control_BaseAddress, XECG_CLASSIFY_TRAINED_CONTROL_ADDR_ISR, Mask);
}

u32 XEcg_classify_trained_InterruptGetEnabled(XEcg_classify_trained *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XEcg_classify_trained_ReadReg(InstancePtr->Control_BaseAddress, XECG_CLASSIFY_TRAINED_CONTROL_ADDR_IER);
}

u32 XEcg_classify_trained_InterruptGetStatus(XEcg_classify_trained *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XEcg_classify_trained_ReadReg(InstancePtr->Control_BaseAddress, XECG_CLASSIFY_TRAINED_CONTROL_ADDR_ISR);
}

