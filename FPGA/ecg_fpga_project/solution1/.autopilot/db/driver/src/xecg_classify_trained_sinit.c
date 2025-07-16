// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.1.2 (64-bit)
// Tool Version Limit: 2024.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef __linux__

#include "xstatus.h"
#ifdef SDT
#include "xparameters.h"
#endif
#include "xecg_classify_trained.h"

extern XEcg_classify_trained_Config XEcg_classify_trained_ConfigTable[];

#ifdef SDT
XEcg_classify_trained_Config *XEcg_classify_trained_LookupConfig(UINTPTR BaseAddress) {
	XEcg_classify_trained_Config *ConfigPtr = NULL;

	int Index;

	for (Index = (u32)0x0; XEcg_classify_trained_ConfigTable[Index].Name != NULL; Index++) {
		if (!BaseAddress || XEcg_classify_trained_ConfigTable[Index].Control_BaseAddress == BaseAddress) {
			ConfigPtr = &XEcg_classify_trained_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XEcg_classify_trained_Initialize(XEcg_classify_trained *InstancePtr, UINTPTR BaseAddress) {
	XEcg_classify_trained_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XEcg_classify_trained_LookupConfig(BaseAddress);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XEcg_classify_trained_CfgInitialize(InstancePtr, ConfigPtr);
}
#else
XEcg_classify_trained_Config *XEcg_classify_trained_LookupConfig(u16 DeviceId) {
	XEcg_classify_trained_Config *ConfigPtr = NULL;

	int Index;

	for (Index = 0; Index < XPAR_XECG_CLASSIFY_TRAINED_NUM_INSTANCES; Index++) {
		if (XEcg_classify_trained_ConfigTable[Index].DeviceId == DeviceId) {
			ConfigPtr = &XEcg_classify_trained_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XEcg_classify_trained_Initialize(XEcg_classify_trained *InstancePtr, u16 DeviceId) {
	XEcg_classify_trained_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XEcg_classify_trained_LookupConfig(DeviceId);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XEcg_classify_trained_CfgInitialize(InstancePtr, ConfigPtr);
}
#endif

#endif

