// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.1.2 (64-bit)
// Tool Version Limit: 2024.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef XECG_CLASSIFY_TRAINED_H
#define XECG_CLASSIFY_TRAINED_H

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#ifndef __linux__
#include "xil_types.h"
#include "xil_assert.h"
#include "xstatus.h"
#include "xil_io.h"
#else
#include <stdint.h>
#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stddef.h>
#endif
#include "xecg_classify_trained_hw.h"

/**************************** Type Definitions ******************************/
#ifdef __linux__
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#else
typedef struct {
#ifdef SDT
    char *Name;
#else
    u16 DeviceId;
#endif
    u64 Control_BaseAddress;
} XEcg_classify_trained_Config;
#endif

typedef struct {
    u64 Control_BaseAddress;
    u32 IsReady;
} XEcg_classify_trained;

typedef u32 word_type;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XEcg_classify_trained_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XEcg_classify_trained_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XEcg_classify_trained_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XEcg_classify_trained_ReadReg(BaseAddress, RegOffset) \
    *(volatile u32*)((BaseAddress) + (RegOffset))

#define Xil_AssertVoid(expr)    assert(expr)
#define Xil_AssertNonvoid(expr) assert(expr)

#define XST_SUCCESS             0
#define XST_DEVICE_NOT_FOUND    2
#define XST_OPEN_DEVICE_FAILED  3
#define XIL_COMPONENT_IS_READY  1
#endif

/************************** Function Prototypes *****************************/
#ifndef __linux__
#ifdef SDT
int XEcg_classify_trained_Initialize(XEcg_classify_trained *InstancePtr, UINTPTR BaseAddress);
XEcg_classify_trained_Config* XEcg_classify_trained_LookupConfig(UINTPTR BaseAddress);
#else
int XEcg_classify_trained_Initialize(XEcg_classify_trained *InstancePtr, u16 DeviceId);
XEcg_classify_trained_Config* XEcg_classify_trained_LookupConfig(u16 DeviceId);
#endif
int XEcg_classify_trained_CfgInitialize(XEcg_classify_trained *InstancePtr, XEcg_classify_trained_Config *ConfigPtr);
#else
int XEcg_classify_trained_Initialize(XEcg_classify_trained *InstancePtr, const char* InstanceName);
int XEcg_classify_trained_Release(XEcg_classify_trained *InstancePtr);
#endif

void XEcg_classify_trained_Start(XEcg_classify_trained *InstancePtr);
u32 XEcg_classify_trained_IsDone(XEcg_classify_trained *InstancePtr);
u32 XEcg_classify_trained_IsIdle(XEcg_classify_trained *InstancePtr);
u32 XEcg_classify_trained_IsReady(XEcg_classify_trained *InstancePtr);
void XEcg_classify_trained_EnableAutoRestart(XEcg_classify_trained *InstancePtr);
void XEcg_classify_trained_DisableAutoRestart(XEcg_classify_trained *InstancePtr);

void XEcg_classify_trained_Set_features(XEcg_classify_trained *InstancePtr, u64 Data);
u64 XEcg_classify_trained_Get_features(XEcg_classify_trained *InstancePtr);
void XEcg_classify_trained_Set_probabilities(XEcg_classify_trained *InstancePtr, u64 Data);
u64 XEcg_classify_trained_Get_probabilities(XEcg_classify_trained *InstancePtr);
void XEcg_classify_trained_Set_predicted_class(XEcg_classify_trained *InstancePtr, u64 Data);
u64 XEcg_classify_trained_Get_predicted_class(XEcg_classify_trained *InstancePtr);

void XEcg_classify_trained_InterruptGlobalEnable(XEcg_classify_trained *InstancePtr);
void XEcg_classify_trained_InterruptGlobalDisable(XEcg_classify_trained *InstancePtr);
void XEcg_classify_trained_InterruptEnable(XEcg_classify_trained *InstancePtr, u32 Mask);
void XEcg_classify_trained_InterruptDisable(XEcg_classify_trained *InstancePtr, u32 Mask);
void XEcg_classify_trained_InterruptClear(XEcg_classify_trained *InstancePtr, u32 Mask);
u32 XEcg_classify_trained_InterruptGetEnabled(XEcg_classify_trained *InstancePtr);
u32 XEcg_classify_trained_InterruptGetStatus(XEcg_classify_trained *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
