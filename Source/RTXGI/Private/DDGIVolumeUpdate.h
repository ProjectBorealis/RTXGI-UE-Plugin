/*
* Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include "CoreMinimal.h"

class FViewInfo;
class FRDGBuilder;
class UDDGIVolumeComponent;
class FScene;

#ifndef WITH_RTXGI
#define WITH_RTXGI 0
#endif

namespace DDGIVolumeUpdate
{
	void Startup();
	void Shutdown();

	void DDGIInitLoadedVolumes_RenderThread(FRDGBuilder& GraphBuilder);
	void DDGIUpdatePerFrame_RenderThread(const FScene& Scene, const FViewInfo& View, FRDGBuilder& GraphBuilder);
}
