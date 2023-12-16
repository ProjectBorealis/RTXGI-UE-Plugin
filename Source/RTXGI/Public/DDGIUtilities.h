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
#include "UObject/ObjectMacros.h"
#include "Engine/EngineTypes.h"
#include "Components/SceneComponent.h"
#include "RHI.h"
#include "RHIResources.h"
#include "RendererInterface.h"

/*
* Helper class to time sections of the GPU work.
* Buffers multiple frames to avoid waiting on the GPU so times are a little lagged.
* Original code from ScenePrivate.h
*/
class FLatentGPUTimerDDGI
{
	FRenderQueryPoolRHIRef TimerQueryPool;
public:
	static const int NumBufferedFrames = 4;

	FLatentGPUTimerDDGI(FRenderQueryPoolRHIRef InTimerQueryPool);
	~FLatentGPUTimerDDGI()
	{
		Release();
	}

	void Release();
	/** Sets the TimerQueryPool. */
	void SetPool(FRenderQueryPoolRHIRef InTimerQueryPool);
	/** Retrieves the most recently ready query results. */
	bool Tick(FRHICommandListImmediate& RHICmdList);
	/** Kicks off the query for the start of the rendering you're timing. */
	void Begin(FRHICommandListImmediate& RHICmdList);
	/** Kicks off the query for the end of the rendering you're timing. */
	void End(FRHICommandListImmediate& RHICmdList);

	/** Returns the most recent time in ms. */
	float GetTimeMS();
	/** Gets the average time in ms. Average is tracked over AvgSamples. */
	float GetAverageTimeMS();

private:
	//Average Tracking;
	int AvgSamples;
	TArray<float> TimeSamples;
	float TotalTime;
	int SampleIndex;

	int QueryIndex;
	bool QueriesInFlight[NumBufferedFrames];
	FRHIPooledRenderQuery StartQueries[NumBufferedFrames];
	FRHIPooledRenderQuery EndQueries[NumBufferedFrames];
	FGraphEventRef QuerySubmittedFences[NumBufferedFrames];
};