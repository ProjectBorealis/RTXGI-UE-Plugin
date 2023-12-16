/*
* Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

using UnrealBuildTool;
using System.IO;

public class RTXGI : ModuleRules
{
	private string ModulePath
	{
		get { return ModuleDirectory; }
	}

	protected virtual bool IsSupportedPlatform(ReadOnlyTargetRules Target)
	{
		return Target.Platform == UnrealTargetPlatform.Win64;
	}

	public RTXGI(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

		bool bPlatformSupportsRTXGI = IsSupportedPlatform(Target);
		PublicDefinitions.Add("WITH_RTXGI=" + (bPlatformSupportsRTXGI ? '1' : '0'));

		PrivateDependencyModuleNames.AddRange(new string[]
		{
			"CoreUObject",
			"Engine",
			"RenderCore",
			"Renderer",
			"DeveloperSettings",
			"RHI",
		});

		PublicDependencyModuleNames.AddRange(new string[]
		{
			"Core",
			"Projects"
		});

		PrivateIncludePaths.AddRange(new string[]
		{
			EngineDirectory + "/Source/Runtime/Renderer/Private",
			EngineDirectory + "/Source/Runtime/RenderCore/Public",
		});

		PublicIncludePaths.AddRange(new string[]
		{
			"../Shaders/Shared"
		});
	}
}
