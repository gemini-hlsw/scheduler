# Resource

The GPP Resource services allows other services in the GPP environment to 
know what is available in the telescope at a giving night, from gratings to 
the status of the status of an instrument. 

This service act like a client for the real Resource and the emulated version for OCS,
that is based in several text files compiled from old OCS systems.
The original files can be checked at `/scheduler/services/resource/src`and the ones
actually used in the Scheduler are in the same path at `validation`.


## Resource structures
:::scheduler.services.resource.ResourceService

:::scheduler.services.resource.FileBasedResourceService

:::scheduler.services.resource.OcsResourceService

:::scheduler.services.resource.resource_manager.ResourceManager

:::scheduler.services.resource.NightConfiguration