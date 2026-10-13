from enum import StrEnum

class ServerMessages(StrEnum):
    """Messages we receive FROM Unity clients."""
    CLIENT_READY = "CLIENT_READY"
    OBJECT_EVENT = "OBJECT_EVENT"
    # Explicit request to run the pipeline again after a PIPELINE_ERROR. A plain
    # reconnect never restarts a failed run on its own any more.
    RETRY = "RETRY"
 
class ClientMessages(StrEnum):
    """Messages we send TO Unity clients."""
    SCENE_INIT     = "SCENE_INIT"
    OBJECT_SPAWN   = "OBJECT_SPAWN"
    OBJECT_UPDATE  = "OBJECT_UPDATE"
    OBJECT_DESTROY = "OBJECT_DESTROY"
    SCENE_PARAM    = "SCENE_PARAM"
    PROGRESS       = "PROGRESS"
    PIPELINE_ERROR = "PIPELINE_ERROR"
    PIPELINE_CANCELLED = "PIPELINE_CANCELLED"
 