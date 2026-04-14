from enum import StrEnum

class ServerMessages(StrEnum):
    """Messages we receive FROM Unity clients."""
    CLIENT_READY = "CLIENT_READY"
    OBJECT_EVENT = "OBJECT_EVENT"
 
class ClientMessages(StrEnum):
    """Messages we send TO Unity clients."""
    SCENE_INIT     = "SCENE_INIT"
    OBJECT_SPAWN   = "OBJECT_SPAWN"
    OBJECT_UPDATE  = "OBJECT_UPDATE"
    OBJECT_DESTROY = "OBJECT_DESTROY"
    SCENE_PARAM    = "SCENE_PARAM"
    PROGRESS       = "PROGRESS"
    PIPELINE_ERROR = "PIPELINE_ERROR"
 